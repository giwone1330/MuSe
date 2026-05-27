"""
Full FHE MUSE inference pipeline.

run_fhe_muse_inference(model, tokenizer, input_text, config, var_profile, verbose)

Encryption boundary:
    plaintext  → [wte + wpe] → encrypt → [MUSE blocks × N] → decrypt → [ln_f + lm_head] → plaintext

D packing: enc_rows is a List[Ciphertext] of length T, each with D=512 slots.
"""

import time
import numpy as np
import torch

from .config import FHEConfig
from .engine import create_fhe_engine, create_keys
from .weights import (
    extract_weights_from_hf_model,
    profile_layernorm_variances,
    precompute_all_blocks,
)
from .muse_block import MuseBlock_FHE


def run_fhe_muse_inference(
    model,
    tokenizer,
    input_text: str = "417+384=801",
    config: FHEConfig = None,
    var_profile: dict = None,
    verbose: bool = True,
):
    """
    Full FHE inference for a MUSE language model.

    Steps:
        1. Tokenize input
        2. PyTorch reference forward pass
        3. Extract + fuse weights (plaintext precomputation)
        4. Create FHE engine and keys
        5. Encrypt embeddings (wte + wpe output) — D packing
        6. Run all MUSE blocks in FHE
        7. Decrypt hidden states
        8. Plaintext ln_f + lm_head; compare predictions

    Parameters
    ----------
    model       : HuggingFace GPT2LMHeadModel (MUSE variant)
    tokenizer   : HuggingFace tokenizer
    input_text  : str
    config      : FHEConfig or None (uses defaults)
    var_profile : dict from profile_layernorm_variances, or None.
                  Providing this improves LayerNorm accuracy significantly.
    verbose     : bool

    Returns
    -------
    dict with: logits_pt, logits_fhe, pred_ids_pt, pred_ids_fhe,
               error_profile, block_timings, t_fhe_total, t_pytorch, t_encrypt
    """
    if config is None:
        config = FHEConfig()

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 80)
    log("FHE MUSE Inference (fused, D packing)")
    log("=" * 80)
    log(f"Input: \"{input_text}\"")
    log(f"\n{config}")

    # ── 1. Tokenize ──
    log("\n" + "─" * 80)
    log("[1/8] Tokenizing...")
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    T = input_ids.shape[1]
    log(f"  Tokens: {input_ids[0].tolist()}  (T={T})")

    # ── 2. PyTorch reference ──
    log("\n" + "─" * 80)
    log("[2/8] PyTorch reference forward pass...")
    t0 = time.time()
    X_embed_np, block_refs, logits_pt_np = _pytorch_forward(model, input_ids)
    t_pytorch = time.time() - t0
    log(f"  Done in {t_pytorch:.4f}s  |  embedding: {X_embed_np.shape}")

    # ── 3. Extract + fuse weights ──
    log("\n" + "─" * 80)
    log("[3/8] Extracting and fusing weights...")
    t0 = time.time()
    weights = extract_weights_from_hf_model(model, config)
    all_block_weights = weights['blocks']
    all_fused = precompute_all_blocks(all_block_weights, config, T)
    log(f"  Fused weights ready in {time.time()-t0:.2f}s")

    # ── 4. FHE engine + keys ──
    log("\n" + "─" * 80)
    log("[4/8] Creating FHE engine and keys...")
    t0 = time.time()
    engine = create_fhe_engine(config, track=True)
    log(f"  Engine: slot_count={engine.slot_count}, max_level={engine.max_level}")
    secret_key = engine.create_secret_key()
    keys = create_keys(engine, secret_key, config)
    log(f"  Keys generated in {time.time()-t0:.2f}s")

    # ── 5. Encrypt embeddings ──
    log("\n" + "─" * 80)
    log("[5/8] Encrypting embedding output (D packing)...")
    t0 = time.time()
    enc_rows = [engine.encrypt(X_embed_np[t].tolist(), secret_key) for t in range(T)]
    t_encrypt = time.time() - t0
    log(f"  Encrypted {T} × {config.embedding_dim} in {t_encrypt:.4f}s  "
        f"|  initial level: {enc_rows[0].level}")

    # ── 6. Build and run FHE blocks ──
    log("\n" + "─" * 80)
    log("[6/8] Building FHE MUSE blocks...")
    fhe_blocks = []
    for l in range(config.num_blocks):
        ln1_bounds = _get_ln_bounds(var_profile, l * 2,     config)
        ln2_bounds = _get_ln_bounds(var_profile, l * 2 + 1, config)

        fhe_blocks.append(MuseBlock_FHE(
            engine=engine, keys=keys, config=config,
            fused_weights=all_fused[l],
            ln_1_var_bounds=ln1_bounds,
            ln_2_var_bounds=ln2_bounds,
        ))
        if ln1_bounds:
            log(f"  Block {l}  ln1: min_var={ln1_bounds['min_var']:.4e}, "
                f"max_var={ln1_bounds['max_var']:.4e}")

    log("\n" + "─" * 80)
    log("[7/8] Running FHE forward pass through all MUSE blocks...")
    error_profile = []
    block_timings = []
    enc_hidden = enc_rows
    t_fhe_start = time.time()

    for l in range(config.num_blocks):
        log(f"\n{'='*60}")
        log(f"  Block {l} / {config.num_blocks}")
        log(f"{'='*60}")
        t_blk = time.time()

        enc_hidden = fhe_blocks[l].forward(enc_hidden, T, block_idx=l)

        t_blk = time.time() - t_blk
        block_timings.append(t_blk)
        log(f"  Block {l} latency: {t_blk:.2f}s")

        # Per-block error vs PyTorch reference
        Y_fhe = _decrypt_rows(engine, secret_key, enc_hidden, T, config.embedding_dim)
        Y_ref = block_refs[l]
        err = _compute_error(Y_ref, Y_fhe, l, t_blk, enc_hidden)
        error_profile.append(err)
        log(f"  max_abs={err['max_abs_error']:.4e}, "
            f"l2_rel={err['l2_rel_error']:.4e}, "
            f"levels={err['ciphertext_levels']}")

    t_fhe_total = time.time() - t_fhe_start

    # ── 7. Decrypt ──
    log("\n" + "─" * 80)
    log("[8/8] Decrypt + plaintext ln_f + lm_head...")
    Y_final = _decrypt_rows(engine, secret_key, enc_hidden, T, config.embedding_dim)

    # Plaintext LayerNorm (ln_f)
    ln_f_w = weights['ln_f_weight']
    ln_f_b = weights['ln_f_bias']
    mean = Y_final.mean(axis=-1, keepdims=True)
    var = Y_final.var(axis=-1, keepdims=True)
    Y_normed = (Y_final - mean) / np.sqrt(var + 1e-5)
    Y_normed = Y_normed * ln_f_w + ln_f_b

    # lm_head (plaintext)
    lm_head_w = weights['lm_head_weight']  # (vocab, D)
    logits_fhe = Y_normed @ lm_head_w.T    # (T, vocab)

    pred_ids_pt = np.argmax(logits_pt_np, axis=-1).tolist()
    pred_ids_fhe = np.argmax(logits_fhe, axis=-1).tolist()
    agreement = sum(a == b for a, b in zip(pred_ids_pt, pred_ids_fhe))

    log(f"\n  PyTorch predicted: {pred_ids_pt}")
    log(f"  FHE     predicted: {pred_ids_fhe}")
    try:
        log(f"  PyTorch decoded:   {tokenizer.decode(pred_ids_pt)}")
        log(f"  FHE     decoded:   {tokenizer.decode(pred_ids_fhe)}")
    except Exception:
        pass
    log(f"  Token agreement:   {agreement}/{T}")

    # ── Summary ──
    log("\n" + "=" * 80)
    log("ERROR ACCUMULATION PROFILE")
    log("=" * 80)
    log(f"\n{'Block':>6}  {'MaxAbsErr':>12}  {'MeanAbsErr':>12}  {'L2RelErr':>12}  "
        f"{'Latency':>10}  {'Levels':>10}")
    log("─" * 80)
    for e in error_profile:
        lvl_range = f"{min(e['ciphertext_levels'])}-{max(e['ciphertext_levels'])}"
        log(f"  {e['block']:>4}  {e['max_abs_error']:>12.4e}  "
            f"{e['mean_abs_error']:>12.4e}  {e['l2_rel_error']:>12.4e}  "
            f"{e['latency_s']:>9.2f}s  {lvl_range:>10}")

    log(f"\n  Total FHE inference:  {t_fhe_total:.2f}s")
    log(f"  PyTorch reference:    {t_pytorch:.4f}s")
    log(f"  Speedup factor:       {t_fhe_total / max(t_pytorch, 1e-6):.0f}×")
    log(f"  Encryption overhead:  {t_encrypt:.4f}s")
    log("=" * 80)

    return {
        'logits_pt': logits_pt_np,
        'logits_fhe': logits_fhe,
        'pred_ids_pt': pred_ids_pt,
        'pred_ids_fhe': pred_ids_fhe,
        'error_profile': error_profile,
        'block_timings': block_timings,
        't_fhe_total': t_fhe_total,
        't_pytorch': t_pytorch,
        't_encrypt': t_encrypt,
        'token_agreement': agreement,
    }


# ============================================================================
# Private helpers
# ============================================================================

def _pytorch_forward(model, input_ids):
    """Run full HuggingFace model forward, capturing per-block hidden states."""
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        ids = input_ids.to(device)
        T = ids.shape[1]

        wte = model.transformer.wte(ids)
        wpe = model.transformer.wpe(torch.arange(T, device=device).unsqueeze(0))
        hidden = wte + wpe
        X_embed_np = hidden[0].cpu().numpy()

        block_refs = []
        for block in model.transformer.h:
            hidden = block(hidden)[0]
            block_refs.append(hidden[0].cpu().numpy())

        final_hidden = model.transformer.ln_f(hidden)
        logits = model.lm_head(final_hidden)
        logits_np = logits[0].cpu().numpy()

    return X_embed_np, block_refs, logits_np


def _decrypt_rows(engine, secret_key, enc_rows, T, D):
    """Decrypt T ciphertexts and stack into (T, D) numpy array."""
    result = np.zeros((T, D), dtype=np.float64)
    for t in range(T):
        dec = engine.decrypt(enc_rows[t], secret_key)
        result[t] = dec[:D]
    return result


def _compute_error(Y_ref, Y_fhe, block_idx, latency_s, enc_rows):
    abs_diff = np.abs(Y_ref - Y_fhe)
    rel_diff = abs_diff / (np.abs(Y_ref) + 1e-12)
    return {
        'block': block_idx,
        'max_abs_error': float(np.max(abs_diff)),
        'mean_abs_error': float(np.mean(abs_diff)),
        'max_rel_error': float(np.max(rel_diff)),
        'l2_rel_error': float(
            np.linalg.norm(Y_ref - Y_fhe) / (np.linalg.norm(Y_ref) + 1e-12)
        ),
        'latency_s': latency_s,
        'ciphertext_levels': [ct.level for ct in enc_rows],
    }


def _get_ln_bounds(var_profile, idx, config):
    """Look up per-LN variance bounds from profiling results."""
    if var_profile is None:
        return None
    per_ln = var_profile.get('per_ln', [])
    if idx < len(per_ln):
        e = per_ln[idx]
        return {'min_var': e['min_var'], 'max_var': e['max_var']}
    return None
