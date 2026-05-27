"""
Full FHE MUSE inference pipeline under N packing.

N packing: D column-ciphertexts of T slots each.

Public API
----------
run_fhe_muse_inference_n(model, tokenizer, input_text, config, var_profile, verbose=True)
    → predicted token string
"""

import gc
import time
import numpy as np
import torch

from .engine_n import (
    create_fhe_engine_n, create_keys_n, next_power_of_2,
    encrypt_matrix_n, decrypt_matrix_n, bootstrap_if_needed, bs_kwargs,
)
from .muse_block_n import MuseBlock_N
from .weights import extract_weights_from_hf_model


def run_fhe_muse_inference_n(model, tokenizer, input_text, config,
                               var_profile=None, verbose=True):
    """
    Run FHE MUSE inference on input_text under N packing.

    Returns
    -------
    predicted_token : str
    """
    if var_profile is None:
        var_profile = {}

    # ── Tokenise and embed ──
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        input_ids = inputs.input_ids
        T = input_ids.shape[1]
        slot_count = next_power_of_2(T)

        if verbose:
            print(f"Input: '{input_text}'  →  T={T}, slot_count={slot_count}")

        embeddings = model.transformer.wte(input_ids)    # (1, T, D)
        pos_emb = model.transformer.wpe(
            torch.arange(T).unsqueeze(0))                # (1, T, D)
        X = (embeddings + pos_emb).squeeze(0).numpy()   # (T, D)

    D = config.embedding_dim

    # ── Create FHE engine with slot_count = T (N packing) ──
    if verbose:
        print(f"Creating FHE engine: slot_count={slot_count}")
    engine, _ = create_fhe_engine_n(T, config, track=True)
    secret_key = engine.create_secret_key()
    keys = create_keys_n(engine, secret_key, config)
    keys['secret_key'] = secret_key

    # ── Extract weights ──
    if verbose:
        print("Extracting weights...")
    all_weights = extract_weights_from_hf_model(model, config)

    # ── Encrypt input matrix in N packing (D column-ciphertexts) ──
    if verbose:
        print(f"Encrypting {D} column-ciphertexts of {slot_count} slots...")
    enc_cols = encrypt_matrix_n(engine, secret_key, X, T, D, slot_count)

    # ── Run transformer blocks ──
    block_layer = MuseBlock_N(engine, keys, config, var_profile)
    num_blocks = config.num_blocks

    for block_idx in range(num_blocks):
        if verbose:
            print(f"\n=== Block {block_idx+1}/{num_blocks} ===")
        t_block = time.time()
        enc_cols_new = block_layer.forward(
            enc_cols,
            all_weights['blocks'][block_idx],
            T, slot_count,
            block_idx=block_idx,
        )
        del enc_cols
        enc_cols = enc_cols_new
        del enc_cols_new
        # Reset TrackedEngine diagnostic lists — each block appends ~1.4M entries
        # to level_trace/bootstrap_log; clearing prevents unbounded CPU RAM growth
        # which slows GC and masks VRAM freed-but-unreleased patterns
        if hasattr(engine, 'reset_counters'):
            engine.reset_counters()
        gc.collect()
        if verbose:
            print(f"=== Block {block_idx+1} done in {time.time()-t_block:.2f}s ===\n")

    # ── Final LayerNorm (ln_f) ──
    if verbose:
        print("Final LayerNorm...")
    from .layernorm_n import layernorm_n
    enc_cols = layernorm_n(
        engine, keys, config, enc_cols, D, T,
        gamma=all_weights['ln_f_weight'], beta=all_weights['ln_f_bias'],
    )

    # ── Decrypt and compute logits ──
    if verbose:
        print("Decrypting...")
    X_out = decrypt_matrix_n(engine, secret_key, enc_cols, T, D)  # (T, D)
    del enc_cols
    gc.collect()

    # Logits from last token: (D,) → multiply by lm_head weights (vocab_size, D)
    # lm_head_weight falls back to wte_weight (tied weights) for AutoModel
    lm_head_weight = all_weights['lm_head_weight']               # (vocab_size, D)
    logits = X_out[-1] @ lm_head_weight.T                        # (vocab_size,)

    predicted_id = int(np.argmax(logits))
    predicted_token = tokenizer.decode([predicted_id])

    if verbose:
        print(f"\nPredicted next token: '{predicted_token}' (id={predicted_id})")

    return predicted_token
