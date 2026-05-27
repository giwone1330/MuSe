"""
Weight extraction, variance profiling, and fused weight precomputation
for the MUSE FHE inference pipeline.

Public API
----------
extract_weights_from_hf_model(model, config) → dict
profile_layernorm_variances(model, tokenizer, input_text, ...) → dict
precompute_fused_weights(block_weights, config, T) → dict
precompute_all_blocks(all_block_weights, config, T) → List[dict]
"""

import numpy as np
import torch
from .muse_mixing import precompute_mixing_P_matrices


# ============================================================================
# Weight extraction
# ============================================================================

def extract_weights_from_hf_model(model, config):
    """
    Extract all weights from a HuggingFace GPT2LMHeadModel with MUSE architecture
    (MixingLayer attn + PolyMlp mlp).

    HuggingFace Conv1D stores weights as (nx, nf).
    We transpose to (nf, nx) = (out_dim, in_dim) for FHE multiply_matrix.

    Returns
    -------
    dict with:
        'blocks'       : List[dict] — per-block weight dicts
        'ln_f_weight'  : np.ndarray (D,)
        'ln_f_bias'    : np.ndarray (D,)
        'lm_head_weight': np.ndarray (vocab, D)
        'wte_weight'   : np.ndarray (vocab, D)
        'wpe_weight'   : np.ndarray (max_pos, D)
    """
    sd = model.state_dict()

    def get_np(key):
        if key in sd:
            return sd[key].detach().cpu().float().numpy()
        return None

    def get_np_required(key):
        assert key in sd, f"Missing weight: {key}"
        return sd[key].detach().cpu().float().numpy()

    wte_weight = get_np_required("transformer.wte.weight")
    wpe_weight = get_np_required("transformer.wpe.weight")
    ln_f_weight = get_np_required("transformer.ln_f.weight")
    ln_f_bias = get_np_required("transformer.ln_f.bias")

    lm_head_weight = get_np("lm_head.weight")
    if lm_head_weight is None:
        lm_head_weight = wte_weight.copy()

    blocks = []
    for l in range(config.num_blocks):
        prefix = f"transformer.h.{l}"
        D = config.embedding_dim
        Hm = config.num_heads * config.max_mixing_length

        # c_attn: Conv1D (D, H*m + D) → .T = (H*m + D, D)
        # Split as [A (H*m), C (D)]
        c_attn_w = get_np_required(f"{prefix}.attn.c_attn.weight").T.copy()
        c_attn_A_w = c_attn_w[:Hm, :]     # (H*m, D)
        c_attn_C_w = c_attn_w[Hm:, :]     # (D, D)

        c_attn_b = get_np(f"{prefix}.attn.c_attn.bias")
        c_attn_A_b = c_attn_b[:Hm] if c_attn_b is not None else None
        c_attn_C_b = c_attn_b[Hm:] if c_attn_b is not None else None

        # c_proj (attn output): Conv1D (D, D) → .T = (D, D)
        c_proj_attn_w = get_np_required(f"{prefix}.attn.c_proj.weight").T.copy()
        c_proj_attn_b = get_np(f"{prefix}.attn.c_proj.bias")

        # PolyMlp weights (all Conv1D, transpose to (out_dim, in_dim))
        c_fc_w = get_np_required(f"{prefix}.mlp.c_fc.weight").T.copy()
        c_fc_b = get_np(f"{prefix}.mlp.c_fc.bias")

        c_fc1_w = get_np_required(f"{prefix}.mlp.c_fc1.weight").T.copy()
        c_fc1_b = get_np(f"{prefix}.mlp.c_fc1.bias")

        c_fc2_w = get_np_required(f"{prefix}.mlp.c_fc2.weight").T.copy()
        c_fc2_b = get_np(f"{prefix}.mlp.c_fc2.bias")

        c_proj_mlp_w = get_np_required(f"{prefix}.mlp.c_proj.weight").T.copy()
        c_proj_mlp_b = get_np(f"{prefix}.mlp.c_proj.bias")

        alpha_raw = get_np(f"{prefix}.mlp.alpha")
        alpha_val = float(
            alpha_raw.item() if alpha_raw is not None and alpha_raw.ndim == 0
            else (alpha_raw[0] if alpha_raw is not None else 1.0)
        )

        ln_1_w = get_np_required(f"{prefix}.ln_1.weight")
        ln_1_b = get_np_required(f"{prefix}.ln_1.bias")
        ln_2_w = get_np_required(f"{prefix}.ln_2.weight")
        ln_2_b = get_np_required(f"{prefix}.ln_2.bias")

        blocks.append({
            'c_attn_A_weight': c_attn_A_w,
            'c_attn_A_bias': c_attn_A_b,
            'c_attn_C_weight': c_attn_C_w,
            'c_attn_C_bias': c_attn_C_b,
            'c_proj_attn_weight': c_proj_attn_w,
            'c_proj_attn_bias': c_proj_attn_b,
            'c_fc_weight': c_fc_w,
            'c_fc_bias': c_fc_b,
            'c_fc1_weight': c_fc1_w,
            'c_fc1_bias': c_fc1_b,
            'c_fc2_weight': c_fc2_w,
            'c_fc2_bias': c_fc2_b,
            'c_proj_mlp_weight': c_proj_mlp_w,
            'c_proj_mlp_bias': c_proj_mlp_b,
            'alpha': alpha_val,
            'ln_1_weight': ln_1_w,
            'ln_1_bias': ln_1_b,
            'ln_2_weight': ln_2_w,
            'ln_2_bias': ln_2_b,
        })

        print(f"  Block {l}: c_attn_A={c_attn_A_w.shape}, c_attn_C={c_attn_C_w.shape}, "
              f"c_fc={c_fc_w.shape}, c_fc1={c_fc1_w.shape}, "
              f"c_fc2={c_fc2_w.shape}, alpha={alpha_val:.4f}")

    return {
        'blocks': blocks,
        'wte_weight': wte_weight,
        'wpe_weight': wpe_weight,
        'ln_f_weight': ln_f_weight,
        'ln_f_bias': ln_f_bias,
        'lm_head_weight': lm_head_weight,
    }


# ============================================================================
# Variance profiling
# ============================================================================

def profile_layernorm_variances(model, tokenizer, input_text,
                                 margin_low=0.5, margin_high=2.0):
    """
    Profile per-LayerNorm input variance on a representative plaintext input.

    Run once before FHE inference to get tight min_var/max_var bounds for
    he_layernorm. These bounds depend on the model and input distribution,
    NOT on any private data.

    Parameters
    ----------
    margin_low  : float < 1 — safety margin below actual min variance.
    margin_high : float > 1 — safety margin above actual max variance.

    Returns
    -------
    dict with:
        'per_ln' : List[dict] — one entry per LayerNorm in order:
                   [b0_ln1, b0_ln2, ..., bN_ln1, bN_ln2, ln_f]
                   Each dict: block, name, var_min, var_max, min_var, max_var
        'global_min_var' : float
        'global_max_var' : float
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        T = input_ids.shape[1]

        wte = model.transformer.wte(input_ids)
        wpe = model.transformer.wpe(
            torch.arange(T, device=input_ids.device).unsqueeze(0))
        hidden = wte + wpe

        per_ln = []
        for l, block in enumerate(model.transformer.h):
            ln1_in = hidden[0].cpu().numpy()
            vars_ln1 = np.array([ln1_in[t].var() for t in range(T)])
            mn, mx = float(vars_ln1.min()), float(vars_ln1.max())
            per_ln.append({
                'block': l, 'name': 'ln_1',
                'var_min': mn, 'var_max': mx,
                'min_var': mn * margin_low,
                'max_var': mx * margin_high,
            })

            ln1_out = block.ln_1(hidden)
            attn_out = block.attn(ln1_out)[0]
            after_attn = hidden + attn_out

            ln2_in = after_attn[0].cpu().numpy()
            vars_ln2 = np.array([ln2_in[t].var() for t in range(T)])
            mn, mx = float(vars_ln2.min()), float(vars_ln2.max())
            per_ln.append({
                'block': l, 'name': 'ln_2',
                'var_min': mn, 'var_max': mx,
                'min_var': mn * margin_low,
                'max_var': mx * margin_high,
            })

            ln2_out = block.ln_2(after_attn)
            mlp_out = block.mlp(ln2_out)
            hidden = after_attn + mlp_out

        lnf_in = hidden[0].cpu().numpy()
        vars_lnf = np.array([lnf_in[t].var() for t in range(T)])
        mn, mx = float(vars_lnf.min()), float(vars_lnf.max())
        per_ln.append({
            'block': 'final', 'name': 'ln_f',
            'var_min': mn, 'var_max': mx,
            'min_var': mn * margin_low,
            'max_var': mx * margin_high,
        })

    return {
        'per_ln': per_ln,
        'global_min_var': min(e['min_var'] for e in per_ln),
        'global_max_var': max(e['max_var'] for e in per_ln),
    }


# ============================================================================
# Fused weight precomputation
# ============================================================================

def precompute_fused_weights(block_weights, config, T):
    """
    Precompute all fused weight matrices for one block.

    MixingLayer fusion:
        P_h[h][i] (T, D) — fused extract + context_norm + transformD
        D_h[i, :] = enc_rows[i] @ P_h[h][i]^T

    PolyMlp fusion:
        W_23 = W_fc2 @ W_fc1   (D_hid, D)
        out2 = x @ W_23^T

    Parameters
    ----------
    block_weights : dict — one block's weights from extract_weights_from_hf_model.
    config        : FHEConfig
    T             : int — sequence length

    Returns
    -------
    dict with:
        'P_h'       : List[H lists] each containing T matrices of shape (T, D)
        'W_C_h'     : List[H] of (hs, D) matrices
        'W_23'      : (D_hid, D) — fused W_fc2 @ W_fc1
        'W_23_bias' : (D_hid,) or None
        'original'  : block_weights (passthrough for non-fused ops)
    """
    D = config.embedding_dim
    H = config.num_heads
    m = config.max_mixing_length
    hs = config.head_size
    w = block_weights

    W_A_full = w['c_attn_A_weight']  # (H*m, D)
    W_C_full = w['c_attn_C_weight']  # (D, D)

    P_h_all = []
    W_C_h_all = []
    for h in range(H):
        W_A_h = W_A_full[h * m:(h + 1) * m, :]   # (m, D)
        P_h_all.append(precompute_mixing_P_matrices(W_A_h, T, m, D))
        W_C_h_all.append(W_C_full[h * hs:(h + 1) * hs, :])   # (hs, D)

    # Fused W_23 = W_fc2 @ W_fc1
    W_fc1 = w['c_fc1_weight']   # (D_bot, D)
    W_fc2 = w['c_fc2_weight']   # (D_hid, D_bot)
    W_23 = W_fc2 @ W_fc1        # (D_hid, D)

    W_23_bias = None
    if w['c_fc1_bias'] is not None or w['c_fc2_bias'] is not None:
        b1 = w['c_fc1_bias'] if w['c_fc1_bias'] is not None else np.zeros(W_fc1.shape[0])
        b2 = w['c_fc2_bias'] if w['c_fc2_bias'] is not None else np.zeros(W_fc2.shape[0])
        W_23_bias = b1 @ W_fc2.T + b2

    return {
        'P_h': P_h_all,
        'W_C_h': W_C_h_all,
        'W_23': W_23,
        'W_23_bias': W_23_bias,
        'original': w,
    }


def precompute_all_blocks(all_block_weights, config, T):
    """Precompute fused weights for every block in the model."""
    return [precompute_fused_weights(w, config, T) for w in all_block_weights]
