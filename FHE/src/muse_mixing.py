"""
FHE MUSE MixingLayer — fused implementation.

MixingLayer is the MUSE replacement for self-attention:
    A = c_attn_A(X)                     # (T, H*m)
    C = c_attn_C(X)                     # (T, D)
    For each head h:
        context_norm(A_h) → D_h (T, T)  # causal mixing matrix
        B_h = D_h @ C_h                  # (T, head_size)
    output = c_proj(concat(B_h))         # (T, D)

Fused version precomputes per-head P_h[i] matrices that collapse
extract_head + context_norm + transformD into a single multiply_matrix:
    D_h[i, :] = enc_rows[i] @ P_h[i]^T

This saves ~2 levels vs the unfused approach.

Public API
----------
precompute_mixing_P_matrices(w_A_h, T, m, D) → List[np.ndarray]
fhe_fused_mixing_layer(engine, keys, config, enc_rows, fused_weights, T)
    → List[Ciphertext]   (length T, each D-slot)
"""

import time
import numpy as np
from .engine import bootstrap_if_needed, bs_kwargs


# ============================================================================
# Precomputation
# ============================================================================

def precompute_mixing_P_matrices(w_A_h, T, m, D):
    """
    Precompute fused P_h[i] matrices for one head.

    For D_h row i:
        D_h[i, j] = X[i] @ P_h[i][:, j]
    where:
        P_h[i][:, j] = W_A_h[i-j, :] / (i+1)   if 0 <= i-j < m and j <= i
                      = 0                          otherwise

    Parameters
    ----------
    w_A_h : (m, D) np.ndarray — per-head mixing weight slice of W_A
    T     : int — sequence length
    m     : int — max mixing length
    D     : int — embedding dimension

    Returns
    -------
    P_matrices : List[np.ndarray] of length T, each of shape (T, D).
        D_h[i, :] = enc_rows[i] @ P_matrices[i].T
    """
    P_matrices = []
    for i in range(T):
        P_i = np.zeros((T, D), dtype=np.float64)
        for j in range(T):
            k = i - j
            if 0 <= k < m and j <= i:
                P_i[j, :] = w_A_h[k, :] / (i + 1)
        P_matrices.append(P_i)
    return P_matrices


# ============================================================================
# ct×ct matmul helper: B[i,:] = sum_j D[i,j] * C[j,:]
# ============================================================================

def _fhe_matmul_ct_ct(engine, keys, enc_D_rows, enc_C_rows, T, col_dim):
    """
    Encrypted matrix multiply: B = D @ C.
    D is represented as T row-ciphertexts (each T-slot prefix used),
    C is represented as T row-ciphertexts (each col_dim-slot prefix used).
    Returns T row-ciphertexts for B, each col_dim-slot prefix.
    """
    enc_B_rows = []
    for i in range(T):
        accumulated = None
        for j in range(T):
            slot_size = max(T, col_dim)
            mask_j = np.zeros(slot_size, dtype=np.float64)
            mask_j[j] = 1.0
            d_ij = engine.multiply(enc_D_rows[i], mask_j)
            if j > 0:
                d_ij = engine.rotate(d_ij, keys['rotation_key'], -j)

            # Replicate scalar d_ij across col_dim slots
            replicated = d_ij
            shift = 1
            while shift < col_dim:
                replicated = engine.add(
                    replicated,
                    engine.rotate(replicated, keys['rotation_key'], shift),
                )
                shift *= 2

            product = engine.multiply(replicated, enc_C_rows[j], keys['relin_key'])
            accumulated = product if accumulated is None else engine.add(accumulated, product)

        enc_B_rows.append(accumulated)
    return enc_B_rows


# ============================================================================
# Fused MixingLayer
# ============================================================================

def fhe_fused_mixing_layer(engine, keys, config, enc_rows, fused_weights, T,
                            label="fused_mix"):
    """
    Run the fused MixingLayer in FHE.

    For each head h:
        D_h[i, :] = enc_rows[i] @ P_h[i]^T          (1 multiply_matrix per row)
        C_h[t]    = enc_rows[t] @ W_C_h^T            (1 multiply_matrix per row)
        B_h       = D_h @ C_h                         (ct×ct)
    output = concat(B_h) @ W_proj^T                   (1 multiply_matrix per row)

    Parameters
    ----------
    enc_rows      : List[Ciphertext] length T — D-slot row ciphertexts.
    fused_weights : dict from precompute_fused_weights (src/weights.py).
    T             : int — sequence length.

    Returns
    -------
    List[Ciphertext] of length T — output rows, each D-slot.
    """
    D = config.embedding_dim
    H = config.num_heads
    hs = config.head_size
    slot_count = engine.slot_count
    w = fused_weights['original']

    enc_B_rows = [None] * T

    for h in range(H):
        if hasattr(engine, 'set_label'):
            engine.set_label(f"{label}_head_{h}")
        t_head = time.time()

        P_matrices = fused_weights['P_h'][h]   # List[T] of (T, D)
        W_C_h = fused_weights['W_C_h'][h]      # (hs, D)

        # ── D_h rows: enc_rows[i] @ P_h[i]^T ──
        enc_Dh = []
        for i in range(T):
            ct = bootstrap_if_needed(engine, enc_rows[i], **bs_kwargs(keys))
            padded = np.zeros((slot_count, slot_count), dtype=np.float64)
            padded[:T, :D] = P_matrices[i]
            enc_Dh.append(engine.multiply_matrix(padded, ct, keys['rotation_key']))

        # ── C_h rows: enc_rows[t] @ W_C_h^T ──
        padded_C = np.zeros((slot_count, slot_count), dtype=np.float64)
        padded_C[:hs, :D] = W_C_h

        c_bias_h = None
        if w['c_attn_C_bias'] is not None:
            c_bias_h = np.zeros(slot_count, dtype=np.float64)
            c_bias_h[:hs] = w['c_attn_C_bias'][h * hs:(h + 1) * hs]

        enc_Ch = []
        for t in range(T):
            ct = bootstrap_if_needed(engine, enc_rows[t], **bs_kwargs(keys))
            result = engine.multiply_matrix(padded_C, ct, keys['rotation_key'])
            if c_bias_h is not None:
                result = engine.add(result, c_bias_h)
            enc_Ch.append(result)

        # Bootstrap before ct×ct matmul
        enc_Dh = [bootstrap_if_needed(engine, ct, **bs_kwargs(keys)) for ct in enc_Dh]
        enc_Ch = [bootstrap_if_needed(engine, ct, **bs_kwargs(keys)) for ct in enc_Ch]

        enc_Bh = _fhe_matmul_ct_ct(engine, keys, enc_Dh, enc_Ch, T, hs)

        # Shift head output to position h*hs and accumulate
        for t in range(T):
            placed = enc_Bh[t]
            if h > 0:
                placed = engine.rotate(placed, keys['rotation_key'], h * hs)
            enc_B_rows[t] = (
                placed if enc_B_rows[t] is None else engine.add(enc_B_rows[t], placed)
            )

        print(f"      [FusedMix] Head {h}/{H} done in {time.time()-t_head:.2f}s")

    # ── Output projection ──
    padded_proj = np.zeros((slot_count, slot_count), dtype=np.float64)
    padded_proj[:D, :D] = w['c_proj_attn_weight']
    proj_bias = np.zeros(slot_count, dtype=np.float64)
    if w['c_proj_attn_bias'] is not None:
        proj_bias[:D] = w['c_proj_attn_bias']

    enc_out = []
    for t in range(T):
        ct = bootstrap_if_needed(engine, enc_B_rows[t], **bs_kwargs(keys))
        result = engine.multiply_matrix(padded_proj, ct, keys['rotation_key'])
        result = engine.add(result, proj_bias)
        enc_out.append(result)

    return enc_out
