"""
FHE MUSE PolyMlp — fused implementation.

PolyMlp is the MUSE replacement for the FFN:
    out1 = c_fc(x)                          [U1: D → D_hid]
    tmp  = c_fc1(x)                         [U2: D → D_bot]
    out2 = c_fc2(tmp)                       [U3: D_bot → D_hid]
    act  = out1 + alpha * (out1 * out2)     [polynomial activation]
    out  = c_proj(act)                      [C: D_hid → D]

Fused version precomputes W_23 = W_fc2 @ W_fc1 (D_hid, D) so that
out2 = x @ W_23^T — skipping one level and one matrix multiply.

Public API
----------
fhe_fused_polymlp(engine, keys, config, enc_rows, fused_weights, T)
    → List[Ciphertext]   (length T, each D-slot)
"""

import numpy as np
from .engine import bootstrap_if_needed, bs_kwargs


def fhe_fused_polymlp(engine, keys, config, enc_rows, fused_weights, T,
                       label="fused_mlp"):
    """
    Run the fused PolyMlp in FHE.

    Instead of:  c_fc → c_fc1 → c_fc2 → poly → c_proj  (5 linear ops)
    Does:        c_fc → W_23 (fused c_fc1+c_fc2) → poly → c_proj  (4 linear ops)

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
    D_hid = config.hidden_features
    slot_count = engine.slot_count
    w = fused_weights['original']

    if hasattr(engine, 'set_label'):
        engine.set_label(f"{label}_c_fc")

    # ── out1 = x @ W_fc^T ──
    padded_fc = np.zeros((slot_count, slot_count), dtype=np.float64)
    padded_fc[:D_hid, :D] = w['c_fc_weight']
    fc_bias = np.zeros(slot_count, dtype=np.float64)
    if w['c_fc_bias'] is not None:
        fc_bias[:D_hid] = w['c_fc_bias']

    enc_out1 = []
    for t in range(T):
        ct = bootstrap_if_needed(engine, enc_rows[t], **bs_kwargs(keys))
        result = engine.multiply_matrix(padded_fc, ct, keys['rotation_key'])
        result = engine.add(result, fc_bias)
        enc_out1.append(result)

    # ── out2 = x @ W_23^T  (fused W_fc2 @ W_fc1) ──
    if hasattr(engine, 'set_label'):
        engine.set_label(f"{label}_W23")

    W_23 = fused_weights['W_23']  # (D_hid, D)
    padded_23 = np.zeros((slot_count, slot_count), dtype=np.float64)
    padded_23[:D_hid, :D] = W_23
    w23_bias = np.zeros(slot_count, dtype=np.float64)
    if fused_weights['W_23_bias'] is not None:
        w23_bias[:D_hid] = fused_weights['W_23_bias']

    enc_out2 = []
    for t in range(T):
        ct = bootstrap_if_needed(engine, enc_rows[t], **bs_kwargs(keys))
        result = engine.multiply_matrix(padded_23, ct, keys['rotation_key'])
        result = engine.add(result, w23_bias)
        enc_out2.append(result)

    # ── polynomial activation: out1 + alpha * (out1 * out2) ──
    if hasattr(engine, 'set_label'):
        engine.set_label(f"{label}_poly")

    alpha = float(w['alpha'])
    alpha_vec = np.full(D_hid, alpha, dtype=np.float64)

    enc_poly = []
    for t in range(T):
        o1 = bootstrap_if_needed(engine, enc_out1[t], **bs_kwargs(keys))
        o2 = bootstrap_if_needed(engine, enc_out2[t], **bs_kwargs(keys))
        product = engine.multiply(o1, o2, keys['relin_key'])
        scaled = engine.multiply(product, alpha_vec) if alpha != 1.0 else product
        combined = engine.add(o1, scaled)
        enc_poly.append(combined)

    # ── output = poly @ W_proj^T ──
    if hasattr(engine, 'set_label'):
        engine.set_label(f"{label}_c_proj")

    padded_proj = np.zeros((slot_count, slot_count), dtype=np.float64)
    padded_proj[:D, :D_hid] = w['c_proj_mlp_weight']
    proj_bias = np.zeros(slot_count, dtype=np.float64)
    if w['c_proj_mlp_bias'] is not None:
        proj_bias[:D] = w['c_proj_mlp_bias']

    enc_out = []
    for t in range(T):
        ct = bootstrap_if_needed(engine, enc_poly[t], **bs_kwargs(keys))
        result = engine.multiply_matrix(padded_proj, ct, keys['rotation_key'])
        result = engine.add(result, proj_bias)
        enc_out.append(result)

    return enc_out
