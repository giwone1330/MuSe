"""
FHE MUSE MixingLayer under N packing.

N packing: D column-ciphertexts of T slots each.
    enc_cols[d] = X[:, d]

MixingLayer:
    A = c_attn_A(X)        (T, H*m) — for each head h: A_h = X @ W_A_h^T
    C = c_attn_C(X)        (T, D)   — C_h  = X @ W_C_h^T  (head h slice)
    D_h = context_norm(A_h) → transformD  → (T, T) plaintext matrix
    B_h = D_h @ C_h        (T, head_size)
    output = concat(B_h) @ W_proj^T

Under N packing:
  Linear (W: out×D):
    Out[:, j] = Σ_d W[j, d] * X[:, d]
    → sum of D scaled ct-pt multiplies (no multiply_matrix)
    → result is out_dim column-cts of T slots

  D_h (T×T) is a PLAINTEXT matrix computed from A (which itself is X @ W_A_h^T).
    BUT A comes from a linear transform on the encrypted X, so A is encrypted.
    Therefore D_h cannot be computed purely in plaintext.

    Solution (same as D packing): compute A_h in FHE as N-packed column-ciphertexts,
    then apply context_norm + transformD in FHE to get an ENCRYPTED D_h,
    then do ct×ct matmul for D_h @ C_h.

    Alternative — fused approach:
    Use the same P_h[i] fusion but in N packing:
        D_h[i, j] = Σ_k P_h[i][j, k] * X_col[k]
    Here D_h is an (encrypted) T×T matrix stored as T row-ciphertexts
    (each T-slot), but under N packing we want column-outputs.

    Cleaner: compute B_h columns directly.
        B_h[:, s] = Σ_i D_h[:, i] * C_h[i, s]
    This still requires knowing D_h encrypted.

  Practical approach chosen here:
    1. Compute A_h_cols (m column-cts) via linear transform on enc_cols.
    2. Apply context_norm per column (element-wise scale, causal mask is position-dependent).
       context_norm: A_h_normed[t, k] = A_h[t, k] / (t+1) if k <= t else 0
       → For each column k of A_h: multiply by position-dependent mask [1/(t+1) for t>=k else 0]
    3. Compute D_h rows from A_h_normed:
       D_h[i, j] = A_h_normed[i, i-j]   (shift pattern)
       Under N packing D_h rows must be assembled.
       → It's more natural to work with D_h as T row-ciphertexts (T slots each).
       → This means we need to convert A_h from N-packed columns to T-packed rows.

  Transposition in FHE is expensive. Instead we keep the FUSED approach:
    Precompute P_h[i] (T×D) as in D packing.
    For each row i of D_h:
        D_h[i, :] = X[i, :] @ P_h[i]^T
    But X[i, :] is a specific ROW — not available in N packing as a single ct.

  Resolution: in N packing, to get row i of X we would need to extract slot i
  from each of the D column-ciphertexts. This is also expensive.

  BEST STRATEGY for N packing MixingLayer:
    Use the plaintext D_h matrix approach with fused P_h:
    B_h[:, s] = Σ_i D_h[:, i] * C_h[:, s]   where i indexes rows of C_h.

    Since D_h is T×T and C_h is T×hs:
    B_h[:, s] = D_h @ C_h[:, s]

    C_h columns (hs of them) are computed as N-packed column-cts (T slots each).
    D_h comes from A which is encrypted — so we must compute D_h in FHE.

    Full N-packing MixingLayer steps:
    a) A_h = X @ W_A_h^T  → m column-ciphertexts (enc_Ah_cols)
    b) Apply context_norm: each column k gets mask [1/(t+1) if t>=k else 0]
    c) TransformD: assemble D_h as T row-cts from A_h_normed columns
       D_h row i: slot j gets A_h_normed[i, i-j] if valid
       → For slot j in D_h[i]: extract slot (i-j) from A_h_normed col (i-j)... complex.

    SIMPLEST CORRECT APPROACH — keep fused D packing for mixing, convert packing:
    Since the mixing layer's D_h @ C_h matmul is T×T, and under N packing
    we have T-slot column ciphertexts, we can do:
        multiply_matrix(D_h_pt, enc_Ch_col[s], rotation_key)
    BUT D_h must be a known plaintext — it's not, since it depends on encrypted A.

  CONCLUSION:
    The cleanest N-packing MixingLayer uses a HYBRID approach:
    - Linear transforms (c_attn_A, c_attn_C, c_proj) use N-packing scale+accumulate.
    - D_h is computed as encrypted T-row ciphertexts (matching slot_count=T).
    - B_h = D_h @ C_h uses:
        For each output column s of B_h (0..hs):
            enc_Bh_col_s = Σ_{row j of C_h} D_h[j] * C_h[j, s]
        Where D_h[j] is a T-slot ct, and C_h[j, s] is a scalar extracted from ct.
      This matches the ct×ct matmul as implemented in D-packing but column-output.
    - After D_h @ C_h, results are naturally in N packing (column-cts).

    For context_norm + transformD: work in "row view" (T-slot cts) internally,
    then results flow back to column view. The D_h itself is computed row-by-row:
        D_h[i, j] = A_h_normed[i, i-j]
    given A_h_normed as m column-ciphertexts.

Public API
----------
fhe_mixing_layer_n(engine, keys, config, enc_cols, block_weights, T, D)
    → List[Ciphertext] of length D (N packed output)
"""

import gc
import time
import numpy as np
try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
from .engine_n import bootstrap_if_needed, bs_kwargs


def _gpu_collect():
    """Run Python GC and flush CUDA memory cache if available."""
    gc.collect()
    if _HAS_TORCH and _torch.cuda.is_available():
        _torch.cuda.empty_cache()


# ============================================================================
# Linear transform under N packing
# ============================================================================

def linear_transform_n(engine, keys, enc_cols_in, W, bias, in_dim, out_dim, T,
                        slot_count, label="linear"):
    """
    Encrypted linear transform Y = X @ W^T under N packing.

    Y[:, j] = Σ_{d=0}^{in_dim-1} W[j, d] * X[:, d]

    Parameters
    ----------
    enc_cols_in : List[Ciphertext] of length in_dim — input column-ciphertexts
    W           : np.ndarray (out_dim, in_dim)
    bias        : np.ndarray (out_dim,) or None
    in_dim      : int
    out_dim     : int
    T           : int — number of active slots per ciphertext

    Returns
    -------
    List[Ciphertext] of length out_dim — output column-ciphertexts
    """
    if hasattr(engine, 'set_label'):
        engine.set_label(label)

    enc_cols_out = []
    for j in range(out_dim):
        # Y[:, j] = Σ_d W[j, d] * X[:, d]
        accumulated = None
        for d in range(in_dim):
            w_jd = float(W[j, d])
            if w_jd == 0.0:
                continue
            scaled = engine.multiply(enc_cols_in[d], w_jd)
            if accumulated is None:
                accumulated = scaled
            else:
                new_acc = engine.add(accumulated, scaled)
                del accumulated, scaled
                accumulated = new_acc

        if accumulated is None:
            # zero output column
            zeros = np.zeros(slot_count, dtype=np.float64)
            accumulated = engine.encrypt(zeros.tolist(), keys['secret_key'])

        if bias is not None:
            bias_vec = np.zeros(slot_count, dtype=np.float64)
            bias_vec[:T] = float(bias[j])
            accumulated = engine.add(accumulated, bias_vec)

        enc_cols_out.append(accumulated)

    return enc_cols_out


# ============================================================================
# Context norm under N packing
# ============================================================================

def context_norm_n(engine, enc_Ah_cols, T, m, slot_count):
    """
    Apply causal context norm to A_h columns under N packing.

    context_norm: A_h_normed[t, k] = A_h[t, k] / (t+1)  if k <= t else 0

    For column k of A_h (T active slots):
        mask[t] = 1/(t+1)  if t >= k else 0

    Parameters
    ----------
    enc_Ah_cols : List[Ciphertext] of length m — column-ciphertexts of A_h

    Returns
    -------
    List[Ciphertext] of length m — context-normed column-ciphertexts
    """
    result = []
    for k in range(m):
        mask = np.zeros(slot_count, dtype=np.float64)
        for t in range(T):
            if t >= k:
                mask[t] = 1.0 / (t + 1)
        normed = engine.multiply(enc_Ah_cols[k], mask)
        result.append(normed)
    return result


# ============================================================================
# TransformD under N packing: output as T row-ciphertexts
# ============================================================================

def transform_D_n(engine, keys, enc_Ah_normed_cols, T, m, slot_count):
    """
    Compute D_h rows from context-normed A_h columns under N packing.

    D_h[i, j] = A_h_normed[i, i-j]   if 0 <= i-j < m and j <= i else 0

    For D_h row i (a T-slot ciphertext):
        D_h[i, j] = slot i of enc_Ah_normed_cols[i-j]   for valid (i,j)

    Strategy: extract slot i from column (i-j), place in slot j of D_h row i.
    "Extract slot i" = multiply by single-slot mask e_i, then rotate by -i to slot 0,
    then rotate by j to place in slot j.

    This is O(T^2 * m) operations — expensive but correct.
    Returns T row-ciphertexts (each T active slots).
    """
    enc_Dh_rows = []
    for i in range(T):
        d_row = None
        for j in range(min(i + 1, T)):
            k = i - j       # column index into A_h_normed
            if k >= m:
                continue
            # Extract slot i from enc_Ah_normed_cols[k]
            mask_i = np.zeros(slot_count, dtype=np.float64)
            mask_i[i] = 1.0
            extracted = engine.multiply(enc_Ah_normed_cols[k], mask_i)
            # Move from slot i to slot j: rotate by -(i-j) = j-i
            delta = j - i
            if delta != 0:
                rotated = engine.rotate(extracted, keys['rotation_key'], delta)
                del extracted
                extracted = rotated
            # Now slot j holds A_h_normed[i, k] = D_h[i, j]
            if d_row is None:
                d_row = extracted
            else:
                new_row = engine.add(d_row, extracted)
                del d_row, extracted
                d_row = new_row

        if d_row is None:
            zeros = np.zeros(slot_count, dtype=np.float64)
            d_row = engine.encrypt(zeros.tolist(), keys['secret_key'])

        enc_Dh_rows.append(d_row)

    return enc_Dh_rows


# ============================================================================
# D_h @ C_h under N packing: column output
# ============================================================================

def matmul_Dh_Ch_n(engine, keys, enc_Dh_rows, enc_Ch_cols, T, hs, slot_count):
    """
    Compute B_h = D_h @ C_h under N packing.

    D_h: T row-ciphertexts (each T active slots) — as returned by transform_D_n
    C_h: hs column-ciphertexts (each T active slots) — N packed

    B_h[:, s] = Σ_j D_h[:, j] * C_h[j, s]

    For each output column s (0..hs-1):
        B_h[:, s] = Σ_j (slot j of D_h[row]) * C_h_col_s

    But D_h rows are ciphertexts — "slot j of enc_Dh_rows[row]" is not available
    as a scalar. We need to multiply each D_h[j] (the j-th row of D_h as a ct)
    by something... wait, the ct×ct matmul needs a different view.

    Correct formulation:
        B_h[i, s] = Σ_j D_h[i, j] * C_h[j, s]

    Under N packing, B_h column s = T-slot ct = B_h[:, s].
    For each slot i of B_h[:, s]:
        B_h[i, s] = Σ_j D_h[i, j] * C_h[j, s]

    This is the standard T×T @ T×hs matmul where we want column outputs.

    Method: for each output column s, we can use multiply_matrix if D_h were
    a plaintext, but D_h is encrypted. Instead we use:

        B_h_col_s = Σ_j (D_h_col_j_as_row_ct) ⊙_replicated C_h_col_s

    Where D_h_col_j = enc values [D_h[0,j], D_h[1,j], ..., D_h[T-1,j]]
    = column j of D_h, which we can extract from the T row-ciphertexts:
        D_h_col_j[i] = slot j of enc_Dh_rows[i]

    So for each j: extract slot j from enc_Dh_rows[i] for all i, stack into col-ct.
    This requires assembling a column from row-cts — O(T) operations per column.

    Full algorithm for one output column s:
        for j in 0..T:
            extract D_h column j from enc_Dh_rows → enc_Dhcol_j  (T-slot ct)
            c_js = C_h[j, s]  scalar from enc_Ch_cols[s] slot j
            ... but both D_h_col_j and C_h_col_s are encrypted.

    Since both D_h and C_h are encrypted, we need a ct×ct approach.
    We use the same "replicate and multiply" strategy as D-packing:

        For output column s:
        B_h[:, s] = Σ_j D_h_col_j * C_h[j, s]

    where D_h_col_j (T-slot ct) needs to be assembled, and C_h[j, s] is a
    single slot in enc_Ch_cols[s].

    To avoid assembling D_h columns, use row-view:
        B_h[i, s] = Σ_j D_h[i, j] * C_h[j, s]
               = dot product of D_h row i with column s of C_h

    D_h row i = enc_Dh_rows[i] (T-slot ct)
    C_h col s = enc_Ch_cols[s]  (T-slot ct, slot j = C_h[j, s])

    This is a T-slot ct dot product for each (i, s) pair, giving a scalar at slot 0.
    Then we assemble B_h[:, s] column by stacking these scalars.

    Dot product via multiply + sum-of-slots:
        dp = enc_Dh_rows[i] * enc_Ch_cols[s]  (element-wise, T slots)
        sum by rotation: accumulate all T slots into slot 0
        Then slot 0 of dp = B_h[i, s]

    For each output column s (0..hs-1):
        For each row i (0..T-1):
            dp = enc_Dh_rows[i] ⊙ enc_Ch_cols[s]   (ct×ct)
            scalar_i = sum_all_slots(dp)  → T-1 rotations + adds
            Place scalar_i in slot i of B_h_col_s  → assemble column

    Total: hs * T * (1 ct×ct mul + T rotations + 1 slot-place)
    This is expensive: O(hs * T^2) operations.

    OPTIMIZATION: Instead of scalar dot products, use the "replicate row" approach:
        For each j (0..T):
            Replicate slot j of enc_Ch_cols[s] to all T slots → scalar ct R_js
            B_h_col_s += enc_Dh_rows[j] ⊙ R_js   (element-wise ct×ct)

    where enc_Dh_rows[j] now plays the role of the j-th column of D_h^T.

    Wait — enc_Dh_rows[j] holds D_h[j, :] (row j), not column j.
    B_h[i, s] = Σ_j D_h[i, j] * C_h[j, s]
              = (D_h @ C_h[:,s])[i]

    This is D_h (T×T) matrix-vector product with C_h[:,s].
    If D_h were a plaintext, we'd just do multiply_matrix(D_h, enc_Ch_cols[s]).
    But D_h is encrypted.

    Use the same ct×ct strategy from D packing (adapted for column output):
        For output column s:
            result = Σ_j  replicate(slot_j(enc_Ch_cols[s]))  ⊙  enc_Dh_rows_col_j

    where enc_Dh_rows stores row-cts, and "column j of D_h" extracted from rows.

    Column j of D_h (a T-slot ct): enc_Dhcol_j[i] = D_h[i, j] = slot j of enc_Dh_rows[i]
    Construct enc_Dhcol_j:
        For each i: extract slot j from enc_Dh_rows[i] → slot 0, then rotate to slot i
        Add all T contributions.

    This is O(T^2) rotations to build each D_h column — too expensive.

    FINAL APPROACH: use the same replicate-and-multiply as D packing but with
    enc_Dh_rows as the "C" side and enc_Ch_cols as the "D" side:

    B_h[i, s] = Σ_j D_h[i, j] * C_h[j, s]

    Treat enc_Dh_rows as "D" (T row-cts) and enc_Ch_cols as "C" (col-cts).

    For each output column s (result is a T-slot column-ct B_h_col_s):
        acc = 0
        for j in 0..T:
            # extract D_h[:, j] from enc_Dh_rows (column j of D_h)
            # = sum_i (mask_j(enc_Dh_rows[i]) rotated to slot i)
            # Too expensive.

    ACTUAL SIMPLEST CORRECT METHOD:
    Since enc_Dh_rows[i] is a T-slot ct and enc_Ch_cols[s] is a T-slot ct,
    use the ELEMENT-WISE approach for column output:

        For row i: B_h[i, s] = inner_product(enc_Dh_rows[i], enc_Ch_cols[s])
        = Σ_j D_h[i,j] * C_h[j,s]

    Compute via: multiply enc_Dh_rows[i] ⊙ enc_Ch_cols[s] slot-wise,
    then sum all T slots into a scalar value at slot i of B_h_col_s.

    Sum T slots: multiply by [1,1,...,1,0,...,0] then rotate-accumulate to slot 0.
    Then the scalar at slot 0 needs to go to slot i in the output column.
    This is O(T * hs * T) = O(T^2 * hs) in total.

    This is the correct and implementable approach below.
    """
    mask0 = np.zeros(slot_count, dtype=np.float64)
    mask0[0] = 1.0

    enc_B_cols = []
    for s in range(hs):
        enc_Bh_col_s = None
        ch_col_s = bootstrap_if_needed(engine, enc_Ch_cols[s], **bs_kwargs(keys))

        for i in range(T):
            dh_row_i = bootstrap_if_needed(engine, enc_Dh_rows[i], **bs_kwargs(keys))

            # Element-wise ct×ct multiply: dp[j] = D_h[i,j] * C_h[j,s]
            dp = engine.multiply(dh_row_i, ch_col_s, keys['relin_key'])

            # Sum all T active slots into slot 0 via rotation accumulation
            step = 1
            while step < T:
                rot = engine.rotate(dp, keys['rotation_key'], -step)
                new_dp = engine.add(dp, rot)
                del dp, rot
                dp = new_dp
                step *= 2
            # Now slot 0 of dp = B_h[i, s] = Σ_j D_h[i,j]*C_h[j,s]

            # Extract slot 0 and place in slot i of the output column
            scalar = engine.multiply(dp, mask0)
            del dp
            # Rotate by +i to move to slot i
            if i > 0:
                rotated = engine.rotate(scalar, keys['rotation_key'], i)
                del scalar
                scalar = rotated

            if enc_Bh_col_s is None:
                enc_Bh_col_s = scalar
            else:
                new_col = engine.add(enc_Bh_col_s, scalar)
                del enc_Bh_col_s, scalar
                enc_Bh_col_s = new_col

        enc_B_cols.append(enc_Bh_col_s)

    return enc_B_cols


# ============================================================================
# Full MixingLayer under N packing
# ============================================================================

def fhe_mixing_layer_n(engine, keys, config, enc_cols, block_weights, T, slot_count,
                        label="mix_n", return_pre_proj=False):
    """
    MixingLayer in FHE under N packing.

    Parameters
    ----------
    enc_cols      : List[Ciphertext] of length D — N-packed input columns
    block_weights : dict from extract_weights_from_hf_model (one block)
    T             : int — sequence length
    slot_count    : int — engine slot count (>= T)

    Returns
    -------
    List[Ciphertext] of length D — N-packed output columns
    """
    D = config.embedding_dim
    H = config.num_heads
    m = config.max_mixing_length
    hs = config.head_size
    w = block_weights

    W_A_full = w['c_attn_A_weight']  # (H*m, D)
    W_C_full = w['c_attn_C_weight']  # (D, D)
    W_proj = w['c_proj_attn_weight'] # (D, D)

    enc_B_cols = [None] * D  # accumulate per-head contributions

    for h in range(H):
        t_head = time.time()
        if hasattr(engine, 'set_label'):
            engine.set_label(f"{label}_h{h}")

        W_A_h = W_A_full[h * m:(h + 1) * m, :]   # (m, D)
        W_C_h = W_C_full[h * hs:(h + 1) * hs, :]  # (hs, D)

        # ── A_h = X @ W_A_h^T → m column-ciphertexts ──
        enc_Ah_cols = linear_transform_n(
            engine, keys, enc_cols, W_A_h,
            w.get('c_attn_A_bias') and w['c_attn_A_bias'][h * m:(h + 1) * m],
            D, m, T, slot_count, label=f"{label}_h{h}_A",
        )

        # ── C_h = X @ W_C_h^T → hs column-ciphertexts ──
        c_bias_h = None
        if w['c_attn_C_bias'] is not None:
            c_bias_h = w['c_attn_C_bias'][h * hs:(h + 1) * hs]
        enc_Ch_cols = linear_transform_n(
            engine, keys, enc_cols, W_C_h, c_bias_h,
            D, hs, T, slot_count, label=f"{label}_h{h}_C",
        )

        # ── context_norm on A_h columns ──
        enc_Ah_normed = context_norm_n(engine, enc_Ah_cols, T, m, slot_count)
        del enc_Ah_cols

        # ── transformD: assemble T row-ciphertexts for D_h (in-place bootstrap) ──
        for i in range(len(enc_Ah_normed)):
            old = enc_Ah_normed[i]
            enc_Ah_normed[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
            if enc_Ah_normed[i] is not old:
                del old
        enc_Dh_rows = transform_D_n(engine, keys, enc_Ah_normed, T, m, slot_count)
        del enc_Ah_normed

        # ── B_h = D_h @ C_h → hs column-ciphertexts (in-place bootstrap) ──
        for i in range(len(enc_Dh_rows)):
            old = enc_Dh_rows[i]
            enc_Dh_rows[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
            if enc_Dh_rows[i] is not old:
                del old
        for i in range(len(enc_Ch_cols)):
            old = enc_Ch_cols[i]
            enc_Ch_cols[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
            if enc_Ch_cols[i] is not old:
                del old
        enc_Bh_cols = matmul_Dh_Ch_n(
            engine, keys, enc_Dh_rows, enc_Ch_cols, T, hs, slot_count)
        del enc_Dh_rows, enc_Ch_cols

        # ── Accumulate into D-wide output at correct head offset ──
        for s in range(hs):
            out_idx = h * hs + s
            if enc_B_cols[out_idx] is None:
                enc_B_cols[out_idx] = enc_Bh_cols[s]
            else:
                old = enc_B_cols[out_idx]
                enc_B_cols[out_idx] = engine.add(old, enc_Bh_cols[s])
                del old
        del enc_Bh_cols
        _gpu_collect()

        print(f"    [MixN] Head {h}/{H} done in {time.time()-t_head:.2f}s")

    # ── Output projection: out = B @ W_proj^T → D column-cts ──
    for i in range(D):
        old = enc_B_cols[i]
        enc_B_cols[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
        if enc_B_cols[i] is not old:
            del old
    enc_out = linear_transform_n(
        engine, keys, enc_B_cols, W_proj,
        w['c_proj_attn_bias'], D, D, T, slot_count, label=f"{label}_proj",
    )

    if return_pre_proj:
        _gpu_collect()
        return enc_B_cols, enc_out

    del enc_B_cols
    _gpu_collect()
    return enc_out
