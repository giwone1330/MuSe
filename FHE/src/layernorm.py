"""
FHE LayerNorm implementation via Newton inverse square root.

LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta

Under D packing: each row is a D-slot ciphertext. To compute per-row LayerNorm
we chunk the row into num_chunks sub-ciphertexts so that he_layernorm can
reduce across the D elements.

Public API
----------
he_invsqrt(...)         — Newton/Goldschmidt inverse sqrt on ciphertexts
he_layernorm(...)       — LayerNorm on a list of num_chunks ciphertexts
apply_layernorm_rows(engine, keys, config, enc_rows, T, gamma, beta,
                    min_var, max_var)
                        — convenience wrapper: splits D-dim rows → layernorm
                          → reassembles, returns List[Ciphertext] of length T
"""

import numpy as np
from .engine import _do_bootstrap, bootstrap_if_needed, bs_kwargs


# ============================================================================
# Newton inverse square root
# ============================================================================

def he_invsqrt(engine, relin_key, rotation_key, conj_key,
               bootstrap_keys, stage_count, use_14_levels,
               numerator, denominator, epsilon, alpha=0.001,
               slot_count=None, iter=0):
    """
    Compute 1/sqrt(denominator) homomorphically via Newton/Goldschmidt iterations.

    Parameters
    ----------
    numerator   : encrypted 1.0 (or scaled version)
    denominator : encrypted variance (scaled)
    epsilon     : initial convergence parameter in (0, 1); ratio min_var/max_var
    alpha       : convergence tolerance (stop when epsilon >= 1 - alpha)
    iter        : if > 0, run exactly this many iterations regardless of epsilon
    """
    d = 0
    an = denominator
    bn = numerator
    en = epsilon

    # Bootstrap safety: bn converges toward 1/sqrt(variance_min) = 1/sqrt(epsilon).
    # At the bootstrap trigger point (after ~3 iters), kn ≈ 1 and
    # bn1 = bn * kn^1.5/2 ≈ (1/sqrt(epsilon)) * 0.5.
    # For small epsilon (e.g. LN2 has epsilon ≈ 0.0075 → max bn1 ≈ 5.8),
    # this exceeds the CKKS bootstrap valid range (~[-1, 1]) and gets corrupted.
    # Fix: prescale bn1 into [-1, 1] before bootstrap, postscale after.
    # an values converge to ~1.0 by the bootstrap point, so no prescale needed there.
    #
    # bn_prescale is a power-of-2 integer (0 levels when used as multiplier).
    _max_bn = 1.0 / max(float(np.sqrt(epsilon)), 1e-6)
    _bn_prescale = max(1, int(2 ** np.ceil(np.log2(3.0 * _max_bn + 1))))

    def should_continue():
        if iter > 0:
            return d < iter
        return en < 1 - alpha

    while should_continue():
        d += 1
        coeffs = [1 - en**3, 6 * en**2 - 6, 9 - 9 * en]
        roots = np.roots(coeffs)
        kn = float(np.real(roots[1]))

        bn1 = engine.multiply(bn, kn**1.5 / 2)

        if bn1.level <= 4 or an.level <= 4:
            an = _do_bootstrap(engine, an, relin_key, conj_key, rotation_key,
                               bootstrap_keys, stage_count, use_14_levels)
            # Prescale bn1 into [-1, 1] before bootstrap to avoid polynomial
            # approximation corruption for large invsqrt values (e.g. LN2).
            if _bn_prescale > 1:
                bn1 = engine.multiply(bn1, 1.0 / _bn_prescale)  # float: level -1 (refreshed by bootstrap)
            bn1 = _do_bootstrap(engine, bn1, relin_key, conj_key, rotation_key,
                                bootstrap_keys, stage_count, use_14_levels)
            if _bn_prescale > 1:
                bn1 = engine.multiply(bn1, _bn_prescale)         # int: zero level cost

        bn2 = engine.subtract(3.0 / kn, an)
        bn = engine.multiply(bn1, bn2, relin_key)

        an1 = engine.multiply(an, kn**3 / 4)
        tmp = engine.subtract(3.0 / kn, an)
        an2 = engine.square(tmp, relin_key)
        an = engine.multiply(an1, an2, relin_key)

        en = kn * en * (3 - kn * en)**2 / 4

    return bn


# ============================================================================
# FHE LayerNorm (chunked input)
# ============================================================================

def he_layernorm(engine, secret_key, relin_key, rotation_key, conj_key,
                 bootstrap_keys, stage_count, use_14_levels,
                 enc_x_list, gamma_list, beta_list,
                 n=512, var_e=1e-5, min_var=0.15, max_var=10.0,
                 invsqrt_iter=0):
    """
    Homomorphic LayerNorm on enc_x_list (num_chunks ciphertexts).

    Parameters
    ----------
    enc_x_list  : List[Ciphertext] of length num_chunks.
                  Each ct holds D//num_chunks active slots at positions [0..elems_per_chunk).
    gamma_list  : List[np.ndarray] — gamma parameter chunks (same shape as enc_x_list).
    beta_list   : List[np.ndarray] — beta parameter chunks.
    n           : int — total feature dimension D.
    var_e       : float — epsilon added to variance for numerical stability.
    min_var     : float — known lower bound on per-row variance (from profiling).
    max_var     : float — known upper bound on per-row variance (from profiling).
    invsqrt_iter: int — Newton iterations (0 = auto convergence).

    Returns
    -------
    List[Ciphertext] — normalized and scaled/shifted chunks.
    """
    slot_count = engine.slot_count
    num_chunks = len(enc_x_list)

    enc_x_list = [
        _do_bootstrap(engine, ct, relin_key, conj_key, rotation_key,
                      bootstrap_keys, stage_count, use_14_levels)
        for ct in enc_x_list
    ]

    epsilon_var1 = min_var / max_var
    w_buffer = 1.05
    max_for_denom = (max_var * w_buffer + var_e) * n**2
    scale = 1.0 / np.sqrt(max_for_denom)

    active_per_chunk = min(n // num_chunks, slot_count)
    mask_vals = np.zeros(slot_count)
    mask_vals[:active_per_chunk] = scale

    enc_l = [engine.multiply(ct, mask_vals.tolist()) for ct in enc_x_list]

    # sum(x) across all chunks
    sum_x = engine.clone(enc_l[0])
    for i in range(1, num_chunks):
        sum_x = engine.add(sum_x, enc_l[i])
    step = 1
    while step < active_per_chunk:
        sum_x = engine.add(sum_x, engine.rotate(sum_x, rotation_key, delta=-step))
        step *= 2
    slot0_mask = np.zeros(slot_count)
    slot0_mask[0] = 1.0
    sum_x = engine.multiply(sum_x, slot0_mask.tolist())

    sq_sum_x = engine.square(sum_x, relin_key)

    # Spread sum_x across active slots
    step = 1
    while step < active_per_chunk:
        sum_x = engine.add(sum_x, engine.rotate(sum_x, rotation_key, delta=step))
        step *= 2

    # numerator: n*x_i - sum(x)
    numerators = []
    for ct in enc_l:
        nx = engine.multiply(ct, float(n))
        numerators.append(engine.subtract(nx, sum_x))

    # sum(x^2)
    sigma_x2 = engine.square(enc_l[0], relin_key)
    for i in range(1, num_chunks):
        sigma_x2 = engine.add(sigma_x2, engine.square(enc_l[i], relin_key))
    step = 1
    while step < active_per_chunk:
        sigma_x2 = engine.add(sigma_x2,
                               engine.rotate(sigma_x2, rotation_key, delta=-step))
        step *= 2
    sigma_x2 = engine.multiply(sigma_x2, slot0_mask.tolist())

    # variance = n*sum(x^2) - sum(x)^2
    variance = engine.subtract(engine.multiply(sigma_x2, float(n)), sq_sum_x)
    variance = engine.add(variance, var_e / max_for_denom)

    enc_one = engine.encrypt([1.0], secret_key, level=variance.level)
    denominator = he_invsqrt(
        engine, relin_key, rotation_key, conj_key,
        bootstrap_keys, stage_count, use_14_levels,
        enc_one, variance, epsilon_var1, alpha=0.001,
        slot_count=slot_count, iter=invsqrt_iter,
    )

    if denominator.level <= 3:
        denominator = _do_bootstrap(engine, denominator, relin_key, conj_key,
                                    rotation_key, bootstrap_keys, stage_count,
                                    use_14_levels)

    # Spread denominator across active slots
    step = 1
    while step < active_per_chunk:
        denominator = engine.add(
            denominator, engine.rotate(denominator, rotation_key, delta=step))
        step *= 2

    results = []
    for i in range(num_chunks):
        gamma_denom = engine.multiply(denominator, gamma_list[i].tolist())
        ln_i = engine.multiply(numerators[i], gamma_denom, relin_key)
        ln_i = engine.add(ln_i, beta_list[i].tolist())
        results.append(ln_i)

    return results


# ============================================================================
# Convenience wrapper: apply layernorm to a list of D-dim row ciphertexts
# ============================================================================

def apply_layernorm_rows(engine, keys, config, enc_rows, T, gamma, beta,
                         min_var=None, max_var=None):
    """
    Apply LayerNorm to each row in enc_rows.

    Each enc_rows[t] is a single ciphertext with D active slots (D packing).
    Internally splits into num_chunks, calls he_layernorm, then reassembles.

    Parameters
    ----------
    enc_rows : List[Ciphertext] of length T; each ciphertext has D slots.
    T        : int — sequence length.
    gamma    : np.ndarray of shape (D,) — scale parameter.
    beta     : np.ndarray of shape (D,) — shift parameter.
    min_var  : float or None — fallback to config.min_var if None.
    max_var  : float or None — fallback to config.max_var if None.

    Returns
    -------
    List[Ciphertext] of length T — layer-normalized rows.
    """
    D = config.embedding_dim
    num_chunks = config.num_chunks
    elems_per_chunk = D // num_chunks
    slot_count = engine.slot_count

    if min_var is None:
        min_var = config.min_var
    if max_var is None:
        max_var = config.max_var

    # Pre-chunk gamma and beta into padded arrays
    def chunk_and_pad(arr):
        chunks = []
        for i in range(num_chunks):
            padded = np.zeros(slot_count, dtype=np.float64)
            segment = arr[i * elems_per_chunk:(i + 1) * elems_per_chunk]
            padded[:len(segment)] = segment
            chunks.append(padded)
        return chunks

    gamma_chunks = chunk_and_pad(gamma)
    beta_chunks = chunk_and_pad(beta)

    result_rows = []
    for t in range(T):
        # Split D-slot row into num_chunks sub-ciphertexts
        enc_chunks = []
        for c in range(num_chunks):
            start = c * elems_per_chunk
            mask = np.zeros(slot_count, dtype=np.float64)
            mask[start:start + elems_per_chunk] = 1.0
            extracted = engine.multiply(enc_rows[t], mask.tolist())
            if start > 0:
                extracted = engine.rotate(extracted, keys['rotation_key'], -start)
            enc_chunks.append(extracted)

        ln_results = he_layernorm(
            engine,
            keys['secret_key'], keys['relin_key'],
            keys['rotation_key'], keys['conj_key'],
            keys['bootstrap_keys'], keys['stage_count'],
            keys['use_14_levels'],
            enc_chunks, gamma_chunks, beta_chunks,
            n=D, var_e=config.var_e, min_var=min_var, max_var=max_var,
            invsqrt_iter=config.invsqrt_iter,
        )

        # Reassemble chunks back into a single D-slot ciphertext
        assembled = None
        for c in range(num_chunks):
            start = c * elems_per_chunk
            piece = ln_results[c]
            if start > 0:
                piece = engine.rotate(piece, keys['rotation_key'], start)
            mask = np.zeros(slot_count, dtype=np.float64)
            mask[start:start + elems_per_chunk] = 1.0
            piece = engine.multiply(piece, mask.tolist())
            assembled = piece if assembled is None else engine.add(assembled, piece)

        result_rows.append(assembled)

    return result_rows
