"""
FHE LayerNorm under N packing.

N packing: D column-ciphertexts of T slots each.
    enc_cols[d] = X[:, d]  (all T positions of feature d)

LayerNorm is per-row (across D features).
Under N packing this is very natural:
  - mean[t]    = (1/D) * sum_d X[t, d]  → sum D column-ciphertexts then scale
  - E[x^2][t]  = (1/D) * sum_d X[t,d]^2 → sum D squared column-cts then scale
  - var[t]     = E[x^2] - mean^2
  - normed[d]  = (X[:, d] - mean) / sqrt(var + eps)
  - out[d]     = gamma[d] * normed[d] + beta[d]   (scalar pt multiply per column)

No chunking needed — the reduction is just addition across D ciphertexts.
The Newton invsqrt operates on a single ciphertext of T slots.

Public API
----------
layernorm_n(engine, keys, config, enc_cols, D, T, gamma, beta, min_var, max_var)
    → List[Ciphertext] of length D
"""

import numpy as np
from .engine_n import _do_bootstrap, bootstrap_if_needed, bs_kwargs
from .layernorm import he_invsqrt  # reuse Newton invsqrt


def layernorm_n(engine, keys, config, enc_cols, D, T,
                gamma, beta, min_var=None, max_var=None):
    """
    Per-row LayerNorm on N-packed ciphertexts.

    Streaming two-pass implementation to avoid holding 3×D ciphertexts
    simultaneously (which would cause CUDA OOM for large D).

    Pass 1 streams over enc_cols to accumulate sum_x and sigma_x2 one
    column at a time (~4 CTs peak instead of 3×D).
    Pass 2 streams over enc_cols again to produce the output columns.

    Parameters
    ----------
    enc_cols : List[Ciphertext] of length D — column-ciphertexts (T active slots each)
    D        : int — embedding dimension
    T        : int — sequence length
    gamma    : np.ndarray (D,) — scale
    beta     : np.ndarray (D,) — shift
    min_var  : float — lower bound on per-row variance (from profiling)
    max_var  : float — upper bound on per-row variance

    Returns
    -------
    List[Ciphertext] of length D — layer-normalised column-ciphertexts
    """
    slot_count = engine.slot_count
    relin_key = keys['relin_key']
    rotation_key = keys['rotation_key']
    conj_key = keys['conj_key']
    bootstrap_keys = keys['bootstrap_keys']
    stage_count = keys['stage_count']
    use_14_levels = keys['use_14_levels']

    if min_var is None:
        min_var = config.min_var
    if max_var is None:
        max_var = config.max_var

    epsilon_var1 = min_var / max_var
    w_buffer = 1.05
    max_for_denom = (max_var * w_buffer + config.var_e) * D**2
    scale = 1.0 / np.sqrt(max_for_denom)

    # Root cause: CKKS bootstrap polynomial approximation is only accurate for
    # plaintext values in roughly [-1, 1].  enc_cols[d] holds X[:,d] which can be
    # O(sqrt(max_var)) — e.g. O(100-700) for LN2 after the mixing layer.
    # Bootstrap silently corrupts these large values, making every subsequent
    # operation (square, multiply) wrong.
    #
    # Fix: scale down by an integer power-of-2 BEFORE bootstrap, then scale back
    # up AFTER bootstrap using an integer multiply (integer multiplies cost 0 levels
    # in desilofhe).  The pre-scale float multiply costs 1 level, but since bootstrap
    # refreshes to level 12 regardless of input level, the net output level is the
    # same as the unscaled path.
    #
    # bs_prescale = smallest power of 2 ≥ 3·sqrt(max_var), giving O(0.3) after scaling.
    bs_prescale = max(4, int(2 ** np.ceil(np.log2(3.0 * np.sqrt(max(max_var, 0.01)) + 1))))

    def _bs_col(ct):
        """Bootstrap a large-valued column ciphertext safely.

        multiply(float) → level -1 (values into [-1,1])
        _do_bootstrap   → level 12 (bootstrap accurate now)
        multiply(int)   → level 12 (integer multiply: zero level cost!)
        """
        ct = engine.multiply(ct, 1.0 / bs_prescale)
        ct = _do_bootstrap(engine, ct, relin_key, conj_key, rotation_key,
                           bootstrap_keys, stage_count, use_14_levels)
        return engine.multiply(ct, bs_prescale)   # int multiply: no level consumed

    # ── Pass 1: stream over columns — accumulate sum_x and sigma_x2 ──
    # variance = D·Σ(X·scale)² - (Σ X·scale)² + var_e/max_fd
    sum_x = None
    sigma_x2 = None

    for d in range(D):
        ct_bs = _bs_col(enc_cols[d])

        # Scale: values go from O(100-700) to O(1e-3) — well within CKKS precision
        scaled_d = engine.multiply(ct_bs, scale)           # level 11
        del ct_bs
        if sum_x is None:
            sum_x = engine.clone(scaled_d)
        else:
            sum_x = engine.add(sum_x, scaled_d)

        sq_d = engine.square(scaled_d, relin_key)          # level 10
        del scaled_d
        if sigma_x2 is None:
            sigma_x2 = engine.clone(sq_d)
        else:
            sigma_x2 = engine.add(sigma_x2, sq_d)
        del sq_d

    # ── Variance + Newton invsqrt ──
    # variance = D·sigma_x2 - sq_sum_x + var_e/max_fd
    #   sigma_x2  at level 10 (Σ(X·scale)²)
    #   sq_sum_x  at level 10 ((Σ X·scale)²)
    # Bring both to level 9 via multiply(1.0), then subtract.
    sq_sum_x   = engine.square(sum_x, relin_key)           # level 10
    sigma_x2_D = engine.multiply(sigma_x2, float(D))       # level  9  (float → rescale)
    sq_sum_x_r = engine.multiply(sq_sum_x, 1.0)            # level  9
    variance   = engine.subtract(sigma_x2_D, sq_sum_x_r)
    variance   = engine.add(variance, config.var_e / max_for_denom)
    del sq_sum_x, sq_sum_x_r, sigma_x2, sigma_x2_D

    ones = np.ones(slot_count, dtype=np.float64)
    enc_one = engine.encrypt(ones.tolist(), keys['secret_key'])
    denominator = he_invsqrt(
        engine, relin_key, rotation_key, conj_key,
        bootstrap_keys, stage_count, use_14_levels,
        enc_one, variance, epsilon_var1, alpha=0.001,
        slot_count=slot_count, iter=config.invsqrt_iter,
    )
    del variance, enc_one

    if denominator.level <= 3:
        denominator = _do_bootstrap(engine, denominator, relin_key, conj_key,
                                    rotation_key, bootstrap_keys, stage_count,
                                    use_14_levels)

    # Use int for D here too; divide by int(D) via multiply by float 1/D is fine
    # because mean_vec is only used in subtract(scaled_d, mean_vec) where both
    # will be at level 11 after the pass-2 bootstrap+multiply(float).
    # Actually: sum_x is at level 11 (after float multiply on bootstrap-12 CT),
    # and 1.0/D is a float so this consumes one level -> mean_vec at level 10.
    # scaled_d in pass2 = bootstrap(12) * float(scale) = level 11.
    # To avoid mismatch, bootstrap sum_x first so it starts at max level.
    sum_x_bs = _do_bootstrap(engine, sum_x, relin_key, conj_key, rotation_key,
                             bootstrap_keys, stage_count, use_14_levels)
    del sum_x
    mean_vec = engine.multiply(sum_x_bs, 1.0 / D)
    del sum_x_bs

    # ── Pass 2: stream over columns again to produce normalised outputs ──
    # Peak memory: denominator + mean_vec + ~3 temporaries per iteration
    result_cols = []
    for d in range(D):
        ct_bs = _bs_col(enc_cols[d])
        scaled_d = engine.multiply(ct_bs, scale)
        del ct_bs

        centered = engine.subtract(scaled_d, mean_vec)
        del scaled_d

        den_bs = bootstrap_if_needed(engine, denominator, **bs_kwargs(keys))
        normed = engine.multiply(centered, den_bs, relin_key)
        del centered
        normed = engine.multiply(normed, int(D))

        gamma_d = float(gamma[d])
        beta_d = np.zeros(slot_count, dtype=np.float64)
        beta_d[:T] = float(beta[d])

        out = engine.multiply(normed, gamma_d)
        out = engine.add(out, beta_d)
        result_cols.append(out)

    return result_cols
