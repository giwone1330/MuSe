"""
N packing utilities: engine creation and key generation.

N packing convention:
    slot_count = T (sequence length, rounded up to power of 2)
    enc_cols: List[Ciphertext] of length D (embedding_dim)
    enc_cols[d] holds X[:, d] — all T positions of feature dimension d

    This is the transpose of D packing:
        D packing: T ciphertexts × D slots  (one row per ct)
        N packing: D ciphertexts × T slots  (one column per ct)

Key properties under N packing:
  - Linear transforms (W: out_dim × D) require O(out_dim × D) ct-pt multiplies
    (no multiply_matrix needed — just scale-and-accumulate across columns)
  - T×T matmuls (e.g. D_h @ C_h in MixingLayer) can use multiply_matrix
    natively since slot_count = T
  - LayerNorm mean/var computed by summing across D column-ciphertexts
"""

import math
import numpy as np


# ============================================================================
# Bootstrap helpers (same logic, imported by N-packing modules)
# ============================================================================

def _do_bootstrap(engine, ct, relin_key, conj_key, rotation_key,
                  bootstrap_keys, stage_count, use_14_levels):
    ct = engine.intt(ct)
    bk = bootstrap_keys
    if isinstance(bk, dict):
        if "bootstrap_key" in bk:
            return engine.bootstrap(ct, relin_key, conj_key, bk["bootstrap_key"])
        else:
            return engine.bootstrap(
                ct, relin_key, conj_key, rotation_key,
                bk["small_bootstrap_key"], stage_count=stage_count,
            )
    return engine.bootstrap(ct, relin_key, conj_key, bk)


def bootstrap_if_needed(engine, ct, relin_key, conj_key, rotation_key,
                        bootstrap_keys, stage_count, use_14_levels,
                        threshold=4):
    if ct.level <= threshold:
        return _do_bootstrap(engine, ct, relin_key, conj_key, rotation_key,
                             bootstrap_keys, stage_count, use_14_levels)
    return ct


def bs_kwargs(keys):
    return dict(
        relin_key=keys['relin_key'],
        conj_key=keys['conj_key'],
        rotation_key=keys['rotation_key'],
        bootstrap_keys=keys['bootstrap_keys'],
        stage_count=keys['stage_count'],
        use_14_levels=keys['use_14_levels'],
    )


# ============================================================================
# Engine creation for N packing
# ============================================================================

def next_power_of_2(n):
    return 1 << math.ceil(math.log2(max(n, 1)))


def create_fhe_engine_n(T, config, track=True):
    """
    Create a desilofhe Engine with slot_count = next_power_of_2(T) (N packing).

    Parameters
    ----------
    T      : int — sequence length
    config : FHEConfig
    track  : bool — wrap in TrackedEngine if True

    Returns
    -------
    engine, slot_count
    """
    from desilofhe import Engine
    from src.engine import TrackedEngine

    sc = next_power_of_2(T)
    if config.use_14_levels:
        raw = Engine(use_bootstrap_to_14_levels=True, slot_count=sc, mode="gpu")
    else:
        raw = Engine(use_bootstrap=True, slot_count=sc, mode="gpu")

    engine = TrackedEngine(raw) if track else raw
    return engine, sc


def _max_stage_count_sparse(slot_count):
    """Return the maximum valid bootstrap stage_count for a given sparse slot_count.

    Based on desilofhe docs sparse bootstrap benchmarks:
      slot_count <= 32   → max 2 stages
      slot_count <= 512  → max 3 stages
      slot_count >= 1024 → max 5 stages (same as full bootstrap)
    """
    if slot_count <= 32:
        return 2
    elif slot_count <= 512:
        return 3
    else:
        return 5


def create_keys_n(engine, secret_key, config):
    """Generate all FHE keys for N packing.

    Stage count is automatically clamped to the maximum valid value for the
    engine's slot_count (sparse bootstrap has tighter limits than full bootstrap).
    """
    slot_count = engine.slot_count
    if config.use_14_levels:
        actual_stage_count = 3
    else:
        max_stage = _max_stage_count_sparse(slot_count)
        actual_stage_count = min(config.bootstrap_stage_count, max_stage)
        if actual_stage_count != config.bootstrap_stage_count:
            print(f"[engine_n] Note: bootstrap_stage_count clamped from "
                  f"{config.bootstrap_stage_count} → {actual_stage_count} "
                  f"(slot_count={slot_count} sparse limit)")

    public_key = engine.create_public_key(secret_key)
    relin_key = engine.create_relinearization_key(secret_key)
    conj_key = engine.create_conjugation_key(secret_key)
    rotation_key = engine.create_rotation_key(secret_key)

    if config.use_14_levels:
        if config.use_small_bootstrap_key:
            bootstrap_key_obj = engine.create_small_bootstrap_key(secret_key)
            bootstrap_keys_dict = {"small_bootstrap_key": bootstrap_key_obj}
        else:
            bootstrap_key_obj = engine.create_bootstrap_key(
                secret_key, size=config.bootstrap_key_size)
            bootstrap_keys_dict = {"bootstrap_key": bootstrap_key_obj}
    else:
        if config.use_small_bootstrap_key:
            bootstrap_key_obj = engine.create_small_bootstrap_key(secret_key)
            bootstrap_keys_dict = {"small_bootstrap_key": bootstrap_key_obj}
        else:
            bootstrap_key_obj = engine.create_bootstrap_key(
                secret_key,
                stage_count=actual_stage_count,
                size=config.bootstrap_key_size,
            )
            bootstrap_keys_dict = {"bootstrap_key": bootstrap_key_obj}

    return {
        'secret_key': secret_key,
        'public_key': public_key,
        'relin_key': relin_key,
        'conj_key': conj_key,
        'rotation_key': rotation_key,
        'bootstrap_keys': bootstrap_keys_dict,
        'stage_count': actual_stage_count,
        'use_14_levels': config.use_14_levels,
    }


# ============================================================================
# Encrypt / decrypt helpers for N packing
# ============================================================================

def encrypt_matrix_n(engine, secret_key, X_np, T, D, slot_count):
    """
    Encrypt input matrix X (T, D) in N packing: D ciphertexts of T slots.

    Parameters
    ----------
    X_np     : np.ndarray (T, D)
    slot_count: int — engine slot count (>= T)

    Returns
    -------
    enc_cols : List[Ciphertext] of length D
    """
    enc_cols = []
    for d in range(D):
        col = np.zeros(slot_count, dtype=np.float64)
        col[:T] = X_np[:, d]
        enc_cols.append(engine.encrypt(col.tolist(), secret_key))
    return enc_cols


def decrypt_matrix_n(engine, secret_key, enc_cols, T, D):
    """
    Decrypt D column-ciphertexts back into (T, D) numpy array.

    Parameters
    ----------
    enc_cols : List[Ciphertext] of length D

    Returns
    -------
    np.ndarray (T, D)
    """
    result = np.zeros((T, D), dtype=np.float64)
    for d in range(D):
        dec = engine.decrypt(enc_cols[d], secret_key)
        result[:T, d] = dec[:T]
    return result
