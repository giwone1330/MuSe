"""
FHE MUSE PolyMlp under N packing.

N packing: D column-ciphertexts of T slots each.
    enc_cols[d] = X[:, d]

MUSE PolyMlp architecture (matches model weights in block_weights):
    out1 = c_fc(x)          (T, D_hid)  — linear D→D_hid
    tmp  = c_fc1(x)         (T, D_bot)  — linear D→D_bot  (bottleneck)
    out2 = c_fc2(tmp)       (T, D_hid)  — linear D_bot→D_hid
    act  = out1 + α*(out1⊙out2)         — polynomial activation
    out  = c_proj(act)      (T, D)      — linear D_hid→D

Where D_hid = config.hidden_features  (= D * hiddenmult)
      D_bot = config.bottleneck_features (= D_hid // 8)

Under N packing:
  - All linear transforms use scale+accumulate (linear_transform_n)
  - ct×ct multiply is element-wise per column
  - Bottleneck path (D→D_bot) uses only D_bot=64 CTs — very cheap

Public API
----------
fhe_polymlp_n(engine, keys, config, enc_cols, block_weights, T, slot_count)
    → List[Ciphertext] of length D (N-packed output)
"""

import gc
import numpy as np
try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
from .engine_n import bootstrap_if_needed, bs_kwargs
from .muse_mixing_n import linear_transform_n


def _gpu_collect():
    """Run Python GC and flush CUDA memory cache if available."""
    gc.collect()
    if _HAS_TORCH and _torch.cuda.is_available():
        _torch.cuda.empty_cache()


def fhe_polymlp_n(engine, keys, config, enc_cols, block_weights, T, slot_count,
                   label="mlp_n"):
    """
    PolyMlp in FHE under N packing.

    Parameters
    ----------
    enc_cols      : List[Ciphertext] of length D — N-packed input columns
    block_weights : dict from extract_weights_from_hf_model (one block)
    T             : int — sequence length
    slot_count    : int — engine slot count

    Returns
    -------
    List[Ciphertext] of length D — N-packed output columns
    """
    D = config.embedding_dim
    D_hid = config.hidden_features     # D * hiddenmult (e.g. 512)
    D_bot = config.bottleneck_features # D_hid // 8     (e.g. 64)

    w = block_weights
    alpha = w['alpha']

    # Bootstrap inputs before expensive operations (in-place)
    for i in range(D):
        old = enc_cols[i]
        enc_cols[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
        if enc_cols[i] is not old:
            del old

    # ── out1 = c_fc(x) → D_hid column-cts ──
    print(f"  [{label}] c_fc(x) [{D}→{D_hid}]")
    enc_out1 = linear_transform_n(
        engine, keys, enc_cols,
        w['c_fc_weight'], w.get('c_fc_bias'),
        D, D_hid, T, slot_count, label=f"{label}_cfc",
    )

    # ── tmp = c_fc1(x) → D_bot column-cts (bottleneck) ──
    print(f"  [{label}] c_fc1(x) [{D}→{D_bot}]")
    enc_tmp = linear_transform_n(
        engine, keys, enc_cols,
        w['c_fc1_weight'], w.get('c_fc1_bias'),
        D, D_bot, T, slot_count, label=f"{label}_cfc1",
    )

    # Bootstrap bottleneck before c_fc2 (only D_bot=64 CTs — fast)
    for i in range(D_bot):
        old = enc_tmp[i]
        enc_tmp[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
        if enc_tmp[i] is not old:
            del old

    # ── out2 = c_fc2(tmp) → D_hid column-cts ──
    print(f"  [{label}] c_fc2(tmp) [{D_bot}→{D_hid}]")
    enc_out2 = linear_transform_n(
        engine, keys, enc_tmp,
        w['c_fc2_weight'], w.get('c_fc2_bias'),
        D_bot, D_hid, T, slot_count, label=f"{label}_cfc2",
    )
    del enc_tmp
    _gpu_collect()

    # Bootstrap before ct×ct multiply (in-place)
    for i in range(D_hid):
        old = enc_out1[i]
        enc_out1[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
        if enc_out1[i] is not old:
            del old
    for i in range(D_hid):
        old = enc_out2[i]
        enc_out2[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
        if enc_out2[i] is not old:
            del old

    # ── act = out1 + α*(out1⊙out2) ──
    print(f"  [{label}] poly activation (α={alpha:.4f})")
    enc_act = []
    for j in range(D_hid):
        prod = engine.multiply(enc_out1[j], enc_out2[j], keys['relin_key'])
        if alpha != 1.0:
            prod = engine.multiply(prod, float(alpha))
        act_j = engine.add(enc_out1[j], prod)
        enc_act.append(act_j)
        enc_out1[j] = None  # drop reference
        enc_out2[j] = None
    del enc_out1, enc_out2
    gc.collect()

    # Bootstrap after ct×ct multiply (in-place)
    for i in range(D_hid):
        old = enc_act[i]
        enc_act[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
        if enc_act[i] is not old:
            del old

    # ── out = c_proj(act) → D column-cts ──
    print(f"  [{label}] c_proj(act) [{D_hid}→{D}]")
    enc_out = linear_transform_n(
        engine, keys, enc_act,
        w['c_proj_mlp_weight'], w.get('c_proj_mlp_bias'),
        D_hid, D, T, slot_count, label=f"{label}_cproj",
    )
    del enc_act
    _gpu_collect()

    return enc_out

