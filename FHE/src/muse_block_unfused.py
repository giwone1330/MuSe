"""
MuseBlock_Unfused — one MUSE transformer block in FHE, unfused version.

Unlike MuseBlock_FHE (which uses precomputed P_h[i] fused matrices),
this implementation computes all operations explicitly:

MixingLayer:
    A_h = X @ W_A_h^T          (multiply_matrix, 1 level)
    context_norm(A_h)           (pt-scalar mask, 0 levels)
    transformD(A_h_normed)      (slot-mask + rotate, ~1 level)
    D_h @ C_h                   (ct×ct matmul, 1 level)

PolyMlp:
    out1 = c_fc(x)              (multiply_matrix)
    tmp  = c_fc1(x)             (multiply_matrix, bottleneck D_bot)
    out2 = c_fc2(tmp)           (multiply_matrix)
    act  = out1 + α*(out1⊙out2) (ct×ct)
    out  = c_proj(act)          (multiply_matrix)

Compared to fused:
  + Simpler, no precomputation
  - ~2 extra FHE levels consumed (no P_h fusion)
  - Explicit transformD rotation overhead

Public API
----------
MuseBlock_Unfused(engine, keys, config, block_weights, ln_1_var_bounds, ln_2_var_bounds)
    .forward(enc_rows, T, block_idx='') → List[Ciphertext]
"""

import time
import numpy as np
from .layernorm import apply_layernorm_rows
from .engine import bootstrap_if_needed, bs_kwargs


# ============================================================================
# Unfused MixingLayer
# ============================================================================

def _linear_transform_rows(engine, keys, enc_rows, weight, bias, T, in_dim, out_dim):
    """
    Encrypted linear transform: out_t = weight @ in_t + bias
    Uses multiply_matrix. weight: (out_dim, in_dim).
    """
    slot_count = engine.slot_count
    padded = np.zeros((slot_count, slot_count), dtype=np.float64)
    padded[:out_dim, :in_dim] = weight

    bias_arr = np.zeros(slot_count, dtype=np.float64)
    if bias is not None:
        bias_arr[:len(bias)] = bias

    enc_out = []
    for t in range(T):
        ct = bootstrap_if_needed(engine, enc_rows[t], **bs_kwargs(keys))
        result = engine.multiply_matrix(padded, ct, keys['rotation_key'])
        result = engine.add(result, bias_arr)
        enc_out.append(result)
    return enc_out


def _apply_context_norm(engine, enc_A_rows, T, m):
    """context_norm: A[i, k] /= (i+1) for k <= i, else 0."""
    result = []
    for i in range(T):
        causal_mask = np.zeros(m, dtype=np.float64)
        for k in range(min(i + 1, m)):
            causal_mask[k] = 1.0 / (i + 1)
        result.append(engine.multiply(enc_A_rows[i], causal_mask))
    return result


def _transform_D(engine, keys, enc_A_rows, T, m):
    """
    transformD: D[i, j] = A_normed[i, i-j] for valid (i,j), else 0.

    For row i of D: slot j gets slot (i-j) of A_normed[i].
    """
    enc_D_rows = []
    for i in range(T):
        d_row = None
        for j in range(min(i + 1, m)):
            k = i - j
            mask_k = np.zeros(m, dtype=np.float64)
            mask_k[k] = 1.0
            extracted = engine.multiply(enc_A_rows[i], mask_k)
            delta = j - k
            if delta != 0:
                extracted = engine.rotate(extracted, keys['rotation_key'], delta)
            d_row = extracted if d_row is None else engine.add(d_row, extracted)
        if d_row is None:
            # position 0: D[0,0] = A_normed[0,0]
            m0 = np.zeros(m, dtype=np.float64)
            m0[0] = 1.0
            d_row = engine.multiply(enc_A_rows[0], m0)
        enc_D_rows.append(d_row)
    return enc_D_rows


def _matmul_ct_ct(engine, keys, enc_D_rows, enc_C_rows, T, col_dim):
    """
    Encrypted B = D @ C where D is T×T (row-cts, T active slots)
    and C is T×col_dim (row-cts, col_dim active slots).
    Returns T row-cts for B, col_dim active slots each.

    For row i: B[i,:] = Σ_j D[i,j] * C[j,:]
    Strategy: replicate scalar D[i,j] across col_dim slots, multiply by C[j].
    """
    enc_D_rows = [bootstrap_if_needed(engine, ct, **bs_kwargs(keys))
                  for ct in enc_D_rows]
    enc_C_rows = [bootstrap_if_needed(engine, ct, **bs_kwargs(keys))
                  for ct in enc_C_rows]

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

            # Replicate slot 0 across col_dim slots
            replicated = d_ij
            shift = 1
            while shift < col_dim:
                rotated = engine.rotate(replicated, keys['rotation_key'], shift)
                replicated = engine.add(replicated, rotated)
                shift *= 2

            product = engine.multiply(replicated, enc_C_rows[j], keys['relin_key'])
            accumulated = product if accumulated is None else engine.add(accumulated, product)

        enc_B_rows.append(accumulated)
    return enc_B_rows


def fhe_unfused_mixing_layer(engine, keys, config, enc_rows, block_weights, T,
                              label="unfused_mix"):
    """
    Unfused MixingLayer in FHE.

    Parameters
    ----------
    enc_rows      : List[Ciphertext] length T — D-slot row ciphertexts
    block_weights : dict from extract_weights_from_hf_model (one block)
    T             : int — sequence length

    Returns
    -------
    List[Ciphertext] length T — output row ciphertexts
    """
    D = config.embedding_dim
    H = config.num_heads
    m = config.max_mixing_length
    hs = config.head_size
    w = block_weights

    out_dim_A = H * m

    # ── A = c_attn_A(X) → T row-cts of H*m active slots ──
    print(f"    [{label}] A = c_attn_A(X)  [{T}×{out_dim_A}]")
    t0 = time.time()
    enc_A_rows = _linear_transform_rows(
        engine, keys, enc_rows,
        w['c_attn_A_weight'], w['c_attn_A_bias'], T, D, out_dim_A)
    print(f"    [{label}] c_attn_A done in {time.time()-t0:.2f}s")

    # ── C = c_attn_C(X) → T row-cts of D active slots ──
    print(f"    [{label}] C = c_attn_C(X)  [{T}×{D}]")
    t0 = time.time()
    enc_C_rows = _linear_transform_rows(
        engine, keys, enc_rows,
        w['c_attn_C_weight'], w['c_attn_C_bias'], T, D, D)
    print(f"    [{label}] c_attn_C done in {time.time()-t0:.2f}s")

    # Bootstrap A and C before head loop
    enc_A_rows = [bootstrap_if_needed(engine, ct, **bs_kwargs(keys)) for ct in enc_A_rows]
    enc_C_rows = [bootstrap_if_needed(engine, ct, **bs_kwargs(keys)) for ct in enc_C_rows]

    enc_B_rows = [None] * T

    for h in range(H):
        print(f"    [{label}] Head {h}/{H}")
        t_head = time.time()

        # Extract A_h: slots [h*m .. (h+1)*m) → shift to [0..m)
        enc_Ah_rows = []
        for t in range(T):
            mask = np.zeros(out_dim_A, dtype=np.float64)
            mask[h * m:(h + 1) * m] = 1.0
            extracted = engine.multiply(enc_A_rows[t], mask)
            if h > 0:
                extracted = engine.rotate(extracted, keys['rotation_key'], -(h * m))
            enc_Ah_rows.append(extracted)

        enc_Ah_normed = _apply_context_norm(engine, enc_Ah_rows, T, m)
        enc_Dh_rows = _transform_D(engine, keys, enc_Ah_normed, T, m)

        # Extract C_h: slots [h*hs .. (h+1)*hs) → shift to [0..hs)
        enc_Ch_rows = []
        for t in range(T):
            mask = np.zeros(D, dtype=np.float64)
            mask[h * hs:(h + 1) * hs] = 1.0
            extracted = engine.multiply(enc_C_rows[t], mask)
            if h > 0:
                extracted = engine.rotate(extracted, keys['rotation_key'], -(h * hs))
            enc_Ch_rows.append(extracted)

        enc_Bh_rows = _matmul_ct_ct(engine, keys, enc_Dh_rows, enc_Ch_rows, T, hs)

        # Shift head output back to correct slot range and accumulate
        for t in range(T):
            placed = enc_Bh_rows[t]
            if h > 0:
                placed = engine.rotate(placed, keys['rotation_key'], h * hs)
            enc_B_rows[t] = (placed if enc_B_rows[t] is None
                             else engine.add(enc_B_rows[t], placed))

        print(f"    [{label}] Head {h} done in {time.time()-t_head:.2f}s")

    # ── Output projection c_proj ──
    print(f"    [{label}] output = c_proj(concat)  [{T}×{D}]")
    t0 = time.time()
    enc_out = _linear_transform_rows(
        engine, keys, enc_B_rows,
        w['c_proj_attn_weight'], w['c_proj_attn_bias'], T, D, D)
    print(f"    [{label}] c_proj done in {time.time()-t0:.2f}s")

    return enc_out


# ============================================================================
# Unfused PolyMlp
# ============================================================================

def fhe_unfused_polymlp(engine, keys, config, enc_rows, block_weights, T,
                         label="unfused_mlp"):
    """
    Unfused PolyMlp in FHE:
        out1 = c_fc(x)               [U1: D→D_hid]
        tmp  = c_fc1(x)              [U2: D→D_bot]
        out2 = c_fc2(tmp)            [U3: D_bot→D_hid]
        act  = out1 + α*(out1⊙out2)
        out  = c_proj(act)           [C: D_hid→D]

    Parameters
    ----------
    enc_rows      : List[Ciphertext] length T
    block_weights : dict from extract_weights_from_hf_model (one block)
    T             : int

    Returns
    -------
    List[Ciphertext] length T
    """
    D = config.embedding_dim
    D_hid = config.hidden_features
    D_bot = config.bottleneck_features
    w = block_weights
    alpha = w['alpha']

    print(f"    [{label}] out1 = c_fc(x)  [{T}×{D}→{D_hid}]")
    t0 = time.time()
    enc_out1 = _linear_transform_rows(
        engine, keys, enc_rows, w['c_fc_weight'], w['c_fc_bias'], T, D, D_hid)
    print(f"    [{label}] c_fc done in {time.time()-t0:.2f}s")

    print(f"    [{label}] tmp = c_fc1(x)  [{T}×{D}→{D_bot}]")
    t0 = time.time()
    enc_tmp = _linear_transform_rows(
        engine, keys, enc_rows, w['c_fc1_weight'], w['c_fc1_bias'], T, D, D_bot)
    print(f"    [{label}] c_fc1 done in {time.time()-t0:.2f}s")

    print(f"    [{label}] out2 = c_fc2(tmp)  [{T}×{D_bot}→{D_hid}]")
    t0 = time.time()
    enc_out2 = _linear_transform_rows(
        engine, keys, enc_tmp, w['c_fc2_weight'], w['c_fc2_bias'], T, D_bot, D_hid)
    print(f"    [{label}] c_fc2 done in {time.time()-t0:.2f}s")

    # Polynomial activation: out1 + α*(out1⊙out2)
    print(f"    [{label}] polynomial activation [ct×ct]")
    t0 = time.time()
    alpha_vec = np.full(D_hid, alpha, dtype=np.float64)
    enc_act = []
    for t in range(T):
        o1 = bootstrap_if_needed(engine, enc_out1[t], **bs_kwargs(keys))
        o2 = bootstrap_if_needed(engine, enc_out2[t], **bs_kwargs(keys))
        prod = engine.multiply(o1, o2, keys['relin_key'])
        scaled = engine.multiply(prod, alpha_vec) if alpha != 1.0 else prod
        act = engine.add(o1, scaled)
        enc_act.append(act)
    print(f"    [{label}] poly_combine done in {time.time()-t0:.2f}s")

    print(f"    [{label}] out = c_proj(act)  [{T}×{D_hid}→{D}]")
    t0 = time.time()
    enc_out = _linear_transform_rows(
        engine, keys, enc_act, w['c_proj_mlp_weight'], w['c_proj_mlp_bias'], T, D_hid, D)
    print(f"    [{label}] c_proj done in {time.time()-t0:.2f}s")

    return enc_out


# ============================================================================
# Unfused MUSE Block
# ============================================================================

class MuseBlock_Unfused:
    """
    One MUSE block using the unfused (explicit) FHE computation path.

    ln1(x) → unfused_mixing → residual → ln2(x) → unfused_polymlp → residual

    Parameters
    ----------
    engine          : TrackedEngine or desilofhe Engine
    keys            : dict from src.engine.create_keys
    config          : FHEConfig
    block_weights   : dict from extract_weights_from_hf_model (one block)
    ln_1_var_bounds : dict {'min_var': float, 'max_var': float} or None
    ln_2_var_bounds : dict {'min_var': float, 'max_var': float} or None
    """

    def __init__(self, engine, keys, config, block_weights,
                 ln_1_var_bounds=None, ln_2_var_bounds=None):
        self.engine = engine
        self.keys = keys
        self.config = config
        self.w = block_weights

        self.ln_1_min_var = ln_1_var_bounds['min_var'] if ln_1_var_bounds else config.min_var
        self.ln_1_max_var = ln_1_var_bounds['max_var'] if ln_1_var_bounds else config.max_var
        self.ln_2_min_var = ln_2_var_bounds['min_var'] if ln_2_var_bounds else config.min_var
        self.ln_2_max_var = ln_2_var_bounds['max_var'] if ln_2_var_bounds else config.max_var

    def forward(self, enc_rows, T, block_idx=""):
        """
        Parameters
        ----------
        enc_rows  : List[Ciphertext] length T — input D-slot row ciphertexts
        T         : int — sequence length
        block_idx : str/int — label for progress printing

        Returns
        -------
        List[Ciphertext] length T
        """
        engine, keys, config, w = self.engine, self.keys, self.config, self.w
        label = f"b{block_idx}"

        # ── LayerNorm 1 ──
        print(f"  [Block {block_idx}] LayerNorm1...")
        t0 = time.time()
        enc_ln1 = apply_layernorm_rows(
            engine, keys, config, enc_rows, T,
            gamma=w['ln_1_weight'], beta=w['ln_1_bias'],
            min_var=self.ln_1_min_var, max_var=self.ln_1_max_var,
        )
        print(f"  [Block {block_idx}] LayerNorm1 done in {time.time()-t0:.2f}s")

        # ── MixingLayer (unfused) ──
        print(f"  [Block {block_idx}] MixingLayer (unfused)...")
        t0 = time.time()
        enc_mix = fhe_unfused_mixing_layer(engine, keys, config, enc_ln1, w, T,
                                            label=f"mix_{label}")
        print(f"  [Block {block_idx}] MixingLayer done in {time.time()-t0:.2f}s")

        # ── Residual 1 ──
        enc_rows = [engine.add(enc_rows[t], enc_mix[t]) for t in range(T)]

        # ── LayerNorm 2 ──
        print(f"  [Block {block_idx}] LayerNorm2...")
        t0 = time.time()
        enc_ln2 = apply_layernorm_rows(
            engine, keys, config, enc_rows, T,
            gamma=w['ln_2_weight'], beta=w['ln_2_bias'],
            min_var=self.ln_2_min_var, max_var=self.ln_2_max_var,
        )
        print(f"  [Block {block_idx}] LayerNorm2 done in {time.time()-t0:.2f}s")

        # ── PolyMlp (unfused) ──
        print(f"  [Block {block_idx}] PolyMlp (unfused)...")
        t0 = time.time()
        enc_mlp = fhe_unfused_polymlp(engine, keys, config, enc_ln2, w, T,
                                       label=f"mlp_{label}")
        print(f"  [Block {block_idx}] PolyMlp done in {time.time()-t0:.2f}s")

        # ── Residual 2 ──
        enc_rows = [engine.add(enc_rows[t], enc_mlp[t]) for t in range(T)]

        return enc_rows
