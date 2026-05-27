"""
MuseBlock_FHE — one MUSE transformer block in FHE.

Block computation:
    ln1_out  = LayerNorm(x)
    attn_out = FusedMixingLayer(ln1_out)
    x        = x + attn_out               (residual)
    ln2_out  = LayerNorm(x)
    mlp_out  = FusedPolyMlp(ln2_out)
    x        = x + mlp_out               (residual)

Input/output: List[Ciphertext] of length T (D packing — one D-slot ct per row).
"""

import time
import numpy as np
from .layernorm import apply_layernorm_rows
from .muse_mixing import fhe_fused_mixing_layer
from .muse_polymlp import fhe_fused_polymlp


class MuseBlock_FHE:
    """
    One MUSE block (ln1 → fused_mixing → residual → ln2 → fused_polymlp → residual).

    Parameters
    ----------
    engine         : TrackedEngine or desilofhe Engine
    keys           : dict from src.engine.create_keys
    config         : FHEConfig
    fused_weights  : dict from src.weights.precompute_fused_weights
    ln_1_var_bounds: dict {'min_var': float, 'max_var': float} or None
    ln_2_var_bounds: dict {'min_var': float, 'max_var': float} or None
    """

    def __init__(self, engine, keys, config, fused_weights,
                 ln_1_var_bounds=None, ln_2_var_bounds=None):
        self.engine = engine
        self.keys = keys
        self.config = config
        self.fused_weights = fused_weights

        w = fused_weights['original']
        self.ln_1_weight = w['ln_1_weight']
        self.ln_1_bias = w['ln_1_bias']
        self.ln_2_weight = w['ln_2_weight']
        self.ln_2_bias = w['ln_2_bias']

        self.ln_1_min_var = ln_1_var_bounds['min_var'] if ln_1_var_bounds else config.min_var
        self.ln_1_max_var = ln_1_var_bounds['max_var'] if ln_1_var_bounds else config.max_var
        self.ln_2_min_var = ln_2_var_bounds['min_var'] if ln_2_var_bounds else config.min_var
        self.ln_2_max_var = ln_2_var_bounds['max_var'] if ln_2_var_bounds else config.max_var

    def forward(self, enc_rows, T, block_idx=""):
        """
        Parameters
        ----------
        enc_rows  : List[Ciphertext] of length T — input D-slot row ciphertexts.
        T         : int — sequence length.
        block_idx : str/int — label for progress printing.

        Returns
        -------
        List[Ciphertext] of length T — output D-slot row ciphertexts.
        """
        engine = self.engine
        keys = self.keys
        config = self.config
        prefix = f"[Block {block_idx}]" if block_idx != "" else "[Block]"

        # LayerNorm 1
        print(f"  {prefix} LayerNorm 1 "
              f"(min_var={self.ln_1_min_var:.4e}, max_var={self.ln_1_max_var:.4e})")
        t0 = time.time()
        enc_ln1 = apply_layernorm_rows(
            engine, keys, config, enc_rows, T,
            self.ln_1_weight, self.ln_1_bias,
            min_var=self.ln_1_min_var, max_var=self.ln_1_max_var,
        )
        print(f"  {prefix} LayerNorm 1 done in {time.time()-t0:.2f}s")

        # Fused MixingLayer
        print(f"  {prefix} Fused MixingLayer")
        t0 = time.time()
        enc_attn = fhe_fused_mixing_layer(
            engine, keys, config, enc_ln1, self.fused_weights, T,
            label=f"b{block_idx}_mix",
        )
        print(f"  {prefix} MixingLayer done in {time.time()-t0:.2f}s")

        # Residual
        enc_rows = [engine.add(enc_rows[t], enc_attn[t]) for t in range(T)]

        # LayerNorm 2
        print(f"  {prefix} LayerNorm 2 "
              f"(min_var={self.ln_2_min_var:.4e}, max_var={self.ln_2_max_var:.4e})")
        t0 = time.time()
        enc_ln2 = apply_layernorm_rows(
            engine, keys, config, enc_rows, T,
            self.ln_2_weight, self.ln_2_bias,
            min_var=self.ln_2_min_var, max_var=self.ln_2_max_var,
        )
        print(f"  {prefix} LayerNorm 2 done in {time.time()-t0:.2f}s")

        # Fused PolyMlp
        print(f"  {prefix} Fused PolyMlp")
        t0 = time.time()
        enc_mlp = fhe_fused_polymlp(
            engine, keys, config, enc_ln2, self.fused_weights, T,
            label=f"b{block_idx}_mlp",
        )
        print(f"  {prefix} PolyMlp done in {time.time()-t0:.2f}s")

        # Residual
        enc_rows = [engine.add(enc_rows[t], enc_mlp[t]) for t in range(T)]

        return enc_rows
