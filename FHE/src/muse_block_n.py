"""
FHE MUSE Block under N packing.

One complete transformer block:
    residual + LayerNorm → MixingLayer → residual
    residual + LayerNorm → PolyMlp → residual

N packing: D column-ciphertexts of T slots each.

Public API
----------
MuseBlock_N.forward(enc_cols, block_weights, T, slot_count, block_idx=0)
    → List[Ciphertext] of length D
"""

import gc
import time
from .engine_n import bootstrap_if_needed, bs_kwargs
from .layernorm_n import layernorm_n
from .muse_mixing_n import fhe_mixing_layer_n
from .muse_polymlp_n import fhe_polymlp_n


class MuseBlock_N:
    """One MUSE transformer block using N packing."""

    def __init__(self, engine, keys, config, var_profile):
        self.engine = engine
        self.keys = keys
        self.config = config
        self.var_profile = var_profile

    def forward(self, enc_cols, block_weights, T, slot_count, block_idx=0):
        """
        Parameters
        ----------
        enc_cols      : List[Ciphertext] of length D
        block_weights : dict from extract_weights_from_hf_model
        T             : int — active sequence length
        slot_count    : int — engine slot count

        Returns
        -------
        List[Ciphertext] of length D
        """
        engine = self.engine
        keys = self.keys
        config = self.config
        w = block_weights

        D = config.embedding_dim

        def _ln_bounds(block_idx, ln_name):
            for e in self.var_profile.get('per_ln', []):
                if e['block'] == block_idx and e['name'] == ln_name:
                    return e['min_var'], e['max_var']
            return None, None

        # ── Layer Norm 1 ──
        print(f"  [Block {block_idx}] LayerNorm1...")
        t0 = time.time()
        ln1_min, ln1_max = _ln_bounds(block_idx, 'ln_1')
        ln1_cols = layernorm_n(
            engine, keys, config, enc_cols, D, T,
            gamma=w['ln_1_weight'], beta=w['ln_1_bias'],
            min_var=ln1_min, max_var=ln1_max,
        )
        print(f"  [Block {block_idx}] LayerNorm1 done in {time.time()-t0:.2f}s")

        # ── MixingLayer ──
        print(f"  [Block {block_idx}] MixingLayer...")
        t0 = time.time()
        mix_cols = fhe_mixing_layer_n(
            engine, keys, config, ln1_cols, w, T, slot_count,
            label=f"mix_n_b{block_idx}",
        )
        del ln1_cols
        print(f"  [Block {block_idx}] MixingLayer done in {time.time()-t0:.2f}s")

        # ── Residual 1 ──
        enc_cols_r1 = [engine.add(enc_cols[d], mix_cols[d]) for d in range(D)]
        del mix_cols
        enc_cols = enc_cols_r1
        del enc_cols_r1
        gc.collect()

        # ── Layer Norm 2 ──
        print(f"  [Block {block_idx}] LayerNorm2...")
        t0 = time.time()
        ln2_min, ln2_max = _ln_bounds(block_idx, 'ln_2')
        ln2_cols = layernorm_n(
            engine, keys, config, enc_cols, D, T,
            gamma=w['ln_2_weight'], beta=w['ln_2_bias'],
            min_var=ln2_min, max_var=ln2_max,
        )
        print(f"  [Block {block_idx}] LayerNorm2 done in {time.time()-t0:.2f}s")

        # ── PolyMlp ──
        print(f"  [Block {block_idx}] PolyMlp...")
        t0 = time.time()
        mlp_cols = fhe_polymlp_n(
            engine, keys, config, ln2_cols, w, T, slot_count,
            label=f"mlp_n_b{block_idx}",
        )
        del ln2_cols
        print(f"  [Block {block_idx}] PolyMlp done in {time.time()-t0:.2f}s")

        # ── Residual 2 ──
        enc_cols_r2 = [engine.add(enc_cols[d], mlp_cols[d]) for d in range(D)]
        del mlp_cols
        enc_cols = enc_cols_r2
        del enc_cols_r2
        gc.collect()

        # ── Bootstrap all columns if needed before next block ──
        for i in range(D):
            old = enc_cols[i]
            enc_cols[i] = bootstrap_if_needed(engine, old, **bs_kwargs(keys))
            if enc_cols[i] is not old:
                del old
        gc.collect()

        return enc_cols
