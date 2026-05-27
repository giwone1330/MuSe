"""
FHE engine utilities: TrackedEngine wrapper, engine creation, key generation,
and bootstrap helpers.

D-packing convention: slot_count = embedding_dim (D).
Each encrypted row is a ciphertext of D slots.
"""

import time
import numpy as np


# ============================================================================
# Bootstrap helpers
# ============================================================================

def _do_bootstrap(engine, ct, relin_key, conj_key, rotation_key,
                  bootstrap_keys, stage_count, use_14_levels):
    """Perform bootstrap using whichever bootstrap key type is available."""
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
    """Bootstrap a ciphertext if its level is at or below threshold."""
    if ct.level <= threshold:
        return _do_bootstrap(engine, ct, relin_key, conj_key, rotation_key,
                             bootstrap_keys, stage_count, use_14_levels)
    return ct


def bs_kwargs(keys):
    """Return keyword arguments dict for bootstrap_if_needed from a keys dict."""
    return dict(
        relin_key=keys['relin_key'],
        conj_key=keys['conj_key'],
        rotation_key=keys['rotation_key'],
        bootstrap_keys=keys['bootstrap_keys'],
        stage_count=keys['stage_count'],
        use_14_levels=keys['use_14_levels'],
    )


# ============================================================================
# Engine creation
# ============================================================================

def create_fhe_engine(config, track=True):
    """
    Create a desilofhe Engine with slot_count = embedding_dim (D packing).

    Parameters
    ----------
    config : FHEConfig
    track  : bool — if True, wraps engine in TrackedEngine for op counting.

    Returns
    -------
    engine : TrackedEngine (or raw Engine if track=False)
    """
    from desilofhe import Engine

    sc = config.embedding_dim
    if config.use_14_levels:
        raw = Engine(use_bootstrap_to_14_levels=True, slot_count=sc, mode="gpu")
    else:
        raw = Engine(use_bootstrap=True, slot_count=sc, mode="gpu")

    return TrackedEngine(raw) if track else raw


def create_keys(engine, secret_key, config):
    """
    Generate all necessary FHE keys and return a keys dict.

    Parameters
    ----------
    engine     : Engine or TrackedEngine
    secret_key : SecretKey returned by engine.create_secret_key()
    config     : FHEConfig

    Returns
    -------
    dict with: secret_key, public_key, relin_key, conj_key, rotation_key,
               bootstrap_keys, stage_count, use_14_levels
    """
    actual_stage_count = (
        3 if config.use_14_levels else config.bootstrap_stage_count
    )

    public_key = engine.create_public_key(secret_key)
    relin_key = engine.create_relinearization_key(secret_key)
    conj_key = engine.create_conjugation_key(secret_key)
    rotation_key = engine.create_rotation_key(secret_key)

    if config.use_14_levels:
        bootstrap_key = engine.create_bootstrap_key(
            secret_key, size=config.bootstrap_key_size
        )
    else:
        bootstrap_key = engine.create_bootstrap_key(
            secret_key,
            stage_count=config.bootstrap_stage_count,
            size=config.bootstrap_key_size,
        )

    return {
        'secret_key': secret_key,
        'public_key': public_key,
        'relin_key': relin_key,
        'conj_key': conj_key,
        'rotation_key': rotation_key,
        'bootstrap_keys': {"bootstrap_key": bootstrap_key},
        'stage_count': actual_stage_count,
        'use_14_levels': config.use_14_levels,
    }


# ============================================================================
# TrackedEngine — instruments every engine call for diagnostics
# ============================================================================

class TrackedEngine:
    """
    Wraps a desilofhe Engine to count operations and track level changes.
    Transparent proxy: any attribute not overridden is forwarded to the real engine.
    """

    def __init__(self, engine):
        self._engine = engine
        self.reset_counters()

    def reset_counters(self):
        self.op_counts = {
            'multiply_ct_ct': 0,
            'multiply_ct_pt': 0,
            'multiply_matrix': 0,
            'add': 0,
            'subtract': 0,
            'rotate': 0,
            'square': 0,
            'bootstrap': 0,
            'encrypt': 0,
            'decrypt': 0,
            'clone': 0,
            'intt': 0,
        }
        self.bootstrap_log = []   # list of (label, level_before, level_after)
        self.level_trace = []     # list of (op_name, result_level)
        self.levels_consumed = 0
        self.levels_restored = 0
        self._current_label = ""
        # Per-type time and level tracking (8 mutually exclusive categories)
        _OP_TYPES = (
            'encrypt', 'decrypt', 'pt_ct_mult', 'ct_ct_mult',
            'matrix_mult', 'rotation', 'add_subtract', 'bootstrap',
        )
        self.times_by_type           = dict.fromkeys(_OP_TYPES, 0.0)
        self.levels_consumed_by_type = dict.fromkeys(_OP_TYPES, 0)
        self.levels_restored_by_type = dict.fromkeys(_OP_TYPES, 0)

    def set_label(self, label):
        self._current_label = label

    @property
    def slot_count(self):
        return self._engine.slot_count

    @property
    def max_level(self):
        return self._engine.max_level

    def summary(self):
        return dict(self.op_counts)

    def print_summary(self, title=""):
        if title:
            print(f"\n  --- {title} ---")
        for k, v in self.op_counts.items():
            if v > 0:
                print(f"    {k:>20}: {v}")
        print(f"    {'levels_consumed':>20}: {self.levels_consumed}")
        print(f"    {'levels_restored':>20}: {self.levels_restored}")
        print(f"    {'net_levels_used':>20}: {self.levels_consumed - self.levels_restored}")
        if self.bootstrap_log:
            from collections import Counter
            unique = Counter()
            for label, lb, la in self.bootstrap_log:
                unique[f"{label} lv{lb}→{la}"] += 1
            print(f"    {'total_bootstraps':>20}: {len(self.bootstrap_log)}")
            for desc, count in unique.most_common():
                print(f"      {desc} (×{count})")

    # ---- Tracked methods ----

    def encrypt(self, data, key, level=None):
        self.op_counts['encrypt'] += 1
        t0 = time.time()
        if level is not None:
            result = self._engine.encrypt(data, key, level=level)
        else:
            result = self._engine.encrypt(data, key)
        self.times_by_type['encrypt'] += time.time() - t0
        return result

    def decrypt(self, ct, key):
        self.op_counts['decrypt'] += 1
        t0 = time.time()
        result = self._engine.decrypt(ct, key)
        self.times_by_type['decrypt'] += time.time() - t0
        return result

    def clone(self, ct):
        self.op_counts['clone'] += 1
        return self._engine.clone(ct)

    def intt(self, ct):
        self.op_counts['intt'] += 1
        t0 = time.time()
        result = self._engine.intt(ct)
        self.times_by_type['bootstrap'] += time.time() - t0
        return result

    def add(self, x, y, out=None):
        self.op_counts['add'] += 1
        t0 = time.time()
        if out is not None:
            result = self._engine.add(x, y, out=out)
        else:
            result = self._engine.add(x, y)
        self.times_by_type['add_subtract'] += time.time() - t0
        return result

    def subtract(self, x, y):
        self.op_counts['subtract'] += 1
        t0 = time.time()
        result = self._engine.subtract(x, y)
        self.times_by_type['add_subtract'] += time.time() - t0
        return result

    def multiply(self, x, y, relin_key=None):
        level_before = x.level if hasattr(x, 'level') else None
        t0 = time.time()
        if relin_key is not None:
            self.op_counts['multiply_ct_ct'] += 1
            result = self._engine.multiply(x, y, relin_key)
            op_type = 'ct_ct_mult'
        else:
            if hasattr(y, 'level'):
                self.op_counts['multiply_ct_ct'] += 1
                op_type = 'ct_ct_mult'
            else:
                self.op_counts['multiply_ct_pt'] += 1
                op_type = 'pt_ct_mult'
            result = self._engine.multiply(x, y)
        self.times_by_type[op_type] += time.time() - t0
        if hasattr(result, 'level'):
            self.level_trace.append((self._current_label, result.level))
            if level_before is not None and result.level < level_before:
                delta = level_before - result.level
                self.levels_consumed += delta
                self.levels_consumed_by_type[op_type] += delta
        return result

    def multiply_matrix(self, matrix, ct, key):
        self.op_counts['multiply_matrix'] += 1
        level_before = ct.level if hasattr(ct, 'level') else None
        t0 = time.time()
        result = self._engine.multiply_matrix(matrix, ct, key)
        self.times_by_type['matrix_mult'] += time.time() - t0
        if hasattr(result, 'level'):
            self.level_trace.append((self._current_label, result.level))
            if level_before is not None and result.level < level_before:
                delta = level_before - result.level
                self.levels_consumed += delta
                self.levels_consumed_by_type['matrix_mult'] += delta
        return result

    def square(self, ct, relin_key=None):
        self.op_counts['square'] += 1
        level_before = ct.level if hasattr(ct, 'level') else None
        t0 = time.time()
        result = (
            self._engine.square(ct, relin_key)
            if relin_key is not None
            else self._engine.square(ct)
        )
        self.times_by_type['ct_ct_mult'] += time.time() - t0
        if level_before is not None and hasattr(result, 'level') and result.level < level_before:
            _lev_delta = level_before - result.level
            self.levels_consumed += _lev_delta
            self.levels_consumed_by_type['ct_ct_mult'] += _lev_delta
        return result

    def rotate(self, ct, key_or_delta, delta=None):
        self.op_counts['rotate'] += 1
        level_before = ct.level if hasattr(ct, 'level') else None
        t0 = time.time()
        result = (
            self._engine.rotate(ct, key_or_delta, delta=delta)
            if delta is not None
            else self._engine.rotate(ct, key_or_delta)
        )
        self.times_by_type['rotation'] += time.time() - t0
        if level_before is not None and hasattr(result, 'level') and result.level < level_before:
            _lev_delta = level_before - result.level
            self.levels_consumed += _lev_delta
            self.levels_consumed_by_type['rotation'] += _lev_delta
        return result

    def bootstrap(self, ct, *args, **kwargs):
        level_before = ct.level
        self.op_counts['bootstrap'] += 1
        t0 = time.time()
        result = self._engine.bootstrap(ct, *args, **kwargs)
        self.times_by_type['bootstrap'] += time.time() - t0
        level_after = result.level
        self.bootstrap_log.append((self._current_label, level_before, level_after))
        if level_after > level_before:
            _lev_delta = level_after - level_before
            self.levels_restored += _lev_delta
            self.levels_restored_by_type['bootstrap'] += _lev_delta
        return result

    def __getattr__(self, name):
        return getattr(self._engine, name)
