"""
FHEConfig — central configuration for MUSE FHE inference.
"""


class FHEConfig:
    """All configurable parameters for the FHE MUSE inference pipeline."""

    def __init__(
        self,
        embedding_dim: int = 512,
        vocab_size: int = 101,
        max_position_embeddings: int = 1024,
        num_blocks: int = 6,
        num_heads: int = 8,
        max_mixing_length: int = 64,
        hiddenmult: int = 1,
        # FHE parameters
        num_chunks: int = 8,
        bootstrap_key_size: str = "medium",
        bootstrap_stage_count: int = 3,
        use_14_levels: bool = False,
        use_small_bootstrap_key: bool = False,
        invsqrt_iter: int = 0,
        var_e: float = 1e-5,
        min_var: float = 0.15,
        max_var: float = 10.0,
    ):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_mixing_length = max_mixing_length
        self.hiddenmult = hiddenmult
        self.head_size = embedding_dim // num_heads
        self.hidden_features = embedding_dim * hiddenmult
        self.bottleneck_features = max(self.hidden_features // 8, 1)

        self.num_chunks = num_chunks
        self.bootstrap_key_size = bootstrap_key_size
        self.bootstrap_stage_count = bootstrap_stage_count
        self.use_14_levels = use_14_levels
        self.use_small_bootstrap_key = use_small_bootstrap_key
        self.invsqrt_iter = invsqrt_iter
        self.var_e = var_e
        self.min_var = min_var
        self.max_var = max_var

        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        )

    def __repr__(self):
        return (
            f"FHEConfig(\n"
            f"  embedding_dim={self.embedding_dim}, vocab_size={self.vocab_size},\n"
            f"  num_blocks={self.num_blocks}, num_heads={self.num_heads},\n"
            f"  max_mixing_length={self.max_mixing_length}, hiddenmult={self.hiddenmult},\n"
            f"  head_size={self.head_size}, hidden_features={self.hidden_features},\n"
            f"  bottleneck_features={self.bottleneck_features},\n"
            f"  num_chunks={self.num_chunks}, bootstrap_key_size='{self.bootstrap_key_size}',\n"
            f"  use_small_bootstrap_key={self.use_small_bootstrap_key},\n"
            f"  bootstrap_stage_count={self.bootstrap_stage_count},\n"
            f"  invsqrt_iter={self.invsqrt_iter}, var_e={self.var_e},\n"
            f"  min_var={self.min_var}, max_var={self.max_var}\n"
            f")"
        )
