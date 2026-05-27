from transformers import PretrainedConfig
from typing import List
import math
import inspect
from dataclasses import dataclass
from dataclasses import dataclass, field # added field for args injection
from typing import Dict, Any # for args injection

import torch
import torch.nn as nn
from torch.nn import functional as F

class NanoGPTConfig(PretrainedConfig):
    model_type = "nanogpt"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 96,
        n_layer: int = 16,
        n_head: int = 16,
        n_embd: int = 1024,
        dropout: float = 0.0,
        bias: bool = True,
        use_flash: bool = True,
        ff_proj: str = 'mlp',  # 'relu', 'gelu', 'gated-gelu' etc.
        normalization_layer: str = 'layernorm',  # 'layernorm', 'rmsnorm'
        layer_norm_position: str = 'pre',  # 'pre', 'post', 'pre_post'
        layer_norm_epsilon: float = 1e-6,
        # ---------- positional embedding key -----------
        positional_embedding: str = "none",
        fire_max_length: int = 0,

        # --------------- 新增的开关 ---------------
        use_deepnorm: bool = True,
        clamp_fp16_activations: bool = True,

        # extra_args: Dict[str, Any] = field(default_factory=dict),
        **kwargs,
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_flash = use_flash
        self.ff_proj = ff_proj
        self.normalization_layer = normalization_layer
        self.layer_norm_position = layer_norm_position
        self.layer_norm_epsilon = layer_norm_epsilon
        self.positional_embedding = positional_embedding
        self.fire_max_length = fire_max_length
        self.use_deepnorm = use_deepnorm
        self.clamp_fp16_activations = clamp_fp16_activations
        # self.extra_args = extra_args
        super().__init__(**kwargs)
