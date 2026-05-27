"""Non-standard embedding implementations."""

import torch
import math

from typing import Tuple
import random
from torch.nn import functional as F



class PositionalEmbedding(torch.nn.Module):
    # https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15C1-L31C37
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = (1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))).float()
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input_ids, bsz=None):
        seq_len = input_ids.shape[1]

        # sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_seq=torch.arange(seq_len, dtype=torch.float, device=input_ids.device)
        tensor_24_17_1 = pos_seq.float().unsqueeze(2)

        vector_512_expanded = self.inv_freq.unsqueeze(0).unsqueeze(1)

        result = torch.matmul(tensor_24_17_1, vector_512_expanded)

        sinusoid_inp = result.squeeze(2)

        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

        return pos_emb


class RandomNoise(torch.nn.Module):

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, input_ids):
        return torch.normal(0, 0.1, size=(input_ids.size(0), input_ids.size(1), self.embedding_dim)).to(input_ids.device)


class RPE(torch.nn.Module):
    # https://jaketae.github.io/study/relative-positional-encoding/
    # def __init__(self, embedding_dim, max_seq_length=5000):
    #     super().__init__()

    # def forward(self, input_ids):
    #     return torch.normal(0, 0.1, size=input_ids.shape)
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `num_heads`")
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.query = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.Er = torch.nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError("sequence length exceeds model capacity")

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = torch.nn.functional.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


# module partially stolen from pytorch examples:
class SinusoidalPositional(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.pe[:, : input_ids.shape[1], :]


class ScaledSinosoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length):
        super().__init__(embedding_dim, max_seq_length)
        self.scale_factor = torch.nn.Parameter(torch.tensor([1.0 / embedding_dim**0.5]))

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.scale_factor * self.pe[:, : input_ids.shape[1], :]


class LearnablePositional(torch.nn.Module):
    """Shorthand for a learnable embedding."""

    def __init__(self, embedding_dim, max_seq_length=1024):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))

    def forward(self, input_ids):
        """This is a batch-first implementation"""
        position_ids = self.position_ids[:, : input_ids.shape[1]]
        return self.embedding(position_ids)


class LearnablePositionalRand(torch.nn.Module):
    """Shorthand for a learnable embedding."""

    def __init__(self, embedding_dim, max_seq_length=1024):
        super().__init__()
        self.max_length = max_seq_length
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))

    def forward(self, input_ids):
        """This is a batch-first implementation"""
        seq_length = input_ids.shape[1]
        device = input_ids.device
        if seq_length > self.max_length:  # max length will be increased to max sequnece length if max length is short
            max_length = seq_length
        else:
            max_length = self.max_length
        position_ids = self.position_ids[:, : input_ids.shape[1]]
        position_ids = torch.sort(torch.randperm(max_length, dtype=torch.long, device=device)[:seq_length]).values
        return self.embedding(position_ids)

# Code stolen from GPT-X:
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, def_seq_length=128, seq_dim: int = 0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.seq_len_cached = def_seq_length
        self.seq_dim = seq_dim
        cos_cache, sin_cache = self._get_cos_sin()
        self.register_buffer("cos_cached", cos_cache, persistent=False)
        self.register_buffer("sin_cached", sin_cache, persistent=False)

        # Force fusions on batched version
        def rotate_half(x: torch.Tensor):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster
            return torch.cat((-x2, x1), dim=-1)

        def rope_fn(cos: torch.Tensor, sin: torch.Tensor, query_layer: torch.Tensor, key_layer: torch.Tensor):
            QK = torch.cat([query_layer, key_layer], dim=1)
            rotated = QK * cos[: QK.shape[0]] + rotate_half(QK) * sin[: QK.shape[0]]
            return torch.split(rotated, query_layer.shape[1], dim=1)

        self.rope_fn = rope_fn  # handle fusion on module level

    @torch.no_grad()
    def get_cos_sin_cache(self, x: torch.Tensor):
        seq_len = x.shape[self.seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = x.shape[self.seq_dim]
            cos_cache, sin_cache = self._get_cos_sin()
            self.cos_cached = cos_cache.to(x.device)
            self.sin_cached = sin_cache.to(x.device)
        return self.cos_cached, self.sin_cached

    def _get_cos_sin(self):
        t = torch.arange(self.seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.seq_dim == 0:
            return emb.cos()[:, None, None, :].detach(), emb.sin()[:, None, None, :].detach()
        else:
            return emb.cos()[None, :, None, :].detach(), emb.sin()[None, :, None, :].detach()

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        cos_cached, sin_cached = self.get_cos_sin_cache(query_layer)
        return self.rope_fn(cos_cached, sin_cached, query_layer, key_layer)

    @torch.jit.export
    def single_forward(self, inputs: torch.Tensor):
        """For cases where shapes of Q and K do not match."""
        cos, sin = self.cos_cached[: inputs.shape[0]], self.sin_cached[: inputs.shape[0]]
        return inputs * cos + self.rotate_half(inputs) * sin

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster

class RotarySanityCheck(torch.nn.Module):
    """not again..."""

    def __init__(self, dim, base=10000, def_seq_length=128, seq_dim: int = 0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.seq_len_cached = def_seq_length
        self.seq_dim = seq_dim
        cos_cache, sin_cache = self._get_cos_sin()
        self.register_buffer("cos_cached", cos_cache, persistent=False)
        self.register_buffer("sin_cached", sin_cache, persistent=False)

    @torch.no_grad()
    def get_cos_sin_cache(self, x: torch.Tensor):
        seq_len = x.shape[self.seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = x.shape[self.seq_dim]
            cos_cache, sin_cache = self._get_cos_sin()
            self.cos_cached = cos_cache.to(x.device)
            self.sin_cached = sin_cache.to(x.device)
        return self.cos_cached, self.sin_cached

    def _get_cos_sin(self):
        t = torch.arange(self.seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.seq_dim == 0:
            return emb.cos()[:, None, None, :].detach(), emb.sin()[:, None, None, :].detach()
        else:
            return emb.cos()[None, :, None, :].detach(), emb.sin()[None, :, None, :].detach()

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        # cos, sin = self.get_cos_sin_cache(key_layer)
        # cos, sin = (cos[offset : query_layer.shape[0] + offset, ...], sin[offset : query_layer.shape[0] + offset, ...])
        cos, sin = self.cos_cached, self.sin_cached
        return (query_layer * cos) + (self.rotate_half(query_layer) * sin), (key_layer * cos) + (self.rotate_half(key_layer) * sin)

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster

    @torch.jit.export
    def single_forward(self, inputs: torch.Tensor):
        """For cases where shapes of Q and K do not match."""
        cos, sin = self.cos_cached[: inputs.shape[0]], self.sin_cached[: inputs.shape[0]]
        return inputs * cos + self.rotate_half(inputs) * sin


# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/rotary.py who adapted from
# Adapted from https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
class RotaryEleutherAI(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    """

    _seq_len_cached: int
    # _cos_cached: Optional[torch.Tensor]
    # _sin_cached: Optional[torch.Tensor]

    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        _cos_cached, _sin_cached = self._update_cos_sin_tables(torch.randn(1, 128, 1), seq_dimension=-2)
        self.register_buffer("_cos_cached", _cos_cached, persistent=False)
        self.register_buffer("_sin_cached", _sin_cached, persistent=False)

    @torch.jit.ignore
    def _update_cos_sin_tables(self, x: torch.Tensor, seq_dimension: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        # if seq_len != self._seq_len_cached:  # or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
        self._seq_len_cached = seq_len
        t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype)
        # Don't do einsum, it converts fp32 to fp16
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        cos_cached = repeat(torch.cos(freqs).to(x.dtype), "... d -> ... (d 2)")
        sin_cached = repeat(torch.sin(freqs).to(x.dtype), "... d -> ... (d 2)")

        return cos_cached, sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dimension: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert seq_dimension in [-2, -3]  # Either (bs, h, s, d) or (bs, s, h, d)
        # self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=seq_dimension)

        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dimension),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dimension),
        )

    def rotate_half(self, x: torch.Tensor):
        x = x.unflatten(dim=-1, sizes=(-1, 2))
        x1, x2 = x.unbind(dim=-1)
        rotated_x = torch.stack((-x2, x1), dim=-1)
        return rotated_x.flatten(start_dim=-2)

    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dimension: int = -2):
        # NOTE: This could probably be moved to Triton

        # Handle a possible sequence length mismatch in between q and k
        cos = cos[: x.shape[seq_dimension], :]
        sin = sin[: x.shape[seq_dimension], :]
        if seq_dimension == -3:
            cos = cos[:, None, :]
            sin = sin[:, None, :]
        return (x * cos) + (self.rotate_half(x) * sin)


class RotaryLLAMA(torch.nn.Module):
    """Facebook implementation of rotary embeddings."""

    def __init__(self, hidden_per_head, base=10000, max_seq_length=512, seq_dim: int = 0):
        super().__init__()
        self.seq_dim: int = seq_dim
        freqs_cis = self.precompute_freqs_cis(dim=hidden_per_head, end=max_seq_length * 2, theta=base)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        return self.apply_rotary_emb(query_layer, key_layer, freqs_cis=self.freqs_cis)

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        freqs_cis = freqs_cis[: x.shape[self.seq_dim]]
        # shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
        # shape = [1, seq_length, 1, hidden_per_head]
        shape = [s if i == self.seq_dim or i == x.ndim - 1 else 1 for i, s in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

class FIRE(torch.nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512.0, eps=1e-6, max_length=0):
        """
        FIRE attention bias module (https://arxiv.org/abs/2310.04418).

        Args:
            num_heads: number of attention heads.
            mlp_width: Width of MLP.
            init_c: initial value of log transformation parameter
            init_L: initial value of thresholding parameter
            eps: small constant for numerical stability
        """
        super(FIRE, self).__init__()
        self.max_length = max_length  # using random PE

        # Define the MLP layers
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, mlp_width), torch.nn.ReLU(), torch.nn.Linear(mlp_width, num_heads))

        # Initialize c (log transformation parameter)
        self.c = torch.nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = torch.nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = torch.nn.Parameter(torch.tensor(1.0))  # learn a multiplier to L

        self.eps = eps

    def forward(self, seq_length, device):
        """
        Compute FIRE attention bias (https://arxiv.org/abs/2310.04418).

        Args:
            x: input sequence, shape [bsz, num_heads, seq_len, hidden_dim]

        Returns:
            attention bias of shape [1, num_heads, seq_len, seq_len]
        """
        if (seq_length > self.max_length) or (
            not self.training
        ):  # max length will be increased to max sequnece length if max length is short
            max_length = seq_length
        else:
            max_length = self.max_length

        # take a subset (of length seq_length) of a random permutation of length max_length, then sort it to
        positions = torch.sort(torch.randperm(max_length, dtype=torch.float, device=device)[:seq_length]).values
        relative_distances = positions[:, None] - positions[None, :]
        
        # Thresholding the normalizer for short sequence modeling
        threshold = torch.abs(self.L_multiplier * self.init_L)
        position_normalizer = torch.max(positions, threshold)[:, None]

        # Amplifying differences among local positions with log transform
        relative_distances = torch.log(torch.abs(self.c * relative_distances) + 1)
        position_normalizer = torch.log(torch.abs(self.c * position_normalizer) + 1)

        # Progressive interpolation
        normalized_distances = relative_distances / (position_normalizer + self.eps)
        fire_bias = self.mlp(normalized_distances.unsqueeze(-1)).unsqueeze(0)
        fire_bias = fire_bias.permute(0, 3, 1, 2)
        
        return fire_bias
"""Implementation of abacus embeddings"""
# Example of how to extract digit tokens to pass into constructor
# digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])

class Abacus(torch.nn.Module):
    """
    Abacus Embeddings, learned emebddings resued for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    """
    def __init__(self, digit_tokens=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26], embedding_dim=1024, max_seq_length=1024, max_k=99):
        """
        digit_tokens (list): list of the tokens for each of the 10 digits, `digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])`
        embedding_dim (int): dimension to embed into
        max_seq_length (int): maximum number of embeddings that can be trained
        max_k (int): maximum k value which we randomly shift by during training
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)

        self.max_k = max_k

    def helper(self, mask, device):
        """
        Converts a binary mask of digit locations into spans of consecutive digits
        """
        mask_shape = mask.shape
        
        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask
        
        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)
        
        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)
        
        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        
        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1
        
        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def forward(self, input_ids):
        """
        input_ids (tensor): a batch of inputs, each row is a sample
        """
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        k=0
        if self.training:
            k = random.randint(0, self.max_k)
            output[output>0] += k # as we already have ones in the tensor, the tensor values will be k+1

        return self.embedding(output)

class KerpleRelativeBias(torch.nn.Module):
    """
    Minimal Kerple implementation:
    - kernel_type = 'log'  =>  b(i,j) = -r1 * log(1 + r2 * |i-j|)
    - kernel_type = 'power'=>  b(i,j) = -r1 * |i-j|^r2
    其中 r1, r2 可设为可学习参数或固定超参。
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 4096,
        kernel_type: str = "log",   # or "power"
        learnable: bool = True
    ):
        """
        Args:
            num_heads: 注意力头数
            max_seq_len: 最大序列长度 (用于预分配)
            kernel_type: 'log' 或 'power'
            learnable: r1, r2 是否可学习
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.kernel_type = kernel_type

        # r1, r2 默认初值，可以调
        r1_init = 0.5
        r2_init = 0.5

        if learnable:
            self.r1 = torch.nn.Parameter(torch.tensor(r1_init, dtype=torch.float32))
            self.r2 = torch.nn.Parameter(torch.tensor(r2_init, dtype=torch.float32))
        else:
            # 如果不想学，可以写成 buffer
            self.register_buffer("r1", torch.tensor(r1_init, dtype=torch.float32))
            self.register_buffer("r2", torch.tensor(r2_init, dtype=torch.float32))

    def forward(self, seq_len: int, device=None):
        """
        返回形状 [1, num_heads, seq_len, seq_len] 的偏置矩阵。
        """
        if device is None:
            device = "cpu"

        # 1) 生成距离矩阵 |i-j|, shape: [seq_len, seq_len]
        arange = torch.arange(seq_len, dtype=torch.long, device=device)
        dist_mat = arange.unsqueeze(0) - arange.unsqueeze(1)  # (1, seq_len, seq_len)
        dist_mat = dist_mat.abs()  # |i-j| => shape: [seq_len, seq_len]

        # 2) 根据 kernel_type 计算 b(i,j)
        if self.kernel_type == "log":
            # b(i,j) = - r1 * log(1 + r2 * |i-j|)
            # 注意要加 1e-8 避免 log(0)
            kerple_bias = - self.r1 * torch.log1p(torch.abs(self.r2) * dist_mat.to(torch.float32) + 1e-8)
        elif self.kernel_type == "power":
            # b(i,j) = - r1 * (|i-j|^ r2)
            kerple_bias = - self.r1 * (dist_mat.to(torch.float32) ** torch.abs(self.r2))
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

        # 3) kerple_bias => [num_heads, seq_len, seq_len]
        #   Kerple 原论文中可以对多头共享，也可多头不同(看实现)。这里最常见是直接共享，
        #   如果想让每个头有不同参数，可以对 r1, r2 做 expand(num_heads)。
        kerple_bias = kerple_bias.unsqueeze(0).expand(self.num_heads, seq_len, seq_len)

        # 4) reshape => [1, num_heads, seq_len, seq_len]
        kerple_bias = kerple_bias.unsqueeze(0)
        return kerple_bias
     
class T5RelativePositionBias(torch.nn.Module):
    """
    Implementation of T5’s RPE
    参考: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    或 https://arxiv.org/abs/1910.10683
    - 主要思想: 把 (i - j) 映射到一个 bucket, 对每个 bucket + 每个 head 学一个可训练偏置
    """
    def __init__(self, num_buckets=32, max_distance=128, n_heads=8, bidirectional=False):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = n_heads

        # 每个bucket都有[n_heads]大小的可训练scalar
        self.relative_attention_bias = torch.nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self,relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        """
        将距离映射到bucket id的函数
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, key_length, device=None, cache_position=None):
        """
        返回 [1, n_heads, seq_len, seq_len]
        """
        query_length = key_length
        if device is None:
            device = "cpu"
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values



class AlibiPositionalBias(torch.nn.Module):
    """
    Implementation of Alibi: https://github.com/ofirpress/attention_with_linear_biases
    需要注意:
    - 每个注意力头对应不同的slope（斜率）。
    - 在 Transformer decoder-only 模型里，一般只对自注意力加 bias。
    """

    def __init__(self, num_heads: int, max_sequence_length: int = 4096):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_sequence_length

        # 生成每个head的 slope, 参考 ALiBi 论文 or code
        slopes = self._get_slopes(num_heads)
        # register_buffer 让它成为一个不会参与训练的常数参数
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n):
        """
        原始代码中使用了一种对数方式生成slopes，
        这里给出最常见的一种计算方法
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-2.0 ** -(math.log2(n) - 3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return torch.FloatTensor(get_slopes_power_of_2(n))
        else:
            # 如果 n 不是 2 的幂，则拆分成 2^floor(log2(n)) + 剩余
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
            remaining = n - closest_power_of_2
            last_val = slopes_power_of_2[-1]
            step = last_val
            slopes_power_of_2 += [last_val * (step ** i) for i in range(1, remaining + 1)]
            return torch.FloatTensor(slopes_power_of_2)

    def forward(self, seq_len: int, device=None):
        """
        返回形状: [1, num_heads, seq_len, seq_len]
        """
        if device is None:
            device = "cpu"
        # distance matrix: [seq_len, seq_len], 值 = j - i
        arange = torch.arange(seq_len, device=device)
        distance_matrix = arange.unsqueeze(0) - arange.unsqueeze(1)  # [seq_len, seq_len]
        distance_matrix = distance_matrix.unsqueeze(0).expand(self.num_heads, seq_len, seq_len)
        # shape: [num_heads, seq_len, seq_len]

        # multiply by slope for each head
        slopes = self.slopes.to(device).unsqueeze(-1).unsqueeze(-1)  # [num_heads, 1, 1]
        alibi_bias = slopes * distance_matrix  # [num_heads, seq_len, seq_len]
        # 把它 reshape 成 [1, num_heads, seq_len, seq_len]
        alibi_bias = alibi_bias.unsqueeze(0)
        return alibi_bias


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RPEBias(nn.Module):
    """
    返回形如 [B, n_head, T, T] 的“相对位置偏置”，供外部在 QK^T 之后加上。
    
    这里延续了你原先的“skew操作 + Er参数”思路（类似 Shaw et al. 2018），
    但去掉了自带的 Q/K/V 线性层，以及内置的 softmax 等。
    """
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `num_heads`")
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head

        # 原先 self.Er: [max_len, d_head]
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        # 可选的 dropout，一般对最终的 att logits 不一定做 dropout，这里保留看你需要
        self.dropout = nn.Dropout(dropout)

    def forward(self, q):
        """
        Args:
            q: [B, n_head, T, d_head]
               ——这是 GPT 里已经做完 `q = q.view(B, T, n_head, head_dim).transpose(1, 2)` 的形状

        Returns:
            rpe_bias: [B, n_head, T, T]
               ——可直接加在 (q @ k^T / sqrt(d)) 上
        """
        B, N, T, Dh = q.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds RPE max_len={self.max_len}")

        # 1) 取出对应长度的 Er_t: shape = [d_head, T]
        start = self.max_len - T
        Er_t = self.Er[start:, :].transpose(0, 1)  
        # Er_t: [d_head, T]

        # 2) 计算 QEr = q · Er_t
        # q:   [B, N, T, Dh]
        # Er_t:[Dh, T]
        # => QEr: [B, N, T, T]
        QEr = torch.matmul(q, Er_t)  # B, N, T, T

        # 3) 做 skew 操作 => 得到 Srel: [B, N, T, T]
        Srel = self.skew(QEr)

        # 4) (可选) dropout
        # 这里如果只当 bias 用，一般不会对它再做 dropout；但你可以保留
        Srel = self.dropout(Srel)

        return Srel

    @staticmethod
    def skew(QEr):
        """
        参考你原先的 skew() 实现，把 QEr[:, :, :, :] 做一个 'shift' 操作。
        """
        # QEr: [B, N, T, T]
        padded = F.pad(QEr, (1, 0))  # => [B, N, T, T+1]
        B, N, T, Tplus1 = padded.shape
        reshaped = padded.reshape(B, N, Tplus1, T)  # => [B, N, T+1, T]
        # 舍弃 reshaped[:,:,0,:]
        Srel = reshaped[:, :, 1:, :]  # => [B, N, T, T]
        return Srel
