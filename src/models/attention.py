from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch
from typing import Optional

def polynomial_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: torch.Tensor,  # required arg
    value: torch.Tensor,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    dropout: float = 0.0,
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    # transform_D
    if key.dim() == 2:
        # Single sequence case
        n, m = key.shape
        attn_weights = torch.zeros(n, n, device=key.device, dtype=key.dtype)
        
        for i in range(n):
            for j in range(n):
                k = j - i  # k such that i - j + k = 0
                if 0 <= k < m:
                    attn_weights[i, j] = key[i, k]
    else:
        # Batch case
        batch_size, n, m = key.shape
        attn_weights = torch.zeros(batch_size, n, n, device=key.device, dtype=key.dtype)
        
        for i in range(n):
            for j in range(n):
                k = j - i
                if 0 <= k < m:
                    attn_weights[:, i, j] = key[:, i, k]

    attn_weights = torch.dropout(attn_weights, dropout, train=True)
    attn_output = attn_weights @ value

    return attn_output, attn_weights



def custom_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: torch.Tensor,  # required arg
    value: torch.Tensor,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    a_new_kwargs = None,  # You can now add as many kwargs as you need
    another_new_kwargs = None,  # You can now add as many kwargs as you need
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    # do your magic!
    attn_output = None
    attn_weights = None
    return attn_output, attn_weights  # attn_weights are optional here




#! usage
# AttentionInterface.register("custom", custom_attention)

# model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="custom")
# # Forward pass with the new kwargs
# model(torch.ones(1, 5, dtype=int), a_new_kwargs=..., another_new_kwargs=...)

