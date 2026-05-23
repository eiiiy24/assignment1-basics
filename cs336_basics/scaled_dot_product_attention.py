import torch
from einops import rearrange, einsum
import math
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    q: torch.Tensor, # (batch_size, ..., n, d_k)
    k: torch.Tensor, # (batch_size, ..., m, d_k)
    v: torch.Tensor, # (batch_size, ..., m, d_v)
    mask: torch.Tensor| None=None # (batch_size, ..., n, m)
) -> torch.Tensor: # (batch_size, ..., n, d_v)
    r"""Scaled dot-product attention: softmax(QK^T / sqrt(d_k) + mask) V.

    Args:
        q: Query tensor (..., n, d_k).
        k: Key tensor (..., m, d_k).
        v: Value tensor (..., m, d_v).
        mask: Boolean mask (..., n, m). True = allow, False = block.

    Returns:
        Output tensor (..., n, d_v).
    """
    d_k = q.size(-1)
    qk_scaled = einsum(q, k, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if mask is not None:
        qk_masked = qk_scaled.masked_fill(~mask, -float("inf"))
    else:
        qk_masked = qk_scaled
    qk_softmax = softmax(qk_masked, dim=-1) # (..., n, m) n = 每个 query token 问："m 个 key 里谁跟我最相关？"
    return einsum(qk_softmax, v, "... n m, ... m d_v -> ... n d_v")