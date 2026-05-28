import torch
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from einops import rearrange, einsum
from cs336_basics.rope import RotaryPositionalEmbedding

class MultiHeadSelfAttention(torch.nn.Module):
    r"""Causal multi-head self-attention with optional RoPE.

    Projects input to Q, K, V via a single batched matrix multiply per projection,
    splits heads, applies causal masking and optional rotary position embeddings,
    then recombines and projects back via the output projection.
    """

    def __init__(
        self,
        d_model: int, # Dimensionality of the Transformer block inputs
        num_heads: int, # Number of heads to use in multi-head self-attention
        theta: float | None=None,
        max_seq_len: int | None=None,
        use_rope: bool = True,  # Set False for NoPE ablation
        device: torch.device | None=None,
        dtype: torch.dtype | None=None,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.device = device
        self.dtype = dtype
        d_k = d_v = self.d_model // self.num_heads
        self.Wq = Linear(self.d_model, self.num_heads * d_k, device=self.device, dtype=self.dtype)
        self.Wk = Linear(self.d_model, self.num_heads * d_k, device=self.device, dtype=self.dtype)
        self.Wv = Linear(self.d_model, self.num_heads * d_v, device=self.device, dtype=self.dtype)
        self.Wo = Linear(self.num_heads * d_v, self.d_model, device=self.device, dtype=self.dtype)
        if self.use_rope:
            if theta is not None and max_seq_len is not None:
                self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, self.device)
            else:
                self.rope = None
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        # (..., seq, d_model) -> (..., seq, d_model)
        q = self.Wq(x) # (..., seq, h*d_k)
        k = self.Wk(x) # (..., seq, h*d_k)
        v = self.Wv(x) # (..., seq, h*d_v)

        n = q.size(-2)
        m = k.size(-2)
        q = rearrange(q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        k = rearrange(k, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        v = rearrange(v, "... seq (h d_v) -> ... h seq d_v", h=self.num_heads)
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal_mask = torch.tril(torch.ones(n, m, device=x.device, dtype=torch.bool)) # device 跟随 x
        heads = scaled_dot_product_attention(q=q, k=k, v=v, mask=causal_mask)  # (..., h, seq, d_v)
        heads = rearrange(heads, "... h seq d_v -> ... seq (h d_v)", h=self.num_heads)

        return self.Wo(heads)
