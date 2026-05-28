import torch
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):
    r"""Pre-norm Transformer block.

    .. math::
        \begin{aligned}
        y &= x + \text{MHA}(\text{RMSNorm}(x)) \\
        y &= y + \text{FFN}(\text{RMSNorm}(y))
        \end{aligned}
    """

    def __init__(
        self,
        d_model: int, # Dimensionality of the Transformer block inputs
        num_heads: int, # Number of heads to use in multi-head self-attention
        d_ff: int, # Dimensionality of the position-wise feed-forward inner layer
        theta: float | None = None,
        max_seq_len: int | None=None,
        use_norm: bool = True,  # Set False for RMSNorm ablation
        norm_position: str = "pre",  # "pre" or "post"
        use_rope: bool = True,  # Set False for NoPE ablation
        device: torch.device | None=None,
        dtype: torch.dtype | None=None,
    ):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.use_norm = use_norm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.device = device
        self.dtype = dtype
        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, self.theta, self.max_seq_len, self.use_rope, device=self.device, dtype=self.dtype)
        if self.use_norm:
            self.ln1 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
            self.ln2 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.ffn = SwiGLU(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
        if self.norm_position == "pre":
            x_temp = x
            x = self.ln1(x) if self.use_norm else x
            x = self.attn(x, token_positions)
            x = x + x_temp
            x_temp = x
            x = self.ln2(x) if self.use_norm else x
            x = self.ffn(x)
            x = x + x_temp
            return x
        else: # post-norm
            x_temp = x
            x = self.attn(x, token_positions)
            x = x + x_temp
            x = self.ln1(x) if self.use_norm else x
            x_temp = x
            x = self.ffn(x)
            x = x + x_temp
            x = self.ln2(x) if self.use_norm else x
            return x
