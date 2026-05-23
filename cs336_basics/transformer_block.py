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
        device: torch.device | None=None,
        dtype: torch.dtype | None=None,
    ):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, self.theta, self.max_seq_len, device=self.device, dtype=self.dtype)
        self.ln1 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.ln2 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.ffn = SwiGLU(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        x_temp = x
        x = self.ln1(x)
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
        x = self.attn(x, token_positions)
        x = x + x_temp
        x_temp = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + x_temp
        return x

