import torch
from cs336_basics.linear import Linear

class SiLU(torch.nn.Module):
    """SiLU (Swish) activation: x * sigmoid(x)."""

    def __init__(self):
        super(SiLU, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (... d_model) -> (... d_model)
        return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    r"""SwiGLU feed-forward network.

    .. math::
        \text{SwiGLU}(x, W_1, W_2, W_3) = W_2 (\text{SiLU}(W_1 x) \odot W_3 x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None,
    ):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        if self.d_ff is None:
            self.d_ff = int(8/3 * self.d_model)
            self.d_ff = ((self.d_ff + 63) // 64) * 64 # 向上取整 64 倍数
        self.w1 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, self.device, self.dtype)
        self.w3 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (... d_model) -> (... d_model)
        silu = self.silu(self.w1(x)) * self.w3(x)
        ffn = self.w2(silu)
        return ffn
