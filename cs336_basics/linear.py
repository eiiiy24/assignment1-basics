import torch
import math
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    """A linear transformation y = x @ W.T without bias."""

    def __init__(
        self,
        in_features: int, # final dimension of the input
        out_features: int, # final dimension of the output
        device: torch.device | None=None, # Device to store the parameters on
        dtype: torch.dtype | None=None # Data type of the parameters
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype)) # 𝑊 ∈ ℝ𝑑out×𝑑in
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # Apply the linear transformation to the input.
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")