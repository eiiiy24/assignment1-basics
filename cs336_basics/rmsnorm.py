import torch

class RMSNorm(torch.nn.Module):
    r"""Root Mean Square Layer Normalization.

    Normalizes the last dimension and scales by a learnable gain parameter g.

    .. math::
        \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \varepsilon}} \cdot g
    """

    def __init__(
        self,
        d_model: int, # Hidden dimension of the model
        eps: float = 1e-5, # Epsilon value for numerical stability
        device: torch.device | None=None, # Device to store the parameters on
        dtype: torch.dtype | None=None # Data type of the parameters
    ):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x / rms * self.g
        return result.to(in_dtype)