import torch
import math

class RotaryPositionalEmbedding(torch.nn.Module):
    r"""Rotary Position Embedding (RoPE).

    .. math::
        \theta_{i,k} = i / \Theta^{2k/d},

        \begin{pmatrix} q'_{2k} \\ q'_{2k+1} \end{pmatrix}
        = \begin{pmatrix} \cos\theta_{i,k} & -\sin\theta_{i,k} \\
                          \sin\theta_{i,k} &  \cos\theta_{i,k} \end{pmatrix}
          \begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix}
    """

    def __init__(
        self,
        theta: float, # Θ value for the RoPE
        d_k: int, # dimension of query and key vectors
        max_seq_len: int, # Maximum sequence length that will be input
        device: torch.device | None=None # Device to store the buffer on
    ):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # theta = torch.empty(self.max_seq_len, self.d_k // 2) # (max_seq_len, d/2)
        # for i in range(self.max_seq_len):
        #     for k in range(self.d_k // 2):
        #         theta[i, k] = i / (self.theta ** (2 * k / self.d_k)) # k∈{1,…,d/2},θi, k=i/Θ(2k−2)/d, 而代码中 k 从 0 开始
        i = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1).to(self.device) # (max_seq_len, 1)
        k = torch.arange(self.d_k // 2, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, d/2)
        theta = i / (self.theta ** (2 * k / self.d_k)) # broadcast to (max_seq_len, d_k//2)
        self.register_buffer("cos_cached", torch.cos(theta), persistent=False)
        self.register_buffer("sin_cached", torch.sin(theta), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len) -> 对应每个 token 的位置编号
        returns:  (..., seq_len, d_k)
        """
        # x_rotated = torch.empty_like(x)
        # for i in range(x.shape[-2]):
        #     pos = token_positions[..., i] # shape (...,)
        #     for k in range(self.d_k // 2):
        #         x1 = x[..., i, 2 * k] # shape (...,)
        #         x2 = x[..., i, 2 * k + 1] # shape (...,)
        #         cos_val = self.cos_cached[pos, k] # shape (...,)
        #         sin_val = self.sin_cached[pos, k] # shape (...,)
        #         x_rotated[..., i, 2 * k] = cos_val * x1 - sin_val * x2 # shape (...,)
        #         x_rotated[..., i, 2 * k + 1] = sin_val * x1 + cos_val * x2 # shape (...,)
        x_even = x[..., 0::2] # shape (..., seq_len, d/2)
        x_odd = x[..., 1::2] # shape (..., seq_len, d/2)
        cos_vals = self.cos_cached[token_positions] # shape (..., seq_len, d/2)
        sin_cached = self.sin_cached[token_positions] # shape (..., seq_len, d/2)
        x_rot_even = x_even * cos_vals - x_odd * sin_cached
        x_rot_odd = x_even * sin_cached + x_odd * cos_vals
        x_rotated = torch.stack([x_rot_even, x_rot_odd], dim=-1) # (..., seq_len, d/2, 2)
        x_rotated = x_rotated.flatten(-2)
        return x_rotated


