import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Numerically stable softmax along a given dimension.

    .. math::
        \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to normalize.

    Returns:
        Tensor of the same shape as ``x`` with values in [0, 1] summing to 1 along ``dim``.
    """
    # res = torch.empty_like(x)                       # (..., d_i, ...)
    # x_max = x.max(dim=dim, keepdim=True)[0]          # (..., 1, ...) — 沿 dim 保留为 1
    # x -= x_max                                        # 数值稳定
    # x_exp = torch.exp(x)                                  # exp(每个)
    # exp_sum = x_exp.sum(dim=dim)                          # (..., ) 不含 dim — sum
    # for i in range(x.shape[dim]):
    #     exp_slice = x_exp.select(dim, i)                  # (..., ) 不含 dim 维，和 exp_sum 同形
    #     res.select(dim, i).copy_(exp_slice / exp_sum) # p_i = e^x_i / Σe^x
    # return res
    x_max = x.max(dim=dim, keepdim=True)[0]
    x = x - x_max
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)
