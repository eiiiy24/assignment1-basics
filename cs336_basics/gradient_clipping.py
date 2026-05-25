import torch
from collections.abc import Iterable

def gradient_clip(
    parameters: Iterable[torch.nn.Parameter], # collection of trainable parameters.
    max_l2_norm: float # a positive value containing the maximum l2-norm.
):
    l2_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        l2_norm += (p.grad.data**2).sum()
    l2_norm = torch.sqrt(l2_norm)
    if l2_norm > max_l2_norm:
        scale_factor = max_l2_norm / (l2_norm + 1e-6)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(scale_factor)
