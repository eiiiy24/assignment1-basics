import numpy as np
import numpy.typing as npt
import torch
import random

def load_data(
    dataset: npt.NDArray, # a numpy array (integer array with token IDs)
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]: # tuple[shape (batch_size, context_length)]
    start_indices = np.random.randint(0, len(dataset) - context_length, batch_size) # shape (batch_size,)
    offsets = np.arange(context_length) # shape (context_length,)
    indices = start_indices[:, None] + offsets # shape (batch_size, 1) + (context_length,) = (batch_size, context_length)
    x = dataset[indices]
    y = dataset[indices + 1]
    return torch.from_numpy(x).to(device=device).int(), torch.from_numpy(y).to(device=device).int()
