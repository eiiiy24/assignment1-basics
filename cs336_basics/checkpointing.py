import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    weights = model.state_dict()
    state = optimizer.state_dict()
    save_dict = dict(weights=weights, state=state, iteration=iteration)
    torch.save(save_dict, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    weights = checkpoint['weights']
    state = checkpoint['state']
    iteration = checkpoint['iteration']
    model.load_state_dict(weights)
    optimizer.load_state_dict(state)
    return iteration