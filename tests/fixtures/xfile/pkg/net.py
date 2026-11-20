import torch
from .blocks import Mlp


def run(batch_size: int, seq_len: int, width: int):
    mlp = Mlp(width, 4 * width)
    x = torch.randn(batch_size, seq_len, width)
    out = mlp.up(x)
    return out
