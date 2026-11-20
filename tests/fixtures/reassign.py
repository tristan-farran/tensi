import torch
from torch import Tensor
from jaxtyping import Float


def step(x: Float[Tensor, "batch seq d_model"], keep: bool):
    if keep:
        x = x.sum(1)
    out = x
    return out


def reduce(x: Float[Tensor, "batch seq d_model"]):
    x = x.mean(1)
    total = x
    return total
