import torch
from layers import Proj


def build(batch_size: int, d_model: int, hidden: int):
    proj = Proj(d_model, hidden)
    x = torch.zeros(batch_size, d_model)
    h = proj(x)
    return h
