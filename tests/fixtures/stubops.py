import torch
from torch import Tensor
from typing import reveal_type
from shape_extensions import IntVar


def f[B: IntVar](x: Tensor[[B, int, 64]], y: Tensor[[int, int, 64]], m: Tensor[[B, int]]):
    reveal_type(torch.cat((y, y), dim=1))
    reveal_type(torch.cat([x, x], dim=1))
    reveal_type(torch.stack([x, x]))
    reveal_type(torch.cat((torch.zeros(2, 3), torch.zeros(2, 4)), dim=1))
    reveal_type(x.masked_fill(m[..., None] > 0, 0.))
    reveal_type(torch.where(m[..., None] > 0, x, 0.))
    reveal_type(x.type(torch.float32))
    reveal_type(x.amax(dim=-1))
