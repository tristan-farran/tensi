import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class Proj(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: Float[Tensor, "batch d_in"]):
        return self.lin(x)
