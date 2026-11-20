import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class Mlp(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        return self.down(self.up(x))
