import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

class Block(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: Float[Tensor, "b t d_model"]):
        return self.proj(x)

def bad(b: int, d: int, v: int):
    blk = Block(d, v)
    x = torch.zeros(b, 7, v)
    blk(x)
    y = torch.zeros(b, d) @ torch.zeros(v, 3)
