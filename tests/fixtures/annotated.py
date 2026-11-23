import torch
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Shaped


class Encoder(nn.Module):
    def __init__(self, vocab, dim, layers=2):
        super().__init__()
        self.wte = nn.Embedding(vocab, dim)
        self.convs = nn.ModuleList(
            [nn.ModuleDict({"ln": nn.LayerNorm(dim), "c1": nn.Conv1d(dim, dim, 3, padding=1)}) for _ in range(layers)]
        )

    def _stack(self, x):
        for blk in self.convs:
            h = blk["ln"](x).transpose(1, 2)
            h = blk["c1"](h).transpose(1, 2)
            x = x + h
        return x

    def forward(self, idx: Shaped[torch.Tensor, "batch seq"]):
        x = self.wte(idx)
        x = self._stack(x)
        return x

    def forward_unnamed(self, idx):
        x = self.wte(idx)
        return self._stack(x)
