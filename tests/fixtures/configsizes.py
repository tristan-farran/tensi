import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        return q


class Mask(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.levels = config.levels
        self.L = sum(self.levels)
        P = config.prefix_len
        self.mask = torch.full((P, self.L), 0.0)


class Base(nn.Module):
    pass


class Quantizer(Base):
    def forward(self, z):
        z = z.float()
        B, L, C = z.shape
        d = (z**2).sum(-1)
        parts = z.split([C // 2, C - C // 2], dim=2)
        return d, parts
