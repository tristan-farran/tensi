import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, None, 1e-6)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        return q + k + v


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd)
        self.attn = Attention(config)

    def forward(self, x):
        return x + self.attn(self.ln(x))


class Encoder(nn.Module):
    def __init__(self, vocab, dim, layers=2):
        super().__init__()
        self.wte = nn.Embedding(vocab, dim)
        cfg = type("Config", (object,), {"n_embd": dim})
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(layers)])
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, 3, padding=1) for _ in range(layers)])
        self.ln = LayerNorm(dim)

    def _conv_stack(self, x):
        for c in self.convs:
            x = x + c(self.ln(x).transpose(1, 2)).transpose(1, 2)
        return x

    def forward(self, idx):
        x = self.wte(idx)
        x = self._conv_stack(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        cfg = type("Config", (object,), {"n_embd": dim})
        self.first = Block(cfg)
        self.rest = nn.ModuleList([Block(cfg)])

    def forward(self, z):
        x = self.first(z)
        for blk in self.rest:
            x = blk(x)
        return x


def lengths(prompts):
    return torch.tensor([len(p) for p in prompts]).unsqueeze(1)
