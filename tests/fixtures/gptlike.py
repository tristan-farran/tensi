"""A small GPT in the style of popular single-file implementations: a config dataclass,
an nn.ModuleDict trunk, unannotated forward() methods. No shape annotations anywhere."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    context: int = 256
    vocab: int = 1000
    width: int = 128
    heads: int = 4
    depth: int = 2
    dropout: float = 0.0


class Norm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, None, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)
        self.heads = config.heads

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        scores = F.softmax((q @ k.transpose(-2, -1)) / 8.0, dim=-1)
        mixed = scores @ v
        merged = mixed.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(merged)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.width, 4 * config.width)
        self.down = nn.Linear(4 * config.width, config.width)

    def forward(self, x):
        x = self.up(x)
        x = F.gelu(x)
        x = self.down(x)
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = Norm(config.width)
        self.attn = SelfAttention(config)
        self.norm2 = Norm(config.width)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trunk = nn.ModuleDict(dict(
            tok=nn.Embedding(config.vocab, config.width),
            pos=nn.Embedding(config.context, config.width),
            layers=nn.ModuleList([Layer(config) for _ in range(config.depth)]),
            norm=Norm(config.width),
        ))
        self.head = nn.Linear(config.width, config.vocab, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long)
        tok_emb = self.trunk.tok(idx)
        pos_emb = self.trunk.pos(pos)
        x = tok_emb + pos_emb
        for layer in self.trunk.layers:
            x = layer(x)
        x = self.trunk.norm(x)
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, steps, temperature=1.0):
        for _ in range(steps):
            logits, _ = self(idx)
            last = logits[:, -1, :] / temperature
            probs = F.softmax(last, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
