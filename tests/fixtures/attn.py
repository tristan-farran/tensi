import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

class Attn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, vocab_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = q @ k.transpose(-2, -1)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = F.softmax(att.masked_fill(~mask, float("-inf")), dim=-1)
        scores = torch.einsum("bhij,bhjd->bhid", att, q)
        merged = scores.transpose(1, 2).reshape(B, T, D)
        logits = self.out(merged)
        return logits

def run(batch_size: int, seq_len: int, d_model: int, n_heads: int):
    m = Attn(d_model, n_heads, 1000)
    x = torch.randn(batch_size, seq_len, d_model)
    y = m(x)
    return y
