import torch
from torch import nn
from torch.nn import functional as F


class LM(nn.Module):
    def __init__(self, vocab, width, block_size):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, idx):
        b, t = idx.size()
        return self.head(self.emb(idx))

    @torch.no_grad()
    def generate(self, idx, steps):
        for _ in range(steps):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
