import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        b, t, d = x.shape
        h = self.fc1(x)
        return self.fc2(h) + x


def run(batch, seq):
    m = MLP(64)
    x = torch.zeros(batch, seq, 64)
    return m(x)
