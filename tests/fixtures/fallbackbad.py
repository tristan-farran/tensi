import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module):
    def __init__(self, dim, other):
        super().__init__()
        self.a = Mlp(dim)
        self.b = Mlp(dim, other)
        self.c = nn.Linear(dim, dim)

    def forward(self, x):
        y = self.a(x)
        z = self.b(x)
        return self.c(y) + z


class Pool(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int | None = None):
        super().__init__()
        self.output_dim = output_dim or embed_dim
        self.to_out = nn.Linear(embed_dim, self.output_dim)

    def forward(self, x):
        return self.to_out(x)


class Combined(nn.Module):
    def __init__(self, embedding_dim: int, text_dim: int):
        super().__init__()
        self.pool = Pool(text_dim, output_dim=embedding_dim)
        self.time = nn.Linear(16, embedding_dim)

    def forward(self, t, h):
        return self.time(t) + self.pool(h)
