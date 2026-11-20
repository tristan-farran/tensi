import torch
from torch import nn
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FF(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())

    def forward(self, x):
        h = self.net(x)
        return h


class Patches(nn.Module):
    def __init__(self, patch_size, dim, channels=3):
        super().__init__()
        ph, pw = pair(patch_size)
        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.Linear(channels * ph * pw, dim),
        )
        self.ff = FF(dim)

    def forward(self, img):
        x = self.embed(img)
        seq = x.shape[1]
        y = x[:, :seq]
        return self.ff(y)


m = Patches(16, 64)
m(torch.randn(2, 3, 32, 32))
