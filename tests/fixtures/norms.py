import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, width, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_classes)
        self.stem = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x, img):
        b, t, c = x.shape
        h = self.norm(x)
        logits = self.head(h)
        flipped = h.mT
        pad = h.new_zeros(b, 1, c)
        feats = self.bn(self.stem(img))
        return logits, flipped, pad, feats


class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x: torch.Tensor):
        y = self.norm(torch.zeros(2, 768))
        z = self.bn(torch.zeros(2, 64, 8, 8))
        return y, z
