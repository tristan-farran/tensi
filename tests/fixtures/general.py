from typing import Optional, Type

import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer: Type[nn.Module] = nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, inplanes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out += identity
        return out


class Head(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.mlp = Mlp(dim, dim * 4)
        self.proj = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.mlp(x)
        return self.proj(x)
