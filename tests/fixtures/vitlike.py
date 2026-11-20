import torch
from torch import nn
from torch.nn import ModuleList
from einops import rearrange
from einops.layers.torch import Rearrange


class ChanNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x * self.g


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        project_out = not (heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(nn.Linear(inner, dim)) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2))
        if mask is not None:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, 0.0)
        out = torch.matmul(dots.softmax(dim=-1), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                nn.Linear(dim, dim),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, dim_head=64, pool='cls'):
        super().__init__()
        self.to_patch = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                                      nn.LazyLinear(dim))
        num_cls = 1 if pool == 'cls' else 0
        self.cls_token = nn.Parameter(torch.randn(num_cls, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head)
        self.norm = ChanNorm(dim)

    def forward(self, img):
        x = self.to_patch(img)
        x = self.transformer(x)
        cls = self.cls_token
        fmap = rearrange(x, 'b n d -> b d n 1')
        y = self.norm(fmap)
        return x, cls, y
