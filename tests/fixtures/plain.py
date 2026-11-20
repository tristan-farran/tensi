import torch

def make(batch_size: int, seq_len: int, d_model: int):
    x = torch.zeros(batch_size, seq_len, d_model)
    w = torch.randn(d_model, 10)
    y = x @ w
    s = y.sum(dim=1)
    flat = x.view(batch_size, -1)
    t = x.transpose(1, 2)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    filled = torch.zeros(seq_len, seq_len).masked_fill(~mask, 0.0)
    sel = x[x > 0]
    return y
