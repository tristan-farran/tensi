import torch; from shape_extensions import Int as _ts_Int, IntVar as _ts_IntVar; from torch import Tensor as _ts_Tensor

def make[_ts_batch_size: _ts_IntVar, _ts_seq_len: _ts_IntVar, _ts_d_model: _ts_IntVar](batch_size: _ts_Int[_ts_batch_size], seq_len: _ts_Int[_ts_seq_len], d_model: _ts_Int[_ts_d_model]):
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
