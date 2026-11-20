import torch; from shape_extensions import Int as _ts_Int, IntVar as _ts_IntVar; from torch import Tensor as _ts_Tensor
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

class Block[_ts_d_model: _ts_IntVar, _ts_vocab_size: _ts_IntVar](nn.Module):
    def __init__(self, d_model: _ts_Int[_ts_d_model], vocab_size: _ts_Int[_ts_vocab_size]):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward[_ts_b: _ts_IntVar, _ts_t: _ts_IntVar](self, x: Tensor[[_ts_b, _ts_t, _ts_d_model]]):
        return self.proj(x)

def bad[_ts_b: _ts_IntVar, _ts_d: _ts_IntVar, _ts_v: _ts_IntVar](b: _ts_Int[_ts_b], d: _ts_Int[_ts_d], v: _ts_Int[_ts_v]):
    blk = Block(d, v)
    x = torch.zeros(b, 7, v)
    blk(x)
    y = torch.zeros(b, d) @ torch.zeros(v, 3)
