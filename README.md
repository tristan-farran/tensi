<p align="center">
  <img src="https://raw.githubusercontent.com/tristan-farran/tensi/main/brand/tensi-logo.svg" alt="tensi" width="440">
</p>

<p align="center">
  <b>tensi sees your tensors.</b><br>
  Hover any PyTorch tensor and see its shape, with every dimension named.
</p>

---

```python
def forward(self, x):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    #   hover k → Tensor[[b, n_head, t, n_embd // n_head]]
```


## Install

**VS Code:** install **tensi** from the Marketplace. It follows each project's interpreter (the Python
extension's choice, else `.venv`), so switching projects or running `uv sync` needs no setup. If tensi is
missing it offers `uv add --dev tensi-shapes` for the project, or `uv tool install tensi-shapes` once for
all projects. It runs beside Pylance and only adds shapes.

**Command line, CI, other editors:**

```bash
uv add --dev tensi-shapes      # or: pip install tensi-shapes, into the project's environment
uv tool install tensi-shapes   # or once for all projects; it reads torch from each project's .venv
tensi doctor                   # checks Pyrefly, the stubs and the interpreter Pyrefly picks, with one real probe
```

The package is `tensi-shapes` on PyPI; the command and the import are `tensi`. It brings Pyrefly 1.5.0
and its torch shape stubs; your project needs `torch` installed.

Any editor that speaks LSP can run `tensi lsp` as a language server for Python files. Python 3.10+;
`rearrange` and friends are shape-aware on Python 3.12+ (Pyrefly's einops stubs need it).

## Try it

```console
$ tensi peek model.py
tensi peek v0.3.0  model.py  2.0s

  CausalSelfAttention.forward
    56  q, k, v   [b, t, n_embd]
    57  k         [b, n_head, t, n_embd // n_head]
    67  att       [b, n_head, t, t]
    72  y         [b, t, n_embd]
  MLP.forward
    88  x         [b, t, 4 * n_embd]
  GPT.forward
   177  tok_emb   [b, t, n_embd]
   178  pos_emb   [t, n_embd]
   186  logits    [b, t, vocab_size]
  GPT.generate
   321  v, _      [b, int]
   326  idx_next  [b, 1]

  37 tensors, 32 fully named
```

```console
$ tensi check model.py
tensi check v0.3.0  model.py  2.0s
  ok no shape errors

$ tensi check bad.py
tensi check v0.3.0  bad.py  819ms
  x bad.py:17:9  Argument `Tensor[[b, 7, v]]` is not assignable to parameter `x` with type `Tensor[[b, 7, d]]` in function `Block.forward`
  x bad.py:18:9  `@` is not supported between `Tensor[[b, d]]` and `Tensor[[v, 3]]`

  2 shape errors
```

`tensi check src/` checks every Python file under a folder and exits 1 when it finds a shape error, so it
can gate CI.

## tensi never asks you to write a shape it can find on its own

| You wrote | tensi reads it as |
| --- | --- |
| `def __init__(self, d_model: int)` or just `(self, d_model)` passed to `nn.Linear` | a dimension called `d_model` |
| `config.n_embd`, with `@dataclass class GPTConfig: n_embd: int = 768` | a dimension called `n_embd` |
| `b, t = idx.size()` at the top of a `forward` | `idx` is `[b, t]` |
| `b, n, _ = x.shape` anywhere, on any tensor | every dim of `x` that was only `int` is now `b` or `n`, from that line on |
| `seq = x.shape[1]` or `n = x.size(1)` | that dim of `x` is `seq` (or `n`), if it was only `int` |
| `hidden_dim = int(dim * mult)` in `__init__` | a dimension called `hidden_dim`, where Pyrefly can only say `int` |
| `ph, pw = pair(patch_size)` with the usual `def pair(t): return t if isinstance(t, tuple) else (t, t)` | `ph` and `pw` are both `patch_size` |
| `pe = posemb_sincos_2d(x)` or `self.split_heads(q)`, a helper in the same file | the helper's parameter takes the shape the caller passes |
| `_, h, w, dim, device = *x.shape, x.device` | `x` is `[int, h, w, dim]` |
| `assert dim % 4 == 0` in a helper that returns `[n, 4 * (dim // 4)]` | its result is `[n, dim]` |
| `self.conv1 = nn.Conv2d(inplanes, planes, 3)`, then `out = self.conv1(x)` in a `forward` nothing calls | `x` is `[int, inplanes, int, int]`: the first layer that reads an input fixes its rank and channels (Conv, BatchNorm, pooling, Embedding) |
| `out_features = out_features or in_features` (or `if d is None: d = dim`) | a dimension called `out_features`, and `Mlp(dim)` with it left out has `out_features` = `dim` |
| `def forward(self, x: torch.Tensor) -> torch.Tensor`, `out: torch.Tensor = ...` | nothing: a bare `Tensor` annotation would erase the shape, so tensi reads past it |
| `act_layer=nn.GELU`, `norm_layer=None` then `norm_layer = norm_layer or nn.LayerNorm`, `nn.BatchNorm2d(c) if use_bn else nn.Identity()` | the layer you'd get by default, with its shapes |
| `self.attn(self.ln_1(x))` | `attn.forward` receives whatever shape `ln_1` gives it |
| `rearrange(img, 'b c (h p1) (w p2) -> ...')` or a `Rearrange(...)` layer as a root module's first step | `img` is `[b, c, int, int]` |
| `self.layers.append(nn.ModuleList([Attention(...), FeedForward(...)]))` in a loop, then `for attn, ff in self.layers` | `attn` is an `Attention`, `ff` a `FeedForward`, each with its own shapes |
| `def forward(self, x: Float[Tensor, "batch seq d_model"])` | exactly that, with `d_model` tied to the constructor's |

The jaxtyping annotation in the last row is the one thing you might write by hand, for a module whose
input no other code in the project produces. tensi tells you which ones: `tensi peek` lists them under
"annotate the input of", and the editor shows a faint hint on that parameter. 

## Settings

| VS Code setting | Env var | Default | |
| --- | --- | --- | --- |
| `tensi.pythonPath` | | empty | leave empty to follow each project's interpreter; set it only to pin one |
| `tensi.hover` | `TENSI_HOVER` | `tensors` | only hovers that carry a shape; `all` for every Pyrefly hover |
| `tensi.diagnostics` | `TENSI_DIAGNOSTICS` | `shapes` | `shapes`, `all` or `off` |
| `tensi.inlayHints` | `TENSI_INLAY_HINTS=1` | off | shapes inline, after each variable |
| `tensi.shadowWorkspace` | `TENSI_SHADOW_WORKSPACE` | on | shapes survive imports between your files |
| | `TENSI_STUBS` | `1` | tensi's patched torch stubs (see below); `0` for Pyrefly's own |
| | `TENSI_PYREFLY` | | the Pyrefly executable, if not the one installed beside tensi |
| | `TENSI_LOG=/path` | | log every LSP message |

## Contributing

See [CONTRIBUTING.md](https://github.com/tristan-farran/tensi/blob/main/CONTRIBUTING.md) for the development setup, tests and releases.

<sub>tensi is built on Pyrefly (MIT) and its torch stubs. The mascot is a small tensor that is very good at
looking.</sub>
