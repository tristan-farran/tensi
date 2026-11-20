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

Any editor that speaks LSP can run `tensi lsp` as a language server for Python files. Python 3.12+;
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

## Contributing

See [CONTRIBUTING.md](https://github.com/tristan-farran/tensi/blob/main/CONTRIBUTING.md) for the development setup, tests and releases.

<sub>tensi is built on Pyrefly (MIT) and its torch stubs. The mascot is a small tensor that is very good at
looking.</sub>
