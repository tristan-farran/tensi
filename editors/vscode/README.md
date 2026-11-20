# tensi for VS Code

**Hover any PyTorch tensor and see its shape, with every dimension named after the variable that sized it.**
No annotations needed, and your files are never modified.

```python
def forward(self, x):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    #   hover k → Tensor[[b, n_head, t, n_embd // n_head]]
```

That's nanoGPT, unmodified. tensi works out that `config.n_embd` is a dimension and follows `x` in from
the modules that call it. Shape errors are underlined in place, in the same names:

```
Argument `Tensor[[b, 7, v]]` is not assignable to parameter `x` with type `Tensor[[b, 7, d]]`
```

Where a size really comes from the data (the batch, the image), tensi says so and suggests the one
jaxtyping annotation that would name it, e.g. `x: Shaped[Tensor, "batch seq in_features"]`.

## Getting started

1. Open a folder with PyTorch code. tensi uses the project's interpreter: the one picked in the Python
   extension, else the project's `.venv` (what `uv sync` creates).
2. If tensi isn't installed yet, it offers two ways:
   * **Add to this project** runs `uv add --dev tensi-shapes` in a uv project (or installs into the
     interpreter otherwise), so `uv sync` keeps it.
   * **Install for all projects** runs `uv tool install tensi-shapes` once. tensi then works in every
     project, and still reads torch and your packages from each project's own environment.
3. Hover a tensor.

Switching projects, changing the interpreter, or running `uv sync` is picked up on its own: there is
nothing to set and no server to restart.

If something looks off, run **tensi: Check setup** from the Command Palette. It checks Pyrefly, the stubs
and the interpreter, runs one real probe, and prints a fix for anything that fails.

tensi runs beside Pylance. By default it only adds hovers that carry a tensor shape, and only reports
shape errors, so nothing is duplicated.

## Settings

| Setting | Default | |
| --- | --- | --- |
| `tensi.pythonPath` | empty | leave empty to follow each project's interpreter; set it only to pin one |
| `tensi.hover` | `tensors` | only hovers that carry a shape; `all` for every Pyrefly hover |
| `tensi.diagnostics` | `shapes` | `shapes`, `all` or `off` |
| `tensi.inlayHints` | off | shapes inline, after each variable |
| `tensi.shadowWorkspace` | on | shapes survive imports between your files |

## Commands

* **tensi: Check setup** runs `tensi doctor` on the workspace.
* **tensi: Restart server**

## Requirements

Python 3.10 or newer with `torch` installed. The same package gives you a CLI: `tensi peek model.py`
lists every tensor's shape, and `tensi check src/` exits 1 on a shape error, for CI.

tensi is built on Pyrefly's experimental tensor-shape support. Source, docs and issues:
[github.com/tristan-farran/tensi](https://github.com/tristan-farran/tensi).
