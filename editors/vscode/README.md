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

1. Open a folder with PyTorch code. tensi uses the project's own `.venv` (what `uv sync` creates), else
   the interpreter picked in the Python extension.
2. If tensi isn't installed there yet, click **Install tensi**. In a uv project that runs
   `uv add --dev tensi-shapes`, so `uv sync` keeps it; otherwise it installs into the project's
   environment. **Install for all projects** (`uv tool install tensi-shapes`) is the other option: tensi
   then works everywhere and still reads torch and your packages from each project's environment.
3. Hover a tensor. The **tensi** item in the status bar spins while it works out a file (a few seconds
   the first time) and settles when shapes are ready.

Switching projects, changing the interpreter, or running `uv sync` is picked up on its own: there is
nothing to set and no server to restart. Any Python from 3.10 up works.

If something looks off, run **tensi: Check setup** from the Command Palette. It checks Pyrefly, the stubs
and the interpreter, runs one real probe, and prints a fix for anything that fails.

tensi runs beside Pylance. By default it only adds hovers that carry a tensor shape, and only reports
shape errors, so nothing is duplicated.

## Settings

| Setting | Default | |
| --- | --- | --- |
| `tensi.pythonPath` | empty | leave empty to follow each project; set it only to pin an interpreter |
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
