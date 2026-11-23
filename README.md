<p align="center">
  <img src="https://raw.githubusercontent.com/tristan-farran/tensi/main/brand/tensi-logo.svg" alt="tensi" width="440">
</p>

<p align="center">
  <b>tensi sees your tensors, and now you do too!</b><br>
  Hover any PyTorch tensor and see its shape, with <i>named dimensions</i>.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/tristan-farran/tensi/main/docs/demo.gif" alt="tensi showing named tensor shapes in nanoGPT">
</p>

## Install


<p align="center">
  <a href="https://marketplace.visualstudio.com/items?itemName=TristanFarran.tensi">VS Code</a> ·
  <a href="https://pypi.org/project/tensi-shapes/">PyPI</a>
</p>

**Command line and CI:** `uv add --dev tensi-shapes` (or `pip install tensi-shapes`).

```bash
tensi peek model.py   # every tensor's shape
tensi check src/      # shape errors; exits 1 if any
```

## What you annotate

tensi works most shapes out on its own: from `b, t = idx.size()`, from callers in the same file,
from einops patterns and layer sizes. What it can't see is the shape of data arriving from outside
the file, so it tells you exactly which inputs need one annotation, and what to write:

```python
from jaxtyping import Shaped

def forward(self, idx: Shaped[torch.Tensor, "batch seq"]):
```

* One name per dimension, in order. A size name from `__init__` (like `n_embd`) links that axis to
  the layer sizes, a number fixes a size, any other name is a new size.
* In VS Code each such input has a blue hint with the line to write and a quick fix (Ctrl+. /
  Cmd+.) that writes it, or annotates every input in the file at once; the status bar counts them,
  and hovering a tensor with no shape says which input it depends on. `tensi peek` lists the same.
* jaxtyping has to be installed in your project (`uv add jaxtyping` or `pip install jaxtyping`).
  `Float`, `Int` and `Bool` work as well as `Shaped`.

tensi says when a newer release is out and how to install it (checked at most once a day, never in CI; `TENSI_NO_UPDATE_CHECK=1` turns it off).

<sub>Built on Pyrefly :)</sub>
