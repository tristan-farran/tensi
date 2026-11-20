# Changelog

## 0.3.1

* Works on projects on any Python from 3.10. tensi's rewritten code uses newer type-parameter syntax,
  so on a 3.10 or 3.11 project Pyrefly's language server answered no hovers at all; tensi now tells
  Pyrefly to accept that syntax (it still reads packages from the project's own interpreter).
* The einops shape stubs ship inside tensi, so `rearrange` and `Rearrange` are shaped whatever Python
  tensi is installed on.
* The shape solver no longer skips silently on Python 3.10 to 3.12.

## 0.3.0

First public release.

* Hovers and shape errors for unannotated PyTorch, with dimensions named after the variables that sized
  them: constructor sizes, `config.n_embd`, `b, t = x.size()`, `x.shape[1]`, `pair(patch_size)`,
  `hidden = int(dim * mult)`, `out_features or in_features`, and shapes flowing in from the modules and
  helpers that call each `forward`.
* Einops patterns and `Rearrange` layers, `nn.ModuleList`/`ModuleDict` loops, layer factories
  (`act_layer=nn.GELU`), and the first layer that reads an input (Conv, BatchNorm, pooling, Embedding).
* `check` reports only shape errors, so it can gate CI: `tensi check src/` exits 1 on any.
* Hints at the one input annotation that would name a size coming from the data, with a suggestion.
* A patch set over Pyrefly's torch stubs (norm layers, `cat`/`stack`, comparisons, `amax`/`amin`, ...).
  `tensi stubs --diff` prints it as a patch for upstream.
* `tensi doctor` checks the setup with one real probe.
* VS Code extension: uses the Python extension's interpreter, offers to install tensi there, and adds a
  **tensi: Check setup** command.
