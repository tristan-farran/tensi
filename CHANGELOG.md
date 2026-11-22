# Changelog

## 0.3.4

* No more false shape errors from a guessed input shape. An input tensi had not seeded yet is
  Unknown to Pyrefly, which then picks a stub's first overload (`nn.Embedding(idx)` reads as 1-d,
  so its output is rank 2). The solver used to push that guess into the modules it was passed to,
  and an `x = blk(x)` loop locked it in, so correct encoder/decoder code reported unpack, split and
  transpose errors. Shapes that flow from such an input no longer seed anything.
* `torch.tensor(ids)` with `ids: list[int]` (or a list comprehension) has an unknown shape instead of
  a 0-d one, so `torch.tensor(lengths).unsqueeze(1)` is no longer an error. Literals keep their
  shape (`torch.tensor([[1, 2], [3, 4]])` is `[2, 2]`).
* `a * b`, `a + b`, `a == b` and the other elementwise operators broadcast both shapes. Pyrefly lets
  a tensor stand in for a Python scalar, so the right operand was read as a scalar and the result
  kept the left one's shape: `self.scale[i] * x` with a 0-d `scale[i]` was 0-d (erasing every
  shape after it in nanochat-style residual mixing), and `[2, 3] + [5, 2, 3]` was `[2, 3]`.
* tensi tells you when a newer release is out, with the command that updates your installation
  (`uv tool upgrade tensi-shapes`, the project's `uv add`, pip, or pipx): after `tensi peek` and
  `tensi check` in a terminal, in `tensi doctor`, and once per session in the editor. PyPI is asked
  at most once a day; the check is off in CI and with `TENSI_NO_UPDATE_CHECK=1`.
* tensi warns (CLI, editor) when the Pyrefly it runs is not 1.5.0, the version it is pinned to and
  tested against, instead of quietly losing shapes. `pyrefly-shape-extensions` is pinned too.

## 0.3.3

* No changes to the Python package; released alongside the VS Code extension.

## 0.3.2

* `tensi doctor` no longer warns that tensi is missing from the project when it is installed there.
* `tensi peek | head` and similar pipes exit quietly instead of printing a traceback.
* README: demo and a one-line install at the top.

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
