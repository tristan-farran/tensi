# Changelog

## 0.3.6

* tensi says exactly what to annotate, and how. Every input whose shape comes from outside the file
  gets a hint: module inputs nothing in the file shapes, and tensor parameters of helpers nothing in
  the file calls. Modules another module here already feeds are skipped: annotating the outer one
  shapes them. Each hint gives the signature line to write in the file's own spelling (`torch.Tensor`
  or `Tensor`), the import it still needs, and how to name axes.
* Editor: the hint is a visible information squiggle (it was a faint hint), with a quick fix that
  writes the annotation and its import, and a source action that annotates every input in the file.
  Hovering a tensor tensi can't shape says which input it depends on and what to write, instead of
  showing nothing. The status bar counts the inputs to annotate.
* `tensi peek` ends with the same list: each input, the line to write, and the import to add once.
* No jaxtyping in your project? The fix imports it under `if TYPE_CHECKING:` (with `from __future__
  import annotations` before Python 3.14), so the code runs without it; with jaxtyping installed it
  is a normal import.
* Suggested axes are safe to accept: the number of dimensions comes from the code (where the input
  is passed, unpacked, transposed or interpolated), and several inputs of one function share their
  leading names (`x` "batch seq dim", `mask` "batch seq"). Accepting every suggestion in three real
  projects gives no shape errors.
* An annotated input's axis names reach the helpers it is passed to (`self._conv_stack(x)`), even
  when another caller only knows `int`; they used to turn into `int` at the first helper.
* A call through a `ModuleDict` loop item (`blk["ln"](x)`) is unknown instead of a guess among the
  dict's layers, which could read as a mismatch; a residual `x = x + h` keeps x's shape.

## 0.3.5

* Sizes read from a plain config object keep shapes. `self.n_head = config.n_head`, `P =
  config.prefix_len` or `self.L = sum(self.levels)` were Unknown to Pyrefly, and one Unknown size
  made every `view`, `split`, `full` or slice after it shapeless. When such a value is used as a size,
  tensi reads it as an `int` in the rewritten code it checks (your file is unchanged).
* `x.split([n, m, m], dim=2)` with sizes that aren't literals keeps each piece's rank.
* `x ** 2` keeps x's shape (Pyrefly typed an int-literal power of a tensor as an `int`).
* `z = z.float()` (or `.to(...)`, `.contiguous()`, ...) before `B, L, C = z.shape` no longer stops
  tensi naming z's dims.
* Modules that subclass your own module base, or an imported one, get the "annotate the input of"
  hint like direct `nn.Module` subclasses.

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
