"""tensi's torch shape stubs: a small, named patch set on top of pyrefly-torch-stubs.

Phase 1 of the fork plan. tensi does not vendor the 13,600 lines of upstream stubs. It copies the
installed `torch-shape-stubs` into a cache folder, applies the patches below, and puts that folder
first on Pyrefly's search path, where it shadows the installed copy. Every patch is an exact text
replacement, so an upstream change that moves the text makes that one patch drop out (and
`tensi stubs` says so) instead of silently doing something else.

Each patch is meant to go upstream; `tensi stubs --diff` prints the whole set as a unified diff.
Disable with TENSI_STUBS=0.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

STUBS_DIR = "torch-shape-stubs"


@dataclass(frozen=True)
class Patch:
    name: str
    file: str  # relative to torch-shape-stubs/
    why: str
    old: str = ""  # exact text, must occur exactly once ("" = append `new` to the file)
    new: str = ""
    fn: Callable[[str], str | None] | None = None  # for edits a single replacement can't express

    def apply(self, text: str) -> str | None:
        """Patched text, or None when the upstream text this patch expects is not there."""
        if self.fn is not None:
            return self.fn(text)
        if not self.old:
            return text.rstrip("\n") + "\n" + self.new
        if text.count(self.old) != 1:
            return None
        return text.replace(self.old, self.new)


# ------------------------------------------------------------------ the patch set

_DSL = """

# ---- tensi patches -----------------------------------------------------------
# Pyrefly 1.5.0 DSL quirks shape these: a subscript compared inline inside a call
# (`is_concrete_int(shape[-1])`) makes the whole call evaluate to Unknown, so it is bound to
# a local first; and touching a gradual `int` argument makes the result a bare Tensor, so the
# modules route an unknown size to a plain shape-preserving overload before the DSL one.

@type_shape_dsl_function
def trailing_extent_shape(shape: IntTuple, extent: Int) -> IntTuple:
    if len(shape) == 0:
        return dsl.Invalid("normalization needs at least a 1D input")
    last = shape[-1]
    if dsl.is_concrete_int(last) and dsl.is_concrete_int(extent) and last != extent:
        return dsl.Invalid("normalized size does not match the input's last dimension")
    return shape

@type_shape_dsl_function
def channel_extent_shape(shape: IntTuple, channels: Int) -> IntTuple:
    if len(shape) < 2:
        return dsl.Invalid("normalization needs a (batch, channels, ...) input")
    actual = shape[1]
    if dsl.is_concrete_int(actual) and dsl.is_concrete_int(channels) and actual != channels:
        return dsl.Invalid("num_features does not match the input's channel dimension")
    return shape
"""


def _nn_imports(text: str) -> str | None:
    old = "        symmetric_pad2d_shape,\n    )"
    if text.count(old) != 1:
        return None
    return text.replace(
        old, "        symmetric_pad2d_shape,\n        trailing_extent_shape,\n        channel_extent_shape,\n    )"
    )


def _typed_norm(cls: str, param: str, dsl: str, base: str = "Module") -> Callable[[str], str | None]:
    """`class LayerNorm(Module)` -> `class LayerNorm[N: _Int = int](Module)`: the constructor's size
    becomes a class type argument (`LayerNorm[n_embd]`) and forward checks it against the input."""

    def fn(text: str) -> str | None:
        m = re.search(rf"^class {cls}\({base}\):\n(.*?)(?=^class |^# =====)", text, re.M | re.S)
        if not m:
            return None
        body = m.group(1)
        new_body, n1 = re.subn(
            rf"(\n        {param}): int( \| list\[int\] \| tuple\[int, \.\.\.\])?,", r"\1: N\2,", body, count=1
        )
        # The size is unknown (`nn.LayerNorm(n)` with a plain int, a tuple, a bare annotation):
        # keep the input shape. A DSL call that touches a gradual int would return a bare Tensor.
        new_body, n2 = re.subn(
            r"    def forward\[S: IntTuple\]\(self, input: Tensor\[S\]\) -> Tensor\[S\]: \.\.\.",
            f"    @overload\n"
            f"    def forward[S: IntTuple](self: {cls}[int], input: Tensor[S]) -> Tensor[S]: ...\n"
            f"    @overload\n"
            f"    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[{dsl}(S, N)]: ...",
            new_body,
            count=1,
        )
        if n1 != 1 or n2 != 1:
            return None
        return text[: m.start()] + f"class {cls}[N: _Int = int]({base}):\n" + new_body + text[m.end() :]

    return fn


def _batchnorm(text: str) -> str | None:
    """_BatchNorm carries the channel count; BatchNorm1d/2d/3d inherit it."""
    fn = _typed_norm("_BatchNorm", "num_features", "channel_extent_shape")
    out = fn(text)
    if out is None:
        return None
    for k in ("1d", "2d", "3d"):
        old = f"class BatchNorm{k}(_BatchNorm):"
        if out.count(old) != 1:
            return None
        out = out.replace(old, f"class BatchNorm{k}[N: _Int = int](_BatchNorm[N]):")
    return out


def _new_varargs(text: str) -> str | None:
    """`x.new_zeros(b, 1, d)` (varargs) is as shaped as `x.new_zeros((b, 1, d))`."""
    pat = re.compile(
        r"(    def (new_zeros|new_ones|new_empty))\(\n        self,\n        size: builtins\.int,\n"
        r"        \*sizes: builtins\.int,\n(.*?)\) -> Tensor:",
        re.S,
    )
    out, n = pat.subn(
        lambda m: (
            f"{m.group(1)}[NewShape: IntTuple](\n        self,\n        *size: *NewShape,\n"
            f"{m.group(3)}) -> Tensor[NewShape]:"
        ),
        text,
    )
    return out if n == 3 else None


# Shape rules for a list of tensors, and the fast path for cat/stack with gradual sizes.
_LIST_DSL = """

@type_shape_dsl_function
def cat_list_shape(shape: IntTuple, dim: int) -> IntTuple:
    if dim < 0 - len(shape) or dim >= len(shape):
        return dsl.Invalid("cat dimension out of range")
    if dim < 0:
        axis = dim + len(shape)
    else:
        axis = dim + 0
    return dsl.IntTuple(
        (dsl.Int.gradual() if index == axis else shape[index] for index in range(len(shape)))
    )

@type_shape_dsl_function
def stack_list_shape(shape: IntTuple, dim: int) -> IntTuple:
    output_rank = len(shape) + 1
    if dim < 0 - output_rank or dim >= output_rank:
        return dsl.Invalid("stack dimension out of range")
    if dim < 0:
        axis = dim + output_rank
    else:
        axis = dim + 0
    return dsl.IntTuple(
        (
            dsl.Int.gradual()
            if index == axis
            else shape[index if index < axis else index - 1]
            for index in range(output_rank)
        )
    )
"""

_INPLACE_OLD = """    offset = len(receiver) - len(other)
    if any(
        other[index] != 1 and other[index] != receiver[offset + index]
        for index in range(len(other))
    ):"""
_INPLACE_NEW = """    offset = len(receiver) - len(other)
    # tensi: `out += identity` with sizes Pyrefly can't compare (a gradual `int`, or `planes` against
    # `inplanes`) has the receiver's shape whenever it runs at all; undecidable used to mean bare Tensor.
    if any(not dsl.is_concrete_int(size) for size in other):
        return receiver
    if any(
        other[index] != 1 and other[index] != receiver[offset + index]
        for index in range(len(other))
    ):"""

_CAT_CHECK = """    if any(mismatch == 1 for mismatch in mismatches):
        return dsl.Invalid(
            "cat expects all tensor sizes to match outside the concatenated dimension"
        )
"""
_CAT_FAST = """    # tensi: when the first tensor has a size that is not a literal (a gradual `int`, typically an
    # unannotated batch), comparing sizes is undecidable and the whole call would come out as a
    # bare Tensor. Keep the rank and the first tensor's sizes instead; literal mismatches are
    # still reported below.
    if any(not dsl.is_concrete_int(size) for size in first):
        return dsl.IntTuple(
            (
                dsl.sum(dsl.IntTuple((shape[axis] for shape in shapes)))
                if index == axis
                else first[index]
                for index in range(len(first))
            )
        )
"""
_STACK_CHECK = """    if any(mismatch == 1 for mismatch in mismatches):
        return dsl.Invalid("stack expects all tensors to have the same shape")
"""
_STACK_FAST = """    if any(not dsl.is_concrete_int(size) for size in first):
        return dsl.IntTuple(
            (
                len(shapes)
                if index == axis
                else first[index if index < axis else index - 1]
                for index in range(output_rank)
            )
        )
"""


def _cat_stack_gradual(text: str) -> str | None:
    if text.count(_CAT_CHECK) != 1 or text.count(_STACK_CHECK) != 1:
        return None
    return (
        text.replace(_CAT_CHECK, _CAT_FAST + _CAT_CHECK).replace(_STACK_CHECK, _STACK_FAST + _STACK_CHECK).rstrip("\n")
        + "\n"
        + _LIST_DSL
    )


def _list_overloads(text: str) -> str | None:
    imp = "from torch._shapes import (\n"
    cat_old = (
        "@overload\ndef cat(\n    tensors: list[Tensor], dim: builtins.int = 0, *, out: Tensor | None = None\n"
        ") -> Tensor: ...\n"
    )
    stack_old = "@overload\ndef stack(tensors: Sequence[Any], dim: builtins.int = 0) -> Tensor: ...\n"
    if any(text.count(o) != 1 for o in (imp, cat_old, stack_old)):
        return None
    cat_new = (
        "@overload\ndef cat[S: IntTuple, Dim: Flag[builtins.int]](\n"
        "    tensors: list[Tensor[S]], dim: Dim = 0, *, out: Tensor | None = None\n"
        ") -> Tensor[cat_list_shape(S, Dim)]: ...\n"
    )
    stack_new = (
        "@overload\ndef stack[S: IntTuple, Dim: Flag[builtins.int]](\n"
        "    tensors: list[Tensor[S]], dim: Dim = 0\n) -> Tensor[stack_list_shape(S, Dim)]: ...\n"
    )
    cat_first = "@overload\ndef cat[Shapes: IntTuples, Dim: Flag[builtins.int]](\n"
    stack_first = "@overload\ndef stack[Shapes: IntTuples, Dim: Flag[builtins.int]](\n"
    if text.count(cat_first) != 1 or text.count(stack_first) != 1:
        return None
    # before the per-member overloads, which accept a list too but can only make it a bare Tensor
    return (
        text.replace(imp, imp + "    cat_list_shape,\n    stack_list_shape,\n", 1)
        .replace(cat_first, cat_new + cat_first)
        .replace(stack_first, stack_new + stack_first)
    )


_ORDERING_OLD = """    def __lt__(self, other: Tensor | builtins.float | builtins.int) -> Any: ...
    def __le__(self, other: Tensor | builtins.float | builtins.int) -> Any: ...
    def __gt__(self, other: Tensor | builtins.float | builtins.int) -> Any: ...
    def __ge__(self, other: Tensor | builtins.float | builtins.int) -> Any: ...
"""
_ORDERING_NEW = "".join(
    f"""    def {op}[OtherShape: _Shape = []](
        self, other: _TensorLike[OtherShape]
    ) -> Tensor[broadcast(Shape, OtherShape)]: ...
"""
    for op in ("__lt__", "__le__", "__gt__", "__ge__")
)

_TYPE_OLD = """        self, dtype: None = None, non_blocking: builtins.bool = False
    ) -> builtins.str: ...
    @overload
    def type(
        self,
        dtype: builtins.str | dtype | type[Tensor],
        non_blocking: builtins.bool = False,
    ) -> Self:"""
_TYPE_NEW = """        self, *, non_blocking: builtins.bool = False
    ) -> builtins.str: ...
    @overload
    def type(
        self,
        dtype: Any,
        non_blocking: builtins.bool = False,
    ) -> Self:"""

_LOGSUMEXP = "    def logsumexp[\n"
_AMAX = "".join(
    f"""    def {name}[
        Shape: IntTuple,
        Dim: Flag[builtins.int | tuple[builtins.int, ...]],
        Keepdim: Flag[builtins.bool],
    ](
        self: Tensor[Shape], dim: Dim, keepdim: Keepdim = False
    ) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]: ...
"""
    for name in ("amax", "amin")
)


def _amax(text: str) -> str | None:
    if text.count(_LOGSUMEXP) != 1 or "    def amax" in text:
        return None
    return text.replace(_LOGSUMEXP, _AMAX + _LOGSUMEXP)


_BINARY_OPS = (
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "iadd",
    "isub",
    "imul",
    "itruediv",
    "ifloordiv",
    "imod",
)
_BINARY_RE = re.compile(
    r"(    def __(?:" + "|".join(_BINARY_OPS) + r")__\[OtherShape: _Shape = \[\]\]\(\n        self, other: )"
    r"_TensorLike\[OtherShape\]"
)


def _binary_ops(text: str) -> str | None:
    """`s * x` with `s` a 0-d tensor: Pyrefly lets a Tensor stand for a Python scalar, so against
    `other: Tensor[OtherShape] | _Scalar` it took the scalar side and `OtherShape` fell back to its
    `[]` default; the result kept only `s`'s shape (`[] * [b, t, d]` was `[]`, `[2, 3] * [5, 2, 3]`
    was `[2, 3]`). Typed as `Tensor[OtherShape]` alone, a tensor binds its shape and a scalar,
    which Pyrefly also accepts there, still takes the `[]` default. `**` is left alone: without the
    union Pyrefly reads `x ** 2` through `int.__rpow__` and makes it an `int`."""
    out, n = _BINARY_RE.subn(r"\1Tensor[OtherShape]", text)
    # upstream's `<`/`>` family only takes this form once `ordering-shaped` has run
    return out if n >= len(_BINARY_OPS) - 4 else None


_SPLIT_SUM_OLD = """    total = dsl.sum(sections)
    if dsl.is_concrete_int(extent) and dsl.is_concrete_int(total) and extent != total:
        return dsl.Invalid("split sections must sum to the selected dimension")
"""
_SPLIT_SUM_NEW = """    if not any(not dsl.is_concrete_int(section) for section in sections):
        total = dsl.sum(sections)
        if dsl.is_concrete_int(extent) and dsl.is_concrete_int(total) and extent != total:
            return dsl.Invalid("split sections must sum to the selected dimension")
"""


PATCHES: list[Patch] = [
    Patch("dsl-norm-checks", "_shapes.pyi", "Shape functions used by the typed norm layers below.", new=_DSL),
    Patch("nn-import-dsl", "nn/__init__.pyi", "Import the two new shape functions.", fn=_nn_imports),
    Patch(
        "layernorm-typed",
        "nn/__init__.pyi",
        "nn.LayerNorm(d) keeps d (`LayerNorm[d]`) and rejects inputs whose last dim is a different size.",
        fn=_typed_norm("LayerNorm", "normalized_shape", "trailing_extent_shape"),
    ),
    Patch(
        "rmsnorm-typed",
        "nn/__init__.pyi",
        "Same as LayerNorm, for nn.RMSNorm.",
        fn=_typed_norm("RMSNorm", "normalized_shape", "trailing_extent_shape"),
    ),
    Patch(
        "batchnorm-typed",
        "nn/__init__.pyi",
        "nn.BatchNorm1d/2d/3d(c) keep c and check it against the input's channel dim.",
        fn=_batchnorm,
    ),
    Patch(
        "instancenorm-typed",
        "nn/__init__.pyi",
        "Same, for nn.InstanceNorm1d/2d/3d.",
        fn=lambda t: _chain(
            t, *(_typed_norm(f"InstanceNorm{k}", "num_features", "channel_extent_shape") for k in ("1d", "2d", "3d"))
        ),
    ),
    Patch(
        "groupnorm-typed",
        "nn/__init__.pyi",
        "nn.GroupNorm(g, c) keeps c and checks the channel dim.",
        fn=_typed_norm("GroupNorm", "num_channels", "channel_extent_shape"),
    ),
    Patch(
        "tensor-mT-roll",
        "__init__.pyi",
        "x.mT swaps the last two dims; x.roll keeps the shape (both were bare).",
        old="    @property\n    def T(self: Tensor[Shape]) -> Tensor[reverse_shape(Shape)]: ...\n",
        new="    @property\n    def T(self: Tensor[Shape]) -> Tensor[reverse_shape(Shape)]: ...\n"
        "    @property\n    def mT(self: Tensor[Shape]) -> Tensor[transpose_shape(Shape, -2, -1)]: ...\n"
        "    def roll(self: Tensor[Shape], shifts: Any, dims: Any = None) -> Tensor[Shape]: ...\n",
    ),
    Patch(
        "tensor-new-varargs",
        "__init__.pyi",
        "x.new_zeros(b, 1, d) keeps its shape like the tuple form.",
        fn=_new_varargs,
    ),
    Patch(
        "cat-stack-gradual",
        "_shapes.pyi",
        "torch.cat/stack keep the rank and sizes when a size is a gradual int (an unannotated batch); "
        "they were a bare Tensor, which erased every shape downstream.",
        fn=_cat_stack_gradual,
    ),
    Patch(
        "inplace-keeps-receiver",
        "_shapes.pyi",
        "x += y keeps x's shape when y's sizes are not literal (was a bare Tensor when undecidable).",
        old=_INPLACE_OLD,
        new=_INPLACE_NEW,
    ),
    Patch(
        "cat-stack-lists",
        "__init__.pyi",
        "torch.cat/stack of a list[Tensor[S]] keep S, with the joined dim unknown.",
        fn=_list_overloads,
    ),
    Patch(
        "ordering-shaped",
        "__init__.pyi",
        "x > 0 (and <, <=, >=) is a shaped bool tensor like ==, so masks built from it keep their shape.",
        old=_ORDERING_OLD,
        new=_ORDERING_NEW,
    ),
    Patch(
        "tensor-type-dtype",
        "__init__.pyi",
        "x.type(torch.float32) keeps the shape (inside Tensor, `dtype` and `type` resolve to the property and the method, so the overload never matched).",
        old=_TYPE_OLD,
        new=_TYPE_NEW,
    ),
    Patch("tensor-amax-amin", "__init__.pyi", "x.amax(dim)/x.amin(dim) reduce like x.max(dim).values.", fn=_amax),
    Patch(
        "tensor-list-unknown-rank",
        "__init__.pyi",
        "torch.tensor(ids) with `ids: list[int]` has an unknown shape; the literal-only overload's `= []` "
        "default made every list that is not a literal a 0-d tensor, so `.unsqueeze(1)` read as an error.",
        old="def tensor[Shape: IntTuple = []](\n    data: Tensor[Shape] | RegularNestedList[Shape, _TensorScalar],\n",
        new="def tensor[Shape: IntTuple](\n    data: Tensor[Shape] | RegularNestedList[Shape, _TensorScalar],\n",
    ),
    Patch(
        "split-gradual-sections",
        "_shapes.pyi",
        "x.split([n, m, m], dim=2) with unannotated sizes keeps every piece's rank (`[b, t, int]`); "
        "the sum check touched the gradual sizes and made the whole result a bare tuple.",
        old=_SPLIT_SUM_OLD,
        new=_SPLIT_SUM_NEW,
    ),
    Patch(
        "binary-ops-broadcast",
        "__init__.pyi",
        "s * x, x + y and x == y broadcast both shapes; a 0-d or lower-rank left operand made the "
        "result its own shape (`self.scale[i] * x` was 0-d, erasing every shape after it).",
        fn=_binary_ops,
    ),
]


def _chain(text: str, *fns: Callable[[str], str | None]) -> str | None:
    for f in fns:
        text = f(text)  # type: ignore[assignment]
        if text is None:
            return None
    return text


# ------------------------------------------------------------------ building the overlay


def upstream_dir() -> Path | None:
    """The installed `torch-shape-stubs` folder (from pyrefly-torch-stubs)."""
    for entry in sys.path:
        p = Path(entry or ".") / STUBS_DIR
        if (p / "__init__.pyi").exists():
            return p
    return None


def _key(upstream: Path) -> str:
    h = hashlib.sha1()
    try:
        from importlib.metadata import version

        h.update(version("pyrefly-torch-stubs").encode())
    except Exception:
        h.update(str((upstream / "__init__.pyi").stat().st_mtime_ns).encode())
    for name, d in sorted(companion_dirs().items()):
        init = next(iter(sorted(d.glob("__init__.py*"))))
        h.update(f"{name}:{init.stat().st_mtime_ns}".encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:12]


# Packages Pyrefly needs beside the torch stubs. Carrying them in the overlay means a project
# whose own virtualenv lacks them (tensi installed with pipx or uv tool) still gets shapes.
COMPANIONS = ("shape_extensions", "einops-shape-stubs")


# Shipped inside tensi, for environments where the stub package can't be installed (the einops
# stubs' package requires Python 3.12, but its `.pyi` files serve a project on any version).
VENDORED = Path(__file__).parent / "_vendor"


def companion_dirs() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in COMPANIONS:
        for entry in [*sys.path, str(VENDORED)]:
            p = Path(entry or ".") / name
            if p.is_dir() and any(p.glob("__init__.py*")):
                found[name] = p
                break
    return found


def build(upstream: Path, dest: Path) -> tuple[list[str], list[str]]:
    """Copy `upstream` to `dest`/torch-shape-stubs and apply PATCHES. Returns (applied, skipped)."""
    target = dest / STUBS_DIR
    shutil.copytree(upstream, target)
    for name, src in companion_dirs().items():
        shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
    applied, skipped = [], []
    for p in PATCHES:
        f = target / p.file
        out = p.apply(f.read_text(encoding="utf-8"))
        if out is None:
            skipped.append(p.name)
            continue
        f.write_text(out, encoding="utf-8")
        applied.append(p.name)
    (dest / "PATCHES").write_text("applied: " + " ".join(applied) + "\nskipped: " + " ".join(skipped) + "\n")
    return applied, skipped


def overlay_dir() -> Path | None:
    """A search-path root holding the patched stubs, built once per (stubs version, patch set)."""
    if os.environ.get("TENSI_STUBS", "1") == "0":
        return None
    upstream = upstream_dir()
    if upstream is None:
        return None
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tensi" / "stubs"
    dest = base / _key(upstream)
    if (dest / STUBS_DIR / "__init__.pyi").exists():
        return dest
    try:
        base.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="build-", dir=base))
        build(upstream, tmp)
        try:
            tmp.rename(dest)
        except OSError:  # another process won the race
            shutil.rmtree(tmp, ignore_errors=True)
        return dest
    except Exception:
        return None


def search_args() -> list[str]:
    """Extra `pyrefly check` arguments that put the patched stubs first."""
    d = overlay_dir()
    return ["--search-path", str(d)] if d else []


def status() -> str:
    d = overlay_dir()
    if d is None:
        return (
            "tensi stubs: off" if os.environ.get("TENSI_STUBS") == "0" else "tensi stubs: pyrefly-torch-stubs not found"
        )
    return f"tensi stubs: {d}\n" + (d / "PATCHES").read_text()


def diff() -> str:
    """The whole patch set as one unified diff against the installed upstream stubs."""
    upstream = upstream_dir()
    d = overlay_dir()
    if upstream is None or d is None:
        return ""
    out = []
    for f in sorted({p.file for p in PATCHES}):
        a = (upstream / f).read_text(encoding="utf-8").splitlines(keepends=True)
        b = (d / STUBS_DIR / f).read_text(encoding="utf-8").splitlines(keepends=True)
        out.extend(difflib.unified_diff(a, b, f"a/{STUBS_DIR}/{f}", f"b/{STUBS_DIR}/{f}"))
    return "".join(out)
