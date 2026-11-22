"""Seed solver: shapes for unannotated `forward` inputs, so nanoGPT-style code needs no labels.

A *seed* says "treat parameter p of Class.method as Tensor[[...]]". Tokens are:
  "n_embd"   a dim already in scope there (a class / config dim) by its user-facing name
  "~b"       a fresh dim local to that function, displayed as `b`
  "int"/"7"  an unnamed / literal size
  "3 * n_embd" an expression over in-scope dims

Sources, in increasing priority:
  1. unpacking: `b, t = idx.size()` / `B, T, C = x.shape` at the top of a function seeds `idx`
     with fresh dims named as the user named them (zero labels for root modules);
  2. same name: an unseeded param shares the seed of a same-named param in another method of the
     same class (`generate(idx)` follows `forward(idx)`);
  3. call sites: Pyrefly reveals what each module call actually receives (`self.attn(self.ln_1(x))`)
     and that shape is pushed into the callee's `forward`, translated into the callee's own dim names
     through the callee's revealed type arguments (`LayerNorm[n_embd]` -> `ndim`);
  4. unification: when Pyrefly reports `Tensor[[B, T, C]]` passed where `Tensor[[@_, @_, n_embd]]`
     is expected, a fresh seeded dim is unified with the known dim.
Rounds repeat until the seeds stop changing.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from . import pyrefly, stubs
from .rewrite import rewrite
from .scratch import scratch_copy

Key = tuple[str, str, str]  # (class name or "", function name, param name)

_SKIP_ROOTS = {"torch", "F", "nn", "math", "np", "numpy", "einops", "os", "sys", "json", "inspect"}
_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _parse_one_tensor(t: str) -> list[str] | None:
    t = t.strip()
    m = re.fullmatch(r"(?:[\w.]*\.)?(?:Tensor|Parameter)\[\[(.*)\]\]", t)
    if not m:
        return None
    dims = _split_top(m.group(1)) if m.group(1).strip() else []
    if any("*" in d and ("tuple" in d or "@" in d) for d in dims):
        return None  # variadic / unsolved
    return dims


def parse_tensor(t: str, strict: bool = False) -> list[str] | None:
    """`torch.Tensor[[a, b]]` -> ["a", "b"]. For seeding (strict=False), union members that are
    Unknown/Any are ignored (a loop's first pass through a not-yet-seeded module) and several
    tensor shapes of one rank are reconciled dim by dim. Bare `Tensor` and variadic members are
    ignored too when the union has a full shape. Anything else -> None."""
    parts = [x.strip() for x in t.split(" | ")]
    if strict and len(parts) > 1:
        return None
    shapes = []
    for part in parts:
        if part in ("Unknown", "Any", "None") or re.fullmatch(r"(?:[\w.]*\.)?Tensor(?:\[Unknown\])?", part):
            continue  # says less than a full shape
        sh = _parse_one_tensor(part)
        if sh is None:
            if len(parts) > 1 and re.fullmatch(r"(?:[\w.]*\.)?(?:Tensor|Parameter)\[.*\]", part):
                continue  # `[*tuple[int, ...], d]`: a loop's first pass through an unseeded module
            return None
        shapes.append(sh)
    if not shapes or len({len(x) for x in shapes}) != 1:
        return None
    return [col[0] if len(set(map(_norm, col))) == 1 else "int" for col in zip(*shapes)]


def parse_generic(t: str) -> tuple[str, list[str] | None] | None:
    """`Block[_ts_a, _ts_b]` -> ("Block", [...]); `Self@GPT` -> ("GPT", None); else None."""
    t = t.strip()
    m = re.fullmatch(r"Self@([\w]+)", t)
    if m:
        return m.group(1), None
    m = re.fullmatch(r"(?:[\w.]*\.)?([A-Za-z_]\w*)(?:\[(.*)\])?", t)
    if not m:
        return None
    return m.group(1), (_split_top(m.group(2)) if m.group(2) else [])


class _Src:
    """Positions helper for one text (ast byte columns -> code points -> absolute offsets)."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.starts = [0]
        for line in self.lines[:-1]:
            self.starts.append(self.starts[-1] + len(line) + 1)

    def cp(self, lineno: int, byte_col: int) -> int:
        return len(self.lines[lineno - 1].encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))

    def abs(self, lineno: int, byte_col: int) -> int:
        return self.starts[lineno - 1] + self.cp(lineno, byte_col)

    def span(self, node) -> tuple[int, int]:
        return self.abs(node.lineno, node.col_offset), self.abs(node.end_lineno, node.end_col_offset)


def _functions(tree: ast.AST):
    """(function node, owner class name or None) for every def."""
    out = []

    def visit(node, cls):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((child, cls))
                visit(child, None)
            else:
                visit(child, cls)

    visit(tree, None)
    return out


_NEUTRAL_ATTRS = {"device", "dtype", "shape", "is_cuda", "requires_grad", "ndim", "size", "dim", "numel"}


def _carries_unknown(e, tainted: set[str]) -> bool:
    """Whether `e` can hold a value Pyrefly only knows through an Unknown: a read of a tainted name
    along the tensor's own data path. Sizes and metadata (`x.shape`, `x.size(1)`, `x.device`) never
    carry it, nor do keyword arguments; a module call (`self.attn(x, mask)`) passes on only what
    its first argument carries, since its seeded `forward` fixes the rest."""
    if isinstance(e, ast.Name):
        return e.id in tainted
    if isinstance(e, ast.Attribute):
        return e.attr not in _NEUTRAL_ATTRS and _carries_unknown(e.value, tainted)
    if isinstance(e, ast.Call):
        f = e.func
        if isinstance(f, ast.Attribute):
            if f.attr in _NEUTRAL_ATTRS:
                return False
            root = f.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "self":
                return bool(e.args) and _carries_unknown(e.args[0], tainted)
            if _carries_unknown(f.value, tainted):
                return True
        elif isinstance(f, ast.Name) and f.id in ("len", "int", "float", "bool", "isinstance", "range"):
            return False
        return any(_carries_unknown(a.value if isinstance(a, ast.Starred) else a, tainted) for a in e.args)
    if isinstance(e, ast.Subscript):
        return _carries_unknown(e.value, tainted)
    if isinstance(e, ast.IfExp):
        return _carries_unknown(e.body, tainted) or _carries_unknown(e.orelse, tainted)
    if isinstance(e, (ast.BinOp,)):
        return _carries_unknown(e.left, tainted) or _carries_unknown(e.right, tainted)
    if isinstance(e, ast.UnaryOp):
        return _carries_unknown(e.operand, tainted)
    if isinstance(e, (ast.Tuple, ast.List)):
        return any(_carries_unknown(x, tainted) for x in e.elts)
    if isinstance(e, ast.NamedExpr):
        return _carries_unknown(e.value, tainted)
    return False


def _unknown_names(fn) -> set[str]:
    """Locals of `fn` that may hold a tensor flowing from a parameter with no annotation: in the
    shadow, an input tensi has not seeded yet. Pyrefly types such a parameter as Unknown, and an
    Unknown argument picks a stub's first overload (`Embedding(idx)` with idx unknown is
    `[T, d]`, the 1-d case), so a shape read downstream of it is a guess, not a fact. Flow
    insensitive: a name assigned from such a value anywhere counts everywhere."""
    a = fn.args
    tainted = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs if x.annotation is None} - {"self", "cls"}
    if not tainted:
        return set()

    def targets(t):
        if isinstance(t, ast.Name):
            yield t.id
        elif isinstance(t, (ast.Tuple, ast.List)):
            for x in t.elts:
                yield from targets(x.value if isinstance(x, ast.Starred) else x)

    flows = []  # (target names, value)
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            flows += [(list(targets(t)), n.value) for t in n.targets]
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and n.value is not None:
            flows.append((list(targets(n.target)), n.value))
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            flows.append((list(targets(n.target)), n.iter))
        elif isinstance(n, ast.NamedExpr):
            flows.append(([n.target.id], n.value))
    changed = True
    while changed:
        changed = False
        for names, value in flows:
            if any(t not in tainted for t in names) and _carries_unknown(value, tainted):
                tainted |= set(names)
                changed = True
    return tainted


def _rank_view(e, p: str) -> bool:
    """`p`, `p[:, -k:]`, or `p if ... else p[:, -k:]`: the parameter itself or a slice of it that
    keeps its rank (nanoGPT's `idx_cond` in `generate`)."""
    if isinstance(e, ast.Name):
        return e.id == p
    if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id == p:
        parts = e.slice.elts if isinstance(e.slice, ast.Tuple) else [e.slice]
        return all(isinstance(x, ast.Slice) or (isinstance(x, ast.Constant) and x.value is Ellipsis) for x in parts)
    if isinstance(e, ast.IfExp):
        return _rank_view(e.body, p) and _rank_view(e.orelse, p)
    return False


# module -> (input rank, index of the type argument that is the input's dim 1)
_FIRST_USE = {
    "Conv1d": (3, 0),
    "Conv2d": (4, 0),
    "Conv3d": (5, 0),
    "ConvTranspose1d": (3, 0),
    "ConvTranspose2d": (4, 0),
    "ConvTranspose3d": (5, 0),
    "BatchNorm2d": (4, 0),
    "BatchNorm3d": (5, 0),
    "InstanceNorm2d": (4, 0),
    "InstanceNorm3d": (5, 0),
    "MaxPool2d": (4, None),
    "AvgPool2d": (4, None),
    "AdaptiveAvgPool2d": (4, None),
    "MaxPool1d": (3, None),
    "AvgPool1d": (3, None),
    "MaxPool3d": (5, None),
    "AvgPool3d": (5, None),
    "Embedding": (2, None),
}


def _seedable_of(info: dict, tree, cls: str | None, fname: str) -> set[str]:
    """Parameters of `cls.fname` (or a module function) that are unannotated or a bare `Tensor`."""
    if cls:
        return set(info.get("methods", {}).get(cls, {}).get(fname, []))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fname:
            return {
                a.arg
                for a in node.args.posonlyargs + node.args.args
                if a.annotation is None or ast.unparse(a.annotation) in ("Tensor", "torch.Tensor")
            }
    return set()


def _unannotated(fn) -> set[str]:
    return {a.arg for a in fn.args.posonlyargs + fn.args.args if a.annotation is None} - {"self", "cls"}


def _positional_params(fn) -> list[str]:
    names = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    return names[1:] if names and names[0] in ("self", "cls") else names


# ------------------------------------------------------------------ seed sources


def _shape_of(v):
    """`x.shape` / `x.size()` -> the `x` node, else None."""
    if isinstance(v, ast.Attribute) and v.attr == "shape":
        return v.value
    if (
        isinstance(v, ast.Call)
        and isinstance(v.func, ast.Attribute)
        and v.func.attr == "size"
        and not v.args
        and not v.keywords
    ):
        return v.func.value
    return None


def shape_unpack(st) -> tuple[ast.AST, list] | None:
    """`b, n, _ = x.shape` / `x.size()`, or `b, n, _, device = *x.shape, x.device`: (x node, the
    target nodes that receive x's dims). None for anything else."""
    if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], (ast.Tuple, ast.List))):
        return None
    elts = st.targets[0].elts
    if any(not isinstance(e, ast.Name) for e in elts):
        return None  # starred or nested target
    base = _shape_of(st.value)
    if base is not None:
        return base, list(elts)
    v = st.value
    if (
        isinstance(v, ast.Tuple)
        and v.elts
        and isinstance(v.elts[0], ast.Starred)
        and not any(isinstance(e, ast.Starred) for e in v.elts[1:])
    ):
        base = _shape_of(v.elts[0].value)
        extra = len(v.elts) - 1
        if base is not None and len(elts) > extra:
            return base, list(elts[: len(elts) - extra])
    return None


def unpack_seeds(source: str, info: dict) -> dict[Key, list[str]]:
    """`b, t = idx.size()` (or `.shape`) in a function's own top-level body seeds `idx`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out: dict[Key, list[str]] = {}
    for fn, cls in _functions(tree):
        seedable = (
            set(info.get("methods", {}).get(cls or "", {}).get(fn.name, []))
            if cls
            else {a.arg for a in fn.args.args if a.annotation is None}
        )
        rebound: set[str] = set()  # params assigned earlier in the body: an unpack then reads the new value
        for st in fn.body:
            su = shape_unpack(st)
            if su is None or not isinstance(su[0], ast.Name) or su[0].id not in seedable or su[0].id in rebound:
                rebound |= {n.id for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                continue
            rebound |= {n.id for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            src, elts = su[0].id, su[1]
            key = (cls or "", fn.name, src)
            out.setdefault(key, ["int" if e.id == "_" else "~" + e.id for e in elts])
    return out


_EINOPS_FUNCS = {"rearrange", "reduce", "repeat"}
_EINOPS_LAYERS = {"Rearrange", "Reduce"}


def _pattern_tokens(pattern: str, class_dims: set[str]) -> list[str] | None:
    """Seed tokens for the input side of an einops pattern: `b c (h p1) (w p2)` ->
    ["~b", "~c", "int", "int"]. A name that is already a class dim stays that dim; a grouped
    axis is a product we can't name, so it is `int`; `...` (any rank) gives up."""
    lhs = pattern.split("->")[0].strip()
    toks = re.findall(r"\([^)]*\)|\.\.\.|\S+", lhs)
    out = []
    for t in toks:
        if t == "...":
            return None
        if t.startswith("("):
            out.append("int")
        elif t.isdigit():
            out.append(t)
        elif re.fullmatch(r"[A-Za-z_]\w*", t):
            out.append(t if t in class_dims else ("int" if t == "_" else "~" + t))
        else:
            return None
    return out or None


def _einops_pattern(call) -> str | None:
    """The pattern string of `Rearrange('...')`, `einops.rearrange(x, '...')`, or the first
    layer of an `nn.Sequential(Rearrange('...'), ...)`."""
    if not isinstance(call, ast.Call):
        return None
    name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
    if (
        name in _EINOPS_LAYERS
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ):
        return call.args[0].value
    if name == "Sequential" and call.args:
        return _einops_pattern(call.args[0])
    return None


def einops_seeds(source: str, info: dict) -> dict[Key, list[str]]:
    """A seedable input whose first use is an einops pattern gets that pattern's axis names:
    `self.to_patch(img)` with `self.to_patch = nn.Sequential(Rearrange('b c (h p1) (w p2) -> ...'))`
    seeds `img` as `[b, c, int, int]`, so a vit-style root module needs no annotation."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    layers: dict[tuple[str, str], str] = {}  # (class, attr) -> pattern
    for fn, cls in _functions(tree):
        if cls and fn.name == "__init__":
            for st in ast.walk(fn):
                if (
                    isinstance(st, ast.Assign)
                    and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Attribute)
                    and isinstance(st.targets[0].value, ast.Name)
                    and st.targets[0].value.id == "self"
                ):
                    pat = _einops_pattern(st.value)
                    if pat:
                        layers[(cls, st.targets[0].attr)] = pat
    out: dict[Key, list[str]] = {}
    for fn, cls in _functions(tree):
        if not cls:
            continue
        seedable = set(info.get("methods", {}).get(cls, {}).get(fn.name, []))
        dims = {user for user, _tp in info.get("classes", {}).get(cls, [])}
        uses: dict[str, list[tuple[int, int, list[str], bool]]] = defaultdict(list)
        for node, guarded in _calls_with_guards(fn):
            if not (node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in seedable):
                continue
            f = node.func
            pat = None
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                pat = layers.get((cls, f.attr))
            elif (
                (
                    (isinstance(f, ast.Name) and f.id in _EINOPS_FUNCS)
                    or (isinstance(f, ast.Attribute) and f.attr in _EINOPS_FUNCS)
                )
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                pat = node.args[1].value
            toks = _pattern_tokens(pat, dims) if pat else None
            if toks:
                uses[node.args[0].id].append((node.lineno, node.col_offset, toks, guarded))
        for name, us in uses.items():
            # `if x.ndim == 2: x = rearrange(x, 'b t -> b 1 t')` says what x *might* be, not what it is;
            # and uses of different ranks mean the input's rank varies
            if any(g for *_r, g in us) or len({len(t) for _l, _c, t, _g in us}) != 1:
                continue
            out[(cls, fn.name, name)] = min(us)[2]
    return out


_RANK_TESTS = re.compile(r"\.ndim\b|\.dim\(\)|\.shape\b|\bndim\b|len\(")


def local_unpacks(tree: ast.AST):
    """`b, n, _ = x.shape` (or `x.size()`) on a local tensor `x`, in a method or function body:
    yields (fn, cls, stmt, var, targets) with None for a `_`/non-name target. A parameter's own
    unpack is the job of `unpack_seeds`."""
    for fn, cls in _functions(tree):
        params = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
        for st in ast.walk(fn):
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
                continue
            v = st.value
            if isinstance(st.targets[0], ast.Name):
                # `seq = x.shape[1]` / `seq = x.size(1)`: one dim, by index
                idx, base = None, None
                if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Attribute) and v.value.attr == "shape":
                    base, sl = v.value.value, v.slice
                    idx = _const_int(sl)
                elif (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Attribute)
                    and v.func.attr == "size"
                    and len(v.args) == 1
                    and not v.keywords
                ):
                    base, idx = v.func.value, _const_int(v.args[0])
                if idx is not None and isinstance(base, ast.Name) and base.id not in params:
                    yield fn, cls, st, base.id, (idx, st.targets[0].id)
                continue
            su = shape_unpack(st)
            if su is None:
                continue
            base, elts = su
            if not isinstance(base, ast.Name) or base.id in params:
                continue
            yield fn, cls, st, base.id, [None if e.id == "_" else e.id for e in elts]


def int_locals(tree: ast.AST):
    """`hidden_dim = int(dim * mult)` at the top of a function body, assigned once: yields
    (fn, cls, stmt, var). When Pyrefly can only say `int`, the variable becomes a dim of its own
    name (a class dim when the function is `__init__`)."""
    for fn, cls in _functions(tree):
        params = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
        stores = defaultdict(int)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                stores[n.id] += 1
        for st in fn.body:
            if not (
                isinstance(st, ast.Assign)
                and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and st.lineno == st.end_lineno
            ):
                continue
            v, var = st.value, st.targets[0].id
            if var in params or stores[var] != 1 or var.startswith("_"):
                continue
            if isinstance(v, ast.BinOp) and isinstance(v.op, (ast.Pow, ast.Div, ast.MatMult)):
                continue
            if not isinstance(v, (ast.BinOp, ast.Call)):
                continue
            yield fn, cls, st, var


_SIMPLE_STMTS = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Expr,
    ast.Return,
    ast.Pass,
    ast.Assert,
    ast.Delete,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.Import,
    ast.ImportFrom,
)


def _is_none(n) -> bool:
    return isinstance(n, ast.Constant) and n.value is None


def _size_expr(n) -> bool:
    if isinstance(n, ast.Constant):
        return isinstance(n.value, int) and not isinstance(n.value, bool)
    if isinstance(n, (ast.Name, ast.Attribute)):
        return True
    if isinstance(n, ast.BinOp):
        return not isinstance(n.op, (ast.Div, ast.Pow, ast.MatMult)) and _size_expr(n.left) and _size_expr(n.right)
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "int"


def fallback_sizes(tree: ast.AST):
    """A size parameter that defaults to another: `out_features = out_features or in_features`,
    `d = d if d is not None else dim`, or `if d is None: d = dim`. Pyrefly sees a union of the two
    and shows whichever; the variable that sized the layer is `out_features`, so that becomes a dim.
    Yields (fn, cls, key stmt, var, anchor): the cast goes after `anchor` when it is the key
    statement, before it (the next statement) for the `if` form, and around it when it is the
    value of `self.attr = var or other`."""
    for fn, cls in _functions(tree):
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        defaults = dict(zip([a.arg for a in fn.args.posonlyargs + fn.args.args][::-1], fn.args.defaults[::-1]))
        defaults.update({a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None})
        ok = {
            a.arg
            for a in args
            if _is_none(defaults.get(a.arg))
            and (a.annotation is None or "int" in ast.unparse(a.annotation))
            and "float" not in (ast.unparse(a.annotation) if a.annotation else "")
        }
        if not ok:
            continue
        stores = defaultdict(int)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                stores[n.id] += 1
        body = fn.body
        for i, st in enumerate(body):
            if (
                isinstance(st, ast.Assign)
                and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Attribute)
                and isinstance(st.targets[0].value, ast.Name)
                and st.targets[0].value.id == "self"
                and isinstance(st.value, ast.BoolOp)
                and isinstance(st.value.op, ast.Or)
                and len(st.value.values) == 2
                and isinstance(st.value.values[0], ast.Name)
                and st.value.values[0].id in ok
                and stores[st.value.values[0].id] == 0
                and _size_expr(st.value.values[1])
            ):
                # `self.output_dim = output_dim or embed_dim`: the cast wraps the value
                yield fn, cls, st, st.value.values[0].id, st.value
                continue
            if (
                isinstance(st, ast.Assign)
                and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and st.lineno == st.end_lineno
            ):
                var, v = st.targets[0].id, st.value
                if var not in ok or stores[var] != 1:
                    continue
                if (
                    isinstance(v, ast.BoolOp)
                    and isinstance(v.op, ast.Or)
                    and len(v.values) == 2
                    and isinstance(v.values[0], ast.Name)
                    and v.values[0].id == var
                    and _size_expr(v.values[1])
                ):
                    yield fn, cls, st, var, st
                elif (
                    isinstance(v, ast.IfExp)
                    and isinstance(v.test, ast.Compare)
                    and len(v.test.ops) == 1
                    and isinstance(v.test.left, ast.Name)
                    and v.test.left.id == var
                    and _is_none(v.test.comparators[0])
                ):
                    other = v.orelse if isinstance(v.test.ops[0], ast.IsNot) else v.body
                    mine = v.body if isinstance(v.test.ops[0], ast.IsNot) else v.orelse
                    if isinstance(mine, ast.Name) and mine.id == var and _size_expr(other):
                        yield fn, cls, st, var, st
            elif (
                isinstance(st, ast.If)
                and not st.orelse
                and len(st.body) == 1
                and isinstance(st.test, ast.Compare)
                and len(st.test.ops) == 1
                and isinstance(st.test.ops[0], ast.Is)
                and isinstance(st.test.left, ast.Name)
                and _is_none(st.test.comparators[0])
                and i + 1 < len(body)
                and isinstance(body[i + 1], _SIMPLE_STMTS)
            ):
                var, a = st.test.left.id, st.body[0]
                if (
                    var in ok
                    and stores[var] == 1
                    and isinstance(a, ast.Assign)
                    and len(a.targets) == 1
                    and isinstance(a.targets[0], ast.Name)
                    and a.targets[0].id == var
                    and _size_expr(a.value)
                ):
                    yield fn, cls, a, var, body[i + 1]


def fallback_seeds(source: str) -> dict:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    return {((cls or ""), fn.name, int_key(st, var)): ["~" + var] for fn, cls, st, var, _a in fallback_sizes(tree)}


def int_key(stmt: ast.stmt, var: str) -> str:
    return f"#{stmt.lineno}:{var}"


def _const_int(node) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return -node.operand.value
    return None


def _base_of(st) -> ast.expr:
    v = st.value
    if isinstance(v, ast.Attribute):
        return v.value
    if isinstance(v, ast.Subscript):
        return v.value.value
    return v.func.value


def _divisors(fn) -> set[int]:
    """k for each `assert x % k == 0` (or `(x % k) == 0`, `not x % k`) in the function."""
    out = set()
    for n in ast.walk(fn):
        if not isinstance(n, ast.Assert):
            continue
        for t in ast.walk(n.test):
            if (
                isinstance(t, ast.BinOp)
                and isinstance(t.op, ast.Mod)
                and isinstance(t.right, ast.Constant)
                and isinstance(t.right.value, int)
                and t.right.value > 1
            ):
                out.add(t.right.value)
    return out


def divisible_returns(tree: ast.AST):
    """Single-line `return expr` in a function that asserts `x % k == 0`: yields (fn, cls, stmt, ks).
    Pyrefly keeps `4 * (dim // 4)` apart from `dim`, so what such a function returns gets cast to
    the simplified shape."""
    for fn, cls in _functions(tree):
        ks = _divisors(fn)
        if not ks:
            continue
        for st in ast.walk(fn):
            if isinstance(st, ast.Return) and st.value is not None and st.lineno == st.end_lineno:
                yield fn, cls, st, ks


def simplify_divisible(d: str, ks: set[int]) -> str:
    for k in ks:
        d = re.sub(rf"\b{k} \* \(([A-Za-z_]\w*) // {k}\)", r"\1", d)
        d = re.sub(rf"\b([A-Za-z_]\w*) // {k} \* {k}\b", r"\1", d)
    return d


def ret_key(stmt: ast.stmt) -> str:
    return f"^{stmt.lineno}"


def local_key(stmt: ast.stmt, var: str) -> str:
    return f"@{stmt.lineno}:{var}"


def _expr_symbols(d: str) -> tuple[str, list[str]] | None:
    """`_ts_w1*_ts_w2` -> ("S*S", ["_ts_w1", "_ts_w2"]): an expression's operator skeleton and symbols."""
    syms = re.findall(r"[A-Za-z_]\w*", d)
    if not syms:
        return None
    return re.sub(r"[A-Za-z_]\w*", "S", d), syms


def _calls_with_guards(fn):
    """Every call in `fn`, with whether it sits under an `if`/ternary whose test inspects a rank."""

    def visit(node, guarded):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            g = guarded or bool(_RANK_TESTS.search(ast.unparse(node.test)))
            yield from visit(node.test, guarded)
            for ch in node.body if isinstance(node.body, list) else [node.body]:
                yield from visit(ch, g)
            for ch in node.orelse if isinstance(node.orelse, list) else [node.orelse]:
                yield from visit(ch, g)
            return
        if isinstance(node, ast.Call):
            yield node, guarded
        for ch in ast.iter_child_nodes(node):
            yield from visit(ch, guarded)

    for st in fn.body:
        yield from visit(st, False)


def translate(dims: list[str], mapping: dict[str, str], keep_fresh: bool = False) -> list[str]:
    """Caller-side shape (shadow names) -> callee-side seed tokens. With `keep_fresh`, an expression
    over the function's own fresh dims stays an expression (`~b * ~x`) instead of becoming `int`."""
    out = []
    for d in dims:
        nd = _norm(d)
        if nd in mapping:
            out.append(mapping[nd])
        elif re.fullmatch(r"-?\d+", nd) or nd == "int":
            out.append(nd)
        elif m := re.fullmatch(r"_ts_([A-Za-z0-9_]+?)(?:__\d+|__p)?", nd):
            out.append("~" + m.group(1))
        elif re.fullmatch(r"[A-Za-z_]\w*", nd):
            out.append("~" + nd)
        else:
            ok = True

            def sub(m):
                nonlocal ok
                sym = m.group(0)
                if sym in mapping and re.fullmatch(r"[A-Za-z_]\w*", mapping[sym]):
                    return mapping[sym]
                fm = re.fullmatch(r"_ts_([A-Za-z0-9_]+?)(?:__\d+|__p)?", sym)
                if keep_fresh and fm:
                    return "~" + fm.group(1)
                ok = False
                return sym

            expr = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", sub, d)
            out.append(
                _norm(expr).replace("*", " * ").replace("//", " // ").replace("+", " + ").replace("-", " - ")
                if ok
                else "int"
            )
    return out


_INT_TOKEN_RE = re.compile(r"-?\d+")


def _reconcile(specs: list[list[str]]) -> list[str] | None:
    if not specs:
        return None
    rank = {len(s) for s in specs}
    if len(rank) != 1:
        return None
    out = []
    for col in zip(*specs):
        if len(set(col)) == 1:
            out.append(col[0])
        elif all(c.startswith("~") for c in col):
            out.append(col[0])
        else:
            out.append("int")
    return out


# ------------------------------------------------------------------ probing


class Probe:
    def __init__(self, kind: str, start: int, end: int, call_id: int, arg_index: int = -1):
        self.kind, self.start, self.end, self.call_id, self.arg_index = kind, start, end, call_id, arg_index
        self.out_line = self.out_col = None
        self.meta = None
        self.type: str | None = None
        self.unknown = False  # the argument flows from an input Pyrefly can only type as Unknown


_TYPE_PARAMS = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:class|def)[ \t]+\w+[ \t]*(?=\[)", re.M)


def parse_shadow(shadow: str) -> ast.Module:
    """`ast.parse` the shadow on any Python. Its `class Mlp[_ts_d: _ts_IntVar = _ts_in]` headers are
    3.12/3.13 syntax, so the type-parameter lists are blanked to spaces first: every other node keeps
    its line and column, and no probe needs the type parameters themselves."""
    out, pos = [], 0
    for m in _TYPE_PARAMS.finditer(shadow):
        start = m.end()
        if start < pos:
            continue
        depth, i = 0, start
        while i < len(shadow):
            depth += {"[": 1, "]": -1}.get(shadow[i], 0)
            if depth == 0:
                break
            i += 1
        out.append(shadow[pos:start])
        out.append(re.sub(r"[^\n]", " ", shadow[start : i + 1]))
        pos = i + 1
    out.append(shadow[pos:])
    return ast.parse("".join(out))


def build_probes(shadow: str) -> tuple[list[Probe], dict[int, tuple]]:
    """Wrap callees and positional args of module-like calls inside functions."""
    tree = parse_shadow(shadow)
    src = _Src(shadow)
    probes: list[Probe] = []
    calls: dict[int, tuple] = {}  # call id -> (enclosing fn name, owner class)
    cid = 0
    top_funcs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    class_methods = {
        c.name: {n.name for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for c in tree.body
        if isinstance(c, ast.ClassDef)
    }
    for fn, cls in _functions(tree):
        fn_params = {a.arg for a in fn.args.posonlyargs + fn.args.args}
        unknown = _unknown_names(fn)

        first_load: dict[str, tuple[int, int]] = {}  # each parameter's first read, in source order
        neutral: set[int] = set()  # reads that say nothing about rank: `identity = x`, `x.device`
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name):
                neutral.add(id(n.value))
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.attr in _NEUTRAL_ATTRS:
                neutral.add(id(n.value))
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in fn_params and id(n) not in neutral:
                if n.id not in first_load or (n.lineno, n.col_offset) < first_load[n.id]:
                    first_load[n.id] = (n.lineno, n.col_offset)
        # `idx_cond = idx if ... else idx[:, -k:]`, assigned once: a call on idx_cond is a first use of idx
        views: dict[str, str] = {}
        stored = defaultdict(int)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                stored[n.id] += 1
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and not isinstance(n.value, ast.Name)
            ):
                for p in fn_params:
                    if _rank_view(n.value, p) and n.lineno <= first_load.get(p, (0, 0))[0] <= n.end_lineno:
                        views[n.targets[0].id] = p
        views = {v: p for v, p in views.items() if stored[v] == 1}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            f = node.func
            root = f
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                continue
            if (
                root.id in _SKIP_ROOTS
                or (isinstance(f, ast.Name) and f.id in _BUILTINS)
                or root.id in ("super", "_ts_cast")
            ):
                continue
            if isinstance(f, ast.Attribute) and root.id != "self":
                continue  # method calls on locals (x.view(...)) are not module calls
            cid += 1
            calls[cid] = (fn.name, cls)
            pr = Probe("callee", *src.span(f), cid)
            if isinstance(f, ast.Name) and f.id in top_funcs:
                pr.meta = ("", f.id)  # a helper function of this module
            elif isinstance(f, ast.Name) and f.id == "self" and cls:
                pr.meta = (cls, "forward")  # `self(idx)` inside `generate`
            elif (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "self"
                and cls
                and f.attr in class_methods.get(cls, set())
                and f.attr != "forward"
            ):
                pr.meta = (cls, f.attr)  # `self.helper(x)`
            probes.append(pr)
            for i, a in enumerate(node.args):
                if isinstance(a, ast.Starred):
                    break
                ap = Probe("arg", *src.span(a), cid, i)
                ap.unknown = _carries_unknown(a, unknown)
                if isinstance(a, ast.Name) and a.id in fn_params and first_load.get(a.id) == (a.lineno, a.col_offset):
                    ap.meta = (a.id, node.lineno, node.col_offset)  # a parameter, not yet reassigned
                elif isinstance(a, ast.Name) and a.id in views:
                    ap.meta = (views[a.id], node.lineno, node.col_offset)
                probes.append(ap)
    for fn, cls, st, var in int_locals(tree):
        cid += 1
        calls[cid] = (fn.name, cls)
        pr = Probe("intlocal", *src.span(st.value), cid)
        pr.meta = (cls or "", fn.name, int_key(st, var), var)
        probes.append(pr)
    for fn, cls, st, ks in divisible_returns(tree):
        target = st.value
        if (
            isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name)
            and target.func.id == "_ts_cast"
            and len(target.args) == 2
        ):
            target = target.args[1]  # tensi's own cast from an earlier round
        cid += 1
        calls[cid] = (fn.name, cls)
        pr = Probe("ret", *src.span(target), cid)
        pr.meta = (cls or "", fn.name, ret_key(st), ks)
        probes.append(pr)
    # local `b, n, _ = x.shape`: probe x as it was before tensi's own cast on that line
    for fn, cls, st, var, targets in local_unpacks(tree):
        target = _base_of(st)
        body = _body_of(fn, st)
        i = body.index(st) if body is not None else -1
        prev = body[i - 1] if i > 0 else None
        if (
            isinstance(prev, ast.Assign)
            and prev.lineno == st.lineno
            and isinstance(prev.value, ast.Call)
            and isinstance(prev.value.func, ast.Name)
            and prev.value.func.id == "_ts_cast"
            and len(prev.value.args) == 2
        ):
            target = prev.value.args[1]
        cid += 1
        calls[cid] = (fn.name, cls)
        pr = Probe("local", *src.span(target), cid)
        pr.meta = (cls or "", fn.name, local_key(st, var), targets)
        probes.append(pr)
    return probes, calls


def _body_of(fn, stmt):
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(node, field, None)
            if isinstance(seq, list) and stmt in seq:
                return seq
    return None


def instrument(shadow: str, probes: list[Probe]) -> str:
    """Insert `reveal_type(...)` around every probe span; record where each probe's expression
    starts in the output (1-based line/col, code points)."""
    events = []  # (pos, order, kind, probe)
    for p in probes:
        events.append((p.start, 1, -(p.end - p.start), "open", p))
        events.append((p.end, 0, (p.end - p.start), "close", p))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    out, pos = [], 0
    for at, _o, _l, kind, p in events:
        out.append(shadow[pos:at])
        pos = at
        if kind == "open":
            out.append("reveal_type(")
            text = "".join(out)
            p.out_line = text.count("\n") + 1
            p.out_col = len(text) - (text.rfind("\n") + 1) + 1
        else:
            out.append(")")
    out.append(shadow[pos:])
    return "from typing import NamedTuple, reveal_type as reveal_type\n" + "".join(out)


def run_pyrefly(text: str, workdir: Path, search_paths: list[str] | None = None) -> list[dict]:
    with scratch_copy(workdir, text, "_tensi_probe_") as (tmp, extra, cwd):
        cmd = [pyrefly(), "check", tmp, "--preset", "default", "--output-format", "json", *extra]
        for sp in search_paths or []:
            cmd += ["--search-path", sp]
        cmd += stubs.search_args()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    try:
        return json.loads(proc.stdout or "{}").get("errors", [])
    except json.JSONDecodeError:
        return []


# ------------------------------------------------------------------ the solver


def _unify(
    errors: list[dict],
    shadow_tree,
    info: dict,
    new: dict,
    new_origin: dict,
    line_offset: int,
    named: dict | None = None,
) -> None:
    """Rename fresh seed dims that a mismatch error shows must equal a known one (in place).
    A name the code itself gave (`b, n, _ = x.shape`) is never renamed: a mismatch there is real."""

    for e in errors:
        if e.get("name") == "bad-argument-type":
            m = re.search(
                r"Argument `(.+?)` is not assignable to parameter `\w+` with type `(.+?)` in function", e["description"]
            )
        elif e.get("name") == "unsupported-operation":
            # `x * self.g` with `x: [b, ~d, h, w]` and `g: [1, dim, 1, 1]`: broadcasting pairs dims
            # from the right
            m = re.search(
                r"`[-+*/@%]+` is not supported between `(.+?)` and `(.+?)`$", e["description"].split("\n", 1)[0]
            )
        else:
            continue
        bm = re.search(
            r"Cannot broadcast dimension (?:Int\[)?(_ts_\w+)\]? with dimension (?:Int\[)?(_ts_\w+)\]?", e["description"]
        )
        if bm:  # `dots.masked_fill(mask, ...)`: the two dims that failed to broadcast
            a, prm = [bm.group(1), bm.group(2)], [bm.group(2), bm.group(1)]
        elif not m:
            continue
        else:
            a, prm = parse_tensor(m.group(1), strict=True), parse_tensor(m.group(2), strict=True)
            if a and prm is None:
                # `Linear.forward(input: Tensor[[*S, In]])`: align the fixed tail from the right
                pm = re.fullmatch(r"(?:[\w.]*\.)?Tensor\[\[(.*)\]\]", m.group(2).strip())
                tail = _split_top(pm.group(1)) if pm else []
                while tail and tail[0].startswith("*"):
                    tail = tail[1:]
                if tail and not any(d.startswith("*") or "@" in d for d in tail) and len(tail) <= len(a):
                    a, prm = a[len(a) - len(tail) :], tail
        if not a or not prm:
            continue
        if e["name"] == "unsupported-operation" and not bm:
            k = min(len(a), len(prm))
            a, prm = a[len(a) - k :], prm[len(prm) - k :]
            # either side may hold the fresh dim
            a, prm = a + prm, prm + a
        elif len(a) != len(prm):
            continue
        where = _shadow_fn_at(shadow_tree, e["line"] - line_offset)  # the reveal_type import line
        if where is None:
            continue
        fn, cls = where
        known = {tp: user for user, tp in info["classes"].get(cls or "", [])}
        fn_keys = [k for k in new if k[0] == (cls or "") and k[1] == fn.name]
        explicit = {t for k in fn_keys if new_origin.get(k) == "local" for t in new[k] if t.startswith("~")}
        explicit |= {t for k in fn_keys for t in (named or {}).get(k, ())}
        # `w1 * w2` against `window_size * window_size`: same expression shape, pair the symbols
        pa, pp = [], []
        for ad, pd in zip(a, prm):
            sa, sp = _expr_symbols(_norm(ad)), _expr_symbols(_norm(pd))
            if sa and sp and sa[0] == sp[0] and len(sa[1]) > 1:
                if set(sa[0]) <= set("S*()"):
                    # a product, which Pyrefly may print in any order: pair what's left over
                    from collections import Counter

                    ca, cp = Counter(sa[1]), Counter(sp[1])
                    la, lp = list((ca - cp).elements()), list((cp - ca).elements())
                    if len(la) == len(lp) and (len(la) == 1 or len(set(la)) == 1 or len(set(lp)) == 1):
                        pa += la
                        pp += lp
                    continue
                pa += sa[1]
                pp += sp[1]
        a, prm = list(a) + pa, list(prm) + pp
        if bm:
            # two fresh dims of one function that must be equal (`mask`'s einops `~b` vs `x`'s
            # `~batch`): the one from an einops pattern takes the other's name
            fresh = []
            for d in (bm.group(1), bm.group(2)):
                dm = re.fullmatch(r"_ts_([A-Za-z0-9_]+?)(?:__\d+)?", d)
                owners = [k for k in fn_keys if dm and "~" + dm.group(1) in new[k]]
                fresh.append((d, owners))
            (d1, o1), (d2, o2) = fresh
            if o1 and o2 and d1 not in known and d2 not in known:
                if all(new_origin.get(k) in ("einops", "unify") for k in o1):
                    a, prm = [d1], [d2]
                else:
                    a, prm = [d2], [d1]
                known[prm[0]] = "~" + re.fullmatch(r"_ts_([A-Za-z0-9_]+?)(?:__\d+)?", prm[0]).group(1)
        for key in fn_keys:
            toks = list(new[key])
            changed = False
            for ad, pd in zip(a, prm):
                am = re.fullmatch(r"_ts_([A-Za-z0-9_]+?)(?:__\d+)?", _norm(ad))
                if am and ("~" + am.group(1)) in toks and _norm(pd) in known and ("~" + am.group(1)) not in explicit:
                    toks = [known[_norm(pd)] if t == "~" + am.group(1) else t for t in toks]
                    changed = True
            if changed:
                new[key], new_origin[key] = toks, "unify"


def _shadow_fn_at(tree: ast.AST, line: int):
    best = None
    for fn, cls in _functions(tree):
        if fn.lineno <= line <= fn.end_lineno and (best is None or fn.lineno >= best[0].lineno):
            best = (fn, cls)
    return best


def solve(
    path: str | Path,
    source: str | None = None,
    *,
    package: str | None = None,
    config_index: dict | None = None,
    search_paths: list[str] | None = None,
    workdir: Path | None = None,
    max_rounds: int = 8,
    trace: list | None = None,
) -> dict[Key, list[str]]:
    path = Path(path)
    source = path.read_text(encoding="utf-8") if source is None else source
    workdir = workdir or path.parent
    try:
        user_tree = ast.parse(source)
    except SyntaxError:
        return {}
    fwd_params = {cls: _positional_params(fn) for fn, cls in _functions(user_tree) if cls and fn.name == "forward"}
    helper_params: dict[tuple[str, str], list[str]] = {}
    helper_free: dict[tuple[str, str], set[str]] = {}  # params with no annotation: the seedable ones
    for node in user_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            helper_params[("", node.name)] = _positional_params(node)
            helper_free[("", node.name)] = _unannotated(node)
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name not in ("forward", "__init__"):
                    if not any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in m.decorator_list):
                        helper_params[(node.name, m.name)] = _positional_params(m)
                        helper_free[(node.name, m.name)] = _unannotated(m)

    base = rewrite(source, package, config_index=config_index)
    info = base.info
    seeds: dict[Key, list[str]] = dict(unpack_seeds(source, info))
    origin = {k: "unpack" for k in seeds}
    for k, v in einops_seeds(source, info).items():
        if k not in seeds:
            seeds[k], origin[k] = v, "einops"
    for k, v in fallback_seeds(source).items():
        seeds[k], origin[k] = v, "local"
    own_seeds = dict(seeds)
    named_toks: dict[Key, set[str]] = defaultdict(set)  # tokens that carry a name the code wrote (`b, n, _ = x.shape`)

    history: dict[Key, list[list[str]]] = defaultdict(list)
    frozen: dict[Key, list[str]] = {}
    for rnd in range(max_rounds):
        res = rewrite(source, package, config_index=config_index, seeds=seeds)
        info = res.info
        try:
            shadow_tree = parse_shadow(res.shadow)
            probes, calls = build_probes(res.shadow)
        except SyntaxError:
            break
        text = instrument(res.shadow, probes)
        errors = run_pyrefly(text, workdir, search_paths)
        reveals = {}
        for e in errors:
            if e.get("name") == "reveal-type":
                m = re.search(r"revealed type: (.*)", e["description"], re.S)
                if m:
                    reveals[(e["line"], e["column"])] = m.group(1).strip()
        for p in probes:
            # +1 line: the prepended reveal_type import; Pyrefly's column points at the `(` before
            # the revealed expression.
            p.type = reveals.get((p.out_line + 1, p.out_col - 1))

        new: dict[Key, list[str]] = {}
        new_origin: dict[Key, str] = {}
        # 3. call sites
        by_call: dict[int, list[Probe]] = defaultdict(list)
        for p in probes:
            by_call[p.call_id].append(p)
        specs: dict[Key, list[list[str]]] = defaultdict(list)
        unreadable: set[Key] = set()  # keys with a call site whose argument Pyrefly can't type yet
        # 3b. first use: `out = self.conv1(x)` with `self.conv1: Conv2d[inplanes, planes]` means the
        #     unseeded `x` is `[int, inplanes, int, int]`. Only rank and channel; the rest is data.
        first_use: dict[Key, tuple] = {}
        for cid, ps in by_call.items():
            callee = next((p for p in ps if p.kind == "callee"), None)
            hparams = None
            if callee is not None and callee.meta:
                hparams = helper_params.get(callee.meta) or (
                    fwd_params.get(callee.meta[0]) if callee.meta[1] == "forward" else None
                )
            if hparams:
                # `return self._forward_impl(x)`, `logits = self(idx)`: x is what the callee's own
                # seed says it is
                caller_fn, caller_cls = calls[cid]
                owner, hname = callee.meta
                for p in ps:
                    if p.kind != "arg" or not p.meta or p.arg_index >= len(hparams):
                        continue
                    hkey = (owner, hname, hparams[p.arg_index])
                    hseed = seeds.get(hkey) if origin.get(hkey) not in ("call", None) else None
                    pname, line, col = p.meta
                    key = (caller_cls or "", caller_fn, pname)
                    if not hseed or pname not in _seedable_of(info, user_tree, caller_cls, caller_fn):
                        continue
                    same = (caller_cls or "") == owner
                    toks = [t if (t == "int" or t.isdigit() or same) else "int" for t in hseed]
                    if not same:  # another scope's fresh names mean nothing here
                        toks = [t if not t.startswith("~") else "int" for t in toks]
                    if key not in first_use or (line, col) < first_use[key][0]:
                        first_use[key] = ((line, col), toks)
                continue
            g = parse_generic(callee.type) if callee is not None and callee.type else None
            if g is None or g[0] not in _FIRST_USE:
                continue
            rank, chan = _FIRST_USE[g[0]]
            arg0 = next((p for p in ps if p.kind == "arg" and p.arg_index == 0), None)
            if arg0 is None or not arg0.meta:
                continue
            caller_fn, caller_cls = calls[cid]
            pname, line, col = arg0.meta
            key = (caller_cls or "", caller_fn, pname)
            if pname not in _seedable_of(info, user_tree, caller_cls, caller_fn):
                continue
            toks = ["int"] * rank
            if chan is not None and g[1] and len(g[1]) > chan:
                ct = translate(
                    [g[1][chan]], {_norm(tp): user for user, tp in info["classes"].get(caller_cls or "", [])}
                )[0]
                toks[1] = ct if not ct.startswith("~") else "int"
            if key not in first_use or (line, col) < first_use[key][0]:
                first_use[key] = ((line, col), toks)
        for cid, ps in by_call.items():
            callee = next((p for p in ps if p.kind == "callee"), None)
            if callee is not None and callee.meta and callee.meta in helper_params:
                # a plain helper (`posemb_sincos_2d(x)`, `self.split_heads(q)`): its tensor
                # parameters take the shapes its callers pass, in the callers' names
                owner, fname = callee.meta
                for p in ps:
                    if p.kind != "arg" or p.arg_index >= len(helper_params[callee.meta]):
                        continue
                    pname = helper_params[callee.meta][p.arg_index]
                    key = (owner, fname, pname)
                    if pname not in helper_free[callee.meta]:
                        continue
                    if p.unknown:
                        unreadable.add(key)
                        continue
                    shape = parse_tensor(p.type) if p.type else None
                    if shape is None:
                        if p.type and "Tensor" in p.type:
                            unreadable.add(key)
                        continue
                    caller_fn, caller_cls = calls[cid]
                    mapping = (
                        {_norm(tp): user for user, tp in info["classes"].get(caller_cls or "", [])}
                        if caller_cls == owner and owner
                        else {}
                    )
                    toks = translate(shape, mapping)
                    caller_named = {
                        t
                        for k, v in seeds.items()
                        if k[:2] == (caller_cls or "", caller_fn)
                        for t in v
                        if origin.get(k) == "local" or t in named_toks.get(k, ())
                    }
                    named_toks[key] |= {t for t in toks if t in caller_named}
                    specs[key].append(toks)
                continue
            if callee is None or not callee.type:
                continue
            g = parse_generic(callee.type)
            if g is None:
                continue
            cname, targs = g
            if cname not in fwd_params:
                continue
            dims = info["classes"].get(cname, [])
            if targs is None:  # Self@C: same vocabulary
                mapping = {_norm(tp): user for user, tp in dims}
            elif len(targs) == len(dims):
                # only symbolic arguments name anything: `PEG[int]` must not turn every `int` dim
                # of the argument into `dim`
                mapping = {}
                lits = [_norm(a) for a in targs]
                for a, (user, _tp) in zip(targs, dims):
                    na = _norm(a)
                    if re.fullmatch(r"[A-Za-z_]\w*", na) and na != "int":
                        mapping.setdefault(na, user)
                    elif re.fullmatch(r"\d+", na) and lits.count(na) == 1:
                        # `MLP(64)` called on `[b, t, 64]`: inside forward that 64 is `dim`
                        mapping[na] = user
            else:
                mapping = {}
            seedable = set(info["methods"].get(cname, {}).get("forward", []))
            for p in ps:
                if p.kind != "arg" or not p.type or p.arg_index >= len(fwd_params[cname]):
                    continue
                pname = fwd_params[cname][p.arg_index]
                if pname not in seedable:
                    continue
                if p.unknown:
                    unreadable.add((cname, "forward", pname))
                    continue
                shape = parse_tensor(p.type)
                if shape is None:
                    unreadable.add((cname, "forward", pname))
                    continue
                toks = translate(shape, mapping)
                caller_fn, caller_cls = calls[cid]
                caller_named = {
                    t
                    for k, v in seeds.items()
                    if k[:2] == (caller_cls or "", caller_fn)
                    for t in v
                    if origin.get(k) == "local" or t in named_toks.get(k, ())
                }
                named_toks[(cname, "forward", pname)] |= {t for t in toks if t in caller_named}
                specs[(cname, "forward", pname)].append(toks)
        for key, sp in specs.items():
            r = _reconcile(sp)
            if r is not None and key in unreadable:
                # another call site may pass any size here: a literal from the readable ones
                # (`[b, 1, d]` from a cls-token call) must not pin the input
                r = ["int" if _INT_TOKEN_RE.fullmatch(t) else t for t in r]
            if r is not None:
                own = own_seeds.get(key)
                if own and len(own) == len(r):
                    # the function's own `b, t = x.size()` names what its callers only know as `int`
                    r = [o if c == "int" else c for c, o in zip(r, own)]
                new[key], new_origin[key] = r, "call"
        # 6. `hidden_dim = int(dim * mult)`, only `int` to Pyrefly: a dim named `hidden_dim`
        for p in probes:
            if p.kind == "intlocal" and p.type == "int":
                cname, fname, ikey, var = p.meta
                if var not in {u for u, _tp in info["classes"].get(cname, [])}:
                    new[(cname, fname, ikey)], new_origin[(cname, fname, ikey)] = ["~" + var], "local"
        # 7. `return pe` where pe is `[n, 4 * (dim // 4)]` and the function asserts `dim % 4 == 0`
        for p in probes:
            if p.kind != "ret" or not p.type:
                continue
            cname, fname, rkey, ks = p.meta
            shape = parse_tensor(p.type)
            if shape is None:
                continue
            simple = [simplify_divisible(d, ks) for d in shape]
            if simple == shape:
                continue
            mapping = {_norm(tp): user for user, tp in info["classes"].get(cname, [])}
            new[(cname, fname, rkey)], new_origin[(cname, fname, rkey)] = (
                translate(simple, mapping, keep_fresh=True),
                "local",
            )
        # 5. locals: `b, n, _ = x.shape` names the dims of x that are only `int` so far
        for p in probes:
            if p.kind != "local" or not p.type:
                continue
            cname, fname, lkey, targets = p.meta
            shape = parse_tensor(p.type)
            if shape is None:
                continue
            if isinstance(targets, tuple):  # `seq = x.shape[1]`
                i, name = targets
                if not -len(shape) <= i < len(shape):
                    continue
                targets = [None] * len(shape)
                targets[i] = name
            if len(shape) != len(targets):
                continue
            mapping = {_norm(tp): user for user, tp in info["classes"].get(cname, [])}
            toks = translate(shape, mapping, keep_fresh=True)
            used = {t for k, v in new.items() if k[:2] == (cname, fname) for t in v} | {
                t for k, v in seeds.items() if k[:2] == (cname, fname) and k[2] != lkey for t in v
            }
            named = False
            for i, (tok, tgt) in enumerate(zip(toks, targets)):
                if tok == "int" and tgt and ("~" + tgt) not in used and tgt not in mapping.values():
                    toks[i] = "~" + tgt
                    used.add("~" + tgt)
                    named = True
            if named:
                new[(cname, fname, lkey)], new_origin[(cname, fname, lkey)] = toks, "local"
        for key, (_pos, toks) in first_use.items():
            if key not in new and key not in seeds:
                new[key], new_origin[key] = toks, "firstuse"
            elif key not in new and origin.get(key) == "firstuse":
                new[key], new_origin[key] = toks, "firstuse"
        # 1. seeds persist: a key that no call site informed this round keeps its last value
        #    (its call site may have been unreadable while an upstream seed was still settling)
        for key, v in seeds.items():
            if key not in new:
                new[key], new_origin[key] = v, origin[key]
        # 4. unification from mismatch errors
        _unify(errors, shadow_tree, info, new, new_origin, 1, named_toks)
        # 5. a seed flipping between two values each round (a call site whose argument depends on
        #    the seed itself) settles on the one in the callee's own names: fewest fresh `~` names
        for key, v in new.items():
            if key in frozen:
                new[key] = frozen[key]
            elif v != seeds.get(key) and v in history[key]:
                frozen[key] = new[key] = min(
                    history[key] + [v], key=lambda t: (sum(x.startswith("~") for x in t), t.count("int"))
                )
            history[key].append(list(new[key]))
        if trace is not None:
            trace.append({"round": rnd, "seeds": dict(new), "origin": dict(new_origin)})
        if new == seeds:
            # Probing wraps expressions in reveal_type, which can hide an error the plain shadow
            # has; one plain run lets those errors unify too.
            plain = run_pyrefly(res.shadow, workdir, search_paths)
            _unify(plain, shadow_tree, info, new, new_origin, 0, named_toks)
            if new == seeds:
                break
        seeds, origin = new, new_origin
    return seeds


# ------------------------------------------------------------------ hints


class Hint(NamedTuple):
    cls: str
    param: str
    line: int  # 1-based
    col: int  # code points
    seed: list[str] | None  # what tensi knows (`["~b", "channels", "int", "int"]`), None for nothing
    axes: str  # suggested jaxtyping axes, e.g. "b channels height width"


_FEATURE_LAYERS = {"Linear", "LayerNorm", "RMSNorm", "LazyLinear", "Bilinear"}


def _feature_axis(cls: ast.ClassDef, fwd, param: str, classes: dict | None = None, depth: int = 0) -> str | None:
    """The size name of the last axis when forward's first use of `param` is a feature layer:
    `self.fc1(x)` with `self.fc1 = nn.Linear(in_features, ...)` (or a Sequential starting with one)
    means x is `[..., in_features]`."""
    layers: dict[str, ast.expr] = {}
    for f in cls.body:
        if isinstance(f, ast.FunctionDef) and f.name == "__init__":
            for n in ast.walk(f):
                if (
                    isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name)
                    and n.targets[0].value.id == "self"
                ):
                    layers[n.targets[0].attr] = n.value

    def first_feature(v) -> str | None:
        if not isinstance(v, ast.Call):
            return None
        name = v.func.attr if isinstance(v.func, ast.Attribute) else getattr(v.func, "id", "")
        if name == "Sequential" and v.args:
            return first_feature(v.args[0])
        a = None
        if name in _FEATURE_LAYERS and v.args:
            a = v.args[0]
        elif classes and name in classes and depth < 4:  # `self.mlp = Mlp(dim, 4 * dim)`: Mlp's own first layer
            sub = classes[name]
            sfwd = next((f for f in sub.body if isinstance(f, ast.FunctionDef) and f.name == "forward"), None)
            sinit = next((f for f in sub.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
            spos = sfwd.args.posonlyargs + sfwd.args.args if sfwd else []
            feat = _feature_axis(sub, sfwd, spos[1].arg, classes, depth + 1) if len(spos) >= 2 else None
            if feat is None or sinit is None:
                return None
            ipos = [x.arg for x in sinit.args.posonlyargs + sinit.args.args][1:]
            kw = {k.arg: k.value for k in v.keywords if k.arg}
            a = kw.get(feat) or (v.args[ipos.index(feat)] if feat in ipos and ipos.index(feat) < len(v.args) else None)
        if a is not None:
            if isinstance(a, ast.Name):
                return a.id
            if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "self":
                return a.attr
        return None

    for n in sorted((n for n in ast.walk(fwd) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)):
        if not any(isinstance(a, ast.Name) and a.id == param for a in n.args[:1]):
            continue
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "self":
            return first_feature(layers.get(n.func.attr))
        return None
    return None


def _suggest_axes(seed: list[str] | None, feature: str | None = None) -> str:
    """Axis names for the annotation a hint suggests: what tensi knows, and a conventional name
    for each `int` (`batch` first, then `seq`/`dim`, `length` or `height width` by rank)."""
    if not seed:
        return f"batch seq {feature}" if feature else "batch seq"
    rank = len(seed)
    spatial = {1: ["length"], 2: ["height", "width"], 3: ["depth", "height", "width"]}
    out = []
    for i, t in enumerate(seed):
        if t == "int":
            if i == 0:
                out.append("batch")
            elif rank in (3, 4, 5) and i >= 2 and seed[1] != "int":  # [batch, channels, *spatial]
                out.append(spatial[rank - 2][i - 2])
            else:
                out.append(["seq", "dim"][min(i - 1, 1)] if rank <= 3 else f"d{i}")
        elif re.fullmatch(r"~?[A-Za-z_]\w*|\d+", t):
            out.append(t.lstrip("~"))
        else:
            out.append(f"d{i}")
    return " ".join(out)


def unshaped_inputs(source: str, seeds: dict[Key, list[str]] | None) -> list[Hint]:
    """Module inputs where one annotation is the boundary annotation the module needs: a `forward`
    input tensi could not shape from anything in the file (no caller, no `x.size()` unpack, no
    einops pattern), or the input of a top-level module (one no other module in the file builds)
    whose sizes are partly `int`, i.e. come from the data."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.split("\n")
    seeds = seeds or {}
    out = []
    built_inside: set[str] = set()  # classes some other class constructs
    for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for n in ast.walk(c):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id != c.name:
                built_inside.add(n.func.id)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if not any(
            (isinstance(b, ast.Attribute) and b.attr.endswith("Module"))
            or (isinstance(b, ast.Name) and b.id.endswith("Module"))
            for b in cls.bases
        ):
            continue
        fwd = next(
            (f for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "forward"),
            None,
        )
        if fwd is None:
            continue
        pos = fwd.args.posonlyargs + fwd.args.args
        if len(pos) < 2:
            continue
        a = pos[1]
        defaults = dict(zip([x.arg for x in pos[len(pos) - len(fwd.args.defaults) :]], fwd.args.defaults))
        d = defaults.get(a.arg)
        if d is not None and not (isinstance(d, ast.Constant) and d.value is None):
            continue
        if a.annotation is not None and not (
            isinstance(a.annotation, (ast.Name, ast.Attribute))
            and (getattr(a.annotation, "id", None) or a.annotation.attr) == "Tensor"
        ):
            continue
        seed = seeds.get((cls.name, "forward", a.arg))
        if seed and ("int" not in seed or cls.name in built_inside):
            continue
        col = len(lines[a.lineno - 1].encode("utf-8")[: a.col_offset].decode("utf-8", errors="replace"))
        feature = None if seed else _feature_axis(cls, fwd, a.arg, classes)
        out.append(Hint(cls.name, a.arg, a.lineno, col, seed or None, _suggest_axes(seed, feature)))
    return out


def _shape_text(seed: list[str]) -> str:
    return "[" + ", ".join(t.lstrip("~") for t in seed) + "]"


def hint_text(h: Hint) -> str:
    example = f'`{h.param}: Shaped[Tensor, "{h.axes}"]`'
    if h.seed:
        return (
            f"`{h.param}` is {_shape_text(h.seed)} here: each `int` is a size of the data itself, which "
            f"nothing in this file fixes. Annotating the input names them, e.g. {example} (use your "
            f"own axis names), and {h.cls}.forward carries those names from there on."
        )
    return (
        f"tensi can't tell what shape `{h.param}` has: nothing in this file passes it, unpacks its size "
        f"or names its axes. One jaxtyping annotation, e.g. {example} with your own axis names, "
        f"shapes {h.cls}.forward from there on."
    )
