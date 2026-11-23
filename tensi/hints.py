"""Which inputs need a shape annotation, and exactly what to write.

tensi works shapes out from the code itself: an unpack (`b, t = idx.size()`), a caller in the
same file, an einops pattern. What it can't see is the shape of data coming in from outside the
file. Each such input gets a hint: the input, why tensi can't shape it, the annotation to write
(with your own axis names), the import it needs, and in the editor a quick fix that writes it.
"""

from __future__ import annotations

import ast
from typing import NamedTuple

from .seeds import Key, _feature_axis, _is_plain_base, _suggest_axes


class Hint(NamedTuple):
    cls: str  # "" for a module-level function
    param: str
    line: int  # 1-based line of the parameter
    col: int  # code points
    seed: list[str] | None  # what tensi knows (`["~b", "channels", "int", "int"]`), None for nothing
    axes: str  # suggested axis names, e.g. "batch channels height width"
    fn: str = "forward"
    def_line: int = 0  # 1-based line of the `def`
    ann_end: int = -1  # code-point column where an existing bare `Tensor` annotation ends, else -1

    @property
    def where(self) -> str:
        return f"{self.cls}.{self.fn}" if self.cls else self.fn


class Fix(NamedTuple):
    annotation: str  # `Shaped[Tensor, "batch seq"]`
    imports: list[str]  # import lines the file still needs
    import_line: int  # 0-based line to insert them before
    snippet: str  # the `def` line as it would read, or `x: ...` when the signature spans lines


# methods and attributes only a tensor (or array) has: a parameter used like this is a tensor
_TENSOR_USE = {
    "shape", "size", "view", "reshape", "transpose", "permute", "unsqueeze", "squeeze", "flatten",
    "contiguous", "float", "long", "half", "bfloat16", "to", "dim", "ndim", "expand", "expand_as",
    "repeat", "masked_fill", "softmax", "sum", "mean", "matmul", "chunk", "split", "narrow", "gather",
    "index_select", "clone", "detach", "type_as", "new_zeros", "new_ones", "new_empty", "numel", "t",
    "T", "mT", "device", "dtype", "cumsum", "argmax", "argmin", "topk", "unbind", "unflatten", "norm",
}  # fmt: skip
_SIZE_FACTORIES = {
    "arange", "zeros", "ones", "empty", "full", "randn", "rand", "randint", "randperm", "eye", "linspace",
    "tensor", "as_tensor", "manual_seed", "save", "load", "is_tensor", "compile",
}  # fmt: skip
_TENSOR_FUNCS = {"torch", "F", "einops", "rearrange", "repeat", "reduce", "einsum"}


def _tensor_like(fn, param: str, methods: frozenset[str] = frozenset()) -> bool:
    """Whether the body uses `param` the way only a tensor is used. `methods` are the class's own
    methods: `self.render(conversation)` says nothing, `self.attn(x)` (a module) does."""
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == param:
            if n.attr in _TENSOR_USE:
                return True
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == param:
            parts = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            # `x[:, None]`, `x[..., :d]`: two or more axes, or a new one; a lone `xs[1:]` may be a list
            multi = len(parts) > 1 and any(isinstance(p, ast.Slice) or isinstance(p, ast.Constant) for p in parts)
            if multi or any(isinstance(p, ast.Constant) and p.value is None for p in parts):
                return True
        elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == param:
            f = n.func
            root = f
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(f, ast.Attribute) and (f.attr in _SIZE_FACTORIES or f.attr in methods):
                continue  # `torch.arange(dim)`: dim is a size; `self.helper(x)`: a method, not a module
            if isinstance(root, ast.Name) and (root.id == "self" or root.id in _TENSOR_FUNCS):
                return True
    return False


_SAME_SHAPE = {"float", "double", "half", "long", "int", "bool", "to", "contiguous", "detach", "clone", "type_as"}


def _same_shape_of(e, param: str) -> bool:
    """`p`, `p.float()`, `p.to(device).contiguous()`: p, or p with its shape unchanged."""
    while isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in _SAME_SHAPE:
        e = e.func.value
    return isinstance(e, ast.Name) and e.id == param


def _guess_rank(fn, param: str) -> int | None:
    """The rank the body implies: `p[:, None]` has 1+, `p.shape[2]` 3+, `b, t, d = p.shape` 3."""
    rank = 0
    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and _same_shape_of(n.value, param):
            parts = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            if not any(isinstance(p, ast.Constant) and p.value is Ellipsis for p in parts):
                rank = max(rank, sum(1 for p in parts if not (isinstance(p, ast.Constant) and p.value is None)))
        idx = None
        if (
            isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Attribute)
            and n.value.attr == "shape"
            and isinstance(n.value.value, ast.Name)
            and n.value.value.id == param
        ):
            idx = n.slice
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "size"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == param
            and n.args
        ):
            idx = n.args[0]
        if isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub) and isinstance(idx.operand, ast.Constant):
            rank = max(rank, int(idx.operand.value))
        elif isinstance(idx, ast.Constant) and type(idx.value) is int:
            rank = max(rank, idx.value + 1)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple):
            v = n.value
            if (
                isinstance(v, ast.Attribute)
                and v.attr == "shape"
                and isinstance(v.value, ast.Name)
                and v.value.id == param
            ) or (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Attribute)
                and v.func.attr == "size"
                and isinstance(v.func.value, ast.Name)
                and v.func.value.id == param
                and not v.args
            ):
                return len(n.targets[0].elts)
    return rank or None


def _axes_for_rank(rank: int | None) -> str:
    return {1: "n", 2: "batch n", 3: "batch seq dim", 4: "batch heads seq dim"}.get(rank or 0, "batch seq")


def _modules(tree) -> set[str]:
    """`nn.Module` subclasses: direct, through a class of this file, or an imported base with a
    `forward` of its own."""
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    out: set[str] = set()
    changed = True
    while changed:
        changed = False
        for c in classes.values():
            has_forward = any(isinstance(f, ast.FunctionDef) and f.name == "forward" for f in c.body)
            if c.name not in out and any(
                (isinstance(b, ast.Attribute) and b.attr.endswith("Module"))
                or (isinstance(b, ast.Name) and (b.id.endswith("Module") or b.id in out))
                or (has_forward and isinstance(b, (ast.Name, ast.Attribute)) and not _is_plain_base(b))
                for b in c.bases
            ):
                out.add(c.name)
                changed = True
    return out


def _abstract(fn) -> bool:
    """A body that only raises NotImplementedError (or is `...`/`pass`), after docstring and `_ = x`."""
    body = [
        st
        for st in fn.body
        if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))
        and not (isinstance(st, ast.Assign) and all(isinstance(t, ast.Name) and t.id == "_" for t in st.targets))
    ]
    return all(isinstance(st, (ast.Raise, ast.Pass)) for st in body)


def _col(lines: list[str], a: ast.arg) -> int:
    return len(lines[a.lineno - 1].encode("utf-8")[: a.col_offset].decode("utf-8", errors="replace"))


def unshaped_inputs(source: str, seeds: dict[Key, list[str]] | None) -> list[Hint]:
    """Every input whose shape has to come from an annotation:

    * a module's `forward` input that nothing in the file shapes (no caller, no `x.size()` unpack,
      no einops pattern), or one whose sizes are partly `int` (they come from the data) in a module
      no other module here builds;
    * a tensor parameter of a helper or method that nothing in the file calls, so no caller's shape
      can reach it (`precompute_rope_cos_sin(positions, ...)`).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.split("\n")
    seeds = seeds or {}
    out: list[Hint] = []
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    modules = _modules(tree)
    built_inside: set[str] = set()  # classes some other class constructs
    called: set[str] = set()  # names of functions and methods called anywhere in the file
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name:
                called.add(name)
    # built by another class or by a helper function (`return PhiConvStack(...)`); a script's
    # top-level `model = Model()` doesn't count
    for scope in [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]:
        for n in ast.walk(scope):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id != getattr(scope, "name", None):
                if not (isinstance(scope, ast.ClassDef) and n.func.id == scope.name):
                    built_inside.add(n.func.id)

    own_methods = {
        c.name: frozenset(f.name for f in c.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)))
        for c in classes.values()
    }

    def defaults_of(fn) -> dict[str, ast.expr]:
        pos = fn.args.posonlyargs + fn.args.args
        return dict(zip([x.arg for x in pos[len(pos) - len(fn.args.defaults) :]], fn.args.defaults))

    def bare_tensor(a: ast.arg) -> bool:
        ann = a.annotation
        return isinstance(ann, (ast.Name, ast.Attribute)) and (getattr(ann, "id", None) or ann.attr) == "Tensor"

    def add(cls: str, fn, a: ast.arg, seed, axes: str):
        ann_end = -1
        if a.annotation is not None and a.annotation.end_lineno == a.lineno:
            line = lines[a.lineno - 1]
            ann_end = len(line.encode("utf-8")[: a.annotation.end_col_offset].decode("utf-8", errors="replace"))
        out.append(Hint(cls, a.arg, a.lineno, _col(lines, a), seed or None, axes, fn.name, fn.lineno, ann_end))

    # 1. module inputs
    for cname in sorted(modules, key=lambda n: classes[n].lineno):
        cls = classes[cname]
        fwd = next(
            (f for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "forward"),
            None,
        )
        if fwd is None or _abstract(fwd):
            continue
        pos = fwd.args.posonlyargs + fwd.args.args
        defaults = defaults_of(fwd)
        for i, a in enumerate(pos[1:]):
            d = defaults.get(a.arg)
            if a.annotation is not None and not bare_tensor(a):
                continue
            seed = seeds.get((cname, "forward", a.arg))
            if i == 0:
                if d is not None and not (isinstance(d, ast.Constant) and d.value is None):
                    continue
                if cname in built_inside or (seed and "int" not in seed):
                    continue  # another module here passes it: annotating that one's input shapes this too
                feature = None if seed else _feature_axis(cls, fwd, a.arg, classes)
                add(cname, fwd, a, seed, _suggest_axes(seed, feature))
            elif (
                d is None
                and not seed
                and cname not in built_inside
                and _tensor_like(fwd, a.arg, own_methods.get(cname, frozenset()))
            ):
                # a second required input (`forward(self, x, targets)`) that nothing shapes either
                add(cname, fwd, a, None, _suggest_axes(None))

    # 2. helpers and other methods nothing in this file calls
    owner: dict[int, str] = {}
    for c in classes.values():
        for f in c.body:
            owner[id(f)] = c.name
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        cname = owner.get(id(fn), "")
        if fn.name.startswith("__") or (cname in modules and fn.name == "forward") or fn.name in called:
            continue
        if cname == "" and fn not in tree.body:
            continue  # a nested function: its enclosing one passes what it gets
        if any(isinstance(d, ast.Name) and d.id in ("property", "staticmethod") for d in fn.decorator_list):
            continue
        pos = fn.args.posonlyargs + fn.args.args
        defaults = defaults_of(fn)
        for a in pos[1:] if cname else pos:
            if a.annotation is not None and not bare_tensor(a):
                continue
            if a.arg in defaults or seeds.get((cname, fn.name, a.arg)):
                continue
            if _tensor_like(fn, a.arg, own_methods.get(cname, frozenset())):
                add(cname, fn, a, None, _axes_for_rank(_guess_rank(fn, a.arg)))
    return sorted(out, key=lambda h: (h.line, h.col))


# ---------------------------------------------------------------- what to write


def fix_for(source: str, h: Hint) -> Fix:
    """The annotation for `h` in this file's own spelling, the imports it still lacks, and how the
    signature line reads with it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    jax_names: set[str] = set()
    jax_modules: set[str] = set()
    tensor_name = None
    torch_module = None
    last_import = None
    for st in tree.body:
        if isinstance(st, ast.ImportFrom):
            last_import = st
            for al in st.names:
                bound = al.asname or al.name
                if st.module == "jaxtyping":
                    jax_names.add(bound)
                if st.module == "torch" and al.name == "Tensor":
                    tensor_name = bound
        elif isinstance(st, ast.Import):
            last_import = st
            for al in st.names:
                if al.name == "jaxtyping":
                    jax_modules.add(al.asname or al.name)
                if al.name == "torch":
                    torch_module = al.asname or "torch"
    imports = []
    if "Shaped" in jax_names:
        dtype = "Shaped"
    elif jax_modules:
        dtype = f"{sorted(jax_modules)[0]}.Shaped"
    else:
        dtype = "Shaped"
        imports.append("from jaxtyping import Shaped")
    if tensor_name:
        tensor = tensor_name
    elif torch_module:
        tensor = f"{torch_module}.Tensor"
    else:
        tensor = "Tensor"
        imports.append("from torch import Tensor")
    annotation = f'{dtype}[{tensor}, "{h.axes}"]'
    if last_import is not None:
        import_line = last_import.end_lineno
    elif (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
    ):
        import_line = tree.body[0].end_lineno  # after the module docstring
    else:
        import_line = 0
    lines = source.split("\n")
    text = lines[h.line - 1] if 0 < h.line <= len(lines) else ""
    end = h.col + len(h.param)
    if h.def_line == h.line and text[h.col : end] == h.param:
        rest = text[h.ann_end :] if h.ann_end >= 0 else text[end:]
        snippet = (text[:end] + f": {annotation}" + rest).strip()
    else:
        snippet = f"{h.param}: {annotation}"
    return Fix(annotation, imports, import_line, snippet)


def _shape_text(seed: list[str]) -> str:
    return "[" + ", ".join(t.lstrip("~") for t in seed) + "]"


def axes_advice(h: Hint) -> str:
    if h.axes == "...":
        return 'one name per dimension, in order (e.g. `"batch seq"`)'
    return "one name per dimension; rename them to match your data"


def hint_text(h: Hint, fix: Fix | None = None) -> str:
    """The editor message: what to annotate, why, and exactly what to write."""
    if fix is None:
        fix = Fix(f'Shaped[Tensor, "{h.axes}"]', [], 0, f'{h.param}: Shaped[Tensor, "{h.axes}"]')
    if h.seed:
        why = (
            f"`{h.param}` is {_shape_text(h.seed)} here; each `int` is a size of the incoming data, "
            "which nothing in this file names"
        )
    elif h.fn == "forward":
        why = (
            f"tensi can't tell what shape `{h.param}` has: nothing in this file passes it, unpacks "
            "its size or names its axes, so tensors computed from it show no shape"
        )
    else:
        why = (
            f"nothing in this file calls {h.where}, so tensi can't tell what shape `{h.param}` has "
            "and tensors computed from it show no shape"
        )
    need = f" It needs `{'`, `'.join(fix.imports)}` (jaxtyping installed in your project)." if fix.imports else ""
    return (
        f"Annotate `{h.param}` to shape {h.where}: {why}. Write `{fix.snippet}` ({axes_advice(h)}; a size "
        f"name from `__init__` such as `n_embd` links to it, a number fixes a size).{need} "
        "The quick fix writes it for you."
    )
