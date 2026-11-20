"""Source -> shadow rewriter.

The shadow is the user's file with tensor-size `int` parameters and jaxtyping
annotations turned into Pyrefly's declared-dimension form. Every edit stays on
the line it came from, so line numbers are identical in both texts and only
columns move (see mapping.py).
"""

from __future__ import annotations

import ast
import functools
import re
from dataclasses import dataclass, field

import parso
from parso.python import tree as pt

PREFIX = "_ts_"
IMPORT_TEXT = (
    "from shape_extensions import Int as _ts_Int, IntVar as _ts_IntVar; from torch import Tensor as _ts_Tensor"
)
_NAME_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_TOKEN = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class Edit:
    """Replace user text [start, end) on `line` (0-based; code-point columns) with `text`."""

    line: int
    start: int
    end: int
    text: str


@dataclass
class RewriteResult:
    shadow: str
    edits: list[Edit]
    notes: list[str] = field(default_factory=list)
    # For the seed solver: class name -> [(user dim name, type-param name)] in declaration order,
    # and class name -> {method name: [seedable tensor params]}.
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigClass:
    """A config object whose int fields are tensor sizes (e.g. nanoGPT's GPTConfig)."""

    name: str
    int_fields: tuple[str, ...]
    module: str | None = None  # dotted module, for files that use it without importing it


_CONFIG_NAME = re.compile(r"(Config|Cfg|Args|Params|HParams|Hparams|Settings)$")


def find_config_classes(source: str, module: str | None = None, tree: ast.AST | None = None) -> dict[str, ConfigClass]:
    """Classes with `int`-annotated class-level fields that are @dataclass or named like configs."""
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
    out: dict[str, ConfigClass] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        deco = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            or (isinstance(d, ast.Call) and _call_name(d.func) == "dataclass")
            for d in node.decorator_list
        )
        if not (deco or _CONFIG_NAME.search(node.name)):
            continue
        fields = tuple(
            st.target.id
            for st in node.body
            if isinstance(st, ast.AnnAssign)
            and isinstance(st.target, ast.Name)
            and isinstance(st.annotation, ast.Name)
            and st.annotation.id == "int"
        )
        if fields:
            out[node.name] = ConfigClass(node.name, fields, module)
    return out


def _grammar():
    for v in ("3.13", "3.12"):
        try:
            return parso.load_grammar(version=v)
        except Exception:  # pragma: no cover - depends on parso version
            continue
    return parso.load_grammar()


_GRAMMAR = _grammar()


# ---------------------------------------------------------------- int-parameter inference
# Which argument positions / keywords of well-known calls are tensor sizes (feature counts,
# channel counts, shape entries). Kernel sizes, strides and paddings are deliberately absent:
# they are often tuples, and calling a tuple an `int` dim would make hovers lie.
ALL = "*"
SIZE_CALLS: dict[str, tuple[object, tuple[str, ...]]] = {
    "Linear": ((0, 1), ("in_features", "out_features")),
    "Bilinear": ((0, 1, 2), ("in1_features", "in2_features", "out_features")),
    "Embedding": ((0, 1), ("num_embeddings", "embedding_dim")),
    "EmbeddingBag": ((0, 1), ("num_embeddings", "embedding_dim")),
    "LayerNorm": ((0,), ("normalized_shape",)),
    "RMSNorm": ((0,), ("normalized_shape",)),
    "BatchNorm1d": ((0,), ("num_features",)),
    "BatchNorm2d": ((0,), ("num_features",)),
    "BatchNorm3d": ((0,), ("num_features",)),
    "InstanceNorm1d": ((0,), ("num_features",)),
    "InstanceNorm2d": ((0,), ("num_features",)),
    "GroupNorm": ((0, 1), ("num_groups", "num_channels")),
    "Conv1d": ((0, 1), ("in_channels", "out_channels")),
    "Conv2d": ((0, 1), ("in_channels", "out_channels")),
    "Conv3d": ((0, 1), ("in_channels", "out_channels")),
    "ConvTranspose1d": ((0, 1), ("in_channels", "out_channels")),
    "ConvTranspose2d": ((0, 1), ("in_channels", "out_channels")),
    "MultiheadAttention": ((0, 1), ("embed_dim", "num_heads")),
    "LSTM": ((0, 1), ("input_size", "hidden_size")),
    "GRU": ((0, 1), ("input_size", "hidden_size")),
    "RNN": ((0, 1), ("input_size", "hidden_size")),
    "zeros": (ALL, ()),
    "ones": (ALL, ()),
    "empty": (ALL, ()),
    "randn": (ALL, ()),
    "rand": (ALL, ()),
    "randint": ((), ()),
    "full": ((0,), ("size",)),
    "arange": (ALL, ("start", "end")),
    "view": (ALL, ()),
    "reshape": (ALL, ()),
    "expand": (ALL, ()),
    "new_zeros": (ALL, ()),
    "new_ones": (ALL, ()),
    "new_empty": (ALL, ()),
    "unflatten": ((1,), ("sizes",)),
    "topk": ((1,), ("k",)),
    "kthvalue": ((1,), ("k",)),
    "narrow": ((3,), ("length",)),
    "multinomial": ((1,), ("num_samples",)),
    "chunk": ((1,), ("chunks",)),
    "rearrange": ((), ("*",)),
    "repeat": ((), ("*",)),
    "reduce": ((), ("*",)),
    "Rearrange": ((), ("*",)),
    "Repeat": ((), ("*",)),
    "Reduce": ((), ("*",)),
}


# Parameter names that are often tuples in PyTorch code, even when their default is an int.
TUPLE_PRONE = {
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "output_padding",
    "image_size",
    "patch_size",
    "size",
    "shape",
    "output_size",
    "scale_factor",
}


def _call_name(func) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_self_attr(node) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"


def _within(node, outer) -> bool:
    return outer.lineno <= node.lineno and node.end_lineno <= outer.end_lineno


def _bare_names(expr) -> set[str]:
    """Names used as values inside `expr` (arithmetic allowed), not as call targets or attribute bases."""
    out: set[str] = set()
    if isinstance(expr, ast.Name):
        out.add(expr.id)
    elif isinstance(expr, ast.BinOp):
        out |= _bare_names(expr.left) | _bare_names(expr.right)
    elif isinstance(expr, ast.UnaryOp):
        out |= _bare_names(expr.operand)
    elif isinstance(expr, (ast.Tuple, ast.List)):
        for e in expr.elts:
            out |= _bare_names(e)
    elif isinstance(expr, ast.Starred):
        out |= _bare_names(expr.value)
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ("min", "max", "int"):
        for a in expr.args:
            out |= _bare_names(a)
    return out


class _FnIndex:
    """One traversal of the stdlib AST: for every function, the calls anywhere inside it
    (nested lambdas/defs included, since they can use its params) and the names it stores
    in its own scope."""

    def __init__(self, tree: ast.AST):
        self.funcs: list[ast.AST] = []
        self.calls: dict[int, list[ast.Call]] = {}
        self.stores: dict[int, set[str]] = {}
        self.classes: list[ast.ClassDef] = []
        self.owner_class: dict[int, str | None] = {}
        self.attr_reads: dict[int, dict[str, set[str]]] = {}  # fn -> name -> {attr} for `name.attr`
        FN = (ast.FunctionDef, ast.AsyncFunctionDef)
        # stack items: (node, enclosing function ids, id of the function owning this scope or None,
        #               name of the class whose body we are directly in)
        stack: list = [(tree, (), None, None)]
        while stack:
            node, fns, owner, cls = stack.pop()
            if isinstance(node, FN):
                self.funcs.append(node)
                self.calls[id(node)] = []
                self.stores[id(node)] = set()
                self.attr_reads[id(node)] = {}
                self.owner_class[id(node)] = cls
                inner_fns, inner_owner = fns + (id(node),), id(node)
                # decorators/defaults/annotations belong to the enclosing scope
                for d in node.decorator_list:
                    stack.append((d, fns, owner, None))
                for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                    stack.append((d, fns, owner, None))
                for b in node.body:
                    stack.append((b, inner_fns, inner_owner, None))
                continue
            if isinstance(node, ast.ClassDef):
                self.classes.append(node)
                for b in node.body:
                    stack.append((b, fns, None, node.name))
                for b in node.bases + node.decorator_list + [k.value for k in node.keywords]:
                    stack.append((b, fns, owner, None))
                continue
            if isinstance(node, ast.Lambda):
                for c in ast.iter_child_nodes(node):
                    stack.append((c, fns, None, None))
                continue
            if isinstance(node, ast.Call):
                for f in fns:
                    self.calls[f].append(node)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and owner is not None:
                self.stores[owner].add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                for f in fns:
                    self.attr_reads[f].setdefault(node.value.id, set()).add(node.attr)
            for c in ast.iter_child_nodes(node):
                stack.append((c, fns, owner, None))


def _param_names(fn) -> list[str]:
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]


def infer_int_params(
    source: str, tree: ast.AST | None = None, index: _FnIndex | None = None
) -> dict[tuple[int, str], set[str]]:
    return _infer_ints(source, tree, index)[0]


def _ast_functions(node):
    return [(n, None) for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def pair_helpers(tree: ast.AST) -> dict[str, int]:
    """`def pair(t): return t if isinstance(t, tuple) else (t, t)` -> {"pair": 2}."""
    out: dict[str, int] = {}
    for fn in getattr(tree, "body", []):
        if not (
            isinstance(fn, ast.FunctionDef)
            and len(fn.args.args) == 1
            and not fn.args.defaults
            and len(fn.body) == 1
            and isinstance(fn.body[0], ast.Return)
        ):
            continue
        t = fn.args.args[0].arg
        r = fn.body[0].value
        if not (
            isinstance(r, ast.IfExp)
            and isinstance(r.body, ast.Name)
            and r.body.id == t
            and isinstance(r.test, ast.Call)
            and isinstance(r.test.func, ast.Name)
            and r.test.func.id == "isinstance"
            and len(r.test.args) == 2
            and isinstance(r.test.args[0], ast.Name)
            and r.test.args[0].id == t
            and isinstance(r.orelse, ast.Tuple)
            and len(r.orelse.elts) >= 2
            and all(isinstance(e, ast.Name) and e.id == t for e in r.orelse.elts)
        ):
            continue
        out[fn.name] = len(r.orelse.elts)
    return out


_INT_CALLS = {"int", "min", "max", "round", "len", "abs", "floor", "ceil", "default"}


def _rebound_to_non_int(fn, names: set[str]) -> set[str]:
    """Params the body rebinds to something that is visibly not an int: `local_window_size = 7` then
    `local_window_size = cast_tuple(local_window_size, 4)` is a tuple from then on."""
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        hit = {t.id for t in targets if isinstance(t, ast.Name) and t.id in names}
        hit |= {
            e.id
            for t in targets
            if isinstance(t, (ast.Tuple, ast.List))
            for e in t.elts
            if isinstance(e, ast.Name) and e.id in names
        }
        if not hit:
            continue
        if isinstance(value, ast.Call):
            cname = (_call_name(value.func) or "").rsplit(".", 1)[-1]
            if cname in _INT_CALLS:
                continue
        elif not isinstance(
            value, (ast.Tuple, ast.List, ast.ListComp, ast.Dict, ast.Set, ast.GeneratorExp, ast.Constant)
        ) or (isinstance(value, ast.Constant) and type(value.value) is int):
            continue
        out |= hit
    return out


def _infer_ints(source: str, tree: ast.AST | None = None, index: _FnIndex | None = None):
    """Unannotated parameters that are almost certainly ints used as sizes.

    Returns {(def line 1-based, function name): {param names}}. A parameter qualifies if its
    default is an int literal, or if it appears (alone or in arithmetic) at a size position of
    a known call in that function's body. Calls to classes and functions defined in the same
    file count too, at the positions of *their* dim parameters (iterated to a fixpoint).
    Parameters defaulting to None/float/str/tuple never qualify.
    """
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}, {}
    index = index or _FnIndex(tree)
    funcs = index.funcs
    calls_of = index.calls
    # local callables: class name -> its __init__, function name -> itself (top level or methods)
    local: dict[str, ast.AST] = {}
    for cls in index.classes:
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                local[cls.name] = item
    for f in funcs:
        local.setdefault(f.name, f)
    pairs = pair_helpers(tree)

    def explicit_ints(fn) -> set[str]:
        a = fn.args
        return {
            x.arg
            for x in a.posonlyargs + a.args + a.kwonlyargs
            if isinstance(x.annotation, ast.Name) and x.annotation.id == "int"
        }

    result: dict[int, set[str]] = {id(f): set() for f in funcs}
    optional: dict[int, set[str]] = {id(f): set() for f in funcs}

    def dims_of(fn) -> set[str]:
        return result.get(id(fn), set()) | explicit_ints(fn)

    for _ in range(6):  # fixpoint over same-file calls
        changed = False
        for fn in funcs:
            a = fn.args
            positional = a.posonlyargs + a.args
            defaults: dict[str, ast.expr] = {}
            for arg, d in zip(positional[len(positional) - len(a.defaults) :], a.defaults):
                defaults[arg.arg] = d
            for arg, d in zip(a.kwonlyargs, a.kw_defaults):
                if d is not None:
                    defaults[arg.arg] = d
            cands = {
                x.arg
                for x in positional + a.kwonlyargs
                if x.annotation is None and x.arg not in ("self", "cls") and x.arg not in TUPLE_PRONE
            }
            # a tuple-prone name piped through `pair()` is an int as far as the shadow is concerned
            pair_cands = {
                x.arg for x in positional + a.kwonlyargs if x.annotation is None and x.arg not in ("self", "cls")
            }
            if not pair_cands:
                continue
            chosen: set[str] = set()
            for name in cands:
                d = defaults.get(name)
                if isinstance(d, ast.Constant) and type(d.value) is int:
                    chosen.add(name)
            bad_default = {n for n, d in defaults.items() if not (isinstance(d, ast.Constant) and type(d.value) is int)}
            bad_default |= _rebound_to_non_int(fn, cands)
            for node in calls_of[id(fn)]:
                cname = _call_name(node.func) or ""
                if cname in pairs and len(node.args) == 1:
                    # `image_height, image_width = pair(image_size)`: an int (or a tuple of them)
                    chosen |= _bare_names(node.args[0]) & pair_cands
                    continue
                rule = SIZE_CALLS.get(cname)
                if rule is None and cname in local and local[cname] is not fn:
                    target = local[cname]
                    tdims = dims_of(target) | optional[id(target)]  # `out_features=None` sizes too
                    if tdims:
                        names = [n for n in _param_names(target) if n not in ("self", "cls")]
                        rule = (tuple(i for i, n in enumerate(names) if n in tdims), tuple(tdims))
                if rule is None:
                    continue
                positions, keywords = rule
                for i, arg in enumerate(node.args):
                    if positions == ALL or i in positions:
                        chosen |= _bare_names(arg) & cands
                for kw in node.keywords:
                    if kw.arg and (kw.arg in keywords or "*" in keywords):
                        chosen |= _bare_names(kw.value) & cands
            none_default = {n for n, d in defaults.items() if isinstance(d, ast.Constant) and d.value is None}
            if (chosen & none_default) - optional[id(fn)]:
                optional[id(fn)] |= chosen & none_default
                changed = True
            chosen -= bad_default
            if chosen - result[id(fn)]:
                result[id(fn)] |= chosen
                changed = True
        if not changed:
            break
    return (
        {(f.lineno, f.name): result[id(f)] for f in funcs if result[id(f)]},
        {(f.lineno, f.name): optional[id(f)] for f in funcs if optional[id(f)]},
    )


def reassigned_params(
    source: str, tree: ast.AST | None = None, index: _FnIndex | None = None
) -> dict[tuple[int, str], set[str]]:
    """{(def line, name): params that the function body rebinds (x = ..., x += ..., for x in ...)}."""
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
    index = index or _FnIndex(tree)
    out: dict[tuple[int, str], set[str]] = {}
    for fn in index.funcs:
        hit = index.stores[id(fn)] & set(_param_names(fn))
        if hit:
            out[(fn.lineno, fn.name)] = hit
    return out


class _Scope:
    """Names of dims visible at a point: maps user dim name -> type-param name."""

    def __init__(self, parent: _Scope | None = None):
        self.parent = parent
        self.dims: dict[str, str] = {}
        self.config_tps: dict[str, tuple[str, list[str]]] = {}  # param -> (config ref, tps)

    def lookup(self, name: str) -> str | None:
        s: _Scope | None = self
        while s is not None:
            if name in s.dims:
                return s.dims[name]
            s = s.parent
        return None

    def all_tp_names(self) -> set[str]:
        out: set[str] = set()
        s: _Scope | None = self
        while s is not None:
            out.update(s.dims.values())
            s = s.parent
        return out


class _Analysis:
    """Everything the rewriter derives from the source alone. The seed solver rewrites the same
    source once per round with different seeds, so this is computed once and shared (read-only)."""

    def __init__(self, source: str, infer_ints: bool):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        self.tree = tree
        self.index = index = _FnIndex(tree) if tree is not None else None
        self.inferred, self.optional_ints = _infer_ints(source, tree, index) if (infer_ints and index) else ({}, {})
        self.reassigned = reassigned_params(source, tree, index) if index else {}
        self.local_configs = find_config_classes(source, tree=tree) if tree is not None else {}
        self.imported_names = frozenset(
            a.asname or a.name
            for n in (ast.walk(tree) if tree else ())
            if isinstance(n, ast.ImportFrom)
            for a in n.names
        )
        self.scans: dict[str, list] = {}
        self._source = source
        self._module = None

    def module(self):
        if self._module is None:
            self._module = _GRAMMAR.parse(self._source, error_recovery=True)
        return self._module


@functools.lru_cache(maxsize=8)
def _analyse(source: str, infer_ints: bool) -> _Analysis:
    return _Analysis(source, infer_ints)


class _Rewriter:
    def __init__(
        self,
        source: str,
        package: str | None = None,
        infer_ints: bool = True,
        cache_key: str | None = None,
        config_index: dict[str, ConfigClass] | None = None,
        seeds: dict | None = None,
    ):
        self.source = source
        self.package = package
        a = _analyse(source, infer_ints)
        self.tree, self.index = a.tree, a.index
        self.inferred, self.optional_ints = a.inferred, a.optional_ints
        self.reassigned = a.reassigned
        self.src_lines = source.split("\n")
        # config classes: this file's own (made generic here) and the workspace's (referenced)
        self.local_configs = a.local_configs
        self.configs = {**(config_index or {}), **self.local_configs}
        self.imported_names = a.imported_names
        self._scans = a.scans
        self.extra_imports: dict[str, str] = {}  # alias -> import statement
        self.int_local_tps: dict[tuple, str] = {}  # (class, fn, "#line:var") -> type param
        self.fallback_params: dict[tuple, str] = {}  # (class, `__init__` param) -> defaulted type param
        self.config_params = self._find_config_params() if self.index else {}
        self.seeds = seeds or {}
        self.info: dict = {"classes": {}, "methods": {}, "configs": {}}
        self.lines = source.splitlines(keepends=True) or [""]
        self.edits: list[Edit] = []
        self.notes: list[str] = []
        self.jaxtyping_names: set[str] = set()  # bare names imported from jaxtyping
        self.jaxtyping_modules: set[str] = set()  # `import jaxtyping as jt` -> {"jt"}
        if cache_key:
            # parso's diff parser: re-parses only what changed since the last call with this key
            self.module = _GRAMMAR.parse(source, error_recovery=True, path=cache_key, diff_cache=True, cache=False)
        else:
            self.module = a.module()

    # ---------- helpers ----------
    def _edit(self, start_pos, end_pos, text):
        (l1, c1), (l2, c2) = start_pos, end_pos
        if l1 != l2:
            raise ValueError("multi-line edit")
        self.edits.append(Edit(l1 - 1, c1, c2, text))

    def _fresh(self, name: str, scope: _Scope) -> str:
        base = PREFIX + name
        taken = scope.all_tp_names()
        if base not in taken:
            return base
        i = 2
        while f"{base}__{i}" in taken:
            i += 1
        return f"{base}__{i}"

    @staticmethod
    def _has_errors(node) -> bool:
        if node.type in ("error_node", "error_leaf"):
            return True
        return any(_Rewriter._has_errors(c) for c in getattr(node, "children", []))

    # ---------- stdlib-ast positions ----------
    def _cp(self, lineno: int, byte_col: int) -> int:
        """ast byte column -> code-point column."""
        line = self.src_lines[lineno - 1] if 0 < lineno <= len(self.src_lines) else ""
        return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))

    def _ast_span(self, node):
        return (
            (node.lineno, self._cp(node.lineno, node.col_offset)),
            (node.end_lineno, self._cp(node.end_lineno, node.end_col_offset)),
        )

    # ---------- config objects ----------
    def _find_config_params(self) -> dict[tuple[int, str], dict[str, ConfigClass]]:
        """{(def line, fn name): {param: config class}} for params read as `param.<int field>`.

        A param qualifies when it is unannotated (or annotated with the bare config class name)
        and the attributes read from it include int fields of exactly one best-matching config."""
        out: dict = {}
        if not self.configs:
            return out
        for fn in self.index.funcs:
            reads = self.index.attr_reads.get(id(fn), {})
            a = fn.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                if arg.arg in ("self", "cls") or arg.arg not in reads:
                    continue
                ann = arg.annotation
                if ann is not None and not (isinstance(ann, ast.Name) and ann.id in self.configs):
                    continue
                attrs = reads[arg.arg]
                if ann is not None:
                    best = [self.configs[ann.id]]
                else:
                    scored = sorted(
                        ((len(attrs & set(c.int_fields)), c.name) for c in self.configs.values()), reverse=True
                    )
                    if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
                        continue
                    best = [self.configs[scored[0][1]]]
                out.setdefault((fn.lineno, fn.name), {})[arg.arg] = best[0]
        return out

    def _config_ref(self, cfg: ConfigClass) -> str:
        """How this file names `cfg`: directly if defined/imported here, else via an alias import."""
        if cfg.name in self.local_configs or cfg.name in self.imported_names or cfg.module is None:
            return cfg.name
        alias = f"_ts_cfg_{cfg.name}"
        self.extra_imports[alias] = f"from {cfg.module} import {cfg.name} as {alias}"
        return alias

    # ---------- nn.ModuleDict ----------
    def _rewrite_module_dicts(self):
        """`self.A = nn.ModuleDict(dict(k=v, ...))` loses per-key types (every access is a union).
        Capture each value with a walrus, expose it as a typed attribute `self._ts_md_A__k` at the
        end of the statement's line, and point `self.A.k` / `self.A["k"]` accesses at it."""
        if self.tree is None:
            return
        for cls in [n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]:
            dicts: dict[str, list[str]] = {}
            for fn in cls.body:
                if not (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "__init__"):
                    continue
                for st in ast.walk(fn):
                    if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
                        continue
                    t = st.targets[0]
                    if not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"):
                        continue
                    v = st.value
                    if not (isinstance(v, ast.Call) and _call_name(v.func) == "ModuleDict" and v.args):
                        continue
                    inner = v.args[0]
                    pairs = []
                    if isinstance(inner, ast.Call) and _call_name(inner.func) == "dict" and not inner.args:
                        pairs = [(k.arg, k.value) for k in inner.keywords if k.arg]
                    elif isinstance(inner, ast.Dict) and all(
                        isinstance(k, ast.Constant) and isinstance(k.value, str) for k in inner.keys
                    ):
                        pairs = [(k.value, val) for k, val in zip(inner.keys, inner.values)]
                    pairs = [(k, val) for k, val in pairs if k.isidentifier()]
                    if not pairs:
                        continue
                    attr = t.attr
                    binds = []
                    for key, val in pairs:
                        local = f"_ts_md_{attr}__{key}"
                        (l1, c1), (l2, c2) = self._ast_span(val)
                        self._edit((l1, c1), (l1, c1), f"({local} := ")
                        self._edit((l2, c2), (l2, c2), ")")
                        binds.append(f"self.{local} = {local}")
                    (_, _), (el, ec) = self._ast_span(st)
                    self._edit((el, ec), (el, ec), "; " + "; ".join(binds))
                    dicts[attr] = [k for k, _ in pairs]
            if not dicts:
                continue
            for node in ast.walk(cls):
                key = None
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"
                    and node.value.attr in dicts
                    and node.attr in dicts[node.value.attr]
                ):
                    key = (node.value.attr, node.attr)
                elif (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"
                    and node.value.attr in dicts
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value in dicts[node.value.attr]
                ):
                    key = (node.value.attr, node.slice.value)
                if key and node.lineno == node.end_lineno:
                    # replace `A.k` / `A["k"]`, keep `self.`: a hover on `k` then maps onto the
                    # typed attribute rather than onto `self`
                    (l1, c1), (l2, c2) = self._ast_span(node)
                    m = re.match(r"self\s*\.\s*", self.src_lines[l1 - 1][c1:c2])
                    if m:
                        self._edit((l1, c1 + m.end()), (l2, c2), f"_ts_md_{key[0]}__{key[1]}")

    # ---------- nn.ModuleList filled by append ----------
    def _rewrite_module_lists(self):
        """`self.layers = nn.ModuleList([])` + `self.layers.append(X)` in a loop leaves
        `self.layers` as `ModuleList[Unknown]`: Pyrefly's first-use inference covers locals,
        not attributes. Turn the one append into `self._ts_ml_layers = [X]` and read
        `self.layers` through it elsewhere, so `for attn, ff in self.layers` sees X's type.
        A literal `nn.ModuleList([a, b])` inside X becomes the tuple `((a, b))`, so unpacking
        keeps each module's own type instead of a union. The shadow is never run."""
        if self.tree is None:
            return
        for cls in [n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]:
            init = next(
                (
                    f
                    for f in cls.body
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            empty: set[str] = set()
            appends: dict[str, list[ast.Call]] = {}
            other_writes: set[str] = set()
            for node in ast.walk(init):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and _is_self_attr(node.targets[0]):
                    v = node.value
                    if (
                        isinstance(v, ast.Call)
                        and _call_name(v.func) == "ModuleList"
                        and not v.keywords
                        and (not v.args or (isinstance(v.args[0], ast.List) and not v.args[0].elts))
                    ):
                        empty.add(node.targets[0].attr)
                    else:
                        other_writes.add(node.targets[0].attr)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and _is_self_attr(node.func.value)
                    and node.func.attr in ("append", "extend", "insert")
                ):
                    appends.setdefault(node.func.value.attr, []).append(node)
            for attr in sorted(empty - other_writes):
                calls = appends.get(attr, [])
                if len(calls) != 1 or calls[0].func.attr != "append" or len(calls[0].args) != 1 or calls[0].keywords:
                    continue
                call = calls[0]
                arg = call.args[0]
                (l1, c1), _ = self._ast_span(call)
                (al, ac), (ael, aec) = self._ast_span(arg)
                (_el, _ec), (cl, cc) = self._ast_span(call)
                if al != l1:
                    continue
                # `self.layers.append(` -> `self._ts_ml_layers = [(`  ...  `)` -> `)]`
                self._edit((l1, c1), (al, ac), f"self._ts_ml_{attr} = [(")
                self._edit((cl, cc - 1), (cl, cc), ")]")
                for inner in ast.walk(arg):
                    if (
                        isinstance(inner, ast.Call)
                        and _call_name(inner.func) == "ModuleList"
                        and len(inner.args) == 1
                        and not inner.keywords
                        and isinstance(inner.args[0], ast.List)
                        and inner.args[0].elts
                    ):
                        (il, ic), (iel, iec) = self._ast_span(inner)
                        (ll, lc), (lel, lec) = self._ast_span(inner.args[0])
                        if il != ll or iel != lel:
                            continue
                        self._edit((il, ic), (ll, lc + 1), "((")
                        self._edit((lel, lec - 1), (iel, iec), "))")
                for node in ast.walk(cls):
                    if (
                        isinstance(node, ast.Attribute)
                        and _is_self_attr(node)
                        and node.attr == attr
                        and not _within(node, init)
                        and node.lineno == node.end_lineno
                    ):
                        (nl, nc), (nel, nec) = self._ast_span(node)
                        line = self.src_lines[nl - 1]
                        m = re.match(r"self\s*\.\s*", line[nc:nec])
                        if m:
                            self._edit((nl, nc + m.end()), (nel, nec), f"_ts_ml_{attr}")

    # ---------- fancy indexing with a one-element list ----------
    def _rewrite_identity_fallbacks(self):
        """`nn.Sequential(...) if project_out else nn.Identity()` is a union, and Pyrefly calls a
        union of modules without their shape rules (the result is a bare `Tensor`), so every later
        line loses its shape. In the shadow the condition becomes `True` (or `False` for
        `nn.Identity() if c else X`), picking the module that changes the shape. Identity is the
        degenerate case where the two shapes agree anyway."""
        if self.tree is None:
            return

        def is_identity(e) -> bool:
            return isinstance(e, ast.Call) and _call_name(e.func) == "Identity" and not e.args

        for node in ast.walk(self.tree):
            if not isinstance(node, ast.IfExp) or is_identity(node.body) == is_identity(node.orelse):
                continue
            t = node.test
            if t.lineno != t.end_lineno:
                continue
            if self._claim(t):
                self._edit(*self._ast_span(t), "False" if is_identity(node.body) else "True")

    def _default_flag_choices(self):
        """`linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear` in an
        `__init__` with `use_conv: bool = False`: a union of layer types loses every shape after it.
        In the shadow the test is the flag's default (`False`), the configuration a caller gets by
        default; factory parameters are read the same way."""
        if self.tree is None:
            return
        for fn in ast.walk(self.tree):
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                continue
            pos = fn.args.posonlyargs + fn.args.args
            flags = {
                a.arg: d.value
                for a, d in zip(pos[len(pos) - len(fn.args.defaults) :], fn.args.defaults)
                if isinstance(d, ast.Constant) and isinstance(d.value, bool)
            }
            flags.update(
                {
                    a.arg: d.value
                    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
                    if isinstance(d, ast.Constant) and isinstance(d.value, bool)
                }
            )
            if not flags:
                continue
            assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            for node in ast.walk(fn):
                if not isinstance(node, ast.IfExp):
                    continue
                t, neg = node.test, False
                if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
                    t, neg = t.operand, True
                if not (isinstance(t, ast.Name) and t.id in flags and t.id not in assigned):
                    continue
                if node.test.lineno == node.test.end_lineno and self._claim(node.test):
                    self._edit(*self._ast_span(node.test), str(flags[t.id] != neg))

    def _unroll_map_lambdas(self):
        """`(a, b), (c, d) = map(lambda t: (t[:, :1], t[:, 1:]), (x, y))` types every target as the
        union of both results, which can read as a false mismatch later. With a literal tuple of
        names, spell the calls out: `((x[:, :1], x[:, 1:]), (y[:, :1], y[:, 1:]))`, same line."""
        if self.tree is None:
            return
        for node in ast.walk(self.tree):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Tuple, ast.List))
            ):
                continue
            v = node.value
            if not (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name)
                and v.func.id == "map"
                and len(v.args) == 2
                and not v.keywords
                and isinstance(v.args[0], ast.Lambda)
                and isinstance(v.args[1], (ast.Tuple, ast.List))
            ):
                continue
            lam, seq = v.args
            a = lam.args
            if len(a.args) != 1 or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults:
                continue
            if (
                len(seq.elts) != len(node.targets[0].elts)
                or len(seq.elts) < 2
                or not all(isinstance(e, ast.Name) for e in seq.elts)
            ):
                continue
            if v.lineno != v.end_lineno:
                continue
            param = a.args[0].arg
            line = self.src_lines[lam.body.lineno - 1]
            (bl, bc), (bel, bec) = self._ast_span(lam.body)
            if bl != bel:
                continue
            body = line[bc:bec]
            uses = sorted(
                (self._ast_span(n)[0][1], self._ast_span(n)[1][1])
                for n in ast.walk(lam.body)
                if isinstance(n, ast.Name) and n.id == param
            )
            if any(isinstance(n, ast.Lambda) for n in ast.walk(lam.body)):
                continue  # a nested lambda could rebind the name

            def subst(name: str) -> str:
                out, pos = [], 0
                for c1, c2 in uses:
                    out.append(body[pos : c1 - bc])
                    out.append(name)
                    pos = c2 - bc
                out.append(body[pos:])
                return "(" + "".join(out) + ")"

            text = "(" + ", ".join(subst(e.id) for e in seq.elts) + ")"
            if self._claim(v):
                self._edit(*self._ast_span(v), text)

    def _scan(self, name: str) -> list:
        """The result of one of the seeds module's whole-file AST scans, computed once per source."""
        if name not in self._scans:
            from . import seeds

            self._scans[name] = list(getattr(seeds, name)(self.tree)) if self.tree is not None else []
        return self._scans[name]

    def _claim(self, node) -> bool:
        """Reserve a node's span for one whole-span replacement; False if it overlaps a claimed one."""
        (l1, c1), (l2, c2) = self._ast_span(node)
        span = ((l1, c1), (l2, c2))
        claimed = self.__dict__.setdefault("_claimed", [])
        if any(not (span[1] <= a or b <= span[0]) for a, b in claimed):
            return False
        claimed.append(span)
        return True

    def _split_star_shapes(self):
        """`b, n, _, device = *x.shape, x.device` -> `(b, n, _), device = x.shape, x.device`: the same
        unpacking (both need exactly rank + extras values), but Pyrefly types each of `b, n` from
        its own dim instead of a union of every dim."""
        if self.tree is None:
            return
        from .seeds import shape_unpack

        for st in ast.walk(self.tree):
            su = shape_unpack(st)
            if su is None or not isinstance(st.value, ast.Tuple) or not isinstance(st.value.elts[0], ast.Starred):
                continue
            if st.lineno != st.end_lineno or not self._claim(st.value.elts[0]):
                continue
            star = st.value.elts[0]
            (l1, c1), _end = self._ast_span(star)
            self._edit((l1, c1), (l1, c1 + 1), "")  # drop the `*`
            n = len(su[1])
            first, last = st.targets[0].elts[0], st.targets[0].elts[n - 1]
            (fl, fc), (fl2, fc2) = self._ast_span(first)
            _s, (ll, lc) = self._ast_span(last)
            # a replacement, not an insert: a rebinding inserted at the statement start sorts first
            self._edit((fl, fc), (fl2, fc2), "(" + self.src_lines[fl - 1][fc:fc2])
            self._edit((ll, lc), (ll, lc), ")")

    _BARE_TENSOR_ANN = re.compile(r"(?:Optional\[)?(?:torch\.)?(?:[A-Z][a-z]*)?Tensor\]?(?:\s*\|\s*None)?")

    def _drop_local_tensor_annotations(self):
        """`inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)` inside a function: the bare
        annotation replaces the shape Pyrefly would infer. In the shadow it is `inputs_embeds = ...`."""
        if self.tree is None:
            return
        for fn in ast.walk(self.tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for st in ast.walk(fn):
                if not (
                    isinstance(st, ast.AnnAssign)
                    and st.value is not None
                    and isinstance(st.target, ast.Name)
                    and st.lineno == st.end_lineno
                    and st.simple
                ):
                    continue
                if not self._BARE_TENSOR_ANN.fullmatch(ast.unparse(st.annotation)):
                    continue
                _s, t_end = self._ast_span(st.target)
                _s2, a_end = self._ast_span(st.annotation)
                if self._claim(st.annotation):
                    self._edit(t_end, a_end, "")

    def _type_layer_factories(self):
        """`act_layer: Type[nn.Module] = nn.GELU` (or `norm_layer=nn.LayerNorm`, or `norm_layer=None`
        then `if norm_layer is None: norm_layer = nn.BatchNorm2d`): the declared `Type[nn.Module]`
        makes every layer built from it shapeless. In the shadow the parameter is typed as its
        default, `type[nn.GELU]`, so `self.act = act_layer()` is a GELU; a caller that passes
        another class only gets a (hidden) typing complaint."""
        if self.tree is None:
            return

        def cls_ref(n):
            if isinstance(n, (ast.Name, ast.Attribute)):
                last = n.id if isinstance(n, ast.Name) else n.attr
                return ast.unparse(n) if last[:1].isupper() else None
            return None

        for fn in ast.walk(self.tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            pos = fn.args.posonlyargs + fn.args.args
            pairs = list(zip(pos[len(pos) - len(fn.args.defaults) :], fn.args.defaults))
            pairs += [(a, d) for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None]
            later = {}
            for st in fn.body:
                if (
                    isinstance(st, ast.If)
                    and not st.orelse
                    and len(st.body) == 1
                    and isinstance(st.test, ast.Compare)
                    and isinstance(st.test.left, ast.Name)
                    and len(st.test.ops) == 1
                    and isinstance(st.test.ops[0], ast.Is)
                    and isinstance(st.test.comparators[0], ast.Constant)
                    and st.test.comparators[0].value is None
                    and isinstance(st.body[0], ast.Assign)
                    and len(st.body[0].targets) == 1
                    and isinstance(st.body[0].targets[0], ast.Name)
                    and st.body[0].targets[0].id == st.test.left.id
                ):
                    later[st.test.left.id] = st.body[0].value
            for a, d in pairs:
                ann = ast.unparse(a.annotation) if a.annotation is not None else ""
                if ann and not re.search(r"\b(Type|type|Callable|LayerType|Module)\b", ann):
                    continue
                ref = cls_ref(d)
                optional = False
                if ref is None and isinstance(d, ast.Constant) and d.value is None and a.arg in later:
                    ref, optional = cls_ref(later[a.arg]), True
                if ref is None or not re.search(r"(layer|norm|act|block|fn|cls|module)", a.arg, re.I):
                    continue
                text = f"type[{ref}]" + (" | None" if optional else "")
                if a.annotation is not None:
                    if a.annotation.lineno != a.annotation.end_lineno or not self._claim(a.annotation):
                        continue
                    self._edit(*self._ast_span(a.annotation), text)
                else:
                    if not self._claim(a):
                        continue
                    _s, end = self._ast_span(a)
                    self._edit(end, end, f": {text}")

    def _inline_pair_calls(self):
        """`pair(image_size)` -> `(image_size, image_size)` in the shadow, so a size passed as a plain
        int keeps its name through the helper (Pyrefly can't narrow the helper's isinstance)."""
        if self.tree is None:
            return
        pairs = pair_helpers(self.tree)
        if not pairs:
            return
        for node in ast.walk(self.tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in pairs
                and len(node.args) == 1
                and not node.keywords
                and node.lineno == node.end_lineno
                and isinstance(node.args[0], (ast.Name, ast.Attribute))
            ):
                continue
            (l1, c1), (l2, c2) = self._ast_span(node.args[0])
            arg = self.src_lines[l1 - 1][c1:c2]
            if self._claim(node):
                self._edit(*self._ast_span(node), "(" + ", ".join([arg] * pairs[node.func.id]) + ")")

    def _inline_none_checks(self):
        """`def exists(val): return val is not None`, then `if exists(mask): ...`: Pyrefly doesn't narrow
        through the helper, so `mask` stays `X | None` inside the branch and a later use reads as a
        mismatch. In the shadow the call becomes `(mask is not None)`, which narrows."""
        if self.tree is None:
            return
        helpers: dict[str, bool] = {}  # name -> True for `is not None`, False for `is None`
        for fn in self.tree.body if isinstance(self.tree, ast.Module) else []:
            if not (
                isinstance(fn, ast.FunctionDef)
                and len(fn.args.args) == 1
                and not fn.decorator_list
                and len(fn.body) == 1
                and isinstance(fn.body[0], ast.Return)
            ):
                continue
            r = fn.body[0].value
            arg = fn.args.args[0].arg
            if (
                isinstance(r, ast.Compare)
                and len(r.ops) == 1
                and isinstance(r.left, ast.Name)
                and r.left.id == arg
                and isinstance(r.comparators[0], ast.Constant)
                and r.comparators[0].value is None
                and isinstance(r.ops[0], (ast.IsNot, ast.Is))
            ):
                helpers[fn.name] = isinstance(r.ops[0], ast.IsNot)
        if not helpers:
            return
        for node in ast.walk(self.tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in helpers
                and len(node.args) == 1
                and not node.keywords
                and node.lineno == node.end_lineno
            ):
                continue
            a = node.args[0]
            if not isinstance(a, (ast.Name, ast.Attribute)):
                continue
            (l1, c1), (l2, c2) = self._ast_span(a)
            argtext = self.src_lines[l1 - 1][c1:c2]
            op = "is not" if helpers[node.func.id] else "is"
            if self._claim(node):
                self._edit(*self._ast_span(node), f"({argtext} {op} None)")

    def _rewrite_list_indices(self):
        """`x[:, [-1], :]` keeps a size-1 dim, but Pyrefly types the list index as `int`.
        Rewrite single-int list indices to the equivalent slice (`-1:`, `i:i+1`): same shape."""
        if self.tree is None:
            return
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Subscript):
                continue
            elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for e in elts:
                if not (isinstance(e, ast.List) and len(e.elts) == 1 and e.lineno == e.end_lineno):
                    continue
                c = e.elts[0]
                val = None
                if isinstance(c, ast.Constant) and type(c.value) is int:
                    val = c.value
                elif (
                    isinstance(c, ast.UnaryOp)
                    and isinstance(c.op, ast.USub)
                    and isinstance(c.operand, ast.Constant)
                    and type(c.operand.value) is int
                ):
                    val = -c.operand.value
                if val is None:
                    continue
                text = "-1:" if val == -1 else (f"{val}:{val + 1}" if val >= 0 else f"{val}:{val + 1}")
                self._edit(*self._ast_span(e), text)

    # ---------- min/max passed as a size ----------
    def _widen_literal_choices(self):
        """`n = 1 if pool == 'cls' else 0` is `Literal[0, 1]`, and a tensor sized by it is a union
        that hovers as one arm (`[b, 0, d]`). `int(...)` around the choice makes it a plain `int`."""
        if self.tree is None:
            return

        def int_lit(e) -> bool:
            return isinstance(e, ast.Constant) and type(e.value) is int

        for node in ast.walk(self.tree):
            v = getattr(node, "value", None)
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(v, ast.IfExp):
                continue
            if int_lit(v.body) and int_lit(v.orelse) and v.body.value != v.orelse.value:
                (l1, c1), (l2, c2) = self._ast_span(v)
                self._edit((l1, c1), (l1, c1), "int(")
                self._edit((l2, c2), (l2, c2), ")")

    def _wrap_minmax_sizes(self):
        """Pyrefly 1.5.0 solves a size type-variable fed an inline `min(k, x.size(-1))` to one of its
        operands (`topk(x, min(k, V))` -> `[.., V]`, wrong). At size positions only ints are valid,
        so `int(min(...))` is behaviour-preserving and makes Pyrefly keep it as `int`."""
        if self.tree is None:
            return
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            rule = SIZE_CALLS.get(_call_name(node.func) or "")
            if rule is None:
                continue
            positions, keywords = rule
            cands = [a for i, a in enumerate(node.args) if positions == ALL or i in positions]
            cands += [k.value for k in node.keywords if k.arg and (k.arg in keywords or "*" in keywords)]
            for a in cands:
                if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in ("min", "max"):
                    (l1, c1), (l2, c2) = self._ast_span(a)
                    self._edit((l1, c1), (l1, c1), "int(")
                    self._edit((l2, c2), (l2, c2), ")")

    # ---------- return annotations that hide what the body computes ----------
    _ERASING = re.compile(
        r"(?:torch\.)?(?:[A-Z][a-z]*)?Tensor|(?:torch\.)?nn\.[A-Z]\w*|Module|[A-Z]\w*(?:Conv|Norm)\w*|"
        r"(?:typing\.)?(?:Optional|Union|Tuple|List|tuple|list)|None"
    )

    def _erases_shape(self, func) -> bool:
        """`-> Tensor`, `-> torch.Tensor`, `-> nn.Conv2d`, `-> Optional[Tensor]`, `-> tuple[Tensor, Tensor]`:
        Pyrefly trusts the annotation over the body, so every caller sees a shapeless tensor or
        module. In the shadow such an annotation is dropped and the return type is inferred."""
        code = func.annotation.get_code().strip()
        if not code or "[[" in code or "'" in code or '"' in code:
            return False
        names = re.findall(r"[A-Za-z_][\w.]*", code)
        if not names or not all(self._ERASING.fullmatch(n) for n in names):
            return False
        if not any("Tensor" in n or "nn." in n or n == "Module" or "Conv" in n or "Norm" in n for n in names):
            return False
        for dec in func.get_decorators() if hasattr(func, "get_decorators") else []:
            if re.search(r"overload|abstract", dec.get_code()):
                return False
        body = func.children[-1].get_code().strip()
        if re.fullmatch(
            r"(?:(?:\"\"\"[\s\S]*?\"\"\"|'\'\'[\s\S]*?'\'\')\s*)?(?:\.\.\.|pass|raise NotImplementedError\S*)", body
        ):
            return False  # a stub: inferring would make it return None
        return True

    def _drop_return_annotation(self, func):
        arrow = next((c for c in func.children if getattr(c, "value", None) == "->"), None)
        ann = func.annotation
        if arrow is None or arrow.start_pos[0] != ann.end_pos[0]:
            return
        self._edit(arrow.start_pos, ann.end_pos, "")

    # ---------- seeded tensor params ----------
    @staticmethod
    def _is_bare_tensor(annotation) -> bool:
        if annotation is None:
            return False
        code = annotation.get_code().strip()
        return code in ("Tensor", "torch.Tensor")

    @staticmethod
    def _literal_default_type(p) -> str | None:
        d = p.default
        if d is None:
            return None
        if d.type == "number":
            return "float" if any(c in d.value.lower() for c in ".e") and not d.value.lower().startswith("0x") else None
        if d.type == "keyword" and d.value in ("True", "False"):
            return "bool"
        if (
            d.type == "factor"
            and d.children[0].value == "-"
            and d.children[1].type == "number"
            and "." in d.children[1].value
        ):
            return "float"
        return None

    def _seed_text(self, tokens: list[str], scope: _Scope, new_params: list[str], fresh: dict) -> str:
        out = []
        for tok in tokens:
            if tok == "int" or _INT_TOKEN.match(tok):
                out.append(tok)
            elif re.fullmatch(r"~[A-Za-z_]\w*", tok):
                name = tok[1:]
                if name not in fresh:
                    tp = self._fresh(name, scope)
                    scope.dims.setdefault(name, tp)
                    fresh[name] = tp
                    new_params.append(tp)
                out.append(fresh[name])
            elif _NAME_TOKEN.match(tok):
                tp = scope.lookup(tok)
                if tp is None:
                    if tok not in fresh:
                        fresh[tok] = self._fresh(tok, scope)
                        scope.dims.setdefault(tok, fresh[tok])
                        new_params.append(fresh[tok])
                    tp = fresh[tok]
                out.append(tp)
            else:  # an expression over known dims, e.g. "3 * n_embd"
                ok = True

                def sub(m):
                    nonlocal ok
                    sym = m.group(0)
                    if sym.startswith("~"):  # the function's own dim, e.g. `~h * ~w`
                        name = sym[1:]
                        if name not in fresh:
                            fresh[name] = self._fresh(name, scope)
                            scope.dims.setdefault(name, fresh[name])
                            new_params.append(fresh[name])
                        return fresh[name]
                    tp = scope.lookup(sym)
                    if tp is None:
                        ok = False
                        return sym
                    return tp

                expr = re.sub(r"~?[A-Za-z_][A-Za-z0-9_]*", sub, tok)
                out.append(expr if ok else "int")
        return f"_ts_Tensor[[{', '.join(out)}]]"

    def _local_casts(self, func, owner, fscope: _Scope, new_params: list[str], fresh: dict):
        """Solver seeds keyed `@line:var`: before `b, n, _ = x.shape`, insert
        `x = _ts_cast(_ts_Tensor[[b, n, d]], x); ` so the unpacked names become dims. Only dims
        that were a bare `int` get the new names; the rest are what Pyrefly already had."""
        if self.tree is None or not self.seeds:
            return
        fname = func.name.value
        line = func.name.start_pos[0]
        from .seeds import int_key, local_key

        ints = {k[2] for k in self.seeds if k[0] == (owner or "") and k[1] == fname and k[2].startswith("#")}
        if ints:
            sites = [(fn, st, var, st, "after") for fn, _c, st, var in self._scan("int_locals")]
            sites += [
                (fn, st, var, anchor, "after" if anchor is st else "wrap" if anchor is st.value else "before")
                for fn, _c, st, var, anchor in self._scan("fallback_sizes")
            ]
            for fn, st, var, anchor, where in sites:
                if fn.name != fname or not (fn.lineno <= line <= fn.end_lineno) or int_key(st, var) not in ints:
                    continue
                tp = self.int_local_tps.get((owner or "", fname, int_key(st, var)))
                if tp is None:
                    if fname == "__init__" and owner:
                        continue  # the class could not take it
                    tp = self._fresh(var, fscope)
                    fscope.dims.setdefault(var, tp)
                    new_params.append(tp)
                if where == "after":
                    _, (l2, c2) = self._ast_span(anchor)
                    self._edit((l2, c2), (l2, c2), f"; {var} = _ts_cast(_ts_Int[{tp}], {var})")
                elif where == "wrap":
                    (l1, c1), (l2, c2) = self._ast_span(anchor)
                    self._edit((l1, c1), (l1, c1), f"_ts_cast(_ts_Int[{tp}], ")
                    self._edit((l2, c2), (l2, c2), ")")
                else:
                    (l1, c1), _ = self._ast_span(anchor)
                    self._edit((l1, c1), (l1, c1), f"{var} = _ts_cast(_ts_Int[{tp}], {var}); ")
                self.extra_imports["_ts_cast"] = "from typing import cast as _ts_cast"
        rets = {
            k[2]: v for k, v in self.seeds.items() if k[0] == (owner or "") and k[1] == fname and k[2].startswith("^")
        }
        if rets:
            from .seeds import ret_key

            for fn, _cls, st, _ks in self._scan("divisible_returns"):
                if fn.name != fname or not (fn.lineno <= line <= fn.end_lineno) or ret_key(st) not in rets:
                    continue
                if any(
                    f is not fn and f.lineno <= st.lineno <= f.end_lineno and f.lineno > fn.lineno
                    for f, _c in _ast_functions(fn)
                ):
                    continue  # belongs to a nested def
                text = self._seed_text(rets[ret_key(st)], fscope, new_params, fresh)
                (l1, c1), (l2, c2) = self._ast_span(st.value)
                self._edit((l1, c1), (l1, c1), f"_ts_cast({text}, ")
                self._edit((l2, c2), (l2, c2), ")")
                self.extra_imports["_ts_cast"] = "from typing import cast as _ts_cast"
        wanted = {
            k[2]: v for k, v in self.seeds.items() if k[0] == (owner or "") and k[1] == fname and k[2].startswith("@")
        }
        if not wanted:
            return
        for fn, _cls, st, var, _targets in self._scan("local_unpacks"):
            if fn.name != fname or not (fn.lineno <= line <= fn.end_lineno):
                continue
            toks = wanted.get(local_key(st, var))
            if not toks:
                continue
            text = self._seed_text(toks, fscope, new_params, fresh)
            (l1, c1), _ = self._ast_span(st)
            self._edit((l1, c1), (l1, c1), f"{var} = _ts_cast({text}, {var}); ")
            self.extra_imports["_ts_cast"] = "from typing import cast as _ts_cast"

    # ---------- imports ----------
    def _collect_jaxtyping_imports(self):
        for imp in self.module.iter_imports():
            try:
                if imp.type == "import_from":
                    if [n.value for n in imp.get_from_names()] == ["jaxtyping"]:
                        for name in imp.get_defined_names():
                            self.jaxtyping_names.add(name.value)
                else:
                    for path, defined in zip(imp.get_paths(), imp.get_defined_names()):
                        if [n.value for n in path] == ["jaxtyping"]:
                            self.jaxtyping_modules.add(defined.value)
            except Exception:
                continue

    def _insert_import(self):
        text = IMPORT_TEXT + "".join("; " + imp for imp in self.extra_imports.values())
        stmts = [c for c in self.module.children if c.type != "endmarker"]
        if not stmts:
            return
        imports = []
        for simple in stmts:
            if simple.type != "simple_stmt":
                continue
            for child in simple.children:
                if child.type in ("import_name", "import_from"):
                    imports.append(child)
        non_future = [
            i
            for i in imports
            if not (i.type == "import_from" and [n.value for n in i.get_from_names()] == ["__future__"])
        ]
        if non_future:
            anchor = non_future[0]
            self._edit(anchor.end_pos, anchor.end_pos, "; " + text)
            return
        if imports:
            anchor = imports[-1]
            self._edit(anchor.end_pos, anchor.end_pos, "; " + text)
            return
        first = stmts[0]
        # Docstring first: append after it on its closing line.
        if first.type == "simple_stmt" and first.children[0].type == "string":
            s = first.children[0]
            self._edit(s.end_pos, s.end_pos, "; " + text)
            return
        if first.type == "simple_stmt":
            self._edit(first.start_pos, first.start_pos, text + "; ")
        else:
            # compound statement first (def/class/if): no safe same-line spot before it,
            # so append to its header line after the colon is not valid either; put the
            # import as a prefix statement on line 1 only if line 1 is blank-free code.
            self.notes.append("no import anchor: file starts with a compound statement")

    def _absolutize_relative_imports(self):
        """`from .layers import X` -> `from pkg.layers import X`, so imports resolve through the
        shadow tree on Pyrefly's search path instead of next to the real file."""
        if not self.package:
            return
        parts = self.package.split(".")
        for imp in self.module.iter_imports():
            if imp.type != "import_from" or imp.level == 0:
                continue
            if imp.level > len(parts):
                continue
            base = parts[: len(parts) - (imp.level - 1)]
            names = [n.value for n in imp.get_from_names()]
            # the leaves between `from` and `import`: dots and the dotted name
            leaves = []
            for c in imp.children[1:]:
                if c.type == "keyword" and c.value == "import":
                    break
                leaves.append(c)
            if not leaves or leaves[0].start_pos[0] != leaves[-1].end_pos[0]:
                continue
            target = ".".join(base + names)
            self._edit(leaves[0].start_pos, leaves[-1].end_pos, target)

    # ---------- annotations ----------
    @staticmethod
    def _is_plain_int(annotation) -> bool:
        return annotation is not None and annotation.type == "name" and annotation.value == "int"

    def _jaxtyping_parts(self, annotation):
        """If `annotation` is `Dtype[TensorExpr, "dims"]` from jaxtyping, return (tensor_code, dims_str)."""
        if annotation is None or annotation.type != "atom_expr":
            return None
        children = annotation.children
        trailer = children[-1]
        if trailer.type != "trailer" or trailer.children[0].value != "[":
            return None
        head = children[:-1]
        if len(head) == 1 and head[0].type == "name":
            if head[0].value not in self.jaxtyping_names:
                return None
        elif (
            len(head) == 2
            and head[0].type == "name"
            and head[0].value in self.jaxtyping_modules
            and head[1].type == "trailer"
            and head[1].children[0].value == "."
        ):
            pass
        else:
            return None
        sub = trailer.children[1]
        if sub.type != "subscriptlist":
            return None
        parts = [c for c in sub.children if not (c.type == "operator" and c.value == ",")]
        if len(parts) != 2 or parts[1].type not in ("string", "strings"):
            return None
        try:
            dims = ast.literal_eval(parts[1].get_code().strip())
        except Exception:
            return None
        if not isinstance(dims, str):
            return None
        return parts[0].get_code().strip(), dims

    def _rewrite_annotation(self, annotation, scope: _Scope, new_params: list[str]) -> bool:
        parts = self._jaxtyping_parts(annotation)
        if parts is None:
            return False
        if "\n" in annotation.get_code().strip():
            self.notes.append(f"line {annotation.start_pos[0]}: multi-line jaxtyping annotation left as-is")
            return False
        tensor_code, dims = parts
        tokens = dims.split()
        out: list[str] = []
        ok = True
        for tok in tokens:
            if _INT_TOKEN.match(tok):
                out.append(tok)
            elif _NAME_TOKEN.match(tok) and tok != "_":
                tp = scope.lookup(tok)
                if tp is None:
                    tp = self._fresh(tok, scope)
                    scope.dims[tok] = tp
                    new_params.append(tp)
                out.append(tp)
            else:
                ok = False
                break
        if ok and tokens:
            text = f"{tensor_code}[[{', '.join(out)}]]"
        else:
            text = tensor_code
            self.notes.append(
                f"line {annotation.start_pos[0]}: jaxtyping dims {dims!r} not supported in v1; left unshaped"
            )
        self._edit(annotation.start_pos, annotation.end_pos, text)
        return bool(ok and tokens)

    # ---------- reassigned shaped parameters ----------
    _COMPOUND = {
        "if_stmt",
        "for_stmt",
        "while_stmt",
        "try_stmt",
        "with_stmt",
        "funcdef",
        "classdef",
        "decorated",
        "async_stmt",
        "async_funcdef",
        "match_stmt",
    }

    def _rebind_site(self, func):
        """Where a reassigned shaped param can be rebound as a plain local, on an existing line:
        ("simple", stmt) -> `x = _ts_x__p; <stmt>`; ("header", expr) -> the always-evaluated
        header of a first `if`/`for` becomes `(x := _ts_x__p, <expr>)[1]`; None -> impossible."""
        suite = func.children[-1]
        first = next((c for c in suite.children if c.type != "newline"), None) if suite.type == "suite" else suite
        if first is None:
            return None
        if first.type not in self._COMPOUND:
            return ("simple", first)
        if first.type == "if_stmt":
            return ("header", first.children[1])
        if first.type == "for_stmt":
            return ("header", first.children[3])
        return None

    def _rebind_params(self, func, names: list[str], site):
        params = {p.name.value: p for p in func.get_params()}
        for n in names:
            self._edit(params[n].name.start_pos, params[n].name.end_pos, f"_ts_{n}__p")
        kind, node = site
        if kind == "simple":
            self._edit(node.start_pos, node.start_pos, "".join(f"{n} = _ts_{n}__p; " for n in names))
        else:
            walrus = ", ".join(f"{n} := _ts_{n}__p" for n in names)
            self._edit(node.start_pos, node.start_pos, f"({walrus}, ")
            self._edit(node.end_pos, node.end_pos, f")[{len(names)}]")

    # ---------- type-param lists ----------
    def _add_type_params(self, node, name_leaf, tps: list[str], defaults: dict[str, str] | None = None):
        """Add `tp: _ts_IntVar` entries to a def/class, creating or merging `[...]`."""
        if not tps:
            return
        defaults = defaults or {}
        entries = ", ".join(f"{tp}: _ts_IntVar" + (f" = {defaults[tp]}" if tp in defaults else "") for tp in tps)
        existing = next((c for c in node.children if c.type == "type_params"), None)
        if existing is None:
            self._edit(name_leaf.end_pos, name_leaf.end_pos, f"[{entries}]")
        else:
            close = existing.children[-1]
            inner = existing.children[1:-1]
            has_items = any(c.type != "operator" or c.value != "," for c in inner)
            trailing_comma = bool(inner) and inner[-1].type == "operator" and inner[-1].value == ","
            sep = "" if (not has_items or trailing_comma) else ", "
            self._edit(close.start_pos, close.start_pos, sep + entries)

    def _int_dim_params(self, func):
        """Params of `func` to treat as int dims: `p: int`, or unannotated ones inferred as sizes."""
        inferred = self.inferred.get((func.name.start_pos[0], func.name.value), set())
        # parso's def line may differ from ast's when decorators exist; match on the name's line
        if not inferred:
            inferred = self.inferred.get((func.start_pos[0], func.name.value), set())
        out = []
        for p in func.get_params():
            if p.star_count:
                continue
            if self._is_plain_int(p.annotation):
                out.append((p, "annot"))
            elif p.annotation is None and p.name.value in inferred:
                out.append((p, "infer"))
        return out

    def _edit_int_param(self, p, how: str, tp: str):
        if how == "annot":
            self._edit(p.annotation.start_pos, p.annotation.end_pos, f"_ts_Int[{tp}]")
        else:
            self._edit(p.name.end_pos, p.name.end_pos, f": _ts_Int[{tp}]")

    # ---------- walking ----------
    def _process_function(
        self, func: pt.Function, scope: _Scope, *, class_scope: _Scope | None = None, owner: str | None = None
    ):
        """Rewrite one def. If `class_scope` is given (an __init__), its int params become class dims."""
        if self._has_errors(func.children[2]) or (func.annotation is not None and self._has_errors(func.annotation)):
            self.notes.append(f"line {func.start_pos[0]}: signature has syntax errors; skipped")
            return
        fscope = _Scope(scope)
        new_params: list[str] = []
        params = [p for p in func.get_params()]
        key = (func.name.start_pos[0], func.name.value)
        # 1) int params first so jaxtyping names can unify with them
        int_params = self._int_dim_params(func)
        handled = {id(p) for p, _ in int_params}
        for p, how in int_params:
            name = p.name.value
            if class_scope is not None:
                tp = class_scope.dims.get(name)
            else:
                tp = self._fresh(name, fscope)
                fscope.dims[name] = tp
                new_params.append(tp)
            self._edit_int_param(p, how, tp)
        # 2) config-object params: `config.n_embd` becomes the named dim `n_embd`
        cfg_params = self.config_params.get(key, {})
        for p in params:
            cfg = cfg_params.get(p.name.value)
            if cfg is None or p.star_count:
                continue
            if class_scope is not None and p.name.value in class_scope.config_tps:
                ref, tps = class_scope.config_tps[p.name.value]
            else:
                ref = self._config_ref(cfg)
                tps = []
                for f in cfg.int_fields:
                    tp = self._fresh(f, fscope)
                    fscope.dims.setdefault(f, tp)
                    new_params.append(tp)
                    tps.append(tp)
            text = f"{ref}[{', '.join(tps)}]"
            if p.annotation is None:
                self._edit(p.name.end_pos, p.name.end_pos, f": {text}")
            else:
                self._edit(p.annotation.start_pos, p.annotation.end_pos, text)
            handled.add(id(p))
        # 3) jaxtyping annotations and seeded (unannotated / bare Tensor) params
        rebound = self.reassigned.get(key, set())
        site = self._rebind_site(func) if rebound else None
        shaped: list[str] = []
        seedable: list[str] = []
        fresh: dict[str, str] = {}
        for i, p in enumerate(params):
            if id(p) in handled or p.star_count:
                continue
            pname = p.name.value
            if i == 0 and pname in ("self", "cls") and owner is not None:
                continue
            ftp = self.fallback_params.get((owner, pname)) if func.name.value == "__init__" else None
            if ftp is not None:
                if p.annotation is None:
                    self._edit(p.name.end_pos, p.name.end_pos, f": _ts_Int[{ftp}] | None")
                else:
                    self._edit(p.annotation.start_pos, p.annotation.end_pos, f"_ts_Int[{ftp}] | None")
                continue
            if p.annotation is None or self._is_bare_tensor(p.annotation):
                seedable.append(pname)
                tokens = self.seeds.get((owner or "", func.name.value, pname))
                if not tokens:
                    # `temperature=1.0` / `flag=False` / `top_k=None` used as a size: Pyrefly leaves
                    # unannotated params Unknown, which poisons (or, through min/max, silently
                    # distorts) every tensor expression they touch.
                    lit = self._literal_default_type(p)
                    if p.name.value in self.optional_ints.get(key, set()):
                        lit = "int | None"
                    if lit and p.annotation is None:
                        self._edit(p.name.end_pos, p.name.end_pos, f": {lit}")
                    continue
                if pname in rebound and site is None:
                    self.notes.append(
                        f"line {p.start_pos[0]}: `{pname}` is reassigned and the body starts "
                        f"with a statement that cannot carry a rebinding; left unshaped"
                    )
                    continue
                text = self._seed_text(tokens, fscope, new_params, fresh)
                if p.default is not None and p.default.type == "keyword" and p.default.value == "None":
                    text += " | None"  # `mask = None`: callers may still pass None
                if p.annotation is None:
                    self._edit(p.name.end_pos, p.name.end_pos, f": {text}")
                else:
                    self._edit(p.annotation.start_pos, p.annotation.end_pos, text)
                shaped.append(pname)
                continue
            if pname in rebound and site is None:
                # Can't rebind: a pinned declared shape would show stale after reassignment.
                if self._jaxtyping_parts(p.annotation) is not None:
                    self._edit(p.annotation.start_pos, p.annotation.end_pos, self._jaxtyping_parts(p.annotation)[0])
                    self.notes.append(
                        f"line {p.start_pos[0]}: `{pname}` is reassigned and the body "
                        f"starts with a statement that cannot carry a rebinding; left unshaped"
                    )
                continue
            if self._rewrite_annotation(p.annotation, fscope, new_params):
                shaped.append(pname)
        if owner is not None:
            self.info["methods"].setdefault(owner, {})[func.name.value] = seedable
        if func.annotation is not None:
            if self._erases_shape(func):
                self._drop_return_annotation(func)
            else:
                self._rewrite_annotation(func.annotation, fscope, new_params)
        self._local_casts(func, owner, fscope, new_params, fresh)
        self._add_type_params(func, func.name, new_params)
        to_rebind = [n for n in shaped if n in rebound]
        if to_rebind and site is not None:
            self._rebind_params(func, to_rebind, site)
        self._walk_suite(func.children[-1], fscope)

    def _process_class(self, cls: pt.Class, scope: _Scope):
        cscope = _Scope(scope)
        name = cls.name.value
        dims: list[tuple[str, str]] = []  # (user dim name, type param), in declaration order
        # a config class of this file: generic over its int fields
        cfg = self.local_configs.get(name)
        if cfg is not None:
            for st in self._suite_items(cls.children[-1]):
                if st.type != "simple_stmt":
                    continue
                expr = st.children[0]
                if expr.type != "expr_stmt" or len(expr.children) < 2 or expr.children[1].type != "annassign":
                    continue
                fname, ann = expr.children[0], expr.children[1].children[1]
                if fname.type == "name" and fname.value in cfg.int_fields and self._is_plain_int(ann):
                    tp = self._fresh(fname.value, cscope)
                    cscope.dims[fname.value] = tp
                    dims.append((fname.value, tp))
                    self._edit(ann.start_pos, ann.end_pos, f"_ts_Int[{tp}]")
            self.info["configs"][name] = [d for d, _ in dims]
        init = None
        for f in cls.iter_funcdefs():
            if f.name.value == "__init__":
                init = f
                break
        if init is not None and not self._has_errors(init.children[2]):
            for p, _how in self._int_dim_params(init):
                tp = self._fresh(p.name.value, cscope)
                cscope.dims[p.name.value] = tp
                dims.append((p.name.value, tp))
            for pname, pcfg in self.config_params.get((init.name.start_pos[0], "__init__"), {}).items():
                ref = self._config_ref(pcfg)
                tps = []
                for f in pcfg.int_fields:
                    tp = self._fresh(f, cscope)
                    cscope.dims.setdefault(f, tp)
                    dims.append((f, tp))
                    tps.append(tp)
                cscope.config_tps[pname] = (ref, tps)
        tp_defaults: dict[str, str] = {}
        if init is not None:
            from .seeds import int_key

            # `out_features = out_features or in_features` with `out_features=None`: the class dim
            # defaults to in_features's (PEP 696), so `Mlp(dim)` is `Mlp[dim, ..., dim]`
            falls = {}
            if self.tree is not None:
                for fn, c, st, var, _a in self._scan("fallback_sizes"):
                    if c == name and fn.name == "__init__":
                        v = st.value
                        other = (
                            v.values[1]
                            if isinstance(v, ast.BoolOp)
                            else (v.orelse if isinstance(v.test.ops[0], ast.IsNot) else v.body)
                            if isinstance(v, ast.IfExp)
                            else v
                        )
                        if isinstance(other, ast.Name):
                            falls[int_key(st, var)] = other.id
            init_params = {p.name.value: p for p in init.get_params()}
            for key in sorted(k for k in self.seeds if k[0] == name and k[1] == "__init__" and k[2].startswith("#")):
                var = key[2].split(":", 1)[1]
                if var in cscope.dims:
                    continue
                tp = self._fresh(var, cscope)
                cscope.dims[var] = tp
                dims.append((var, tp))
                self.int_local_tps[(name, "__init__", key[2])] = tp
                other = falls.get(key[2])
                p = init_params.get(var)
                if (
                    other in cscope.dims
                    and p is not None
                    and p.default is not None
                    and p.default.type == "keyword"
                    and p.default.value == "None"
                    and (
                        p.annotation is None
                        or p.annotation.get_code().strip() in ("Optional[int]", "int | None", "typing.Optional[int]")
                    )
                ):
                    tp_defaults[tp] = cscope.dims[other]
                    self.fallback_params[(name, var)] = tp
        if tp_defaults:  # PEP 696: no type param without a default after one with a default
            dims = [d for d in dims if d[1] not in tp_defaults] + [d for d in dims if d[1] in tp_defaults]
        if dims:
            self._add_type_params(cls, cls.name, [tp for _, tp in dims], tp_defaults)
        self.info["classes"][name] = dims
        suite = cls.children[-1]
        for child in self._suite_items(suite):
            is_init = init is not None and (child is init or (child.type == "decorated" and child.children[-1] is init))
            self._dispatch(child, cscope, init_owner=cscope if is_init else None, owner=name)

    @staticmethod
    def _suite_items(suite):
        if suite.type == "suite":
            return suite.children
        return [suite]

    def _dispatch(self, node, scope: _Scope, init_owner: _Scope | None = None, owner: str | None = None):
        if node.type == "funcdef":
            self._process_function(node, scope, class_scope=init_owner, owner=owner)
        elif node.type == "async_stmt":
            for c in node.children:
                self._dispatch(c, scope, owner=owner)
        elif node.type == "decorated":
            target = node.children[-1]
            self._dispatch(target, scope, init_owner=init_owner if target.type == "funcdef" else None, owner=owner)
        elif node.type == "classdef":
            self._process_class(node, scope)
        elif node.type in (
            "if_stmt",
            "for_stmt",
            "while_stmt",
            "try_stmt",
            "with_stmt",
            "suite",
            "async_funcdef",
            "match_stmt",
            "case_block",
        ):
            for c in node.children:
                self._dispatch(c, scope)

    def _walk_suite(self, suite, scope: _Scope):
        for c in self._suite_items(suite):
            self._dispatch(c, scope)

    # ---------- entry ----------
    def run(self) -> RewriteResult:
        self._collect_jaxtyping_imports()
        for child in self.module.children:
            self._dispatch(child, _Scope())
        shape_edits = bool(self.edits)
        self._rewrite_module_dicts()
        self._rewrite_module_lists()
        self._rewrite_identity_fallbacks()
        self._default_flag_choices()
        self._unroll_map_lambdas()
        self._inline_none_checks()
        self._inline_pair_calls()
        self._split_star_shapes()
        self._drop_local_tensor_annotations()
        self._type_layer_factories()
        self._rewrite_list_indices()
        self._wrap_minmax_sizes()
        self._widen_literal_choices()
        self._absolutize_relative_imports()
        if shape_edits:
            self._insert_import()
        return RewriteResult(
            apply_edits(self.source, self.edits),
            sorted(dict.fromkeys(self.edits), key=_edit_key),
            self.notes,
            self.info,
        )


def _edit_key(e: Edit):
    return (e.line, e.start, e.end)


def apply_edits(source: str, edits: list[Edit]) -> str:
    lines = source.split("\n")
    by_line: dict[int, list[Edit]] = {}
    # stable: edits at the same position apply in the order they were made, whatever the hash seed
    for e in sorted(dict.fromkeys(edits), key=_edit_key):
        by_line.setdefault(e.line, []).append(e)
    for ln, es in by_line.items():
        s = lines[ln]
        out, pos = [], 0
        for e in es:
            if e.start < pos:
                raise ValueError(f"overlapping edits on line {ln}")
            out.append(s[pos : e.start])
            out.append(e.text)
            pos = e.end
        out.append(s[pos:])
        lines[ln] = "".join(out)
    return "\n".join(lines)


def rewrite(
    source: str,
    package: str | None = None,
    infer_ints: bool = True,
    cache_key: str | None = None,
    config_index: dict | None = None,
    seeds: dict | None = None,
) -> RewriteResult:
    """Rewrite `source` into its shadow. Never raises on bad input: falls back to no edits.

    `package` is the dotted package the file belongs to (e.g. "pkg.sub" for pkg/sub/mod.py);
    when given, relative imports are made absolute.
    """
    try:
        return _Rewriter(source, package, infer_ints, cache_key, config_index, seeds).run()
    except Exception as exc:  # defensive: the proxy must never die on user text
        return RewriteResult(source, [], [f"rewrite failed: {exc!r}"])
