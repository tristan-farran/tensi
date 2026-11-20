"""Command line: `tensi peek`, `tensi check`, `tensi shadow`, `tensi stubs`, `tensi doctor`, `tensi lsp`."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from . import __version__, pyrefly, stubs
from .mapping import PositionMap, clean_display
from .rewrite import rewrite
from .scratch import scratch_copy


def is_shape_diagnostic(message: str, name: str | None = None) -> bool:
    """`shapes` diagnostics mode keeps only errors that are about tensor shapes.

    A type mismatch counts only when the type it wanted is shaped too: a shaped tensor where code
    annotates `LongTensor`, `int | ndarray` or `tuple[int]` is a typing complaint, not a shape error.
    Nullability errors that merely mention a shaped type (`Tensor[[n]] | None` passed where
    `Tensor[[n]]` is expected) and override checks are not about shapes either.
    """
    if name == "bad-override" or re.search(r"\| None` is not assignable", message):
        return False
    if name == "bad-unpacking" or message.startswith("Cannot unpack"):
        return "IntTuple[" in message  # `b, t, c = x.shape` on a tensor of another rank
    first = message.split("\n", 1)[0]
    m = re.search(r"is not assignable to (?:variable|parameter|declared return type|attribute|type)?(.*)", first)
    if m:
        return "[[" in first[: m.start()] and "[[" in m.group(1)
    m = re.search(r"No matching overload found for function .*? called with arguments: (.*)", first)
    if m:
        return "[[" in m.group(1)
    return (
        "Tensor[[" in message
        or "shape DSL" in message
        or ("shape" in message.lower() and "Tensor" in message)
        or bool(re.search(r"\b(rank|dimensions?|broadcast)\b", first))
    )


def is_rewrite_artifact(message: str) -> bool:
    """Errors caused by the shadow itself, e.g. a config field `n_embd: Int[_ts_n_embd] = 768`."""
    return bool(
        re.search(
            r"is not assignable to `(?:_ts_)?Int\[_ts_|Unnecessary `int\(\)` call|Cannot use type parameter (?:lists|defaults)",
            message,
        )
    )


def is_imprecise(message: str) -> bool:
    """Errors that only exist because Pyrefly is unsure: a union of candidate types, Unknown/Any,
    or unsolved type variables (`@_`), e.g. attribute access through an nn.ModuleDict.
    Not confident enough to underline."""
    return bool(re.search(r"\]\] \| |\| (Any|Unknown)\b|\bUnknown\b|@_", message)) or _divisible_only(message)


_ROUNDED = re.compile(r"\b(\d+) \* \((\w+) // \1\)|\b(\w+) // (\d+) \* \4\b")


def _divisible_only(message: str) -> bool:
    """`dim` against `4 * (dim // 4)`: equal whenever the code divides evenly (it usually asserts
    `dim % 4 == 0`), which Pyrefly cannot know. Not an error worth underlining."""
    for m in _ROUNDED.finditer(message):
        name = m.group(2) or m.group(3)
        if re.search(rf"(?<![\w/*] )(?<![\w(])\b{name}\b(?! //)", _ROUNDED.sub("", message)):
            return True
    return False


def drop_imprecise(diags: list[dict], msg_key: str, range_key) -> list[dict]:
    """Drop imprecise diagnostics and everything else reported at the same range."""
    bad_ranges = {range_key(d) for d in diags if is_imprecise(d[msg_key])}
    return [d for d in diags if range_key(d) not in bad_ranges]


def shadow_for(path: str, solve_seeds: bool = True):
    """(source, rewrite result) for `path`, with forward-input seeds solved unless disabled."""
    from .seeds import solve

    with open(path, encoding="utf-8") as f:
        src = f.read()
    seeds = solve(path, src) if solve_seeds else None
    return src, rewrite(src, seeds=seeds)


def check_file(path: str, shapes_only: bool = True, preset: str = "default", solve_seeds: bool = True) -> list[dict]:
    """Run Pyrefly on the shadow of `path`; return diagnostics in user coordinates and names."""
    src, res = shadow_for(path, solve_seeds)
    pm = PositionMap(src, res.shadow, res.edits)
    folder = os.path.dirname(os.path.abspath(path))
    with scratch_copy(folder, res.shadow, "_tensi_tmp_") as (tmp, extra, cwd):
        proc = subprocess.run(
            [pyrefly(), "check", tmp, "--preset", preset, "--output-format", "json", *extra, *stubs.search_args()],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    # Fail closed: a Pyrefly crash must not read as "no shape errors" in CI.
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        raise SystemExit(f"tensi: pyrefly failed (exit {proc.returncode}):\n{proc.stdout}{proc.stderr}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"tensi: pyrefly failed:\n{proc.stdout}\n{proc.stderr}") from None
    out = []
    errors = data.get("errors", [])
    errors = [e for e in errors if not is_rewrite_artifact(e.get("description", ""))]
    if shapes_only:
        errors = drop_imprecise(
            errors, "description", lambda e: (e["line"], e["column"], e["stop_line"], e["stop_column"])
        )
    for e in errors:
        msg = clean_display(e.get("concise_description") or e.get("description", ""))
        if shapes_only and not is_shape_diagnostic(e.get("description", ""), e.get("name")):
            continue
        l0, l1 = e["line"] - 1, e["stop_line"] - 1
        c0 = pm.shadow_to_user_cp(l0, e["column"] - 1)
        c1 = pm.shadow_to_user_cp(l1, e["stop_column"] - 1)
        out.append(
            {
                "path": path,
                "line": l0 + 1,
                "column": c0 + 1,
                "stop_line": l1 + 1,
                "stop_column": max(c1, c0) + 1,
                "name": e.get("name"),
                "message": msg,
            }
        )
    return out


def _elapsed(t0: float) -> str:
    import time

    dt = time.perf_counter() - t0
    return f"{dt * 1000:.0f}ms" if dt < 1 else f"{dt:.1f}s"


def cmd_peek(path: str) -> int:
    import time

    from .peek import color_shape, peek, style_for

    st = style_for(sys.stdout)
    t0 = time.perf_counter()
    rows, src = peek(path)
    print(f"{st.plum('tensi')} {st.dim('peek v' + __version__)}  {path}  {st.dim(_elapsed(t0))}\n")
    width = max((len(", ".join(r.names)) for r in rows), default=0)
    width = min(width, 18)
    last_fn = None
    for r in rows:
        fn = _enclosing_def(src, r.line)
        if fn != last_fn:
            print(f"  {st.dim(fn or '<module>')}")
            last_fn = fn
        names = ", ".join(r.names)
        print(f"  {st.dim(f'{r.line:>4}')}  {names:<{width}}  {color_shape(r.shape, st)}")
    named = sum(1 for r in rows if is_fully_named(r.shape))
    print(f"\n  {len(rows)} tensors, {st.violet(str(named))} fully named")
    hints = getattr(rows, "hints", [])
    if hints:
        print(f"\n  {st.dim('to name more, annotate the input of:')}")
        for h in hints:
            example = f'e.g. {h.param}: Shaped[Tensor, "{h.axes}"]'
            known = f"  (now {'[' + ', '.join(t.lstrip('~') for t in h.seed) + ']'})" if h.seed else ""
            print(f"  {st.dim(f'{h.line:>4}')}  {h.cls}.forward({h.param})  {st.dim(example + known)}")
    return 0


def is_fully_named(shape: str) -> bool:
    """`[b, n_points]` yes; `[b, int]`, `[?]`, `[2, 3]` no. (A dim name may contain "int".)"""
    return not re.search(r"\bint\b|\?", shape) and bool(re.search(r"[A-Za-z_]", shape))


def _enclosing_def(src: list[str], line: int) -> str | None:
    """`Class.method` / `func` of the innermost def above `line` (by indentation)."""
    cur_indent = None
    names = []
    for i in range(line - 1, -1, -1):
        text = src[i]
        if not text.strip():
            continue
        indent = len(text) - len(text.lstrip())
        m = re.match(r"\s*(?:async\s+)?(def|class)\s+(\w+)", text)
        if m and (cur_indent is None or indent < cur_indent):
            names.append(m.group(2))
            cur_indent = indent
            if indent == 0:
                break
    return ".".join(reversed(names)) if names else None


def python_files(paths: list[str]) -> list[str]:
    """Files named on the command line, with folders expanded to the `.py` files under them
    (skipping hidden folders, virtualenvs and caches)."""
    skip = {"__pycache__", "node_modules", "site-packages", "build", "dist"}
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d not in skip
                    and not os.path.exists(os.path.join(root, d, "pyvenv.cfg"))
                )
                out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".py")]
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit(f"tensi: no such file or folder: {p}")
    return out


def cmd_check(paths: list[str], show_all: bool, solve_seeds: bool) -> int:
    import time

    from .peek import style_for

    st = style_for(sys.stdout)
    files = python_files(paths)
    total = 0
    for path in files:
        t0 = time.perf_counter()
        diags = check_file(path, shapes_only=not show_all, solve_seeds=solve_seeds)
        print(f"{st.plum('tensi')} {st.dim('check v' + __version__)}  {path}  {st.dim(_elapsed(t0))}")
        for d in diags:
            loc = f"{d['path']}:{d['line']}:{d['column']}"
            print(f"  {st.red('x')} {st.bold(loc)}  {d['message']}  {st.dim(d['name'] or '')}")
        total += len(diags)
    kind = "" if show_all else "shape "
    where = f" in {len(files)} files" if len(files) > 1 else ""
    if total:
        print(f"\n  {st.red(str(total))} {kind}error{'s' if total != 1 else ''}{where}")
        return 1
    print(f"  {st.green('ok')} no {kind}errors{where}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tensi", description="tensi: see your tensors. Named shapes for PyTorch, on Pyrefly."
    )
    ap.add_argument("--version", action="version", version=f"tensi {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("peek", help="print the shape of every tensor in a file")
    pk.add_argument("file")
    ck = sub.add_parser("check", help="report tensor-shape errors in files or folders (exit 1 if any)")
    ck.add_argument("paths", nargs="+", metavar="path")
    ck.add_argument("--all", action="store_true", help="all type errors, not only shape errors")
    ck.add_argument("--no-solve", action="store_true", help="skip the forward-input seed solver")
    sh = sub.add_parser("shadow", help="print the rewritten text Pyrefly sees (debugging)")
    sh.add_argument("file")
    sh.add_argument("--notes", action="store_true", help="print rewriter notes to stderr")
    sh.add_argument("--no-solve", action="store_true", help="skip the forward-input seed solver")
    stb = sub.add_parser("stubs", help="show tensi's torch stub patches (--diff: as a patch for upstream)")
    stb.add_argument("--diff", action="store_true")
    dr = sub.add_parser("doctor", help="check the setup (Pyrefly, stubs, torch) from a folder")
    dr.add_argument("folder", nargs="?", default=".")
    lsp = sub.add_parser("lsp", help="run the language server on stdio")
    lsp.add_argument("--pyrefly", default=pyrefly())
    args = ap.parse_args(argv)

    if args.cmd == "peek":
        python_files([args.file])  # a clean message for a missing file
        return cmd_peek(args.file)
    if args.cmd == "check":
        return cmd_check(args.paths, args.all, not args.no_solve)
    if args.cmd == "shadow":
        _src, res = shadow_for(args.file, not args.no_solve)
        sys.stdout.write(res.shadow)
        if args.notes:
            for n in res.notes:
                print(f"note: {n}", file=sys.stderr)
        return 0
    if args.cmd == "stubs":
        if args.diff:
            sys.stdout.write(stubs.diff())
            return 0
        print(stubs.status(), end="")
        for p in stubs.PATCHES:
            print(f"  {p.name:<20} {p.why}")
        return 0
    if args.cmd == "doctor":
        from .doctor import run

        return run(args.folder)
    if args.cmd == "lsp":
        from .server import serve

        return serve(args.pyrefly)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
