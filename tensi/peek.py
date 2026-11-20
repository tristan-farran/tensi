"""`tensi peek FILE`: the shape of every tensor assignment in a file, line by line."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import pyrefly, stubs
from .mapping import clean_display
from .rewrite import rewrite
from .scratch import scratch_copy
from .seeds import solve, unshaped_inputs


@dataclass
class Row:
    line: int
    names: list[str]
    shape: str


_TENSOR_PART = re.compile(r"(?:[\w.]*\.)?(?:Tensor|Parameter)(?:\[\[.*\]\])?|None")


def short_shape(t: str) -> str | None:
    """`torch.Tensor[[b, t, n_embd]]` -> `[b, t, n_embd]`; non-tensors -> None.
    Only types made entirely of tensors (optionally `| None`) count."""
    t = clean_display(t)
    parts = [x.strip() for x in t.split(" | ")]
    if not all(_TENSOR_PART.fullmatch(x) for x in parts) or all(x == "None" for x in parts):
        return None
    t = re.sub(r"(?:[\w.]*\.)?(?:Tensor|Parameter)\[\[(.*?)\]\]", r"[\1]", t)
    t = re.sub(r"(?:[\w.]*\.)?(?:Tensor|Parameter)\b", "[?]", t)
    return t


def _comment_columns(text: str) -> dict[int, int]:
    """0-based line -> column where a trailing `#` comment starts."""
    import io
    import tokenize

    out: dict[int, int] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0] - 1] = tok.start[1]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def peek(path: str | Path) -> tuple[list[Row], list[str]]:
    p = Path(path).resolve()
    src = p.read_text(encoding="utf-8")
    seeds = solve(p, src)
    shadow = rewrite(src, seeds=seeds).shadow
    lines = shadow.split("\n")
    targets: dict[int, list[str]] = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and node.lineno == node.end_lineno:
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in tg for n in ast.walk(t) if isinstance(n, ast.Name)]
            if names:
                targets.setdefault(node.lineno - 1, []).extend(dict.fromkeys(names))
    comments = _comment_columns(shadow)
    for ln, names in targets.items():
        code = lines[ln]
        cut = comments.get(ln, len(code))
        head, tail = code[:cut].rstrip(), code[cut:]
        if head.endswith((":", "\\", ",", "(", "[", "{")):
            continue
        lines[ln] = head + "".join(f"; reveal_type({n})" for n in names) + ("  " + tail if tail else "")
    text = "from typing import reveal_type\n" + "\n".join(lines)
    with scratch_copy(p.parent, text, "_tensi_peek_") as (tmp, extra, cwd):
        out = subprocess.run(
            [pyrefly(), "check", tmp, "--preset", "default", "--output-format", "json", *extra, *stubs.search_args()],
            capture_output=True,
            text=True,
            cwd=cwd,
        ).stdout
    reveals: dict[int, list[str]] = {}
    for e in json.loads(out or "{}").get("errors", []):
        if e["name"] == "reveal-type":
            m = re.search(r"revealed type: (.*)", e["description"], re.S)
            if m:
                reveals.setdefault(e["line"] - 1, []).append(m.group(1).strip())
    rows: list[Row] = []
    for ln, types in sorted(reveals.items()):
        names = targets.get(ln - 1, [])
        groups: dict[str, list[str]] = {}
        for name, t in zip(names, types):
            sh = short_shape(t)
            if sh is not None:
                groups.setdefault(sh, []).append(name)
        for sh, ns in groups.items():
            rows.append(Row(ln, ns, sh))
    rows = _Rows(rows)
    rows.hints = unshaped_inputs(src, seeds)
    return rows, src.split("\n")


class _Rows(list):
    """peek's rows, plus `hints`: module inputs that one annotation would shape."""

    hints: list = []


# ---------------------------------------------------------------- pretty printing


class Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _c(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.on else s

    def dim(self, s):
        return self._c("2", s)

    def bold(self, s):
        return self._c("1", s)

    def violet(self, s):
        return self._c("38;5;141", s)

    def green(self, s):
        return self._c("32", s)

    def red(self, s):
        return self._c("31", s)

    def plum(self, s):
        return self._c("1;38;5;97", s)


def style_for(stream) -> Style:
    return Style(hasattr(stream, "isatty") and stream.isatty() and not os.environ.get("NO_COLOR"))


def color_shape(sh: str, st: Style) -> str:
    """Names in violet, literals and `?` dimmed: the named dims are what tensi adds."""

    def tok(m):
        w = m.group(0)
        if w in ("int", "?") or w.isdigit():
            return st.dim(w)
        return st.violet(w)

    return re.sub(r"[A-Za-z_]\w*|\?|\d+", tok, sh)
