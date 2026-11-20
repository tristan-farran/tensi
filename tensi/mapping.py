"""Position mapping between user text and shadow text, and display-name cleanup.

Lines are identical in both texts; only columns on edited lines move. Edits are
expressed in code-point columns; LSP positions are UTF-16 code units, so the
public functions convert at the boundary using each side's own line text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .rewrite import Edit


def _utf16_to_cp(line: str, col16: int) -> int:
    units = 0
    for i, ch in enumerate(line):
        if units >= col16:
            return i
        units += 2 if ord(ch) > 0xFFFF else 1
    return len(line)


def _cp_to_utf16(line: str, col: int) -> int:
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in line[:col])


@dataclass
class _LineMap:
    # parallel lists of segments: (user_start, user_end, shadow_start, shadow_end)
    segs: list[tuple[int, int, int, int]]


class PositionMap:
    def __init__(self, user_text: str, shadow_text: str, edits: list[Edit]):
        self.user_lines = user_text.split("\n")
        self.shadow_lines = shadow_text.split("\n")
        self.lines: dict[int, _LineMap] = {}
        by_line: dict[int, list[Edit]] = {}
        for e in edits:
            by_line.setdefault(e.line, []).append(e)
        for ln, es in by_line.items():
            es.sort(key=lambda e: (e.start, e.end))
            segs = []
            shift = 0
            for e in es:
                s_start = e.start + shift
                s_end = s_start + len(e.text)
                segs.append((e.start, e.end, s_start, s_end))
                shift += len(e.text) - (e.end - e.start)
            self.lines[ln] = _LineMap(segs)

    # ---- code-point level ----
    def user_to_shadow_cp(self, line: int, col: int) -> int:
        lm = self.lines.get(line)
        if lm is None:
            return col
        shift = 0
        for us, ue, ss, se in lm.segs:
            if us == ue:  # pure insertion: user char at `us` now sits after it
                if col >= us:
                    shift += se - ss
                    continue
                break
            if col < us:
                break
            if col < ue:  # inside a replaced span -> start of its replacement
                return ss
            shift += (se - ss) - (ue - us)
        return col + shift

    def shadow_to_user_cp(self, line: int, col: int) -> int:
        lm = self.lines.get(line)
        if lm is None:
            return col
        delta = 0  # user_col - shadow_col for text between segments
        for us, ue, ss, se in lm.segs:
            if col < ss:
                break
            if col < se:  # inside generated text -> the user span it came from
                return us
            delta = ue - se
        return col + delta

    # ---- UTF-16 (LSP) level ----
    def user_to_shadow(self, line: int, char16: int) -> int:
        u = self.user_lines[line] if line < len(self.user_lines) else ""
        s = self.shadow_lines[line] if line < len(self.shadow_lines) else ""
        cp = _utf16_to_cp(u, char16)
        return _cp_to_utf16(s, self.user_to_shadow_cp(line, cp))

    def shadow_to_user(self, line: int, char16: int) -> int:
        u = self.user_lines[line] if line < len(self.user_lines) else ""
        s = self.shadow_lines[line] if line < len(self.shadow_lines) else ""
        cp = _utf16_to_cp(s, char16)
        return _cp_to_utf16(u, self.shadow_to_user_cp(line, cp))

    def shadow_range_to_user(self, rng: dict) -> dict:
        st, en = rng["start"], rng["end"]
        sl, sc = st["line"], self.shadow_to_user(st["line"], st["character"])
        el, ec = en["line"], self.shadow_to_user(en["line"], en["character"])
        # An end inside generated text maps to the start of its user span; widen it to that span's end.
        lm = self.lines.get(el)
        if lm is not None:
            s_line = self.shadow_lines[el] if el < len(self.shadow_lines) else ""
            u_line = self.user_lines[el] if el < len(self.user_lines) else ""
            end_cp = _utf16_to_cp(s_line, en["character"])
            for us, ue, ss, se in lm.segs:
                if ss < end_cp <= se:
                    ec = _cp_to_utf16(u_line, ue if ue > us else us)
                    break
        if (el, ec) < (sl, sc):
            el, ec = sl, sc
        return {"start": {"line": sl, "character": sc}, "end": {"line": el, "character": ec}}


# ---------------- display names ----------------

_INT_DIM = re.compile(r"\b(?:_ts_)?Int\[_ts_\w+\]")
# A size used as a class type argument (`LayerNorm[Int[_ts_d]]`, `BatchNorm2d[Int[64]]`, from tensi's
# typed norm stubs) reads as the dim itself; a parameter's `n: Int[_ts_n]` stays `int`.
_INT_TARG = re.compile(r"(?:(?<=\[)|(?<=, ))(?:_ts_)?Int\[(_ts_\w+|-?\d+)\](?=[\],])")
_INTVAR = re.compile(r"\b_ts_IntVar\b")
_MD = re.compile(r"\b_ts_md_[A-Za-z0-9_]+?__([A-Za-z0-9_]+)\b")  # ModuleDict attribute -> its key
_CFG = re.compile(r"\b_ts_cfg_([A-Za-z0-9_]+)\b")  # aliased config import -> class name
_TENSOR = re.compile(r"\b_ts_Tensor\b")
_TP = re.compile(r"\b_ts_([A-Za-z0-9_]+?)(?:__\d+|__p)?\b")
_TP_DECL = re.compile(r"\b_ts_[A-Za-z0-9_]+: (?:_ts_)?IntVar(?:, )?")


def clean_display(text: str) -> str:
    """Turn shadow-side names back into what the user wrote."""
    text = text.replace("Cannot evaluate type-level shape DSL call: ", "")
    text = _INT_TARG.sub(lambda m: m.group(1), text)
    text = _INT_DIM.sub("int", text)
    if "IntVar" in text:
        text = _TP_DECL.sub("", text)
        text = re.sub(r",\s*\]\(", "](", text)  # `[T, ](` -> `[T](`
        text = re.sub(r"\[\]\(", "(", text)  # `f[](` -> `f(`; leaves scalar shapes `Tensor[[]]` alone
    text = _INTVAR.sub("IntVar", text)
    text = _MD.sub(lambda m: m.group(1), text)
    text = _CFG.sub(lambda m: m.group(1), text)
    text = _TENSOR.sub("Tensor", text)
    text = _TP.sub(lambda m: m.group(1), text)
    # `Tensor[[*tuple[int, ...], dim]]` is "any leading dims, then dim": jaxtyping writes it `... dim`
    text = text.replace("*tuple[int, ...]", "...")
    return text


def clean_json(obj):
    """Apply clean_display to every string inside a JSON-like value."""
    if isinstance(obj, str):
        return clean_display(obj)
    if isinstance(obj, list):
        return [clean_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    return obj
