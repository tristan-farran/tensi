"""The project's Python, as Pyrefly sees it, and the Python version tensi asks Pyrefly to assume.

tensi's rewritten code uses type-parameter syntax (`class Mlp[_ts_d: _ts_IntVar = _ts_in]`) that
needs Python 3.13. Pyrefly reads the language version from the project's interpreter, so on a 3.10
project it rejects that syntax, and its language server then answers no hovers at all. tensi
therefore tells Pyrefly to assume at least 3.13 while it keeps using the project's own
interpreter for site-packages (torch, einops, the project's dependencies). Your code is never
run on that version; it only sets which syntax and typeshed entries Pyrefly accepts.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from . import pyrefly

MIN_VERSION = (3, 13)
_ROOT_MARKERS = ("pyrefly.toml", "pyproject.toml", "setup.py", "setup.cfg", ".git")


def project_root(path: str | Path) -> Path:
    """The nearest folder above `path` that looks like a project root (else `path`'s folder)."""
    p = Path(path).resolve()
    start = p if p.is_dir() else p.parent
    for d in [start, *start.parents]:
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d
    return start


def has_own_config(root: Path) -> bool:
    """The project configures Pyrefly itself (pyrefly.toml or `[tool.pyrefly]`)."""
    if (root / "pyrefly.toml").is_file():
        return True
    try:
        return "[tool.pyrefly" in (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return False


@functools.lru_cache(maxsize=32)
def interpreter(root: Path) -> str | None:
    """The interpreter Pyrefly picks for files in `root` (from `pyrefly dump-config`)."""
    try:
        fd, tmp = tempfile.mkstemp(prefix="_tensi_env_", suffix=".py", dir=root)
        cwd = root
    except OSError:
        cwd = Path(tempfile.mkdtemp(prefix="tensi-env-"))
        fd, tmp = tempfile.mkstemp(prefix="_tensi_env_", suffix=".py", dir=cwd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("import torch\n")
        out = subprocess.run([pyrefly(), "dump-config", tmp], capture_output=True, text=True, cwd=cwd, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    m = re.search(r"Using interpreter: (.+)", out.stdout + out.stderr)
    return m.group(1).strip() if m else None


@functools.lru_cache(maxsize=32)
def interpreter_version(python: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [python, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        major, minor = out.stdout.split()
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def analysis_version(root: Path) -> str:
    """The Python version Pyrefly should assume: the project's own, but at least 3.13."""
    python = interpreter(root)
    v = interpreter_version(python) if python else None
    major, minor = max(v or MIN_VERSION, MIN_VERSION)
    return f"{major}.{minor}"


@functools.lru_cache(maxsize=256)
def _version_args(root: Path) -> tuple[str, ...]:
    if os.environ.get("TENSI_PYTHON_VERSION") == "project":
        return ()
    return ("--python-version", analysis_version(root))


def version_args(path: str | Path) -> list[str]:
    """`pyrefly check` arguments that make Pyrefly accept tensi's rewritten syntax."""
    return list(_version_args(project_root(path)))


def lsp_config(root: Path, dest: Path) -> Path | None:
    """A Pyrefly config for the language server, for a project without its own: the project's
    interpreter and import roots, plus `python-version` (see the module docstring). None when the
    project configures Pyrefly itself, which tensi then leaves alone."""
    if has_own_config(root) or os.environ.get("TENSI_PYTHON_VERSION") == "project":
        return None
    python = interpreter(root)
    roots = [root] + [root / "src"] * (root / "src").is_dir()
    lines = [
        "# Written by tensi for the language server; see tensi/env.py. Your project is unchanged.",
        f"python-version = {json.dumps(analysis_version(root))}",
        f"project-includes = [{json.dumps(str(root))}]",
        f"search-path = [{', '.join(json.dumps(str(r)) for r in roots)}]",
    ]
    if python:
        lines.append(f"python-interpreter-path = {json.dumps(python)}")
    try:
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "pyrefly.toml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except OSError:
        return None
