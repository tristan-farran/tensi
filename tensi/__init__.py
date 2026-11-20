"""tensi: named tensor shapes for PyTorch, on Pyrefly."""

import os
import shutil
import sys

__version__ = "0.3.3"


def pyrefly() -> str:
    """The Pyrefly executable: `$TENSI_PYREFLY`, else the one installed beside this Python (so a
    venv's Pyrefly wins even when the venv is not activated), else `pyrefly` on PATH."""
    env = os.environ.get("TENSI_PYREFLY")
    if env:
        return env
    exe = "pyrefly.exe" if os.name == "nt" else "pyrefly"
    beside = os.path.join(os.path.dirname(sys.executable), exe)
    return beside if os.path.isfile(beside) else (shutil.which("pyrefly") or "pyrefly")
