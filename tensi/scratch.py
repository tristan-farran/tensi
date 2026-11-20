"""Where the CLI puts the rewritten copy of a file while Pyrefly checks it.

Next to the original is best: Pyrefly then finds the project's config and interpreter, and
sibling and relative imports resolve exactly as they do for the original. Files start with
`_tensi_` and are removed as soon as Pyrefly returns. When the folder is read-only (an installed
package, a mounted volume), the copy goes to a scratch folder instead, with the original folder
on the search path so its siblings still resolve.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .env import version_args


@contextmanager
def scratch_copy(folder: Path, text: str, prefix: str):
    """Yield (path of a temp .py holding `text`, extra pyrefly args, cwd to run Pyrefly in). The extra
    args include the Python version tensi's rewritten syntax needs (see `env`)."""
    folder = Path(folder)
    made_dir: Path | None = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=".py", dir=folder)
        extra: list[str] = []
        cwd = folder
    except OSError:
        made_dir = Path(tempfile.mkdtemp(prefix="tensi-"))
        fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=".py", dir=made_dir)
        extra = ["--search-path", str(folder)]
        cwd = made_dir
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        yield tmp, extra + version_args(folder), cwd
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if made_dir is not None:
            shutil.rmtree(made_dir, ignore_errors=True)
