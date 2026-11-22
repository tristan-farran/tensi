"""Tell the user when a newer tensi is out, and how to install it, and when the Pyrefly in use is
not the one tensi is pinned to.

The update check asks PyPI for the latest `tensi-shapes` release at most once a day (the answer is
cached under the tensi cache folder), never blocks for more than a moment, and sends nothing but
the request itself. It is off in CI and with `TENSI_NO_UPDATE_CHECK=1`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from . import PYREFLY_VERSION, __version__, pyrefly

PACKAGE = "tensi-shapes"
URL = f"https://pypi.org/pypi/{PACKAGE}/json"
DAY = 24 * 3600


def _cache_file() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tensi"
    return base / "latest.json"


def enabled() -> bool:
    if os.environ.get("TENSI_NO_UPDATE_CHECK", "").strip() not in ("", "0"):
        return False
    return not os.environ.get("CI")


def _parse(v: str) -> tuple[int, ...] | None:
    """`0.3.4` -> (0, 3, 4); pre-releases and anything odd -> None (never offered)."""
    return tuple(int(x) for x in v.split(".")) if re.fullmatch(r"\d+(\.\d+){0,3}", v) else None


def newer(latest: str, have: str = __version__) -> bool:
    a, b = _parse(latest), _parse(have)
    return a is not None and b is not None and a > b


def _fetch(timeout: float) -> str | None:
    req = urllib.request.Request(URL, headers={"User-Agent": f"tensi/{__version__}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (a fixed https URL)
        return str(json.load(r)["info"]["version"])


def latest(timeout: float = 2.0, max_age: float = DAY) -> str | None:
    """The newest release on PyPI, from the cache when it is under a day old. None when the check
    is off or PyPI can't be reached; a failed check is cached too, so an offline machine asks once
    a day, not on every command."""
    if not enabled():
        return None
    f = _cache_file()
    try:
        cached = json.loads(f.read_text())
        if time.time() - float(cached["checked"]) < max_age:
            return cached.get("latest")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        found = _fetch(timeout)
    except Exception:
        found = None
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"checked": time.time(), "latest": found}))
    except OSError:
        pass
    return found


def upgrade_command(prefix: str = sys.prefix, executable: str = sys.executable) -> str:
    """The command that upgrades this installation of tensi."""
    p = prefix.replace("\\", "/")
    if "/uv/tools/" in p:
        return f"uv tool upgrade {PACKAGE}"
    if "/pipx/venvs/" in p:
        return f"pipx upgrade {PACKAGE}"
    project = Path(prefix).parent
    if shutil.which("uv") and (project / "uv.lock").exists() and Path(prefix).name == ".venv":
        return f"uv add --dev --upgrade-package {PACKAGE} {PACKAGE}"
    if shutil.which("uv"):
        return f"uv pip install --upgrade --python {executable} {PACKAGE}"
    return f"{executable} -m pip install --upgrade {PACKAGE}"


def update_message(found: str | None) -> str | None:
    if found is None or not newer(found):
        return None
    return f"tensi {found} is available (you have {__version__}). Update with: {upgrade_command()}"


class Check:
    """An update check started in the background, so a command never waits on the network;
    `message()` gives it a moment more at the end and then moves on without it."""

    def __init__(self):
        self.found: str | None = None
        self._t = threading.Thread(target=self._run, daemon=True) if enabled() else None
        if self._t:
            self._t.start()

    def _run(self):
        self.found = latest()

    def message(self, wait: float = 0.5) -> str | None:
        if self._t is None:
            return None
        self._t.join(wait)
        return update_message(self.found)


@lru_cache(maxsize=8)
def pyrefly_version(exe: str | None = None) -> str | None:
    try:
        out = subprocess.run([exe or pyrefly(), "--version"], capture_output=True, text=True, timeout=30).stdout
        return out.split()[-1] if out.split() else None
    except (OSError, subprocess.SubprocessError):
        return None


def pyrefly_mismatch(exe: str | None = None) -> str | None:
    """A warning when the Pyrefly tensi will run is not the pinned version: tensi's rewriting and
    stub patches are made for exactly that one, and another version can quietly lose shapes."""
    ver = pyrefly_version(exe)
    if ver is None or ver == PYREFLY_VERSION:
        return None
    return (
        f"tensi is using Pyrefly {ver} ({exe or pyrefly()}), but it is built and tested against "
        f"{PYREFLY_VERSION}; shapes may be missing or wrong. Fix: install 'pyrefly=={PYREFLY_VERSION}' "
        f"beside tensi, or set TENSI_PYREFLY to a Pyrefly {PYREFLY_VERSION}."
    )
