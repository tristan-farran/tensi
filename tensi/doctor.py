"""`tensi doctor [DIR]`: find the setup problems that otherwise show up only as missing shapes.

Most ways tensi can be misconfigured fail silently: if Pyrefly runs with an interpreter that has
no torch, every tensor is `Unknown`, and hovers simply show nothing. The doctor checks each
piece and then runs one real probe from DIR, the way the editor and `tensi check` would.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import env, pyrefly, stubs

PYREFLY_VERSION = "1.5.0"
PROBE = """import torch
from typing import reveal_type
reveal_type(torch.zeros(2, 3).transpose(0, 1))
"""


class Report:
    def __init__(self, st):
        self.st = st
        self.failed = 0
        self.warned = 0

    def ok(self, what: str, detail: str = "") -> None:
        print(f"  {self.st.green('ok')}    {what}" + (f"  {self.st.dim(detail)}" if detail else ""))

    def warn(self, what: str, fix: str) -> None:
        self.warned += 1
        print(f"  {self.st.plum('warn')}  {what}\n        {self.st.dim('fix: ' + fix)}")

    def fail(self, what: str, fix: str) -> None:
        self.failed += 1
        print(f"  {self.st.red('fail')}  {what}\n        {self.st.dim('fix: ' + fix)}")


def _scratch(folder: Path) -> tuple[int, str, Path]:
    """A temp .py file in `folder` (so Pyrefly finds the project's config), or elsewhere if read-only."""
    try:
        fd, tmp = tempfile.mkstemp(prefix="_tensi_doctor_", suffix=".py", dir=folder)
        return fd, tmp, folder
    except OSError:
        cwd = Path(tempfile.mkdtemp(prefix="tensi-doctor-"))
        fd, tmp = tempfile.mkstemp(prefix="_tensi_doctor_", suffix=".py", dir=cwd)
        return fd, tmp, cwd


def pyrefly_interpreter(folder: Path) -> str | None:
    """The Python interpreter Pyrefly picks for files in `folder`."""
    return env.interpreter(env.project_root(folder))


def _prefix(interpreter: str) -> str | None:
    try:
        return subprocess.run(
            [interpreter, "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def _has_module(interpreter: str, module: str) -> bool | None:
    code = f"import importlib.util as u, sys; sys.exit(0 if u.find_spec({module!r}) else 3)"
    try:
        r = subprocess.run([interpreter, "-c", code], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.returncode == 0


def probe(folder: Path) -> tuple[str | None, str]:
    """Reveal one shaped torch expression from `folder`. Returns (revealed type, raw output)."""
    fd, tmp, cwd = _scratch(folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(PROBE)
        proc = subprocess.run(
            [pyrefly(), "check", tmp, "--preset", "default", "--output-format", "json", *stubs.search_args()],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, repr(exc)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    try:
        errors = json.loads(proc.stdout or "{}").get("errors", [])
    except json.JSONDecodeError:
        return None, proc.stdout + proc.stderr
    for e in errors:
        if e.get("name") == "reveal-type":
            m = re.search(r"revealed type: (.*)", e["description"], re.S)
            if m:
                return m.group(1).strip(), proc.stdout
    return None, proc.stdout + proc.stderr


def run(folder: str | None = None) -> int:
    from . import __version__
    from .peek import style_for

    st = style_for(sys.stdout)
    where = Path(folder or ".").resolve()
    print(f"{st.plum('tensi')} {st.dim('doctor v' + __version__)}  {where}")
    r = Report(st)

    # 1. Pyrefly itself
    exe = shutil.which(pyrefly()) or (pyrefly() if Path(pyrefly()).exists() else None)
    if exe is None:
        r.fail(f"Pyrefly not found (`{pyrefly()}`)", "pip install 'pyrefly==1.5.0', or set TENSI_PYREFLY to its path")
        print(f"\n  {st.red('1')} problem; nothing else can be checked without Pyrefly")
        return 1
    try:
        ver = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout.split()[-1]
    except Exception:
        ver = "?"
    if ver == PYREFLY_VERSION:
        r.ok(f"Pyrefly {ver}", exe)
    else:
        r.warn(
            f"Pyrefly {ver}; tensi is built and tested against {PYREFLY_VERSION}",
            f"pip install 'pyrefly=={PYREFLY_VERSION}'",
        )

    # 2. tensi's stubs
    if os.environ.get("TENSI_STUBS") == "0":
        r.warn("tensi's torch stub patches are off (TENSI_STUBS=0)", "unset TENSI_STUBS")
    elif stubs.upstream_dir() is None:
        r.fail("pyrefly-torch-stubs is not installed next to tensi", "pip install 'pyrefly-torch-stubs==1.5.0'")
    else:
        d = stubs.overlay_dir()
        text = (d / "PATCHES").read_text() if d else ""
        skipped = re.search(r"skipped: *(.*)", text)
        skipped_names = skipped.group(1).split() if skipped else []
        carried = sorted(n for n in stubs.COMPANIONS if d and (d / n).is_dir())
        if skipped_names:
            r.warn(
                f"{len(skipped_names)} stub patch(es) no longer apply: {', '.join(skipped_names)}",
                "your pyrefly-torch-stubs differs from 1.5.0; pin it, or report the patch names",
            )
        else:
            r.ok(
                f"torch stubs with {len(stubs.PATCHES)} tensi patches",
                f"{d}  (also carries {', '.join(carried) or 'nothing else'})",
            )
        if "shape_extensions" not in carried:
            r.fail("shape_extensions not found", "pip install 'pyrefly-shape-extensions==1.5.0'")

    # 3. The interpreter Pyrefly uses for this folder
    interp = pyrefly_interpreter(where)
    if interp is None:
        r.warn("could not tell which Python Pyrefly uses here", f"run `pyrefly dump-config` in {where}")
    else:
        r.ok(
            f"Pyrefly uses {interp}",
            "same as tensi's" if _prefix(interp) == sys.prefix else f"tensi runs on {sys.executable}",
        )
        v = env.interpreter_version(interp)
        if v and v < env.MIN_VERSION:
            r.ok(
                f"project Python {v[0]}.{v[1]}",
                f"tensi has Pyrefly read it as {env.analysis_version(env.project_root(where))} for its own syntax; "
                "packages still come from this interpreter",
            )
        if _has_module(interp, "torch") is False:
            # tensi's overlay carries the shape stubs, so shapes still work; torch's other types don't
            r.warn(
                f"torch is not installed for {interp}; tensi's stubs cover shapes, anything they don't stub is Unknown",
                "install torch into that environment, or set `python-interpreter-path` in "
                "pyrefly.toml / [tool.pyrefly] to the one that has it",
            )
        uses_einops = _has_module(interp, "einops")
        if uses_einops and not (stubs.overlay_dir() and (stubs.overlay_dir() / "einops-shape-stubs").is_dir()):
            r.warn(
                "einops is installed but its shape stubs are not; rearrange/repeat results are unshaped",
                "reinstall tensi: it ships these stubs",
            )

    # 4. One real probe, the way `tensi check` and the editor run Pyrefly
    revealed, raw = probe(where)
    if revealed and "[3, 2]" in revealed.replace(" ", "").replace(",", ", "):
        r.ok("probe: torch.zeros(2, 3).transpose(0, 1) is " + revealed)
    elif revealed:
        r.fail(
            f"probe: torch.zeros(2, 3).transpose(0, 1) is `{revealed}`, expected Tensor[[3, 2]]",
            "the checks above should say why; if they all pass, run `tensi stubs` and report this",
        )
    else:
        r.fail("probe: Pyrefly produced no result", raw.strip()[:400] or "run pyrefly check by hand")

    # 5. The editor launches `tensi lsp`
    if shutil.which("tensi") is None:
        r.warn(
            "`tensi` is not on PATH; the VS Code extension won't find it",
            "add the folder with the tensi script to PATH, or set tensi.path in VS Code settings",
        )

    print()
    if r.failed:
        print(
            f"  {st.red(str(r.failed))} problem{'s' if r.failed != 1 else ''}"
            + (f", {r.warned} warning{'s' if r.warned != 1 else ''}" if r.warned else "")
        )
        return 1
    print(f"  {st.green('ok')} ready" + (f", {r.warned} warning{'s' if r.warned != 1 else ''}" if r.warned else ""))
    return 0
