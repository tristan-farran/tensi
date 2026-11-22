"""The update notice and the Pyrefly pin."""

import json
import os
import re
import time
from pathlib import Path

import pytest

import tensi
from tensi import update


@pytest.fixture
def checks_on(monkeypatch, tmp_path):
    monkeypatch.delenv("TENSI_NO_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


def test_newer_compares_releases_only():
    assert update.newer("0.3.10", "0.3.9")
    assert not update.newer("0.3.4", "0.3.4")
    assert not update.newer("0.3.3", "0.3.4")
    assert not update.newer("0.4.0rc1", "0.3.4")  # pre-releases are never offered


def test_latest_is_cached_for_a_day(checks_on, monkeypatch):
    calls = []
    monkeypatch.setattr(update, "_fetch", lambda timeout: calls.append(1) or "9.9.9")
    assert update.latest() == "9.9.9" and update.latest() == "9.9.9"
    assert calls == [1]
    f = checks_on / "tensi" / "latest.json"
    f.write_text(json.dumps({"checked": time.time() - 2 * update.DAY, "latest": "9.9.9"}))
    update.latest()
    assert calls == [1, 1]


def test_offline_asks_once_a_day(checks_on, monkeypatch):
    calls = []

    def boom(timeout):
        calls.append(1)
        raise OSError("offline")

    monkeypatch.setattr(update, "_fetch", boom)
    assert update.latest() is None and update.latest() is None
    assert calls == [1]


def test_off_in_ci_and_by_env(checks_on, monkeypatch):
    monkeypatch.setattr(update, "_fetch", lambda timeout: pytest.fail("must not fetch"))
    monkeypatch.setenv("CI", "true")
    assert update.latest() is None and update.Check().message() is None
    monkeypatch.delenv("CI")
    monkeypatch.setenv("TENSI_NO_UPDATE_CHECK", "1")
    assert update.latest() is None


def test_message_names_the_upgrade_command(checks_on, monkeypatch):
    monkeypatch.setattr(update, "_fetch", lambda timeout: "99.0.0")
    msg = update.Check().message(wait=5)
    assert msg and "tensi 99.0.0 is available" in msg and "Update with:" in msg
    monkeypatch.setattr(update, "_fetch", lambda timeout: tensi.__version__)
    (checks_on / "tensi" / "latest.json").unlink()
    assert update.Check().message(wait=5) is None


def test_upgrade_command_matches_the_installation(monkeypatch, tmp_path):
    assert update.upgrade_command("/Users/me/.local/share/uv/tools/tensi-shapes", "x") == "uv tool upgrade tensi-shapes"
    assert update.upgrade_command("/Users/me/.local/pipx/venvs/tensi-shapes", "x") == "pipx upgrade tensi-shapes"
    (tmp_path / "uv.lock").write_text("")
    monkeypatch.setattr(update.shutil, "which", lambda name: "/usr/bin/uv")
    assert update.upgrade_command(str(tmp_path / ".venv"), "py").startswith("uv add --dev --upgrade-package")
    monkeypatch.setattr(update.shutil, "which", lambda name: None)
    assert update.upgrade_command("/opt/env", "/opt/env/bin/python") == "/opt/env/bin/python -m pip install --upgrade tensi-shapes"


def test_pyrefly_pins_agree():
    """pyproject pins Pyrefly and its stub packages to the one version tensi checks at runtime."""
    text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    pins = dict(re.findall(r'"(pyrefly[\w-]*)==([\d.]+)"', text))
    assert set(pins) == {"pyrefly", "pyrefly-torch-stubs", "pyrefly-shape-extensions"}, pins
    assert set(pins.values()) == {tensi.PYREFLY_VERSION}, pins


@pytest.mark.skipif(os.name == "nt", reason="uses a shell script as a fake Pyrefly")
def test_pyrefly_mismatch_warns(tmp_path):
    fake = tmp_path / "pyrefly"
    fake.write_text("#!/bin/sh\necho pyrefly 9.9.9\n")
    fake.chmod(0o755)
    msg = update.pyrefly_mismatch(str(fake))
    assert msg and "Pyrefly 9.9.9" in msg and f"pyrefly=={tensi.PYREFLY_VERSION}" in msg
    real = tmp_path / "pyrefly-ok"
    real.write_text(f"#!/bin/sh\necho pyrefly {tensi.PYREFLY_VERSION}\n")
    real.chmod(0o755)
    assert update.pyrefly_mismatch(str(real)) is None
