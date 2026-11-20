"""tensi's torch stub patch set (phase 1 of the fork plan)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensi import env, stubs
from tensi.mapping import clean_display

FIX = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(stubs.upstream_dir() is None, reason="pyrefly-torch-stubs not installed")


def run(*args, **env):
    return subprocess.run(
        [sys.executable, "-m", "tensi.cli", *args],
        capture_output=True,
        text=True,
        cwd=FIX,
        env={**os.environ, "NO_COLOR": "1", **env},
    )


def test_every_patch_applies_to_the_pinned_upstream():
    d = stubs.overlay_dir()
    assert d is not None and (d / stubs.STUBS_DIR / "__init__.pyi").exists()
    applied, skipped = (d / "PATCHES").read_text().splitlines()
    assert skipped.strip() == "skipped:"
    assert applied.split()[1:] == [p.name for p in stubs.PATCHES]


def test_patches_are_exact_replacements():
    upstream = stubs.upstream_dir()
    for p in stubs.PATCHES:
        text = (upstream / p.file).read_text(encoding="utf-8")
        out = p.apply(text)
        assert out is not None and out != text, p.name
    assert stubs.Patch("x", "f", "w", old="nowhere in the file", new="y").apply("text") is None


def test_diff_is_a_unified_diff_for_upstream():
    d = stubs.diff()
    assert d.startswith("--- a/torch-shape-stubs/")
    assert "+class LayerNorm[N: _Int = int](Module):" in d
    assert "+def trailing_extent_shape(" in d


def test_peek_uses_patched_stubs():
    out = " ".join(run("peek", "norms.py").stdout.split())
    for row in ["15 h [b, t, width]", "17 flipped [b, width, t]", "18 pad [b, 1, width]"]:
        assert row in out, row
    off = " ".join(run("peek", "norms.py", TENSI_STUBS="0").stdout.split())
    assert "17 flipped [?]" in off and "18 pad [?]" in off


def test_check_catches_norm_size_mismatches():
    r = run("check", "norms.py")
    assert r.returncode == 1, r.stdout
    assert "norms.py:30:22" in r.stdout and "normalized size does not match the input's last dimension" in r.stdout
    assert "norms.py:31:20" in r.stdout and "num_features does not match the input's channel dimension" in r.stdout
    assert "Cannot evaluate" not in r.stdout
    assert run("check", "norms.py", TENSI_STUBS="0").returncode == 0  # upstream can't see these


def test_check_has_no_false_alarms_from_patches():
    for f in ["gptlike.py", "attn.py", "plain.py"]:
        r = run("check", f)
        assert r.returncode == 0, (f, r.stdout)


def test_norm_type_arguments_display_as_dims():
    assert clean_display("LayerNorm[Int[_ts_n_embd]]") == "LayerNorm[n_embd]"
    assert clean_display("BatchNorm2d[Int[64]]") == "BatchNorm2d[64]"
    assert clean_display("(parameter) n: Int[_ts_n]") == "(parameter) n: int"


def test_ops_keep_shapes_with_gradual_sizes():
    """cat/stack with an unknown batch or of a list, masks from `x > 0`, `.type(dtype)`, amax."""
    import json

    proc = subprocess.run(
        [
            os.environ.get("TENSI_PYREFLY", "pyrefly"),
            "check",
            "stubops.py",
            "--output-format",
            "json",
            *stubs.search_args(),
            *env.version_args(FIX),
        ],
        capture_output=True,
        text=True,
        cwd=FIX,
    )
    got = [
        e["description"].split("revealed type: ")[-1]
        for e in json.loads(proc.stdout)["errors"]
        if e["name"] == "reveal-type"
    ]
    assert got == [
        "Tensor[[int, int, 64]]",
        "Tensor[[B, int, 64]]",
        "Tensor[[int, B, int, 64]]",
        "Tensor[[2, 7]]",
        "Tensor[[B, int, 64]]",
        "Tensor[[B, int, 64]]",
        "Tensor[[B, int, 64]]",
        "Tensor[[B, int]]",
    ], got
