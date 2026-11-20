"""The `tensi` command line: peek, check, shadow, --version."""

import subprocess
import sys
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"


def run(*args, cwd=FIX):
    return subprocess.run(
        [sys.executable, "-m", "tensi.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**__import__("os").environ, "NO_COLOR": "1"},
    )


def test_version():
    r = run("--version")
    assert r.returncode == 0 and r.stdout.strip().startswith("tensi ")


def test_peek_gptlike_zero_annotations():
    r = run("peek", "gptlike.py")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "SelfAttention.forward" in out and "TinyGPT.generate" in out
    for line_shape in [
        "scores    [b, heads, t, t]",
        "tok_emb   [b, t, width]",
        "nxt       [b, 1]",
        "logits    [b, 1, vocab]",
    ]:
        assert " ".join(line_shape.split()) in " ".join(out.split()), line_shape
    assert "_ts_" not in out and "Self@" not in out


def test_check_exit_codes():
    ok = run("check", "gptlike.py")
    assert ok.returncode == 0 and "no shape errors" in ok.stdout
    bad = run("check", "bad.py")
    assert bad.returncode == 1 and "2 shape errors" in bad.stdout
    assert "bad.py:17:9" in bad.stdout and "Tensor[[b, 7, v]]" in bad.stdout


def test_shadow_prints_rewritten_text():
    r = run("shadow", "plain.py")
    assert r.returncode == 0 and "_ts_Int[_ts_batch_size]" in r.stdout


def test_check_fails_closed_when_pyrefly_crashes(tmp_path):
    fake = tmp_path / "pyrefly"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    fake.chmod(0o755)
    import os

    bad = subprocess.run(
        [sys.executable, "-m", "tensi.cli", "check", "plain.py", "--no-solve"],
        capture_output=True,
        text=True,
        cwd=FIX,
        env={**os.environ, "NO_COLOR": "1", "TENSI_PYREFLY": str(fake)},
    )
    assert bad.returncode != 0 and "no shape errors" not in bad.stdout
    assert "pyrefly failed" in bad.stderr


def test_peek_counts_dims_containing_int_as_named():
    from tensi.cli import is_fully_named

    assert is_fully_named("[b, n_points]") and is_fully_named("[b, hint]")
    assert not is_fully_named("[b, int]") and not is_fully_named("[?]") and not is_fully_named("[2, 3]")


def test_doctor_reports_ready():
    r = run("doctor")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "probe: torch.zeros(2, 3).transpose(0, 1) is Tensor[[3, 2]]" in r.stdout
    assert "ready" in r.stdout


def test_doctor_fails_without_pyrefly(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "tensi.cli", "doctor", str(tmp_path)],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "NO_COLOR": "1", "TENSI_PYREFLY": str(tmp_path / "nope")},
    )
    assert r.returncode == 1 and "Pyrefly not found" in r.stdout


def test_project_venv_without_shape_packages_still_gets_shapes(tmp_path):
    """A project whose own .venv lacks shape_extensions and the stubs (tensi installed with pipx/uv
    tool): Pyrefly picks the project's interpreter, and tensi's overlay must carry what it needs."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(tmp_path / ".venv")], check=True)
    (tmp_path / "m.py").write_text((FIX / "plain.py").read_text())
    r = run("peek", str(tmp_path / "m.py"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    import re

    m = re.search(r"(\d+) tensors?, (\d+) fully named", r.stdout)
    assert m and int(m.group(2)) > 0, r.stdout
    d = run("doctor", str(tmp_path), cwd=tmp_path)
    assert str(tmp_path / ".venv") in d.stdout and "torch is not installed" in d.stdout


def test_scratch_copy_falls_back_when_the_folder_is_not_writable(tmp_path):
    from tensi.scratch import scratch_copy

    missing = tmp_path / "read-only"  # mkstemp fails here like it does in a read-only folder
    with scratch_copy(missing, "x = 1\n", "_tensi_t_") as (tmp, extra, cwd):
        assert Path(tmp).read_text() == "x = 1\n" and extra == ["--search-path", str(missing)]
        assert Path(cwd) != missing
    assert not Path(tmp).exists() and not Path(cwd).exists()


def test_peek_and_check_leave_no_temp_files(tmp_path):
    (tmp_path / "m.py").write_text((FIX / "plain.py").read_text())
    run("peek", str(tmp_path / "m.py"))
    run("check", str(tmp_path / "m.py"))
    assert [p.name for p in tmp_path.iterdir()] == ["m.py"]


def test_peek_vitlike_zero_annotations():
    """vit-pytorch idioms with no annotations: einops patterns name the root input, ModuleList
    stacks built with .append keep their modules' types, `X if c else nn.Identity()` keeps X's shape
    rule, an optional `mask=None` input, and `1 if pool == 'cls' else 0` sizes."""
    r = run("peek", "vitlike.py")
    assert r.returncode == 0, r.stderr
    out = " ".join(r.stdout.split())
    for row in [
        "28 x [b, int, dim]",
        "31 k [b, heads, int, dim_head]",
        "35 mask [b, 1, 1, j]",
        "38 out [b, int, heads * dim_head]",
        "54 x [b, int, dim]",
        "55 x [b, int, dim]",
        "71 x [b, int, dim]",
        "72 cls [int, dim]",
        "74 y [b, dim, int, 1]",
    ]:
        assert row in out, row
    assert run("check", "vitlike.py").returncode == 0


def test_peek_names_the_input_that_needs_an_annotation():
    out = " ".join(run("peek", "norms.py").stdout.split())
    assert "annotate the input of: 29 Broken.forward(x)" in out, out
    assert "Encoder.forward" not in out.split("annotate the input of:")[1]  # shaped from x.shape


def test_local_shape_unpack_names_dims(tmp_path):
    """`b, n, _ = x.shape` on a local: the dims that were only `int` take the unpacked names."""
    (tmp_path / "m.py").write_text(
        "import torch\n"
        "from torch import nn\n"
        "from einops import rearrange\n"
        "\n"
        "class M(nn.Module):\n"
        "    def __init__(self, dim):\n"
        "        super().__init__()\n"
        "        self.proj = nn.Linear(dim, dim)\n"
        "\n"
        "\n"
        "    def forward(self, img):\n"
        "        x = rearrange(img, 'b h w d -> b (h w) d')\n"
        "        x = self.proj(x)\n"
        "        b, n, _ = x.shape\n"
        "        y = x.view(b, n, 2, -1)\n"
        "        z = x[:, :int(n * 0.5)]\n"
        "        bz, kept, _ = z.shape\n"
        "        w = z.reshape(bz * kept, -1)\n"
        "        return y, w\n"
    )
    out = " ".join(run("peek", str(tmp_path / "m.py")).stdout.split())
    assert "15 y [b, h * w, 2, dim // 2]" in out, out  # already named: left alone
    assert "16 z [b, int, dim]" in out and "18 w [b * kept, dim]" in out, out


def test_follows_ints_pair_shape_index_and_int_locals():
    """`pair(p)` is inlined, `seq = x.shape[1]` names a dim, and `hidden_dim = int(dim * mult)`
    becomes a dim of its own name."""
    out = " ".join(run("peek", "followints.py").stdout.split())
    assert "17 h [b, seq, hidden_dim]" in out, out
    assert "34 y [b, seq, dim]" in out, out
    shadow = run("shadow", "followints.py").stdout
    assert "pair(patch_size)" not in shadow and "(patch_size, patch_size)" in shadow
    assert len(shadow.splitlines()) == len((FIX / "followints.py").read_text().splitlines())
    assert run("check", "followints.py").returncode == 0


def test_helper_functions_follow_their_callers():
    """`posemb_sincos_2d(x)`: the helper takes x's shape from its caller, `*patches.shape` names h, w,
    dim, and its `4 * (dim // 4)` result is cast back to `dim` (it asserts `dim % 4 == 0`)."""
    out = " ".join(run("peek", "helpers.py").stdout.split())
    assert "15 pe [h * w, 4 * (dim // 4)]" in out, out
    assert "29 pe [int, dim]" in out and "30 x [b, int, dim]" in out, out  # not [?]: `x + pe` keeps its shape
    assert 'Net.forward(img) e.g. img: Shaped[Tensor, "b c height width"] (now [b, c, int, int])' in out
    assert run("check", "helpers.py").returncode == 0
    for seed in ("1", "2", "3"):  # edits at one position must not depend on hash order
        env = {**__import__("os").environ, "NO_COLOR": "1", "PYTHONHASHSEED": seed}
        r = subprocess.run(
            [sys.executable, "-m", "tensi.cli", "peek", "helpers.py"], capture_output=True, text=True, cwd=FIX, env=env
        )
        assert (
            " ".join(r.stdout.split()).split("helpers.py", 1)[1].split(" ", 2)[2]
            == out.split("helpers.py", 1)[1].split(" ", 2)[2]
        )


def test_ordinary_style_modules_name_their_shapes():
    """timm/torchvision style: `-> torch.Tensor` and `out: torch.Tensor` would erase shapes and are
    dropped, `x = x or y` sizes keep their own name (and `Mlp(dim)` defaults them to `dim`), `act_layer`
    factories and `use_bn` flags don't make the layer gradual, a Conv2d first use gives the input its rank
    and channels, and `out += identity` keeps `out`'s shape."""
    out = " ".join(run("peek", "general.py").stdout.split())
    assert "17 x [..., hidden_features] 18 x [..., hidden_features]" in out, out
    assert "30 identity [int, inplanes, int, int] 31 out [int, planes, int, int]" in out, out
    assert "34 out [int, inplanes, int, int]" in out, out
    assert "45 x [..., dim]" in out, out
    # hints name the feature axis from the first layer that reads the input, through user modules too
    assert 'Mlp.forward(x) e.g. x: Shaped[Tensor, "batch seq in_features"]' in out, out
    assert 'Head.forward(x) e.g. x: Shaped[Tensor, "batch seq dim"]' in out, out
    assert 'BasicBlock.forward(x) e.g. x: Shaped[Tensor, "batch inplanes height width"]' in out, out
    assert run("check", "general.py").returncode == 0


def test_defaulted_size_params_still_catch_errors():
    """`Mlp(dim)` and `Mlp(dim, other)` are different shapes; adding their outputs is an error."""
    out = " ".join(run("peek", "fallbackbad.py").stdout.split())
    assert "23 y [..., dim] 24 z [..., other]" in out, out
    r = run("check", "fallbackbad.py")
    assert r.returncode == 1 and "fallbackbad.py:25:16" in r.stdout, r.stdout
    # `self.output_dim = output_dim or embed_dim` is `output_dim`, so `Combined`'s `+` is fine
    assert "1 shape error" in r.stdout, r.stdout


def test_only_shape_diagnostics_count():
    from tensi.cli import is_shape_diagnostic as shape

    assert shape("Argument `Tensor[[b, 3]]` is not assignable to parameter `x` with type `Tensor[[b, 4]]`")
    assert not shape("Argument `Tensor[[b, 3]]` is not assignable to parameter `x` with type `LongTensor`")
    assert not shape("Argument `Tensor[[b, 3]] | None` is not assignable to parameter `x` with type `Tensor[[b, 3]]`")
    assert not shape(
        "Class member `A.forward` overrides parent class `B` in an inconsistent manner: Tensor[[b]]", "bad-override"
    )
    assert shape("Cannot unpack `IntTuple[b, n]` (of size 2) into 3 values", "bad-unpacking")
    assert not shape("Cannot unpack `tuple[int, int]` (of size 2) into 3 values", "bad-unpacking")
    assert not shape("No matching overload found for function `f` called with arguments: (str)")


def test_generate_loop_names_the_batch_through_a_cropped_view():
    """`idx_cond = idx if ... else idx[:, -k:]` then `self(idx_cond)`: generate's idx has forward's rank."""
    out = " ".join(run("peek", "generate.py").stdout.split())
    assert "22 probs [b, vocab]" in out and "23 nxt [b, 1]" in out, out


def test_literal_sizes_from_a_caller_are_the_class_dims():
    """`MLP(64)` called on `[batch, seq, 64]`: inside forward that 64 is `dim`, so `fc2(h) + x` is fine,
    and passing a `[batch, seq, 32]` to the same module is an error."""
    assert run("check", "literal.py").returncode == 0
    assert "13 h [batch, seq, hidden]" in " ".join(run("peek", "literal.py").stdout.split())
    r = run("check", "literalbad.py")
    assert r.returncode == 1 and "literalbad.py:21:14" in r.stdout, r.stdout
