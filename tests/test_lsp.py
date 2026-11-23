"""Acceptance tests from the spec, run through the real proxy with a scripted LSP client."""

import json
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from lspclient import LspClient  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
SERVER = [sys.executable, "-m", "tensi.cli", "lsp"]


def norm(s: str | None) -> str:
    return re.sub(r"\s+", "", (s or "").replace("torch.Tensor", "Tensor"))


def pos(text: str, needle: str, at: int = 0, nth: int = 1):
    """(line, utf16 char) of `needle` (its `at`-th char) in `text`."""
    idx = -1
    for _ in range(nth):
        idx = text.index(needle, idx + 1)
    idx += at
    line = text.count("\n", 0, idx)
    col = idx - (text.rfind("\n", 0, idx) + 1)
    prefix = text.split("\n")[line][:col]
    return line, len(prefix.encode("utf-16-le")) // 2


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    d = tmp_path_factory.mktemp("ws")
    for f in ["plain.py", "attn.py", "bad.py", "reassign.py", "gptlike.py", "norms.py"]:
        shutil.copy(FIX / f, d / f)
    for f in ["layers.py", "model.py"]:
        shutil.copy(FIX / "xfile" / f, d / f)
    shutil.copytree(FIX / "xfile" / "pkg", d / "pkg")
    return d


@pytest.fixture(scope="module")
def client(ws):
    c = LspClient(SERVER, ws)
    yield c
    c.close()


def open_and_wait(client, ws, name):
    uri = client.open(ws / name)
    client.diagnostics(uri, timeout=30)
    return uri


def test_advertises_only_hover_and_sync(client):
    caps = client.init_result["capabilities"]
    assert caps["hoverProvider"] is True
    assert caps["textDocumentSync"]["change"] == 1
    assert "completionProvider" not in caps and "definitionProvider" not in caps
    assert client.init_result["serverInfo"]["name"] == "tensi"


PLAIN = [
    ("    x = ", 4, "Tensor[[batch_size, seq_len, d_model]]"),
    ("    y = ", 4, "Tensor[[batch_size, seq_len, 10]]"),
    ("    s = ", 4, "Tensor[[batch_size, 10]]"),
    ("    flat = ", 4, "Tensor[[batch_size, seq_len * d_model]]"),
    ("    t = ", 4, "Tensor[[batch_size, d_model, seq_len]]"),
    ("    filled = ", 4, "Tensor[[seq_len, seq_len]]"),
]

ATTN = [
    ("        qkv = ", 8, 1, "Tensor[[batch, seq, 3 * d_model]]"),
    ("        q = ", 8, 1, "Tensor[[batch, n_heads, seq, d_model // n_heads]]"),
    ("        att = ", 8, 2, "Tensor[[batch, n_heads, seq, seq]]"),
    ("        scores = ", 8, 1, "Tensor[[batch, n_heads, seq, d_model // n_heads]]"),
    ("        logits = ", 8, 1, "Tensor[[batch, seq, vocab_size]]"),
    ("    y = m(x)", 4, 1, "Tensor[[batch_size, seq_len, 1000]]"),
]


@pytest.mark.parametrize("needle,at,expected", PLAIN)
def test_plain_hovers(client, ws, needle, at, expected):
    text = (ws / "plain.py").read_text()
    uri = open_and_wait(client, ws, "plain.py")
    h = client.hover_text(uri, *pos(text, needle, at))
    assert norm(expected) in norm(h), h
    assert "_ts_" not in h


@pytest.mark.parametrize("needle,at,nth,expected", ATTN)
def test_attn_hovers(client, ws, needle, at, nth, expected):
    text = (ws / "attn.py").read_text()
    uri = open_and_wait(client, ws, "attn.py")
    h = client.hover_text(uri, *pos(text, needle, at, nth))
    assert norm(expected) in norm(h), h
    assert "_ts_" not in h


def test_value_dependent_mask_is_unshaped(client, ws):
    text = (ws / "plain.py").read_text()
    uri = open_and_wait(client, ws, "plain.py")
    # tensors-only hover mode hides non-shaped hovers entirely
    assert client.hover_text(uri, *pos(text, "    sel = ", 4)) is None


def test_hover_same_at_first_middle_last_char(client, ws):
    text = (ws / "plain.py").read_text()
    uri = open_and_wait(client, ws, "plain.py")
    results = {norm(client.hover_text(uri, *pos(text, "    flat = ", 4 + k))) for k in (0, 1, 3)}
    assert len(results) == 1 and "seq_len*d_model" in results.pop()


def test_no_diagnostics_on_clean_files(client, ws):
    for name in ("plain.py", "attn.py"):
        uri = client.open(ws / name)
        assert client.diagnostics(uri, timeout=30) == []


def test_bad_py_diagnostics(client, ws):
    text = (ws / "bad.py").read_text()
    uri = client.open(ws / "bad.py")
    diags = client.diagnostics(uri, timeout=30, min_count=2)
    assert len(diags) == 2, diags
    by_line = {d["range"]["start"]["line"]: d for d in diags}
    l, c = pos(text, "blk(x)", 4)
    d = by_line[l]
    assert d["range"] == {"start": {"line": l, "character": c}, "end": {"line": l, "character": c + 1}}
    assert "Tensor[[b, 7, v]]" in d["message"] and "Tensor[[b, 7, d]]" in d["message"]
    l2, _ = pos(text, "torch.zeros(b, d) @")
    assert "Tensor[[b, d]]" in by_line[l2]["message"] and "Tensor[[v, 3]]" in by_line[l2]["message"]
    assert all("_ts_" not in d["message"] for d in diags)


def test_cross_file_types_through_background_shadow(client, ws):
    text = (ws / "model.py").read_text()
    uri = open_and_wait(client, ws, "model.py")
    h = client.hover_text(uri, *pos(text, "    h = ", 4))
    assert norm("Tensor[[batch_size, hidden]]") in norm(h), h


def test_cross_file_relative_import_in_package(client, ws):
    text = (ws / "pkg" / "net.py").read_text()
    uri = client.open(ws / "pkg" / "net.py")
    client.diagnostics(uri, timeout=30)
    h = client.hover_text(uri, *pos(text, "    out = ", 4))
    assert norm("Tensor[[batch_size, seq_len, 4 * width]]") in norm(h), h


def test_saving_an_imported_file_updates_hovers(client, ws):
    layers = ws / "layers.py"
    original = layers.read_text()
    text = (ws / "model.py").read_text()
    uri = open_and_wait(client, ws, "model.py")
    try:
        layers.write_text(original.replace("nn.Linear(d_in, d_out)", "nn.Linear(d_in, d_in)"))
        client.notify("textDocument/didSave", {"textDocument": {"uri": layers.resolve().as_uri()}})
        deadline, h = time.time() + 20, ""
        while time.time() < deadline:
            h = client.hover_text(uri, *pos(text, "    h = ", 4)) or ""
            if "d_model" in h and "hidden" not in h:
                break
            time.sleep(0.5)
        assert norm("Tensor[[batch_size, d_model]]") in norm(h), h
    finally:
        layers.write_text(original)
        client.notify("textDocument/didSave", {"textDocument": {"uri": layers.resolve().as_uri()}})


def test_non_ascii_comment_on_rewritten_line(client, ws):
    text = (ws / "attn.py").read_text().replace('"batch seq d_model"]):', '"batch seq d_model"]):  # résumé \U0001f680')
    uri = (ws / "attn.py").resolve().as_uri()
    client.change(uri, text, version=50)
    time.sleep(1.0)
    h = client.hover_text(uri, *pos(text, "        logits = ", 8))
    assert norm("Tensor[[batch, seq, vocab_size]]") in norm(h)
    # hover inside the comment after the emoji does not crash
    line, _ = pos(text, "# résumé")
    end = len(text.split("\n")[line].encode("utf-16-le")) // 2
    client.hover(uri, line, end - 1)
    client.change(uri, (ws / "attn.py").read_text(), version=51)


def test_syntax_error_mid_session_recovers(client, ws):
    good = (ws / "attn.py").read_text()
    uri = (ws / "attn.py").resolve().as_uri()
    broken = good.replace("self.qkv = nn.Linear(d_model, 3 * d_model)", "self.qkv = nn.Linear(d_model, 3 * d_model", 1)
    client.change(uri, broken, version=60)
    time.sleep(1.0)
    h = client.hover_text(uri, *pos(broken, "    m = Attn", 4))  # unaffected function: still answers
    client.change(uri, good, version=61)
    time.sleep(1.0)
    h = client.hover_text(uri, *pos(good, "        logits = ", 8))
    assert norm("Tensor[[batch, seq, vocab_size]]") in norm(h)


def test_hover_latency(client, ws):
    text = (ws / "attn.py").read_text()
    uri = open_and_wait(client, ws, "attn.py")
    p = pos(text, "        scores = ", 8)
    client.hover(uri, *p)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        client.hover(uri, *p)
        times.append(time.perf_counter() - t0)
    p95 = statistics.quantiles(times, n=20)[18]
    assert p95 < 0.3, times


def test_hover_all_mode_shows_int_params(ws):
    c = LspClient(SERVER, ws, env={"TENSI_HOVER": "all", "TENSI_SHADOW_WORKSPACE": "0"})
    try:
        text = (ws / "plain.py").read_text()
        uri = open_and_wait(c, ws, "plain.py")
        h = c.hover_text(uri, *pos(text, "batch_size: int", 2))
        assert "batch_size: int" in h and "_ts_" not in h
    finally:
        c.close()


def test_without_background_shadows_cross_file_is_lost(ws):
    """Documents why shadowWorkspace exists."""
    c = LspClient(SERVER, ws, env={"TENSI_SHADOW_WORKSPACE": "0", "TENSI_HOVER": "all"})
    try:
        text = (ws / "model.py").read_text()
        uri = open_and_wait(c, ws, "model.py")
        h = c.hover_text(uri, *pos(text, "    h = ", 4)) or ""
        assert "hidden" not in h
    finally:
        c.close()


VSCODE_LIKE_CAPS = {
    "general": {"positionEncodings": ["utf-16"], "markdown": {"parser": "marked", "version": "1.1.0"}},
    "workspace": {
        "configuration": True,
        "workspaceFolders": True,
        "didChangeConfiguration": {"dynamicRegistration": True},
        "didChangeWatchedFiles": {"dynamicRegistration": True, "relativePatternSupport": True},
        "diagnostics": {"refreshSupport": True},
        "inlayHint": {"refreshSupport": True},
    },
    "textDocument": {
        "synchronization": {"dynamicRegistration": True, "willSave": True, "willSaveWaitUntil": True, "didSave": True},
        "hover": {"dynamicRegistration": True, "contentFormat": ["markdown", "plaintext"]},
        "publishDiagnostics": {
            "relatedInformation": True,
            "versionSupport": False,
            "tagSupport": {"valueSet": [1, 2]},
            "codeDescriptionSupport": True,
            "dataSupport": True,
        },
        "diagnostic": {"dynamicRegistration": True, "relatedDocumentSupport": False},
        "inlayHint": {"dynamicRegistration": True},
        "completion": {"dynamicRegistration": True},
    },
    "window": {"workDoneProgress": True, "showMessage": {}, "showDocument": {"support": True}},
}


def test_vscode_like_client_still_gets_pushed_diagnostics(ws):
    c = LspClient(SERVER, ws, client_caps=VSCODE_LIKE_CAPS)
    try:
        assert "diagnosticProvider" not in c.init_result["capabilities"]
        uri = c.open(ws / "bad.py")
        diags = c.diagnostics(uri, timeout=30, min_count=2)
        assert len(diags) == 2
        # Pyrefly's own dynamic registrations must not leak to the editor
        assert not [m for m in c.server_requests if m["method"] == "client/registerCapability"]
    finally:
        c.close()


def test_inlay_hints_show_named_shapes(ws):
    c = LspClient(SERVER, ws, env={"TENSI_INLAY_HINTS": "1"}, client_caps=VSCODE_LIKE_CAPS)
    try:
        assert c.init_result["capabilities"].get("inlayHintProvider") is True
        text = (ws / "plain.py").read_text()
        uri = open_and_wait(c, ws, "plain.py")
        n = len(text.split("\n"))
        hints = c.request(
            "textDocument/inlayHint",
            {
                "textDocument": {"uri": uri},
                "range": {"start": {"line": 0, "character": 0}, "end": {"line": n, "character": 0}},
            },
        )
        labels = {
            h["position"]["line"]: (
                h["label"] if isinstance(h["label"], str) else "".join(p["value"] for p in h["label"])
            )
            for h in hints
        }
        flat_line, flat_col = pos(text, "flat", 0)
        assert "seq_len * d_model" in labels.get(flat_line, ""), labels
        assert all("_ts_" not in v for v in labels.values())
        # hint sits right after the variable name in user coordinates
        h = next(h for h in hints if h["position"]["line"] == flat_line)
        assert h["position"]["character"] == flat_col + len("flat")
    finally:
        c.close()


def test_reassigned_parameter_hovers_follow_the_new_shape(client, ws):
    text = (ws / "reassign.py").read_text()
    uri = client.open(ws / "reassign.py")
    assert client.diagnostics(uri, timeout=30) == []
    h = client.hover_text(uri, *pos(text, "    total = ", 4))
    assert norm("Tensor[[batch, d_model]]") in norm(h), h
    h = client.hover_text(uri, *pos(text, "    out = ", 4))
    assert norm("Tensor[[batch, seq, d_model]] | Tensor[[batch, d_model]]") in norm(h) or (
        norm("Tensor[[batch, d_model]] | Tensor[[batch, seq, d_model]]") in norm(h)
    ), h
    # the renamed parameter still hovers under the user's name
    h = client.hover(uri, *pos(text, "x: Float", 0, nth=2))
    assert h is None or "_ts_" not in json.dumps(h)


def test_uncertain_union_hovers_are_hidden():
    from tensi.server import Proxy, Settings

    p = Proxy.__new__(Proxy)
    p.settings = Settings()
    guess = {
        "contents": {"kind": "markdown", "value": "```python\n(variable) x: torch.Tensor[[int, b, t]] | Unknown\n```"}
    }
    legit = {
        "contents": {
            "kind": "markdown",
            "value": "```python\n(variable) x: torch.Tensor[[n]] | torch.Tensor[[2 * n]]\n```",
        }
    }
    assert p.fix_hover(guess, None) is None
    assert p.fix_hover(legit, None) is not None


GPTLIKE = [  # zero annotations in the file: config dims, ModuleDict, solved forward inputs
    ("        q = q.view", 8, "Tensor[[b, heads, t, width // heads]]"),
    ("        scores = ", 8, "Tensor[[b, heads, t, t]]"),
    ("        merged = ", 8, "Tensor[[b, t, width]]"),
    ("        x = self.up(x)", 8, "Tensor[[b, t, 4 * width]]"),
    ("        x = x + self.ff", 8, "Tensor[[b, t, width]]"),
    ("        tok_emb = ", 8, "Tensor[[b, t, width]]"),
    ("        pos_emb = ", 8, "Tensor[[t, width]]"),
    ("            x = layer(x)", 12, "Tensor[[b, t, width]]"),
    ("            logits = self.head(x)", 12, "Tensor[[b, t, vocab]]"),
    ("            logits = self.head(x[:, [-1]", 12, "Tensor[[b, 1, vocab]]"),
    ("            last = ", 12, "Tensor[[b, vocab]]"),
    ("            nxt = ", 12, "Tensor[[b, 1]]"),
]


@pytest.fixture(scope="module")
def gpt_uri(client, ws):
    n = len(client.notifications)
    uri = client.open(ws / "gptlike.py")
    assert client.wait_ready(uri, timeout=90, after=n)
    time.sleep(1.0)
    return uri


@pytest.mark.parametrize("needle,at,expected", GPTLIKE)
def test_gptlike_needs_no_annotations(client, ws, gpt_uri, needle, at, expected):
    text = (ws / "gptlike.py").read_text()
    h = client.hover_text(gpt_uri, *pos(text, needle, at))
    assert norm(expected) in norm(h), h
    assert "_ts_" not in h


def test_gptlike_moduledict_attribute_hover(client, ws, gpt_uri):
    text = (ws / "gptlike.py").read_text()
    c = LspClient(SERVER, ws, env={"TENSI_HOVER": "all", "TENSI_SHADOW_WORKSPACE": "0"})
    try:
        uri = c.open(ws / "gptlike.py")
        assert c.wait_ready(uri, timeout=90)
        time.sleep(1.0)
        h = c.hover_text(uri, *pos(text, "self.trunk.tok(idx)", len("self.trunk.")))
        assert "Embedding[vocab, width]" in h and "_ts_" not in h, h
        h = c.hover_text(uri, *pos(text, "    def __init__(self, config):", len("    def __init__(self, c")))
        assert "_ts_" not in (h or "")
    finally:
        c.close()


def test_gptlike_has_no_diagnostics(client, ws, gpt_uri):
    time.sleep(1.5)
    assert client.diagnostics(gpt_uri, timeout=30) == []


def test_patched_stubs_reach_the_language_server(client, ws):
    """tensi's stub overlay is on Pyrefly's search path in the editor too, not just the CLI."""
    uri = client.open(ws / "norms.py")
    diags = client.diagnostics(uri, timeout=30)
    text = (ws / "norms.py").read_text()
    assert norm("flipped: Tensor[[b, width, t]]") in norm(client.hover_text(uri, *pos(text, "        flipped = ", 8)))
    assert norm("pad: Tensor[[b, 1, width]]") in norm(client.hover_text(uri, *pos(text, "        pad = ", 8)))
    msgs = sorted(d["message"] for d in diags if d.get("code") != "annotate-input")
    assert len(msgs) == 2 and "num_features does not match" in msgs[1], msgs
    hints = [d for d in diags if d.get("code") == "annotate-input"]
    # Broken.forward(x) has no caller and no size unpack: the one place an annotation would help
    assert len(hints) == 1 and "`x`" in hints[0]["message"] and hints[0]["severity"] == 3, hints
    assert hints[0]["range"]["start"] == {"line": 28, "character": 22}
    assert 'def forward(self, x: Shaped[torch.Tensor, "batch seq"]):' in hints[0]["message"], hints
    # the quick fix replaces the bare `torch.Tensor` and adds the import once
    rng = {"start": {"line": 28, "character": 22}, "end": {"line": 28, "character": 23}}
    actions = client.request(
        "textDocument/codeAction",
        {"textDocument": {"uri": uri}, "range": rng, "context": {"diagnostics": hints}},
    )
    fix = next(a for a in actions if a["kind"] == "quickfix")
    edits = fix["edit"]["changes"][uri]
    assert fix["title"] == 'tensi: annotate `x` as Shaped[torch.Tensor, "batch seq"]', fix
    span = {"start": {"line": 28, "character": 23}, "end": {"line": 28, "character": 37}}  # `: torch.Tensor`
    assert {"newText": ': Shaped[torch.Tensor, "batch seq"]', "range": span} in edits, edits
    assert any(e["newText"] == "from jaxtyping import Shaped\n" for e in edits), edits


def test_unsaved_edit_to_an_imported_file_updates_hovers(client, ws):
    """Pyrefly reads imports from disk; tensi mirrors open buffers into the shadow tree instead."""
    layers = ws / "layers.py"
    original = layers.read_text()
    text = (ws / "model.py").read_text()
    uri = open_and_wait(client, ws, "model.py")
    luri = client.open(layers)
    try:
        client.change(luri, original.replace("nn.Linear(d_in, d_out)", "nn.Linear(d_in, d_in)"), 2)
        deadline, h = time.time() + 20, ""
        while time.time() < deadline:
            h = client.hover_text(uri, *pos(text, "    h = ", 4)) or ""
            if "d_model" in h and "hidden" not in h:
                break
            time.sleep(0.5)
        assert norm("Tensor[[batch_size, d_model]]") in norm(h), h
        assert layers.read_text() == original  # nothing written to the user's file
    finally:
        client.notify("textDocument/didClose", {"textDocument": {"uri": luri}})
    deadline = time.time() + 20
    while time.time() < deadline:  # closing without saving restores the disk version
        h = client.hover_text(uri, *pos(text, "    h = ", 4)) or ""
        if "hidden" in h:
            break
        time.sleep(0.5)
    assert norm("Tensor[[batch_size, hidden]]") in norm(h), h


def test_big_workspace_does_not_block_startup(tmp_path):
    """Mirroring the workspace into the shadow tree runs in the background: the editor's
    `initialize` is answered at once, and cross-file shapes still arrive once the tree is built."""
    for i in range(300):
        shutil.copy(FIX / "gptlike.py", tmp_path / f"m{i}.py")
    for f in ["layers.py", "model.py"]:
        shutil.copy(FIX / "xfile" / f, tmp_path / f)
    t0 = time.time()
    c = LspClient(SERVER, tmp_path)
    try:
        assert time.time() - t0 < 5, "initialize waited for the shadow tree"
        text = (tmp_path / "model.py").read_text()
        uri = c.open(tmp_path / "model.py")
        c.diagnostics(uri, timeout=60)
        deadline = time.time() + 60
        hover = ""
        while time.time() < deadline:
            hover = c.hover_text(uri, *pos(text, "    h = ", 4)) or ""
            if norm("Tensor[[batch_size, hidden]]") in norm(hover):
                break
            time.sleep(1)
        assert norm("Tensor[[batch_size, hidden]]") in norm(hover), hover
    finally:
        c.close()


def test_hovering_an_unknown_tensor_says_what_to_annotate(client, ws):
    """A tensor tensi can't shape gets a hover naming the input it depends on and the line to write,
    instead of nothing."""
    uri = client.open(ws / "norms.py")
    client.diagnostics(uri, timeout=30)
    text = (ws / "norms.py").read_text()
    deadline, h = time.time() + 20, ""
    while time.time() < deadline:
        h = (
            client.hover_text(uri, *pos(text, "    def forward(self, x: torch.Tensor):", len("    def forward(self, ")))
            or ""
        )
        if "tensi" in h:
            break
        time.sleep(0.5)
    assert "shape unknown" in h and 'x: Shaped[torch.Tensor, "batch seq"]' in h, h
