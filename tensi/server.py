"""tensi language server: an LSP proxy in front of `pyrefly lsp`.

Editor  <->  tensi  <->  pyrefly lsp
user text     rewrite      shadow text

The editor only ever sees user text, user positions and user names; Pyrefly only
ever sees shadow text. tensi advertises just hover, diagnostics and (optionally)
inlay hints, so it can run beside Pylance / basedpyright without duplicating them.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from . import PYREFLY_VERSION, env, stubs, update
from .cli import drop_imprecise, is_rewrite_artifact, is_shape_diagnostic
from .hints import fix_for, hint_text, unshaped_inputs
from .mapping import PositionMap, clean_json
from .rewrite import find_config_classes, rewrite
from .seeds import solve

PYREFLY_SETTINGS = {"typeCheckingMode": "default", "displayTypeErrors": "force-on"}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "site-packages",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    ".ipynb_checkpoints",
}


# ---------------------------------------------------------------- utilities


def uri_to_path(uri: str) -> str | None:
    """`file:///home/a/m.py` -> `/home/a/m.py`; `file:///c%3A/a/m.py` -> `c:\\a\\m.py` on Windows."""
    p = urlparse(uri)
    if p.scheme != "file":
        return None
    return url2pathname(p.path) if os.name == "nt" else unquote(p.path)


def package_of(path: Path) -> str | None:
    """Dotted package containing `path` (pkg/sub/mod.py -> "pkg.sub"), or None outside a package."""
    parts = []
    d = path.parent
    while (d / "__init__.py").exists():
        parts.append(d.name)
        if d.parent == d:
            break
        d = d.parent
    return ".".join(reversed(parts)) or None


def import_root_of(path: Path) -> Path:
    """Directory from which `path` is importable: the parent of its top-level package."""
    d = path.parent
    while (d / "__init__.py").exists() and d.parent != d:
        d = d.parent
    return d


class ShadowTree:
    """A mirror of the workspace's Python files, rewritten, in a cache folder outside the project.

    Pyrefly resolves imports from disk only (unsaved buffers of other files are ignored), so
    cross-file shapes need the shadows of imported modules on disk. The tree goes first on
    Pyrefly's search path via its `pyrefly.extraPaths` setting.
    """

    def __init__(self, roots: list[Path], max_files: int):
        import hashlib

        key = hashlib.sha1("|".join(str(r) for r in roots).encode()).hexdigest()[:12]
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tensi" / key
        self.dir = base / "tree"
        self.roots = roots
        self.max_files = max_files
        self.config_index: dict = {}
        self.seeds: dict[str, dict] = {}  # path -> solved seeds

    def target(self, path: Path) -> Path:
        return self.dir / path.relative_to(import_root_of(path))

    def write(self, path: Path, text: str | None = None) -> Path | None:
        try:
            if text is None:
                if path.stat().st_size > 500_000:
                    return None
                text = path.read_text(encoding="utf-8")
            t = self.target(path)
            t.parent.mkdir(parents=True, exist_ok=True)
            shadow = (
                rewrite(text, package_of(path), config_index=self.config_index, seeds=self.seeds.get(str(path))).shadow
                if path.suffix == ".py"
                else text
            )
            if not t.exists() or t.read_text(encoding="utf-8") != shadow:
                t.write_text(shadow, encoding="utf-8")
            return t
        except Exception as exc:
            log("shadow tree write failed", path, repr(exc))
            return None

    def files(self):
        for root in self.roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    if fn.endswith(".py"):
                        yield Path(dirpath) / fn

    def index_configs(self) -> dict:
        """Config classes across the workspace, with the dotted module to import them from."""
        index: dict = {}
        for n, path in enumerate(self.files()):
            if n >= self.max_files:
                break
            try:
                if path.stat().st_size > 500_000:
                    continue
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if "int" not in text:
                continue
            rel = path.relative_to(import_root_of(path)).with_suffix("")
            module = ".".join(rel.parts)
            for name, cfg in find_config_classes(text, module).items():
                index.setdefault(name, cfg)
        self.config_index = index
        return index

    def build(self) -> list[Path]:
        """Write every workspace module's shadow (unchanged ones are left alone, so a restart is
        cheap) and remove shadows whose source is gone. Returns the shadow files."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_configs()
        written: list[Path] = []
        capped = False
        for root in self.roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    if fn.endswith((".py", ".pyi")):
                        t = self.write(Path(dirpath) / fn)
                        if t is not None:
                            written.append(t)
                        if len(written) >= self.max_files:
                            capped = True
                            break
                if capped:
                    break
            if capped:
                log("shadow tree cap reached")
                break
        if not capped:
            keep = set(written)
            for dirpath, _dirs, filenames in os.walk(self.dir):
                for fn in filenames:
                    f = Path(dirpath) / fn
                    if f not in keep:
                        try:
                            f.unlink()
                        except OSError:
                            pass
        log(f"shadow tree: {len(written)} files in {self.dir}")
        return written


class Log:
    def __init__(self):
        path = os.environ.get("TENSI_LOG")
        self.f = open(path, "a", encoding="utf-8") if path else None
        self.lock = threading.Lock()

    def __call__(self, *parts):
        if self.f:
            with self.lock:
                self.f.write(time.strftime("%H:%M:%S ") + " ".join(str(p) for p in parts) + "\n")
                self.f.flush()


log = Log()


class Channel:
    """Content-Length framed JSON-RPC over a pair of binary streams."""

    def __init__(self, reader, writer, name: str):
        self.reader, self.writer, self.name = reader, writer, name
        self.lock = threading.Lock()

    def read(self) -> dict | None:
        headers = {}
        while True:
            line = self.reader.readline()
            if not line:
                return None
            line = line.decode("ascii", "replace").strip()
            if not line:
                break
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length", "0"))
        body = b""
        while len(body) < n:
            chunk = self.reader.read(n - len(body))
            if not chunk:
                return None
            body += chunk
        msg = json.loads(body)
        log(f"<- {self.name}", json.dumps(msg)[:2000])
        return msg

    def write(self, msg: dict):
        body = json.dumps(msg).encode()
        log(f"-> {self.name}", json.dumps(msg)[:2000])
        with self.lock:
            self.writer.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
            self.writer.flush()


@dataclass
class Doc:
    uri: str
    user_text: str
    shadow: str
    pm: PositionMap
    version: int  # the editor's version
    editor_open: bool
    jaxtyping_lines: set[int] = field(default_factory=set)
    pversion: int = 0  # the version Pyrefly saw (re-seeding changes the shadow, not the editor version)
    hints: list = field(default_factory=list)  # forward inputs one annotation would shape

    @classmethod
    def build(
        cls,
        uri: str,
        text: str,
        version: int,
        editor_open: bool,
        seeds: dict | None = None,
        config_index: dict | None = None,
    ) -> Doc:
        path = uri_to_path(uri)
        res = rewrite(
            text, package_of(Path(path)) if path else None, cache_key=path, config_index=config_index, seeds=seeds
        )
        jl = {i for i, l in enumerate(text.split("\n")) if "jaxtyping" in l and "import" in l}
        hints = unshaped_inputs(text, seeds) if seeds is not None else []
        return cls(
            uri, text, res.shadow, PositionMap(text, res.shadow, res.edits), version, editor_open, jl, hints=hints
        )


@dataclass
class Settings:
    hover: str = "tensors"  # "tensors" | "all"
    diagnostics: str = "shapes"  # "shapes" | "all" | "off"
    inlay_hints: bool = False
    shadow_workspace: bool = True
    max_workspace_files: int = 2000

    @classmethod
    def from_init(cls, opts: dict | None) -> Settings:
        s = cls()
        env = os.environ
        s.hover = env.get("TENSI_HOVER", s.hover)
        s.diagnostics = env.get("TENSI_DIAGNOSTICS", s.diagnostics)
        s.inlay_hints = env.get("TENSI_INLAY_HINTS", "0") == "1"
        s.shadow_workspace = env.get("TENSI_SHADOW_WORKSPACE", "1") == "1"
        opts = opts or {}
        s.hover = opts.get("hover", s.hover)
        s.diagnostics = opts.get("diagnostics", s.diagnostics)
        s.inlay_hints = bool(opts.get("inlayHints", s.inlay_hints))
        s.shadow_workspace = bool(opts.get("shadowWorkspace", s.shadow_workspace))
        s.max_workspace_files = int(opts.get("maxWorkspaceFiles", s.max_workspace_files))
        return s


# ---------------------------------------------------------------- the proxy


class Proxy:
    def __init__(self, pyrefly_cmd: list[str]):
        self.pyrefly_cmd = pyrefly_cmd
        self.editor = Channel(sys.stdin.buffer, sys.stdout.buffer, "editor")
        self.docs: dict[str, Doc] = {}
        self.docs_lock = threading.RLock()
        self.pending: dict = {}  # editor request id -> (method, uri, pm)
        self.pyrefly_pending: dict = {}  # pyrefly->editor request id -> method
        self.settings = Settings()
        self.editor_supports_config = False
        self.init_params: dict | None = None
        self.shutting_down = False
        self.restarts: list[float] = []
        self.child: subprocess.Popen | None = None
        self.server: Channel | None = None
        self.ready = threading.Event()
        self.tree: ShadowTree | None = None
        self.stubs_dir = stubs.overlay_dir()
        self._pyrefly_config: Path | None = None
        self._pyrefly_config_done = False
        self.tree_lock = threading.Lock()
        self.tree_timers: dict[str, threading.Timer] = {}
        self.seeds: dict[str, dict] = {}  # uri -> solved seeds
        self.pcounter = 0
        self.hover_pos: dict = {}  # hover request id -> the editor's position
        self.solve_queue: list[str] = []
        self.solve_cv = threading.Condition()
        threading.Thread(target=self.solver_loop, daemon=True).start()

    def notify_setup(self):
        """Once per session, off the request path: a Pyrefly other than the pinned one (a warning,
        since shapes can quietly go missing), and a newer tensi on PyPI (with its install command)."""
        for kind, text in (
            (2, update.pyrefly_mismatch(self.pyrefly_cmd[0])),
            (3, update.update_message(update.latest())),
        ):
            if text:
                self.editor.write(
                    {"jsonrpc": "2.0", "method": "window/showMessage", "params": {"type": kind, "message": text}}
                )

    # ---------- child process ----------
    def start_child(self):
        self.child = subprocess.Popen(
            self.pyrefly_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.server = Channel(self.child.stdout, self.child.stdin, "pyrefly")
        threading.Thread(target=self.pyrefly_loop, args=(self.server, self.child), daemon=True).start()

    def restart_child(self):
        now = time.time()
        self.restarts = [t for t in self.restarts if now - t < 60] + [now]
        if len(self.restarts) > 3:
            self.editor.write(
                {
                    "jsonrpc": "2.0",
                    "method": "window/showMessage",
                    "params": {"type": 1, "message": "tensi: Pyrefly keeps crashing; giving up."},
                }
            )
            return
        for rid in list(self.pending):
            self.editor.write({"jsonrpc": "2.0", "id": rid, "error": {"code": -32803, "message": "pyrefly restarted"}})
        self.pending.clear()
        self.start_child()
        params = self.child_init_params(self.init_params or {})
        self.server.write({"jsonrpc": "2.0", "id": "tensi-reinit", "method": "initialize", "params": params})
        # the reply is swallowed in pyrefly_loop; then re-send state
        self.ready.wait(30)
        with self.docs_lock:
            for d in self.docs.values():
                self.send_open(d)

    def child_init_params(self, params: dict) -> dict:
        p = json.loads(json.dumps(params))
        caps = p.setdefault("capabilities", {})
        ws = caps.setdefault("workspace", {})
        ws["configuration"] = True
        td = caps.setdefault("textDocument", {})
        td.pop("diagnostic", None)  # make Pyrefly push diagnostics
        td.setdefault("publishDiagnostics", {})["relatedInformation"] = True
        caps.setdefault("general", {})["positionEncodings"] = ["utf-16"]
        p.pop("initializationOptions", None)
        return p

    # ---------- editor -> pyrefly ----------
    def editor_loop(self):
        while True:
            msg = self.editor.read()
            if msg is None:
                break
            try:
                self.on_editor(msg)
            except Exception as exc:  # never die on one bad message
                log("editor handler error", repr(exc))
                if "id" in msg and "method" in msg:
                    self.editor.write(
                        {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32603, "message": f"tensi: {exc}"}}
                    )
        self.shutting_down = True
        if self.child:
            self.child.kill()

    def on_editor(self, msg: dict):
        method = msg.get("method")
        if method is None:  # a response to a request Pyrefly made
            rid = msg.get("id")
            kind = self.pyrefly_pending.pop(rid, None)
            if kind == "workspace/configuration":
                items = msg.get("result") or []
                msg = {**msg, "result": [self.merge_config(x) for x in items]}
            self.server.write(msg)
            return

        if method == "initialize":
            self.init_params = msg["params"]
            self.settings = Settings.from_init(msg["params"].get("initializationOptions"))
            self.editor_supports_config = bool(
                msg["params"].get("capabilities", {}).get("workspace", {}).get("configuration")
            )
            if self.settings.shadow_workspace and self.workspace_roots():
                self.tree = ShadowTree(self.workspace_roots(), self.settings.max_workspace_files)
                self.tree.dir.mkdir(parents=True, exist_ok=True)
                # A big workspace takes seconds to mirror; the editor gets Pyrefly right away and
                # cross-file shapes arrive when the tree is done.
                threading.Thread(target=self.build_tree, daemon=True).start()
            self.pending[msg["id"]] = ("initialize", None, None)
            self.server.write({**msg, "params": self.child_init_params(msg["params"])})
            return
        if method == "initialized":
            self.server.write(msg)
            threading.Thread(target=self.notify_setup, daemon=True).start()
            return
        if method == "shutdown":
            self.shutting_down = True
            self.pending[msg["id"]] = ("shutdown", None, None)
            self.server.write(msg)
            return
        if method == "exit":
            self.shutting_down = True
            try:
                self.server.write(msg)
                self.child.wait(timeout=5)
            except Exception:
                self.child.kill()
            sys.exit(0)

        if method == "textDocument/didOpen":
            td = msg["params"]["textDocument"]
            if td.get("languageId", "python") != "python":
                return
            with self.docs_lock:
                doc = self.build_doc(td["uri"], td["text"], td["version"])
                self.docs[td["uri"]] = doc
                self.send_open(doc)
            self.request_solve(td["uri"])
            return
        if method == "textDocument/didChange":
            uri = msg["params"]["textDocument"]["uri"]
            with self.docs_lock:
                old = self.docs.get(uri)
                if old is None:
                    return
                text = old.user_text
                for ch in msg["params"]["contentChanges"]:
                    if "range" in ch:  # we advertise Full sync, but be lenient
                        text = apply_incremental(text, ch)
                    else:
                        text = ch["text"]
                doc = self.build_doc(uri, text, msg["params"]["textDocument"]["version"])
                self.docs[uri] = doc
                self.send_change(doc)
            self.schedule_tree_write(uri)
            return
        if method == "textDocument/didClose":
            uri = msg["params"]["textDocument"]["uri"]
            with self.docs_lock:
                if self.docs.pop(uri, None) is None:
                    return
                self.server.write(msg)
            self.schedule_tree_write(uri)  # closed without saving: importers go back to the disk text
            self.editor.write(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {"uri": uri, "diagnostics": []},
                }
            )
            return
        if method == "textDocument/didSave":
            self.server.write(msg)
            if msg["params"]["textDocument"]["uri"] in self.docs:
                self.request_solve(msg["params"]["textDocument"]["uri"])
            path = uri_to_path(msg["params"]["textDocument"]["uri"])
            if self.tree is not None and path:
                t = self.tree.write(Path(path))
                if t is not None:
                    self.server.write(
                        {
                            "jsonrpc": "2.0",
                            "method": "workspace/didChangeWatchedFiles",
                            "params": {"changes": [{"uri": t.as_uri(), "type": 2}]},
                        }
                    )
            return

        if method == "textDocument/hover":
            self.hover_pos[msg["id"]] = msg["params"]["position"]
            self.forward_positional(msg, "hover")
            return
        if method == "textDocument/codeAction":
            self.editor.write({"jsonrpc": "2.0", "id": msg["id"], "result": self.code_actions(msg["params"])})
            return
        if method == "textDocument/inlayHint":
            if not self.settings.inlay_hints:
                self.editor.write({"jsonrpc": "2.0", "id": msg["id"], "result": []})
                return
            uri = msg["params"]["textDocument"]["uri"]
            doc = self.docs.get(uri)
            if doc is None:
                self.editor.write({"jsonrpc": "2.0", "id": msg["id"], "result": []})
                return
            rng = msg["params"]["range"]
            params = {
                **msg["params"],
                "range": {
                    "start": {
                        "line": rng["start"]["line"],
                        "character": doc.pm.user_to_shadow(rng["start"]["line"], rng["start"]["character"]),
                    },
                    "end": {
                        "line": rng["end"]["line"],
                        "character": doc.pm.user_to_shadow(rng["end"]["line"], rng["end"]["character"]),
                    },
                },
            }
            self.pending[msg["id"]] = ("inlayHint", uri, doc.pm)
            self.server.write({**msg, "params": params})
            return
        if method == "$/cancelRequest":
            self.server.write(msg)
            return
        if method == "workspace/didChangeConfiguration" or method.startswith("workspace/"):
            if "id" in msg:
                self.editor.write({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            else:
                self.server.write(msg)
            return
        # anything else we did not advertise
        if "id" in msg:
            self.editor.write({"jsonrpc": "2.0", "id": msg["id"], "result": None})

    def forward_positional(self, msg: dict, kind: str):
        uri = msg["params"]["textDocument"]["uri"]
        doc = self.docs.get(uri)
        if doc is None:
            self.editor.write({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            return
        pos = msg["params"]["position"]
        mapped = {"line": pos["line"], "character": doc.pm.user_to_shadow(pos["line"], pos["character"])}
        self.pending[msg["id"]] = (kind, uri, doc.pm)
        self.server.write({**msg, "params": {**msg["params"], "position": mapped}})

    def _next_pversion(self, doc: Doc):
        self.pcounter += 1
        doc.pversion = self.pcounter

    def send_open(self, doc: Doc):
        self._next_pversion(doc)
        self.server.write(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": doc.uri,
                        "languageId": "python",
                        "version": doc.pversion,
                        "text": doc.shadow,
                    }
                },
            }
        )

    def send_change(self, doc: Doc):
        self._next_pversion(doc)
        self.server.write(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": doc.uri, "version": doc.pversion},
                    "contentChanges": [{"text": doc.shadow}],
                },
            }
        )

    def build_doc(self, uri: str, text: str, version: int) -> Doc:
        return Doc.build(
            uri,
            text,
            version,
            editor_open=True,
            seeds=self.seeds.get(uri),
            config_index=self.tree.config_index if self.tree else None,
        )

    def build_tree(self):
        tree = self.tree
        try:
            written = tree.build()
        except Exception as exc:
            log("shadow tree build failed", repr(exc))
            return
        self.ready.wait(60)  # Pyrefly has answered initialize
        # The build wrote files as saved on disk; open buffers win, and open documents get the
        # workspace's config classes now that they are indexed.
        with self.docs_lock:
            uris = list(self.docs)
        for uri in uris:
            self.write_tree_now(uri, notify=False)
            with self.docs_lock:
                cur = self.docs.get(uri)
                if cur is not None:
                    nd = self.build_doc(uri, cur.user_text, cur.version)
                    if nd.shadow != cur.shadow:
                        self.docs[uri] = nd
                        self.send_change(nd)
        if written:
            self.server.write(
                {
                    "jsonrpc": "2.0",
                    "method": "workspace/didChangeWatchedFiles",
                    "params": {"changes": [{"uri": t.as_uri(), "type": 1} for t in written]},
                }
            )
        # solve forward seeds for workspace modules in the background, so modules imported
        # from other files carry their shapes through the shadow tree
        n = 0
        for f in tree.files():
            try:
                if "def forward" not in f.read_text(encoding="utf-8", errors="replace"):
                    continue
            except OSError:
                continue
            self.request_solve(f.resolve().as_uri())
            n += 1
            if n >= 30:
                break

    # ---------- unsaved buffers -> shadow tree ----------
    def schedule_tree_write(self, uri: str, delay: float = 0.3):
        """Pyrefly resolves imports from disk only, so other files would see this module as last
        saved. Mirror the buffer (or, once closed, the disk text) into the shadow tree, debounced."""
        if self.tree is None or not uri_to_path(uri):
            return
        with self.tree_lock:
            old = self.tree_timers.pop(uri, None)
            if old is not None:
                old.cancel()
            t = threading.Timer(delay, self.write_tree_now, (uri,))
            t.daemon = True
            self.tree_timers[uri] = t
            t.start()

    def write_tree_now(self, uri: str, notify: bool = True):
        with self.tree_lock:
            self.tree_timers.pop(uri, None)
        with self.docs_lock:
            doc = self.docs.get(uri)
            text = doc.user_text if doc else None
        path = uri_to_path(uri)
        t = self.tree.write(Path(path), text) if self.tree is not None and path else None
        if t is not None and notify:
            self.server.write(
                {
                    "jsonrpc": "2.0",
                    "method": "workspace/didChangeWatchedFiles",
                    "params": {"changes": [{"uri": t.as_uri(), "type": 2}]},
                }
            )

    # ---------- the seed solver (background) ----------
    def request_solve(self, uri: str):
        with self.solve_cv:
            if uri not in self.solve_queue:
                self.solve_queue.append(uri)
            self.solve_cv.notify()

    def solver_loop(self):
        while True:
            with self.solve_cv:
                while not self.solve_queue:
                    self.solve_cv.wait()
                uri = self.solve_queue.pop(0)
            try:
                self.solve_one(uri)
            except Exception as exc:
                log("solver failed", uri, repr(exc))

    def solve_one(self, uri: str):
        path = uri_to_path(uri)
        with self.docs_lock:
            doc = self.docs.get(uri)
            text = doc.user_text if doc else None
        if text is None and path and os.path.exists(path):  # a background (not open) workspace file
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        if not path or text is None:
            return
        self.editor.write({"jsonrpc": "2.0", "method": "$/tensi/status", "params": {"uri": uri, "state": "solving"}})
        p = Path(path)
        workdir, search = p.parent, None
        if self.tree is not None:
            try:
                workdir = self.tree.target(p).parent
                workdir.mkdir(parents=True, exist_ok=True)
                search = [str(self.tree.dir)]
            except ValueError:
                pass
        seeds = solve(
            p,
            text,
            package=package_of(p),
            config_index=self.tree.config_index if self.tree else None,
            search_paths=search,
            workdir=workdir,
        )
        changed = seeds != self.seeds.get(uri)
        self.seeds[uri] = seeds
        if self.tree is not None:
            self.tree.seeds[path] = seeds
        if changed:
            with self.docs_lock:
                cur = self.docs.get(uri)
                if cur is not None:
                    nd = self.build_doc(uri, cur.user_text, cur.version)
                    self.docs[uri] = nd
                    self.send_change(nd)
            if self.tree is not None:
                t = self.tree.write(p)
                if t is not None:
                    self.server.write(
                        {
                            "jsonrpc": "2.0",
                            "method": "workspace/didChangeWatchedFiles",
                            "params": {"changes": [{"uri": t.as_uri(), "type": 2}]},
                        }
                    )
        self.editor.write(
            {
                "jsonrpc": "2.0",
                "method": "$/tensi/status",
                "params": {
                    "uri": uri,
                    "state": "ready",
                    "seeds": len(seeds),
                    "toAnnotate": len(self.docs[uri].hints) if uri in self.docs else 0,
                },
            }
        )

    def pyrefly_config(self) -> Path | None:
        """The config tensi hands Pyrefly's language server (see `env.lsp_config`), made once."""
        if not self._pyrefly_config_done:
            self._pyrefly_config_done = True
            roots = self.workspace_roots()
            if roots and self.tree is not None:
                self._pyrefly_config = env.lsp_config(roots[0], self.tree.dir.parent)
                log("pyrefly config", self._pyrefly_config)
        return self._pyrefly_config

    def merge_config(self, item):
        item = item if isinstance(item, dict) else {}
        pyrefly = {**(item.get("pyrefly") or {}), **PYREFLY_SETTINGS}
        config = self.pyrefly_config()
        if config is not None and not pyrefly.get("configPath"):
            pyrefly["configPath"] = str(config)
        ours = ([str(self.tree.dir)] if self.tree is not None else []) + (
            [str(self.stubs_dir)] if self.stubs_dir is not None else []
        )
        if ours:
            pyrefly["extraPaths"] = ours + [p for p in (pyrefly.get("extraPaths") or []) if p not in ours]
        return {**item, "pyrefly": pyrefly}

    def workspace_roots(self) -> list[Path]:
        p = self.init_params or {}
        roots = [uri_to_path(f["uri"]) for f in (p.get("workspaceFolders") or [])]
        if not roots and p.get("rootUri"):
            roots = [uri_to_path(p["rootUri"])]
        return [Path(r) for r in roots if r]

    # ---------- pyrefly -> editor ----------
    def pyrefly_loop(self, chan: Channel, child: subprocess.Popen):
        while True:
            msg = chan.read()
            if msg is None:
                break
            try:
                self.on_pyrefly(msg)
            except Exception as exc:
                log("pyrefly handler error", repr(exc))
        if not self.shutting_down and child is self.child:
            log("pyrefly exited unexpectedly; restarting")
            self.ready.clear()
            threading.Thread(target=self.restart_child, daemon=True).start()

    def on_pyrefly(self, msg: dict):
        method = msg.get("method")
        if method is None:  # response to an editor request
            rid = msg.get("id")
            if rid == "tensi-reinit":
                self.server.write({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                self.ready.set()
                return
            kind, uri, pm = self.pending.pop(rid, (None, None, None))
            if kind == "initialize":
                self.ready.set()
                msg = {**msg, "result": self.editor_init_result(msg.get("result") or {})}
            elif kind == "hover":
                result = self.fix_hover(msg.get("result"), pm)
                pos = self.hover_pos.pop(rid, None)
                if result is None and pos is not None:
                    result = self.unknown_shape_hover(msg.get("result"), uri, pos)
                msg = {**msg, "result": result}
            elif kind == "inlayHint":
                msg = {**msg, "result": self.fix_inlay(msg.get("result"), pm)}
            self.editor.write(msg)
            return

        if "id" in msg:  # a request from Pyrefly to the client
            if method == "workspace/configuration":
                if self.editor_supports_config:
                    self.pyrefly_pending[msg["id"]] = method
                    self.editor.write(msg)
                else:
                    items = msg["params"].get("items", [])
                    self.server.write(
                        {"jsonrpc": "2.0", "id": msg["id"], "result": [self.merge_config(None) for _ in items]}
                    )
                return
            if method == "workspace/inlayHint/refresh" and self.settings.inlay_hints:
                self.pyrefly_pending[msg["id"]] = method
                self.editor.write(msg)
                return
            # registerCapability, progress, other refreshes: answer ourselves so Pyrefly's
            # other features never reach the editor.
            self.server.write({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            return

        if method == "textDocument/publishDiagnostics":
            out = self.fix_diagnostics(msg["params"])
            if out is not None:
                self.editor.write({"jsonrpc": "2.0", "method": method, "params": out})
            return
        if method == "window/logMessage":
            self.editor.write(clean_json(msg))
            return
        if method == "window/showMessage" and msg["params"].get("type") == 1:
            self.editor.write(clean_json(msg))
            return
        # $/progress, telemetry, other showMessage: dropped

    def editor_init_result(self, result: dict) -> dict:
        caps = {
            "positionEncoding": "utf-16",
            "textDocumentSync": {"openClose": True, "change": 1, "save": {"includeText": False}},
            "hoverProvider": True,
            "codeActionProvider": {"codeActionKinds": ["quickfix", "source"]},
        }
        if self.settings.inlay_hints:
            caps["inlayHintProvider"] = True
        from . import __version__

        return {"capabilities": caps, "serverInfo": {"name": "tensi", "version": __version__}}

    # ---------- annotation hints: quick fixes and hovers ----------
    def _hint_edits(self, doc: Doc, hints: list) -> list[dict]:
        """TextEdits that annotate `hints` and add the imports they need, once."""
        lines = doc.user_text.split("\n")

        def u16(line: int, cp: int) -> int:
            text = lines[line] if line < len(lines) else ""
            return len(text[:cp].encode("utf-16-le")) // 2

        edits, imports, import_line = [], {}, 0
        for h in hints:
            fix = fix_for(doc.user_text, h)
            import_line = fix.import_line
            imports.update(dict.fromkeys(fix.imports))
            start = h.col + len(h.param)
            end = h.ann_end if h.ann_end >= 0 else start
            edits.append(
                {
                    "range": {
                        "start": {"line": h.line - 1, "character": u16(h.line - 1, start)},
                        "end": {"line": h.line - 1, "character": u16(h.line - 1, end)},
                    },
                    "newText": f": {fix.annotation}",
                }
            )
        if imports:
            at = {"line": import_line, "character": 0}
            edits.append({"range": {"start": at, "end": at}, "newText": "".join(f"{i}\n" for i in imports)})
        return edits

    def code_actions(self, params: dict) -> list:
        """A quick fix per input to annotate (on its hint), and one action for all of them."""
        uri = params["textDocument"]["uri"]
        doc = self.docs.get(uri)
        if doc is None or not doc.hints:
            return []
        first, last = params["range"]["start"]["line"], params["range"]["end"]["line"]
        diags = [d for d in params.get("context", {}).get("diagnostics", []) if d.get("code") == "annotate-input"]
        out = []
        for h in doc.hints:
            if not (first <= h.line - 1 <= last):
                continue
            fix = fix_for(doc.user_text, h)
            match = [d for d in diags if d["range"]["start"]["line"] == h.line - 1]
            out.append(
                {
                    "title": f"tensi: annotate `{h.param}` as {fix.annotation}",
                    "kind": "quickfix",
                    "isPreferred": True,
                    **({"diagnostics": match} if match else {}),
                    "edit": {"changes": {uri: self._hint_edits(doc, [h])}},
                }
            )
        if len(doc.hints) > 1:
            out.append(
                {
                    "title": f"tensi: annotate all {len(doc.hints)} inputs in this file",
                    "kind": "source",
                    "edit": {"changes": {uri: self._hint_edits(doc, doc.hints)}},
                }
            )
        return out

    def unknown_shape_hover(self, result, uri: str | None, pos: dict) -> dict | None:
        """Hovering a tensor tensi can't shape: say which input it depends on and what to write,
        instead of showing nothing."""
        doc = self.docs.get(uri) if uri else None
        if not result or doc is None or not doc.hints:
            return None
        contents = result.get("contents")
        text = contents.get("value", "") if isinstance(contents, dict) else json.dumps(contents)
        if not re.search(r"\bTensor\b", text) or re.search(r"\bTensor\[\[", text):
            return None
        line = pos["line"] + 1
        try:
            tree = ast.parse(doc.user_text)
        except SyntaxError:
            return None
        fns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= (n.end_lineno or n.lineno)
        ]
        fn = max(fns, key=lambda n: n.lineno) if fns else None
        mine = [h for h in doc.hints if fn is not None and h.def_line == fn.lineno]
        if mine:
            parts = []
            for h in mine:
                fix = fix_for(doc.user_text, h)
                parts.append(f"```python\n{fix.snippet}\n```")
            names = ", ".join(f"`{h.param}`" for h in mine)
            value = (
                f"**tensi: shape unknown.** It depends on {names}, whose shape comes from outside this file. "
                "Annotate it (the quick fix on the hint writes this for you):\n\n" + "\n".join(parts)
            )
        else:
            roots = ", ".join(f"{h.where}({h.param})" for h in doc.hints[:3]) + (" …" if len(doc.hints) > 3 else "")
            value = (
                "**tensi: shape unknown.** tensi couldn't work this one out. If it comes from an input of "
                f"this file, annotating the inputs tensi flags ({roots}) shapes it: see their hints, or the "
                'quick fix "annotate all inputs".'
            )
        return {"contents": {"kind": "markdown", "value": value}}

    # ---------- result fixers ----------
    def fix_hover(self, result, pm: PositionMap | None):
        if not result:
            return result
        contents = result.get("contents")
        text = contents.get("value", "") if isinstance(contents, dict) else json.dumps(contents)
        if self.settings.hover == "tensors":
            if not re.search(r"\bTensor\[\[", text):
                return None
            # A tensor type unioned with Any/Unknown is Pyrefly guessing among candidates
            # (e.g. attribute access through an nn.ModuleDict): show nothing rather than a guess.
            first = text.split("```")[1] if text.count("```") >= 2 else text
            if re.search(r"\| (Any|Unknown)\b|\bUnknown \|", first):
                return None
        out = clean_json(result)
        if pm is not None and out.get("range"):
            out["range"] = pm.shadow_range_to_user(out["range"])
        if isinstance(out.get("contents"), dict) and isinstance(out["contents"].get("value"), str):
            # drop Pyrefly's "Go to [Tensor](file://...stubs...)" footer: it points into stubs
            out["contents"]["value"] = re.sub(r"\n\nGo to \[[^\]]*\]\([^)]*\)\s*$", "", out["contents"]["value"])
        return out

    def fix_inlay(self, result, pm: PositionMap | None):
        if not result or pm is None:
            return result or []
        out = []
        for hint in result:
            label = hint.get("label")
            text = label if isinstance(label, str) else "".join(p.get("value", "") for p in label or [])
            if self.settings.hover == "tensors" and "Tensor[[" not in text:
                continue
            h = clean_json(hint)
            pos = h["position"]
            h["position"] = {"line": pos["line"], "character": pm.shadow_to_user(pos["line"], pos["character"])}
            h.pop("textEdits", None)  # would insert shadow-side text
            if isinstance(h.get("label"), list):
                for part in h["label"]:
                    part.pop("location", None)
            out.append(h)
        return out

    def fix_diagnostics(self, params: dict) -> dict | None:
        uri = params["uri"]
        with self.docs_lock:
            doc = self.docs.get(uri)
        if doc is None or not doc.editor_open:
            return None
        if params.get("version") is not None and params["version"] != doc.pversion:
            return None  # stale: computed for an older shadow
        mode = self.settings.diagnostics
        diags = []
        if mode != "off":
            raw = [d for d in params.get("diagnostics", []) if not is_rewrite_artifact(d.get("message", ""))]
            if mode == "shapes":
                raw = drop_imprecise(raw, "message", lambda d: json.dumps(d["range"], sort_keys=True))
            for d in raw:
                raw_msg = d.get("message", "")
                if mode == "shapes" and not is_shape_diagnostic(raw_msg, d.get("code")):
                    continue
                if d.get("code") == "unused-import" and d["range"]["start"]["line"] in doc.jaxtyping_lines:
                    continue  # the rewrite consumed the jaxtyping dtype names
                nd = clean_json(d)
                nd["range"] = doc.pm.shadow_range_to_user(d["range"])
                nd["source"] = "tensi"
                rel = []
                for ri in nd.get("relatedInformation") or []:
                    other = self.docs.get(ri["location"]["uri"])
                    if other is not None:
                        ri["location"]["range"] = other.pm.shadow_range_to_user(ri["location"]["range"])
                    rel.append(ri)
                if rel:
                    nd["relatedInformation"] = rel
                diags.append(nd)
        if mode != "off":
            lines = doc.user_text.split("\n")
            for h in doc.hints:
                line, col, param = h.line, h.col, h.param
                prefix = lines[line - 1][:col] if line - 1 < len(lines) else ""
                c16 = len(prefix.encode("utf-16-le")) // 2
                n16 = len(param.encode("utf-16-le")) // 2
                diags.append(
                    {
                        "range": {
                            "start": {"line": line - 1, "character": c16},
                            "end": {"line": line - 1, "character": c16 + n16},
                        },
                        "severity": 3,
                        "source": "tensi",
                        "code": "annotate-input",
                        "message": hint_text(h, fix_for(doc.user_text, h)),
                    }
                )
        out = {"uri": uri, "diagnostics": diags}
        if params.get("version") is not None:
            out["version"] = doc.version
        return out


def apply_incremental(text: str, change: dict) -> str:
    """Apply one ranged contentChange (UTF-16 positions) to `text`."""
    lines = text.split("\n")

    def offset(pos):
        line = min(pos["line"], len(lines) - 1) if lines else 0
        before = sum(len(l) + 1 for l in lines[: pos["line"]]) if pos["line"] < len(lines) else len(text)
        if pos["line"] >= len(lines):
            return len(text)
        s = lines[line]
        units, cp = 0, 0
        while cp < len(s) and units < pos["character"]:
            units += 2 if ord(s[cp]) > 0xFFFF else 1
            cp += 1
        return before + cp

    a, b = offset(change["range"]["start"]), offset(change["range"]["end"])
    return text[:a] + change["text"] + text[b:]


def serve(pyrefly: str = "pyrefly") -> int:
    proxy = Proxy([pyrefly, "lsp"])
    try:
        proxy.start_child()
    except OSError as exc:
        print(
            f"tensi: cannot start Pyrefly ({pyrefly}): {exc}. Install it with "
            f"`pip install pyrefly=={PYREFLY_VERSION}` in this environment, or set TENSI_PYREFLY.",
            file=sys.stderr,
        )
        return 1
    proxy.editor_loop()
    return 0
