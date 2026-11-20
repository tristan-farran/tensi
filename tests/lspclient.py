"""Minimal scripted LSP client for tests (talks JSON-RPC over a child's stdio)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path


class LspClient:
    def __init__(self, cmd: list[str], root: str | Path, env: dict | None = None, client_caps: dict | None = None):
        self.root = Path(root).resolve()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.root,
            env={**os.environ, **(env or {})},
        )
        self._next = 0
        self._responses: dict[int, dict] = {}
        self._cv = threading.Condition()
        self.notifications: list[dict] = []
        self.server_requests: list[dict] = []
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._stderr = []
        threading.Thread(target=lambda: self._stderr.extend(self.proc.stderr), daemon=True).start()
        self.init_result = self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self.root.as_uri(),
                "workspaceFolders": [{"uri": self.root.as_uri(), "name": self.root.name}],
                "capabilities": client_caps
                or {
                    "textDocument": {
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "publishDiagnostics": {"relatedInformation": True},
                        "synchronization": {"didSave": True},
                    },
                    "workspace": {"configuration": True, "workspaceFolders": True},
                },
            },
            timeout=60,
        )
        self.notify("initialized", {})

    # ---- transport ----
    def _send(self, msg: dict):
        body = json.dumps(msg).encode()
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.proc.stdin.flush()

    def _read(self):
        out = self.proc.stdout
        while True:
            headers = {}
            while True:
                line = out.readline()
                if not line:
                    return
                line = line.decode().strip()
                if not line:
                    break
                k, v = line.split(":", 1)
                headers[k.lower()] = v.strip()
            msg = json.loads(out.read(int(headers["content-length"])))
            with self._cv:
                if "id" in msg and "method" not in msg:
                    self._responses[msg["id"]] = msg
                elif "id" in msg:
                    self.server_requests.append(msg)
                    self._answer_server_request(msg)
                else:
                    self.notifications.append(msg)
                self._cv.notify_all()

    def _answer_server_request(self, msg):
        result = None
        if msg["method"] == "workspace/configuration":
            result = [None for _ in msg["params"].get("items", [])]
        self._send({"jsonrpc": "2.0", "id": msg["id"], "result": result})

    # ---- API ----
    def request(self, method: str, params, timeout: float = 30):
        self._next += 1
        rid = self._next
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        end = time.time() + timeout
        with self._cv:
            while rid not in self._responses:
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError(f"{method} timed out; stderr: {b''.join(self._stderr)[-2000:]!r}")
                self._cv.wait(left)
            msg = self._responses.pop(rid)
        if "error" in msg:
            raise RuntimeError(msg["error"])
        return msg.get("result")

    def notify(self, method: str, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def open(self, path: str | Path, text: str | None = None, version: int = 1) -> str:
        p = Path(path).resolve()
        uri = p.as_uri()
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": version,
                    "text": p.read_text() if text is None else text,
                }
            },
        )
        return uri

    def change(self, uri: str, text: str, version: int):
        self.notify(
            "textDocument/didChange",
            {"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]},
        )

    def hover(self, uri: str, line: int, character: int):
        return self.request(
            "textDocument/hover", {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}}
        )

    def hover_text(self, uri: str, line: int, character: int) -> str | None:
        r = self.hover(uri, line, character)
        if not r:
            return None
        c = r["contents"]
        if isinstance(c, dict):
            return c.get("value")
        if isinstance(c, list):
            return "\n".join(x if isinstance(x, str) else x.get("value", "") for x in c)
        return c

    def diagnostics(self, uri: str, timeout: float = 20, settle: float = 1.5, min_count: int = 0):
        """Latest publishDiagnostics for `uri`, once no new ones arrived for `settle` seconds."""
        end = time.time() + timeout
        last_seen, last_change = None, time.time()
        while time.time() < end:
            with self._cv:
                found = [
                    n
                    for n in self.notifications
                    if n["method"] == "textDocument/publishDiagnostics" and n["params"]["uri"] == uri
                ]
            cur = found[-1]["params"]["diagnostics"] if found else None
            if found and len(found) != last_seen:
                last_seen, last_change = len(found), time.time()
            if cur is not None and len(cur) >= min_count and time.time() - last_change > settle:
                return cur
            time.sleep(0.1)
        return cur

    def wait_ready(self, uri: str, timeout: float = 60, after: int = 0) -> bool:
        """Wait for the server's `$/tensi/status` ready notice for `uri` (the seed solver finished)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._cv:
                ready = [
                    n
                    for n in self.notifications[after:]
                    if n["method"] == "$/tensi/status" and n["params"]["uri"] == uri and n["params"]["state"] == "ready"
                ]
            if ready:
                return True
            time.sleep(0.1)
        return False

    def close(self):
        try:
            self.request("shutdown", None, timeout=10)
            self.notify("exit", None)
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
