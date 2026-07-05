"""HTTP adapter (stdlib only — no web-framework dependency to keep the image
lean). POST / or /task with a JSON body -> JSON answer. GET /health -> ok."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .common import decision_to_response, parse_task


def run(router, port: int = 8080) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (stdlib naming)
            if self.path.rstrip("/") in ("", "/health"):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                task = parse_task(raw)
                if task is None:
                    self._send(400, {"error": "empty or unparseable task"})
                    return
                decision = router.route(task)
                self._send(200, decision_to_response(task, decision))
            except Exception as exc:  # never let a request kill the server
                self._send(500, {"error": str(exc)})

        def log_message(self, fmt, *args):  # route stdlib logging to stderr
            print(f"[http] {fmt % args}", file=sys.stderr)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[cheaproute] HTTP adapter listening on :{port}", file=sys.stderr, flush=True)
    server.serve_forever()
