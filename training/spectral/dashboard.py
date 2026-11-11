"""Read-only localhost view of the spectral experiment."""

import mimetypes
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from training.spatial.common import ROOT
from training.temporal_v2.dashboard import read_shared

ASSETS = Path(__file__).parent / "web"


class LocalDashboard(ThreadingHTTPServer):
    def server_bind(self):
        if os.name == "nt":
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def handler_for(run):
    root = Path(run).resolve()
    assets = {"/": ASSETS / "index.html", "/app.js": ASSETS / "app.js",
              "/style.css": ASSETS / "style.css", "/icons.js": ROOT / "demo/web/vendor/lucide.min.js"}
    api = {"/api/state": root / "live.json", "/api/report": root / "final_report.json"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            port = self.server.server_port
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self.send_error(403)
                return
            route = urlsplit(self.path).path
            path = assets.get(route) or api.get(route)
            try:
                if route == "/api/log":
                    path = root / "training.log"
                    with path.open("rb") as stream:
                        length = stream.seek(0, 2)
                        stream.seek(max(0, length-65536))
                        content = "\n".join(stream.read().decode("utf-8", "replace").splitlines()[-80:]).encode("utf-8")
                elif path is not None:
                    content = read_shared(path) if path.is_file() else None
                else:
                    self.send_error(404)
                    return
            except FileNotFoundError:
                content = None
            except OSError:
                self.send_error(503, "Snapshot temporarily unavailable")
                return
            if content is None:
                if route in api:
                    content = b"{}"
                elif route == "/api/log":
                    content = b""
                else:
                    self.send_error(404)
                    return
            content_type = "text/plain; charset=utf-8" if route == "/api/log" else (
                mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def server_for(run, port):
    if not 0 <= port <= 65535:
        raise ValueError("Invalid dashboard port")
    for candidate in range(port, min(port+20, 65536)):
        try:
            return LocalDashboard(("127.0.0.1", candidate), handler_for(run))
        except OSError:
            continue
    raise OSError("No free localhost port for spectral dashboard")


def serve(run, port=8767):
    with server_for(run, port) as server:
        print(f"Spectral dashboard: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Run directory: {Path(run).resolve()}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
