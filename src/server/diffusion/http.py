"""HTTP transport for the manifest-driven diffusion renderer."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .renderer import AnchorMorphRenderer


def make_handler(renderer: AnchorMorphRenderer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/manifest":
                self.send_error(404, "expected GET /manifest")
                return
            self._send_json(
                {
                    "ok": True,
                    **renderer.manifest,
                    "server": {
                        "kind": "general_stable_diffusion",
                        "render_endpoint": "/render",
                    },
                }
            )

        def do_POST(self) -> None:
            if self.path != "/render":
                self.send_error(404, "expected POST /render")
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                png = renderer.render_png(payload)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

            self.send_response(200)
            self.send_header("content-type", "image/png")
            self.send_header("content-length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[anchor-morph-server] {self.address_string()} - {fmt % args}")

    return Handler


class DiffusionServer:
    def __init__(self, renderer: AnchorMorphRenderer, host: str, port: int) -> None:
        self.renderer = renderer
        self.host = host
        self.port = port
        self.httpd = ThreadingHTTPServer((host, port), make_handler(renderer))

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def close(self) -> None:
        self.httpd.server_close()
