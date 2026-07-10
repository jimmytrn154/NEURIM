#!/usr/bin/env python3
"""HTTP server that runs the local diffusion model for a remote NEURIM client.

Run this on the GPU machine, then set the client config:

  generator.backend: "remote_diffusion"
  generator.remote_diffusion_url: "http://GPU_HOST:8766"
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator.diffusion_pipeline import DiffusionGenerator


class DiffusionRenderServer:
    def __init__(self, generator: DiffusionGenerator):
        self.generator = generator
        self.lock = threading.Lock()

    def render_png(self, payload: dict) -> bytes:
        prompt = payload.get("prompt") or "a little brown puppy"
        frame_size = int(payload.get("frame_size", 512))
        with self.lock:
            image = self.generator.render_from_prompt(prompt)
        if image.size != (frame_size, frame_size):
            image = image.resize((frame_size, frame_size))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def make_handler(render_server: DiffusionRenderServer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/render":
                self.send_error(404, "expected POST /render")
                return

            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                png = render_server.render_png(payload)
            except Exception as exc:  # noqa: BLE001 - report server-side failures to the client.
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(200)
            self.send_header("content-type", "image/png")
            self.send_header("content-length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[diffusion-server] {self.address_string()} - {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    generator = DiffusionGenerator(
        model_id=args.model_id,
        num_inference_steps=args.steps,
        device=args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(DiffusionRenderServer(generator)))
    print(f"[diffusion-server] listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
