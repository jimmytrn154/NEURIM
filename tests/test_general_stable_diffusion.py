import io
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image
import pytest

import scripts.run_general_stable_diffusion as general_server


def _png_bytes(color=(0, 255, 0)):
    image = Image.new("RGB", (8, 8), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_softmax_weights_sum_to_one():
    weights = general_server.softmax_weights(np.array([0.5, -1.2, 0.1]), temperature=4.0)

    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)


def test_top_anchors_orders_by_weight():
    summary = general_server.top_anchors(
        ["crowd", "portrait", "goalkeeper"],
        np.array([0.1, 0.75, 0.15]),
        n=2,
    )

    assert summary == "portrait 0.75 | goalkeeper 0.15"


def test_render_server_rejects_wrong_z_shape():
    server = general_server.AnchorMorphRenderServer(
        pipe=None,
        anchor_labels=["a0", "a1", "a2", "a3", "a4", "a5", "a6"],
        anchor_prompts=["p0", "p1", "p2", "p3", "p4", "p5", "p6"],
        anchor_embeds=None,
        anchor_latents=None,
        frame_size=512,
        device="cpu",
        dtype=None,
        num_inference_steps=1,
        guidance_scale=0.0,
        softmax_temperature=3.0,
        log_weights_every=0,
        target_anchor=None,
    )

    with pytest.raises(RuntimeError, match="expected \\(7,\\)"):
        server.render_png({"z": [0.1, 0.2], "frame_size": 128})


def test_make_handler_accepts_render_payload_shape():
    class FakeRenderServer:
        def __init__(self):
            self.payloads = []

        def render_png(self, payload):
            self.payloads.append(payload)
            return _png_bytes()

    fake_server = FakeRenderServer()
    httpd = general_server.ThreadingHTTPServer(("127.0.0.1", 0), general_server.make_handler(fake_server))
    thread = threading.Thread(target=httpd.handle_request, daemon=True)
    thread.start()
    try:
        body = json.dumps({"z": [0.1] * 7, "frame_size": 128}).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/render",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert response.read().startswith(b"\x89PNG")
    finally:
        thread.join(timeout=5)
        httpd.server_close()

    assert fake_server.payloads == [{"z": [0.1] * 7, "frame_size": 128}]


def test_make_handler_serves_manifest_metadata():
    class FakeRenderServer:
        manifest = {
            "user_prompt": "world cup players",
            "anchor_count": 7,
            "anchor_labels": [f"axis_{i}" for i in range(7)],
            "model": {"provider": "openai", "name": "gpt-test"},
        }

        def render_png(self, payload):
            return _png_bytes()

    httpd = general_server.ThreadingHTTPServer(("127.0.0.1", 0), general_server.make_handler(FakeRenderServer()))
    thread = threading.Thread(target=httpd.handle_request, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/manifest", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/json"
    finally:
        thread.join(timeout=5)
        httpd.server_close()

    assert payload == {
        "ok": True,
        "user_prompt": "world cup players",
        "anchor_count": 7,
        "anchor_labels": [f"axis_{i}" for i in range(7)],
        "model": {"provider": "openai", "name": "gpt-test"},
        "server": {
            "kind": "general_stable_diffusion",
            "render_endpoint": "/render",
        },
    }


def test_make_handler_rejects_unknown_get_path():
    class FakeRenderServer:
        manifest = {"anchor_count": 7, "anchor_labels": [f"axis_{i}" for i in range(7)]}

        def render_png(self, payload):
            return _png_bytes()

    httpd = general_server.ThreadingHTTPServer(("127.0.0.1", 0), general_server.make_handler(FakeRenderServer()))
    thread = threading.Thread(target=httpd.handle_request, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/unknown", timeout=5)
    finally:
        thread.join(timeout=5)
        httpd.server_close()

    assert exc_info.value.code == 404


def test_cli_smoke_loads_manifest_and_initializes_server(tmp_path):
    manifest_path = tmp_path / "session.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "user_prompt": "world cup players",
                "anchor_count": 7,
                "scaffold": "shared scaffold",
                "prompt_template": "shared template {anchor}",
                "anchor_labels": [f"axis_{i}" for i in range(7)],
                "realized_prompts": [f"prompt {i}" for i in range(7)],
                "notes": "none",
                "model": {"provider": "openai", "name": "gpt-test"},
            }
        ),
        encoding="utf-8",
    )

    class FakePipe:
        def __init__(self):
            self.unet = type("Unet", (), {"config": type("Cfg", (), {"in_channels": 4})()})()
            self.vae_scale_factor = 8

    captured = {}

    def fake_load_pipeline(model_id, device=None):
        captured["model_id"] = model_id
        return FakePipe(), "cpu", "float32"

    def fake_encode_anchor_prompts(pipe, prompts, device):
        captured["prompts"] = list(prompts)
        return "encoded-prompts"

    def fake_make_anchor_latents(n, generator, device, dtype, size, channels, vae_scale_factor):
        captured["latents"] = {
            "n": n,
            "device": device,
            "dtype": dtype,
            "size": size,
            "channels": channels,
            "vae_scale_factor": vae_scale_factor,
        }
        return "anchor-latents"

    class FakeHTTPServer:
        def __init__(self, addr, handler):
            captured["server_addr"] = addr
            captured["handler"] = handler

        def serve_forever(self):
            captured["served"] = True

    original_load_pipeline = general_server.load_pipeline
    original_encode = general_server.encode_anchor_prompts
    original_make_latents = general_server.make_anchor_latents
    original_make_generator = general_server.make_cpu_generator
    original_http_server = general_server.ThreadingHTTPServer
    general_server.load_pipeline = fake_load_pipeline
    general_server.encode_anchor_prompts = fake_encode_anchor_prompts
    general_server.make_anchor_latents = fake_make_anchor_latents
    general_server.make_cpu_generator = lambda seed: {"seed": seed}
    general_server.ThreadingHTTPServer = FakeHTTPServer
    try:
        general_server.main(
            ["--session-manifest", str(manifest_path), "--host", "127.0.0.1", "--port", "9999", "--size", "512"]
        )
    finally:
        general_server.load_pipeline = original_load_pipeline
        general_server.encode_anchor_prompts = original_encode
        general_server.make_anchor_latents = original_make_latents
        general_server.make_cpu_generator = original_make_generator
        general_server.ThreadingHTTPServer = original_http_server

    assert captured["model_id"] == "stabilityai/sd-turbo"
    assert captured["prompts"] == [f"prompt {i}" for i in range(7)]
    assert captured["latents"]["n"] == 7
    assert captured["server_addr"] == ("127.0.0.1", 9999)
    assert captured["served"] is True
