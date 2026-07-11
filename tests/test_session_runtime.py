import io

import numpy as np
from PIL import Image

from src.session.diffusion_client import DiffusionClient
from src.session.frame_store import FrameStore


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "green").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status_code=200, content=b"", payload=None, text=""):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(("GET", url, timeout))
        return FakeResponse(payload={"anchor_count": 7})

    def post(self, url, json, timeout):
        self.calls.append(("POST", url, json, timeout))
        return FakeResponse(content=_png_bytes())


def test_diffusion_client_uses_manifest_and_render_contracts():
    session = FakeSession()
    client = DiffusionClient("http://gpu:8766/", timeout=4, session=session)

    assert client.manifest() == {"anchor_count": 7}
    assert client.render(np.array([0.1, 0.2]), 256).startswith(b"\x89PNG")
    assert session.calls == [
        ("GET", "http://gpu:8766/manifest", 4),
        ("POST", "http://gpu:8766/render", {"z": [0.1, 0.2], "frame_size": 256}, 4),
    ]


def test_frame_store_writes_live_and_snapshots(tmp_path):
    store = FrameStore(tmp_path)
    png = _png_bytes()

    store.save_live(png, capture_start=True)
    store.save_live(png, capture_start=True)
    store.save_end(png)

    assert (tmp_path / "live_frame.png").read_bytes() == png
    assert (tmp_path / "session_start.png").read_bytes() == png
    assert (tmp_path / "session_end.png").read_bytes() == png
