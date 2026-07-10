"""Client for running diffusion on a separate machine (the GPU/SSH server).

The local loop keeps EEG, FAA, and optimization close to the headset. Once the
optimizer has a candidate latent, this client sends that low-dim search vector
`z` to the diffusion server, which projects it to a full prompt embedding and
renders a genuine latent-morph keyframe.

Hybrid smoothing (see the NEURIM morph discussion):
  - A remote SDXL-Turbo render costs ~50-300ms, so we can only pull ~3-8 unique
    keyframes/sec - nowhere near target_fps (30). Rendering every display frame
    over the network is impossible.
  - So a background thread fetches keyframes as fast as the server sustains,
    tracking the latest z, and render() (called at target_fps) returns a
    CROSSFADE between the two most recent keyframes. Because the server pins the
    diffusion seed, consecutive keyframes are visually close, so the crossfade
    reads as one object smoothly morphing rather than a dissolve between two.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import numpy as np
from PIL import Image


class RemoteDiffusionClient:
    def __init__(
        self,
        url: str,
        timeout_s: float = 30.0,
        frame_size: int = 512,
        seed: int = 0,
        keyframe_interval_s: float = 0.2,
        session: Any | None = None,
    ):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.frame_size = frame_size
        self.seed = seed
        # Lower bound on time between server calls (~1/keyframe_interval_s fps of
        # true renders). Also the crossfade window: alpha reaches 1.0 over roughly
        # this long, i.e. by the time the next keyframe is expected to land.
        self.keyframe_interval_s = max(keyframe_interval_s, 1e-3)
        self.session = session

        self._lock = threading.Lock()
        self._pending_z: np.ndarray | None = None       # latest z the loop wants
        self._prev_keyframe: Image.Image | None = None   # crossfade from
        self._curr_keyframe: Image.Image | None = None   # crossfade to
        self._curr_keyframe_time = 0.0
        self._last_rendered_z: np.ndarray | None = None
        self._worker: threading.Thread | None = None
        self._blank = Image.new("RGB", (frame_size, frame_size), (0, 0, 0))

    # -- public API ---------------------------------------------------------

    def render(
        self,
        z: np.ndarray,
        step_index: int = 0,
        state: str = "explore",
        reward_estimate: float = 0.0,
        prompt: str | None = None,
    ) -> Image.Image:
        z = np.asarray(z, dtype=float)
        with self._lock:
            self._pending_z = z

        # First call: fetch one keyframe synchronously so we never show a blank
        # frame, then start the background fetcher for everything after.
        if self._curr_keyframe is None:
            image = self._fetch(z)
            with self._lock:
                self._prev_keyframe = image
                self._curr_keyframe = image
                self._curr_keyframe_time = time.monotonic()
                self._last_rendered_z = z
            self._ensure_worker()
            return image.copy()

        return self._current_blend()

    # -- crossfade ----------------------------------------------------------

    def _current_blend(self) -> Image.Image:
        with self._lock:
            prev = self._prev_keyframe
            curr = self._curr_keyframe
            age = time.monotonic() - self._curr_keyframe_time
        if curr is None:
            return self._blank.copy()
        if prev is None or prev is curr:
            return curr.copy()
        alpha = min(1.0, age / self.keyframe_interval_s)
        return Image.blend(prev, curr, alpha)

    # -- background keyframe fetcher ---------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="remote-diffusion-fetch", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            start = time.monotonic()
            with self._lock:
                z = self._pending_z
                last = self._last_rendered_z
            # Skip a render if z hasn't moved since the last keyframe (nothing to
            # morph toward) - avoids burning GPU on identical frames at rest.
            if z is not None and (last is None or not np.array_equal(z, last)):
                try:
                    image = self._fetch(z)
                    with self._lock:
                        self._prev_keyframe = self._curr_keyframe
                        self._curr_keyframe = image
                        self._curr_keyframe_time = time.monotonic()
                        self._last_rendered_z = z
                except Exception as exc:  # keep the loop alive; a dropped frame is survivable
                    print(f"[remote-diffusion] keyframe fetch failed: {exc}")
            elapsed = time.monotonic() - start
            sleep_for = self.keyframe_interval_s - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # -- transport ----------------------------------------------------------

    def _fetch(self, z: np.ndarray) -> Image.Image:
        payload = {
            "z": list(map(float, z)),
            "seed": int(self.seed),
            "frame_size": int(self.frame_size),
        }
        return self._post_render(payload)

    def _post_render(self, payload: dict) -> Image.Image:
        requests = self.session
        if requests is None:
            import requests

        response = requests.post(f"{self.url}/render", json=payload, timeout=self.timeout_s)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            import base64

            data = response.json()
            image_b64 = data.get("frame_b64") or data.get("image_b64")
            if not image_b64:
                raise RuntimeError("remote diffusion response JSON did not include frame_b64")
            image_bytes = base64.b64decode(image_b64)
        else:
            image_bytes = response.content

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.size != (self.frame_size, self.frame_size):
            image = image.resize((self.frame_size, self.frame_size), Image.Resampling.LANCZOS)
        return image
