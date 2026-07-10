"""Client for running diffusion on a separate machine.

The local loop keeps EEG, FAA, and optimization close to the headset. Once the
optimizer has a candidate latent, this client sends that post-FAA state to a
diffusion server and uses the PNG/JPEG image it returns as the rendered frame.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


class RemoteDiffusionClient:
    def __init__(
        self,
        url: str,
        timeout_s: float = 30.0,
        frame_size: int = 512,
        session: Any | None = None,
    ):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.frame_size = frame_size
        self.session = session
        self._cache: dict[tuple[int, str], Image.Image] = {}
        self._last_image: Image.Image | None = None

    def render(
        self,
        z: np.ndarray,
        prompt: str,
        step_index: int,
        state: str,
        reward_estimate: float,
    ) -> Image.Image:
        cache_key = (step_index, prompt)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        payload = {
            "z": list(map(float, z)),
            "prompt": prompt,
            "step_index": int(step_index),
            "state": state,
            "reward_estimate": float(reward_estimate),
            "frame_size": int(self.frame_size),
        }
        image = self._post_render(payload)
        self._cache[cache_key] = image
        self._last_image = image
        return image.copy()

    def _post_render(self, payload: dict) -> Image.Image:
        requests = self.session
        if requests is None:
            import requests

        response = requests.post(f"{self.url}/render", json=payload, timeout=self.timeout_s)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
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
