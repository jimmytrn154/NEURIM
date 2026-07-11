"""HTTP client for a private manifest-driven diffusion server."""

from __future__ import annotations

from typing import Any

import numpy as np


class DiffusionClient:
    def __init__(self, base_url: str, timeout: float = 30.0, session=None) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def manifest(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/manifest", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def render(self, z: np.ndarray, frame_size: int) -> bytes:
        response = self.session.post(
            f"{self.base_url}/render",
            json={"z": [float(value) for value in z], "frame_size": frame_size},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"server returned {response.status_code}: {response.text[:200]}")
        return response.content
