"""Same-machine diffusion process management for API-started sessions."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


class DiffusionProcessManager:
    """Starts one manifest-driven diffusion server process at a time.

    This is intentionally same-machine only. If the GPU server is on another
    host, run the diffusion server manually or add a remote supervisor later.
    """

    def __init__(
        self,
        repo_root: Path,
        host: str = "127.0.0.1",
        port: int = 8766,
        python_executable: str | None = None,
        cuda_visible_devices: str | None = None,
        model: str = "stabilityai/sd-turbo",
        steps: int = 2,
        guidance_scale: float = 0.0,
        temperature: float = 8.0,
        size: int | None = None,
        seed: int | None = None,
        startup_timeout_s: float = 300.0,
    ) -> None:
        self.repo_root = repo_root
        self.host = host
        self.port = port
        self.python_executable = python_executable or sys.executable
        self.cuda_visible_devices = cuda_visible_devices
        self.model = model
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.temperature = temperature
        self.size = size
        self.seed = seed
        self.startup_timeout_s = startup_timeout_s
        self._process: subprocess.Popen | None = None
        self._manifest_path: Path | None = None
        self._last_seed: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def restart(self, manifest_path: Path, _manifest: Any | None = None) -> dict[str, Any]:
        self.stop()
        # A fixed seed pins the anchor noise so every generation renders the same
        # base images. When no seed is configured, draw a fresh one per restart so
        # each session's morph looks visually distinct.
        render_seed = self.seed if self.seed is not None else int.from_bytes(os.urandom(4), "big") % (2**31)
        self._last_seed = render_seed
        command = [
            self.python_executable,
            str(self.repo_root / "scripts" / "run_general_stable_diffusion.py"),
            "--session-manifest",
            str(manifest_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model",
            self.model,
            "--steps",
            str(self.steps),
            "--guidance-scale",
            str(self.guidance_scale),
            "--temperature",
            str(self.temperature),
        ]
        if self.size is not None:
            command.extend(["--size", str(self.size)])
        command.extend(["--seed", str(render_seed)])
        env = os.environ.copy()
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        self._manifest_path = manifest_path
        self._process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            env=env,
        )
        return self.wait_until_ready()

    def wait_until_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error = "not checked"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"diffusion server exited early with code {self._process.returncode}")
            try:
                response = requests.get(f"{self.base_url}/manifest", timeout=2.0)
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:160]}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            time.sleep(1.0)
        raise RuntimeError(
            f"diffusion server did not become ready at {self.base_url} "
            f"within {self.startup_timeout_s:.0f}s: {last_error}"
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def status(self) -> dict[str, Any]:
        process = self._process
        return {
            "managed": True,
            "running": process is not None and process.poll() is None,
            "pid": process.pid if process is not None else None,
            "base_url": self.base_url,
            "manifest_path": str(self._manifest_path) if self._manifest_path else None,
            "seed": self._last_seed,
        }
