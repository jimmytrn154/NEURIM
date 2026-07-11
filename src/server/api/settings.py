"""Environment-backed settings for the local API server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    max_log_lines: int = 2000
    manage_diffusion: bool = False
    diffusion_host: str = "127.0.0.1"
    diffusion_port: int = 8766
    diffusion_startup_timeout_s: float = 300.0
    diffusion_cuda_visible_devices: str | None = None
    diffusion_python: str | None = None
    diffusion_model: str = "stabilityai/sd-turbo"
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    @classmethod
    def from_env(cls) -> "ApiSettings":
        defaults = cls()
        origins = os.environ.get("NEURIM_API_CORS_ORIGINS")
        configured_origins = (
            tuple(origin.strip() for origin in origins.split(",") if origin.strip())
            if origins
            else defaults.cors_origins
        )
        try:
            port = int(os.environ.get("NEURIM_API_PORT", defaults.port))
        except ValueError:
            port = defaults.port
        try:
            diffusion_port = int(os.environ.get("NEURIM_DIFFUSION_PORT", defaults.diffusion_port))
        except ValueError:
            diffusion_port = defaults.diffusion_port
        try:
            startup_timeout = float(
                os.environ.get(
                    "NEURIM_DIFFUSION_STARTUP_TIMEOUT_S",
                    defaults.diffusion_startup_timeout_s,
                )
            )
        except ValueError:
            startup_timeout = defaults.diffusion_startup_timeout_s
        manage_diffusion = os.environ.get("NEURIM_MANAGE_DIFFUSION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            host=os.environ.get("NEURIM_API_HOST", defaults.host),
            port=port,
            max_log_lines=defaults.max_log_lines,
            manage_diffusion=manage_diffusion,
            diffusion_host=os.environ.get("NEURIM_DIFFUSION_HOST", defaults.diffusion_host),
            diffusion_port=diffusion_port,
            diffusion_startup_timeout_s=startup_timeout,
            diffusion_cuda_visible_devices=os.environ.get("NEURIM_DIFFUSION_CUDA_VISIBLE_DEVICES"),
            diffusion_python=os.environ.get("NEURIM_DIFFUSION_PYTHON"),
            diffusion_model=os.environ.get("NEURIM_DIFFUSION_MODEL", defaults.diffusion_model),
            cors_origins=configured_origins,
        )
