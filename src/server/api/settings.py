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
        return cls(
            host=os.environ.get("NEURIM_API_HOST", defaults.host),
            port=port,
            max_log_lines=defaults.max_log_lines,
            cors_origins=configured_origins,
        )
