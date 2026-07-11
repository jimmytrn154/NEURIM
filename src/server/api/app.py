"""FastAPI application factory for the local frontend bridge."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .manager import SessionManager
from .models import StartSessionRequest
from .settings import ApiSettings


def create_app(
    session_manager: SessionManager | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings.from_env()
    manager = session_manager or SessionManager(max_log_lines=api_settings.max_log_lines)
    application = FastAPI(title="NEURIM API", version="0.2.0")
    application.state.session_manager = manager
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(api_settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @application.post("/session/start")
    def start_session(request: StartSessionRequest) -> dict[str, Any]:
        return manager.start(request)

    @application.post("/session/stop")
    def stop_session() -> dict[str, Any]:
        return manager.stop()

    @application.get("/session/status")
    def session_status() -> dict[str, Any]:
        return manager.status()

    @application.get("/session/logs")
    def session_logs(lines: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        return manager.logs(lines)

    return application


app = create_app()
