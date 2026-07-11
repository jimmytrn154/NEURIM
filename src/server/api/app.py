"""FastAPI application factory for the local frontend bridge."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .eeg import EEGConnectionManager
from .manager import SessionManager
from .models import StartSessionRequest
from .settings import ApiSettings


def create_app(
    session_manager: SessionManager | None = None,
    eeg_manager: EEGConnectionManager | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings.from_env()
    eeg = eeg_manager or EEGConnectionManager()
    manager = session_manager or SessionManager(
        max_log_lines=api_settings.max_log_lines,
        eeg_manager=eeg,
    )

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        eeg.start()
        try:
            yield
        finally:
            manager.stop()
            eeg.close()

    application = FastAPI(title="NEURIM API", version="0.3.0", lifespan=lifespan)
    application.state.session_manager = manager
    application.state.eeg_manager = eeg
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(api_settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @application.get("/eeg/status")
    def eeg_status() -> dict[str, Any]:
        return eeg.status()

    @application.post("/eeg/retry")
    def eeg_retry() -> dict[str, Any]:
        return eeg.retry_now()

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
