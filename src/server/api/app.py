"""FastAPI application factory for the local frontend bridge."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .diffusion_process import DiffusionProcessManager
from .diffusion_supervisor_client import RemoteDiffusionSupervisorClient
from .eeg import EEGConnectionManager
from .manager import SessionManager
from .models import StartSessionRequest
from .settings import REPO_ROOT, ApiSettings


def create_app(
    session_manager: SessionManager | None = None,
    eeg_manager: EEGConnectionManager | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings.from_env()
    eeg = eeg_manager or EEGConnectionManager()
    diffusion_process_manager = None
    if api_settings.diffusion_supervisor_url:
        diffusion_process_manager = RemoteDiffusionSupervisorClient(
            api_settings.diffusion_supervisor_url,
            timeout_s=api_settings.diffusion_startup_timeout_s,
        )
    elif api_settings.manage_diffusion:
        diffusion_process_manager = DiffusionProcessManager(
            repo_root=REPO_ROOT,
            host=api_settings.diffusion_host,
            port=api_settings.diffusion_port,
            python_executable=api_settings.diffusion_python,
            cuda_visible_devices=api_settings.diffusion_cuda_visible_devices,
            model=api_settings.diffusion_model,
            steps=api_settings.diffusion_steps,
            guidance_scale=api_settings.diffusion_guidance_scale,
            temperature=api_settings.diffusion_temperature,
            size=api_settings.diffusion_size,
            seed=api_settings.diffusion_seed,
            startup_timeout_s=api_settings.diffusion_startup_timeout_s,
        )
    manager = session_manager or SessionManager(
        max_log_lines=api_settings.max_log_lines,
        eeg_manager=eeg,
        diffusion_process_manager=diffusion_process_manager,
    )

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        eeg.start()
        try:
            yield
        finally:
            manager.stop()
            if diffusion_process_manager is not None:
                diffusion_process_manager.stop()
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

    @application.post("/session/finalize/retry")
    def retry_session_finalization() -> dict[str, Any]:
        return manager.retry_finalization()

    @application.get("/session/status")
    def session_status() -> dict[str, Any]:
        return manager.status()

    @application.get("/session/logs")
    def session_logs(lines: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        return manager.logs(lines)

    return application


app = create_app()
