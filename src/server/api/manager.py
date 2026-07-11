"""Optimizer session lifecycle management."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from src.common.config import Config
from src.generator.prompt_curation import DEFAULT_TEXT_MODEL, PromptCurationManifest
from src.session.curation import PromptCurationService
from src.session.diffusion_client import DiffusionClient
from src.session.frame_store import FrameStore
from src.session.optimizer_loop import OptimizerRenderLoop
from src.signal_service.eeg_sources import MockEEGSource
from src.signal_service.service import FAARewardSource, build_faa_service

from .eeg import EEGConnectionManager
from .models import StartSessionRequest
from .settings import REPO_ROOT

DiffusionClientFactory = Callable[[str, float], DiffusionClient]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:48] or "prompt"


class ProcessLogStore:
    def __init__(self, max_lines: int = 2000) -> None:
        self._lines: deque[str] = deque(maxlen=max_lines)
        self.max_lines = max_lines

    def clear(self) -> None:
        self._lines.clear()

    def append(self, line: str) -> None:
        self._lines.append(line)

    def tail(self, lines: int) -> list[str]:
        count = max(1, min(lines, self.max_lines))
        return list(self._lines)[-count:]

    def extend(self, lines) -> None:
        self._lines.extend(lines)


class SessionManager:
    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        max_log_lines: int = 2000,
        eeg_manager: EEGConnectionManager | None = None,
        curation_service: PromptCurationService | None = None,
        diffusion_client_factory: DiffusionClientFactory | None = None,
        config: Config | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.config = config or Config.load()
        self.eeg_manager = eeg_manager
        self.curation_service = curation_service or PromptCurationService(
            self.config.generator.openai_text_model or DEFAULT_TEXT_MODEL
        )
        self.diffusion_client_factory = diffusion_client_factory or (
            lambda server_url, timeout: DiffusionClient(server_url, timeout)
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._started_at: str | None = None
        self._prompt: str | None = None
        self._last_exit_code: int | None = None
        self._logs = ProcessLogStore(max_log_lines)
        self._manifest_path: str | None = None
        self._mock_source: MockEEGSource | None = None

    def start(self, request: StartSessionRequest) -> dict[str, Any]:
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        with self._lock:
            self._refresh_locked()
            if self._thread is not None:
                raise HTTPException(status_code=409, detail="session already running")
            self._logs.clear()
            self._logs.append(f"[api] prompt: {prompt}")

        reward_source = self._build_reward_source(request)
        manifest = self._curate_manifest(prompt)
        manifest_path = self._write_manifest(manifest)
        client = self.diffusion_client_factory(request.server_url, 30.0)
        remote_manifest = self._load_remote_manifest(client)
        self._validate_remote_manifest(manifest, remote_manifest)
        loop = OptimizerRenderLoop(
            self.config,
            frames_per_step=6,
            client=client,
            frame_store=FrameStore(),
            capture_snapshots=not request.mock,
        )
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_session,
            args=(loop, reward_source, stop_event, request.mock),
            name="neurim-optimizer-session",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
            self._stop_event = stop_event
            self._started_at = _utc_now()
            self._prompt = prompt
            self._last_exit_code = None
            self._manifest_path = str(manifest_path)
            self._logs.append(f"[api] manifest: {manifest_path}")
            self._logs.append(f"[api] render server: {request.server_url}")
            thread.start()
            return self._status_locked()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            thread = self._thread
            stop_event = self._stop_event
            if thread is None or stop_event is None:
                return self._status_locked()
            stop_event.set()

        thread.join(timeout=5)
        with self._lock:
            if not thread.is_alive():
                self._last_exit_code = 0 if self._last_exit_code is None else self._last_exit_code
                self._thread = None
                self._stop_event = None
            return self._status_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return self._status_locked()

    def logs(self, lines: int) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return {"lines": self._logs.tail(lines)}

    def _curate_manifest(self, prompt: str) -> PromptCurationManifest:
        try:
            manifest = self.curation_service.curate(prompt)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"prompt curation failed: {exc}") from exc
        with self._lock:
            self._logs.append("[api] prompt manifest curated")
        return manifest

    def _write_manifest(self, manifest: PromptCurationManifest) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.repo_root / "data" / "processed" / "prompt_sessions" / f"{stamp}-{_slug(manifest.user_prompt)}.json"
        self.curation_service.write(manifest, output_path)
        return output_path

    @staticmethod
    def _load_remote_manifest(client: DiffusionClient) -> dict[str, Any]:
        try:
            return client.manifest()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"diffusion manifest fetch failed: {exc}") from exc

    @staticmethod
    def _validate_remote_manifest(
        manifest: PromptCurationManifest, remote_manifest: dict[str, Any]
    ) -> None:
        expected = manifest.to_dict()
        keys = ["user_prompt", "anchor_count", "anchor_labels", "realized_prompts"]
        mismatches = {
            key: {"expected": expected.get(key), "remote": remote_manifest.get(key)}
            for key in keys
            if key in remote_manifest and remote_manifest.get(key) != expected.get(key)
        }
        missing = [key for key in keys if key not in remote_manifest]
        if missing or mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "diffusion server manifest does not match curated prompt",
                    "missing": missing,
                    "mismatches": mismatches,
                },
            )

    def _build_reward_source(self, request: StartSessionRequest) -> FAARewardSource:
        if request.mock:
            self._mock_source = MockEEGSource(self.config.eeg.channels, self.config.eeg.sample_rate_hz)
            return build_faa_service(self.config, self._mock_source).reward_source  # type: ignore[return-value]
        if self.eeg_manager is None:
            raise HTTPException(status_code=503, detail="EEG manager is not configured")
        try:
            return self.eeg_manager.require_ready_reward_source()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": str(exc), "eeg": self.eeg_manager.status()},
            ) from exc

    def _run_session(
        self,
        loop: OptimizerRenderLoop,
        reward_source: FAARewardSource,
        stop_event: threading.Event,
        mock: bool,
    ) -> None:
        optimizer = loop.optimizer
        exit_code = 0
        try:
            while not stop_event.is_set():
                message = reward_source.read_reward()
                if message is None:
                    continue
                result = optimizer.observe_reward(message.r)
                if result is None:
                    continue
                loop.render_candidate(np.asarray(result.z, dtype=float))
                raw = message.raw_faa if message.raw_faa is not None else float("nan")
                self._append_log(
                    f"{result.step_index:>4} {result.state:>9} "
                    f"{result.reward_estimate:>+7.2f} {raw:>+7.2f} {loop.frame_count}"
                )
                if optimizer.state_machine.should_stop():
                    break
                if mock:
                    time.sleep(max(0.0, self.config.faa.update_interval_s))
            loop.save_final_frame()
        except Exception as exc:  # noqa: BLE001
            exit_code = 1
            self._append_log(f"[api] session error: {exc}")
        finally:
            with self._lock:
                self._last_exit_code = exit_code
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._stop_event = None

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _refresh_locked(self) -> None:
        if self._thread is not None and not self._thread.is_alive():
            if self._last_exit_code is None:
                self._last_exit_code = 0
            self._thread = None
            self._stop_event = None

    def _status_locked(self) -> dict[str, Any]:
        thread = self._thread
        return {
            "running": thread is not None,
            "pid": None,
            "started_at": self._started_at,
            "prompt": self._prompt,
            "exit_code": self._last_exit_code if thread is None else None,
            "manifest_path": self._manifest_path,
        }
