#!/usr/bin/env python3
"""Minimal HTTP API for launching NEURIM sessions from the frontend."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_LOG_LINES = 2000


class StartSessionRequest(BaseModel):
    prompt: str | None = None
    mock: bool = True
    baseline_seconds: float = Field(default=5.0, ge=0)
    server_url: str = "http://localhost:8766"

    @field_validator("server_url")
    @classmethod
    def _validate_server_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("server_url must start with http:// or https://")
        return cleaned

    @field_validator("prompt")
    @classmethod
    def _clean_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._started_at: str | None = None
        self._prompt: str | None = None
        self._last_exit_code: int | None = None
        self._logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._reader_threads: list[threading.Thread] = []

    def start(self, request: StartSessionRequest) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._process is not None:
                raise HTTPException(status_code=409, detail="session already running")

            cmd = [
                sys.executable,
                "scripts/run_real_eeg_optimizer.py",
                "--baseline",
                str(request.baseline_seconds),
                "--server-url",
                request.server_url,
            ]
            if request.mock:
                cmd.insert(2, "--mock")

            self._logs.clear()
            self._logs.append("$ " + " ".join(cmd))
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._process = process
            self._started_at = datetime.now(UTC).isoformat()
            self._prompt = request.prompt
            self._last_exit_code = None
            self._reader_threads = [self._start_reader(process)]
            return self._status_locked()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            process = self._process
            if process is None:
                return self._status_locked()

            process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                process.send_signal(signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)

        with self._lock:
            self._last_exit_code = process.poll()
            self._process = None
            return self._status_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return self._status_locked()

    def logs(self, lines: int) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            count = max(1, min(lines, MAX_LOG_LINES))
            return {"lines": list(self._logs)[-count:]}

    def _refresh_locked(self) -> None:
        if self._process is not None:
            exit_code = self._process.poll()
            if exit_code is not None:
                self._last_exit_code = exit_code
                self._process = None

    def _status_locked(self) -> dict[str, Any]:
        process = self._process
        return {
            "running": process is not None,
            "pid": process.pid if process is not None else None,
            "started_at": self._started_at,
            "prompt": self._prompt,
            "exit_code": self._last_exit_code if process is None else process.poll(),
        }

    def _start_reader(self, process: subprocess.Popen[str]) -> threading.Thread:
        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                with self._lock:
                    self._logs.append(line.rstrip("\n"))
            with self._lock:
                exit_code = process.poll()
                if exit_code is not None:
                    self._last_exit_code = exit_code
                    if self._process is process:
                        self._process = None

        thread = threading.Thread(target=read_output, name="neurim-session-log-reader", daemon=True)
        thread.start()
        return thread


manager = SessionManager()
app = FastAPI(title="NEURIM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/session/start")
def start_session(request: StartSessionRequest) -> dict[str, Any]:
    return manager.start(request)


@app.post("/session/stop")
def stop_session() -> dict[str, Any]:
    return manager.stop()


@app.get("/session/status")
def session_status() -> dict[str, Any]:
    return manager.status()


@app.get("/session/logs")
def session_logs(lines: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
    return manager.logs(lines)
