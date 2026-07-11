"""Optimizer subprocess lifecycle management."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

from .models import StartSessionRequest
from .settings import REPO_ROOT

ProcessFactory = Callable[..., subprocess.Popen[str]]


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
        process_factory: ProcessFactory = subprocess.Popen,
        max_log_lines: int = 2000,
    ) -> None:
        self.repo_root = repo_root
        self.process_factory = process_factory
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._started_at: str | None = None
        self._prompt: str | None = None
        self._last_exit_code: int | None = None
        self._logs = ProcessLogStore(max_log_lines)
        self._reader_threads: list[threading.Thread] = []

    def start(self, request: StartSessionRequest) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._process is not None:
                raise HTTPException(status_code=409, detail="session already running")

            command = self._build_command(request)
            env = os.environ.copy()
            if request.prompt is not None:
                env["NEURIM_SESSION_PROMPT"] = request.prompt

            self._logs.clear()
            self._logs.append("$ " + " ".join(command))
            if request.prompt is not None:
                prompt_log = f"[api] prompt: {request.prompt}"
                print(prompt_log, flush=True)
                self._logs.append(prompt_log)

            process = self.process_factory(
                command,
                cwd=self.repo_root,
                env=env,
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
            return {"lines": self._logs.tail(lines)}

    @staticmethod
    def _build_command(request: StartSessionRequest) -> list[str]:
        command = [
            sys.executable,
            "scripts/run_real_eeg_optimizer.py",
            "--baseline",
            str(request.baseline_seconds),
            "--server-url",
            request.server_url,
        ]
        if request.mock:
            command.insert(2, "--mock")
        return command

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
