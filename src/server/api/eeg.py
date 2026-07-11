"""API-owned EMOTIV connection and calibration lifecycle."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from src.common.config import Config, emotiv_credentials
from src.signal_service.baseline import calibrate_baseline
from src.signal_service.eeg_sources import EmotivCortexSource
from src.signal_service.service import FAARewardSource, build_faa_service


SourceFactory = Callable[[], EmotivCortexSource]
Calibrator = Callable[[Any, Any, float], Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class EEGConnectionManager:
    """Owns the real EEG source while the API process is alive."""

    def __init__(
        self,
        config: Config | None = None,
        source_factory: SourceFactory | None = None,
        calibrator: Calibrator = calibrate_baseline,
        calibration_seconds: float = 30.0,
        retry_interval_s: float = 60.0,
    ) -> None:
        self.config = config or Config.load()
        self.source_factory = source_factory or self._default_source
        self.calibrator = calibrator
        self.calibration_seconds = calibration_seconds
        self.retry_interval_s = retry_interval_s
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: EmotivCortexSource | None = None
        self._reward_source: FAARewardSource | None = None
        self._state = "disconnected"
        self._last_error: str | None = None
        self._last_connected_at: datetime | None = None
        self._last_calibrated_at: datetime | None = None
        self._next_retry_at: datetime | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._wake.set()
            self._thread = threading.Thread(target=self._run, name="neurim-eeg-connector", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            self._close_source_locked()
            self._state = "disconnected"

    def retry_now(self) -> dict[str, Any]:
        with self._lock:
            if self._state not in {"connecting", "calibrating", "ready"}:
                self._state = "disconnected"
                self._next_retry_at = _utc_now()
                self._wake.set()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "connected": self._source is not None and self._state in {"connected", "calibrating", "ready"},
                "calibrated": self._reward_source is not None and self._state == "ready",
                "calibration_seconds": self.calibration_seconds,
                "last_error": self._last_error,
                "last_connected_at": _iso(self._last_connected_at),
                "last_calibrated_at": _iso(self._last_calibrated_at),
                "next_retry_at": _iso(self._next_retry_at),
            }

    def require_ready_reward_source(self) -> FAARewardSource:
        with self._lock:
            if self._state != "ready" or self._reward_source is None:
                raise RuntimeError("EEG is not ready")
            return self._reward_source

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._seconds_until_retry())
            self._wake.clear()
            if self._stop.is_set():
                break
            if not self._retry_due():
                continue
            with self._lock:
                if self._state in {"connecting", "calibrating", "ready"}:
                    continue
            self._connect_and_calibrate()

    def _connect_and_calibrate(self) -> None:
        source: EmotivCortexSource | None = None
        try:
            with self._lock:
                self._state = "connecting"
                self._last_error = None
                self._next_retry_at = None
                self._close_source_locked()
            source = self.source_factory()
            source.connect()
            signal_service = build_faa_service(self.config, source)
            reward_source = signal_service.reward_source
            with self._lock:
                self._source = source
                self._reward_source = reward_source  # type: ignore[assignment]
                self._state = "calibrating"
                self._last_connected_at = _utc_now()
            self.calibrator(reward_source.computer, source.stream(), self.calibration_seconds)
            with self._lock:
                self._state = "ready"
                self._last_calibrated_at = _utc_now()
                self._next_retry_at = None
        except Exception as exc:  # noqa: BLE001
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            with self._lock:
                self._source = None
                self._reward_source = None
                self._state = "error"
                self._last_error = str(exc)
                self._next_retry_at = _utc_now() + timedelta(seconds=self.retry_interval_s)

    def _seconds_until_retry(self) -> float:
        with self._lock:
            if self._next_retry_at is None:
                return self.retry_interval_s
            return max(0.0, (self._next_retry_at - _utc_now()).total_seconds())

    def _retry_due(self) -> bool:
        with self._lock:
            return self._next_retry_at is None or _utc_now() >= self._next_retry_at

    def _close_source_locked(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass
        self._source = None
        self._reward_source = None

    @staticmethod
    def _default_source() -> EmotivCortexSource:
        client_id, client_secret = emotiv_credentials()
        return EmotivCortexSource(client_id, client_secret)
