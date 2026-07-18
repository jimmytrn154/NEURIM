"""Acquisition runner: drains a multi-stream backend into a ring buffer without
blocking the presentation loop, and tracks contact quality + head motion.

Design: a `Backend` yields typed `Record`s (eeg / contact-quality / motion). The
`Acquisition` runner ingests them - either on a background thread (`start()`, for
live Cortex, so the render/present loop never waits on the socket) or pulled
synchronously (`pump()`, for deterministic tests). Markers are stamped in the
*stream* timebase (the latest EEG sample time), so `extract_epoch` around a
marker returns exactly the samples that post-date it.

Two backends:
  - `CortexBackend` wraps the existing `EmotivCortexSource` and subscribes to
    `eeg` + `dev` (contact quality) + `mot` (motion). Reuses the parent's
    auth/session handshake; only parsing differs. (Not unit-tested - no hardware.)
  - `MockBackend` synthesizes realistic eeg/quality/motion and injects a
    settable satisfaction signal, so the whole pipeline runs offline.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from src.acquisition.markers import MarkerLog
from src.acquisition.quality import QualityResult, QualityThresholds, evaluate_epoch
from src.acquisition.ring_buffer import Epoch, RingBuffer


@dataclass
class Record:
    """One item from a backend. Exactly one payload field is set."""

    t: float
    eeg: dict[str, float] | None = None
    contact: dict[str, float] | None = None   # channel -> 0-4 contact quality
    motion_mag: float | None = None           # scalar head-motion magnitude


class Backend(Protocol):
    def connect(self) -> None: ...
    def records(self) -> Iterator[Record]: ...
    def close(self) -> None: ...


class Acquisition:
    def __init__(
        self,
        backend: Backend,
        channels: list[str],
        sample_rate_hz: float,
        capacity_s: float = 30.0,
        motion_window_s: float = 2.0,
        thresholds: QualityThresholds | None = None,
    ):
        self.backend = backend
        self.channels = list(channels)
        self.fs = float(sample_rate_hz)
        self.ring = RingBuffer(channels, sample_rate_hz, capacity_s)
        self.markers = MarkerLog(clock=self.now)
        self.thresholds = thresholds or QualityThresholds()
        self._contact: dict[str, float] = {}
        self._motion: deque[tuple[float, float]] = deque()
        self._motion_window_s = motion_window_s
        self._last_t: float = 0.0
        self._it: Iterator[Record] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        self.backend.connect()
        self._it = self.backend.records()

    def now(self) -> float:
        """Current stream time = timestamp of the most recent EEG sample."""
        return self._last_t

    def _ingest(self, rec: Record) -> None:
        self._last_t = rec.t
        if rec.eeg is not None:
            self.ring.push(rec.t, rec.eeg)
        if rec.contact is not None:
            self._contact = dict(rec.contact)
        if rec.motion_mag is not None:
            self._motion.append((rec.t, float(rec.motion_mag)))
            cutoff = rec.t - self._motion_window_s
            while self._motion and self._motion[0][0] < cutoff:
                self._motion.popleft()

    def pump(self, max_records: int) -> int:
        """Synchronously ingest up to `max_records`. Returns the count ingested.
        Used by the collection driver and tests instead of a live thread."""
        assert self._it is not None, "call connect() first"
        n = 0
        for _ in range(max_records):
            try:
                rec = next(self._it)
            except StopIteration:
                break
            self._ingest(rec)
            n += 1
        return n

    def pump_seconds(self, seconds: float) -> int:
        """Ingest roughly `seconds` worth of samples at the nominal rate."""
        return self.pump(max(1, int(round(seconds * self.fs))))

    def start(self) -> None:
        """Run a background reader thread (live mode)."""
        assert self._it is not None, "call connect() first"
        self._stop.clear()

        def _run() -> None:
            assert self._it is not None
            for rec in self._it:
                if self._stop.is_set():
                    break
                self._ingest(rec)

        self._thread = threading.Thread(target=_run, name="acquisition", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        self.stop()
        self.backend.close()

    # -- reads -------------------------------------------------------------
    def latest_contact(self) -> dict[str, float]:
        return dict(self._contact)

    def motion_rms(self, t0: float, t1: float) -> float | None:
        vals = [m for (t, m) in self._motion if t0 <= t < t1]
        if not vals:
            return None
        return float(np.sqrt(np.mean(np.square(vals))))

    def extract_epoch(self, onset: float, start_offset: float, end_offset: float) -> Epoch:
        return self.ring.extract_around(onset, start_offset, end_offset)

    def evaluate(self, epoch: Epoch) -> QualityResult:
        """Quality-gate an epoch using the latest contact quality + motion."""
        return evaluate_epoch(
            epoch,
            self.channels,
            thresholds=self.thresholds,
            contact_quality=self.latest_contact() or None,
            motion_rms=self.motion_rms(epoch.t0, epoch.t1),
        )


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------
class MockBackend:
    """Synthetic multi-stream EEG with an injectable satisfaction signal.

    `set_satisfaction(s)` in [-1, 1] modulates parietal/occipital band power so a
    decoder has a separable (but noisy) signal - the offline analogue of a real
    satisfied/dissatisfied response. Contact quality and motion default to good;
    `inject_bad_contact` / `inject_motion` exercise the quality gate.
    """

    def __init__(
        self,
        channels: list[str],
        sample_rate_hz: float = 128.0,
        amp: float = 20.0,
        noise: float = 0.15,
        signal_gain: float = 0.5,
        seed: int = 0,
        contact_rate_hz: float = 2.0,
        motion_rate_hz: float = 32.0,
    ):
        self.channels = list(channels)
        self.fs = float(sample_rate_hz)
        self.amp = amp
        self.noise = noise
        self.signal_gain = signal_gain
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._sat = 0.0
        self._contact_every = max(1, int(self.fs / contact_rate_hz))
        self._motion_every = max(1, int(self.fs / motion_rate_hz))
        self._k = 0
        self.inject_bad_contact: set[str] = set()
        self.inject_motion: float = 0.0
        # Parietal/occipital carry the satisfaction modulation.
        self._modulated = {"P7", "P8", "O1", "O2", "AF3", "AF4"}

    def set_satisfaction(self, s: float) -> None:
        self._sat = float(np.clip(s, -1.0, 1.0))

    def connect(self) -> None:
        self._t = 0.0
        self._k = 0

    def close(self) -> None:
        pass

    def _eeg_sample(self) -> dict[str, float]:
        t = self._t
        out: dict[str, float] = {}
        for ch in self.channels:
            phase = (hash(ch) % 100) / 100.0 * 2 * np.pi
            gain = 1.0 + (self.signal_gain * self._sat if ch in self._modulated else 0.0)
            sig = (np.sin(2 * np.pi * 5.0 * t + phase)
                   + 0.8 * gain * np.sin(2 * np.pi * 10.0 * t + phase)
                   + 0.4 * np.sin(2 * np.pi * 18.0 * t + phase))
            out[ch] = float(self.amp * sig + self.noise * self.amp * self._rng.standard_normal())
        return out

    def records(self) -> Iterator[Record]:
        dt = 1.0 / self.fs
        while True:
            yield Record(t=self._t, eeg=self._eeg_sample())
            if self._k % self._contact_every == 0:
                cq = {c: (0.0 if c in self.inject_bad_contact else 4.0) for c in self.channels}
                yield Record(t=self._t, contact=cq)
            if self._k % self._motion_every == 0:
                base = 0.2 + self.inject_motion
                yield Record(t=self._t, motion_mag=float(abs(self._rng.normal(base, 0.05))))
            self._t += dt
            self._k += 1


# ---------------------------------------------------------------------------
# Cortex backend (live; not unit-tested - requires hardware)
# ---------------------------------------------------------------------------
class CortexBackend:
    """Live EPOC X backend: eeg + dev (contact quality) + mot (motion)."""

    def __init__(self, client_id, client_secret, channels: list[str], headset_id=None):
        from src.signal_service.eeg_sources import EmotivCortexSource

        self._src = EmotivCortexSource(client_id, client_secret, headset_id=headset_id)
        self.channels = list(channels)
        self._cols: dict[str, list[str]] = {}

    def connect(self) -> None:
        self._src._open_session()
        sub = self._src._subscribe(["eeg", "dev", "mot"])
        for item in sub.get("success", []):
            if item.get("cols"):
                self._cols[item["streamName"]] = list(item["cols"])
        self._src._eeg_cols = self._cols.get("eeg")

    def _parse_contact(self, values: list) -> dict[str, float]:
        """dev CQ is a per-channel sublist; map it to our channel order."""
        cols = self._cols.get("dev") or []
        d = dict(zip(cols, values))
        cq = d.get("cq") or d.get("CQ")
        if isinstance(cq, list):
            return {c: float(v) for c, v in zip(self.channels, cq)
                    if isinstance(v, (int, float))}
        # Some Cortex versions flatten CQ into per-channel columns.
        return {c: float(d[c]) for c in self.channels
                if isinstance(d.get(c), (int, float))}

    def _parse_motion(self, values: list) -> float | None:
        cols = self._cols.get("mot") or []
        d = dict(zip(cols, values))
        acc = [d.get(k) for k in ("ACCX", "ACCY", "ACCZ", "Q0", "Q1", "Q2", "Q3")]
        acc = [float(a) for a in acc if isinstance(a, (int, float))]
        return float(np.linalg.norm(acc)) if acc else None

    def records(self) -> Iterator[Record]:
        import json
        import time as _time

        ws = self._src._ws
        assert ws is not None, "call connect() first"
        eeg_cols = self._cols.get("eeg")
        while True:
            msg = json.loads(ws.recv())
            t = msg.get("time", _time.time())
            if "eeg" in msg and eeg_cols:
                sample = {c: v for c, v in zip(eeg_cols, msg["eeg"])
                          if self._src._is_eeg_sensor_col(c, self.channels)
                          and isinstance(v, (int, float))}
                yield Record(t=t, eeg=sample)
            elif "dev" in msg:
                yield Record(t=t, contact=self._parse_contact(msg["dev"]))
            elif "mot" in msg:
                mag = self._parse_motion(msg["mot"])
                if mag is not None:
                    yield Record(t=t, motion_mag=mag)

    def close(self) -> None:
        self._src.close()
