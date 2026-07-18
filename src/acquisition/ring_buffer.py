"""Timestamped multi-channel EEG ring buffer with marker-aligned epoch extraction.

The closed-loop presentation must record EEG continuously in the background and,
after a candidate has been held static, pull out exactly the epoch between two
event markers (e.g. CANDIDATE_ONSET .. CANDIDATE_OFFSET) as a `[channels,
samples]` array. Nothing in the repo did this: `EEGFeatureExtractor` keeps a
short per-channel deque for the *trailing* window only, with no timestamps and no
way to select an arbitrary [t0, t1] slice.

`RingBuffer` holds the last `capacity_s` seconds in a fixed channel order, records
each sample's timestamp, and extracts a time-bounded epoch plus the gap/coverage
diagnostics the quality gate needs. It never blocks: pushing is O(1) and
extraction copies only the requested slice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Epoch:
    """A time-bounded slice of the buffer, in a fixed channel order.

    `data` is `[n_channels, n_samples]`. `times` are the per-sample timestamps.
    `max_gap_s` is the largest inter-sample interval inside the slice; `coverage`
    is the fraction of the requested [t0, t1) window that actually contains
    samples (1.0 = no missing stretches at the nominal rate).
    """

    data: np.ndarray
    times: np.ndarray
    channels: list[str]
    t0: float
    t1: float
    fs: float
    max_gap_s: float
    coverage: float

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1]) if self.data.ndim == 2 else 0

    @property
    def duration_s(self) -> float:
        return float(self.t1 - self.t0)


class RingBuffer:
    """Fixed-channel, timestamped EEG ring buffer.

    Channel order is pinned at construction so every extracted epoch has the same
    row order regardless of dict iteration order in incoming samples. Missing
    channels in a pushed sample are filled with NaN (and surfaced by the quality
    gate as flat/absent), never silently dropped or reordered.
    """

    def __init__(self, channels: list[str], sample_rate_hz: float, capacity_s: float = 30.0):
        if not channels:
            raise ValueError("RingBuffer needs a non-empty channel list")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.channels = list(channels)
        self._index = {c: i for i, c in enumerate(self.channels)}
        self.fs = float(sample_rate_hz)
        self.capacity_s = float(capacity_s)
        maxlen = max(1, int(round(self.fs * self.capacity_s)))
        self._times: deque[float] = deque(maxlen=maxlen)
        self._rows: deque[np.ndarray] = deque(maxlen=maxlen)
        self._n_pushed = 0

    # -- writing -----------------------------------------------------------
    def push(self, t: float, sample: dict[str, float]) -> None:
        """Append one timestamped multi-channel sample. O(1), non-blocking."""
        row = np.full(len(self.channels), np.nan, dtype=np.float32)
        for ch, v in sample.items():
            i = self._index.get(ch)
            if i is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                row[i] = float(v)
        self._times.append(float(t))
        self._rows.append(row)
        self._n_pushed += 1

    def clear(self) -> None:
        self._times.clear()
        self._rows.clear()

    # -- reading -----------------------------------------------------------
    @property
    def n_pushed(self) -> int:
        return self._n_pushed

    def span(self) -> tuple[float, float] | None:
        """(oldest_t, newest_t) currently buffered, or None if empty."""
        if not self._times:
            return None
        return self._times[0], self._times[-1]

    def extract(self, t0: float, t1: float) -> Epoch:
        """Return the epoch of samples with t0 <= timestamp < t1.

        Raises ValueError if the window is empty or falls outside the buffer;
        callers align [t0, t1] to markers (e.g. CANDIDATE_ONSET + [0.5, 2.5]s).
        """
        if not self._times:
            raise ValueError("RingBuffer is empty")
        if not t1 > t0:
            raise ValueError(f"invalid epoch window: t0={t0} t1={t1}")

        times = np.fromiter(self._times, dtype=float, count=len(self._times))
        lo = int(np.searchsorted(times, t0, side="left"))
        hi = int(np.searchsorted(times, t1, side="left"))
        if hi <= lo:
            raise ValueError(
                f"no samples in [{t0:.3f}, {t1:.3f}); buffer spans {self.span()}"
            )
        rows = list(self._rows)[lo:hi]
        sel_times = times[lo:hi]
        data = np.stack(rows, axis=1)  # [channels, samples]

        diffs = np.diff(sel_times)
        max_gap = float(diffs.max()) if diffs.size else 0.0
        expected = (t1 - t0) * self.fs
        coverage = float(min(1.0, data.shape[1] / expected)) if expected > 0 else 0.0
        return Epoch(
            data=data,
            times=sel_times,
            channels=list(self.channels),
            t0=t0,
            t1=t1,
            fs=self.fs,
            max_gap_s=max_gap,
            coverage=coverage,
        )

    def extract_around(self, onset: float, start_offset: float, end_offset: float) -> Epoch:
        """Epoch aligned to a marker onset: [onset+start_offset, onset+end_offset)."""
        return self.extract(onset + start_offset, onset + end_offset)
