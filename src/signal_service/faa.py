"""Frontal alpha asymmetry: the entire "reward" signal.

FAA = ln(alpha_power(F4)) - ln(alpha_power(F3)), z-scored against a per-subject
resting baseline and clipped to [-1, 1]. Higher = more left-frontal activation
= approach motivation, per the standard EEG asymmetry literature. This module
only knows about numbers in, numbers out - it never sees the image generator,
the optimizer, or anything about images.

F3 = Channel_3.csv and F4 = Channel_12.csv.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


def band_power(samples: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Welch PSD power in `band` (Hz) for a single-channel 1D signal."""
    if samples.size < 8:
        return 0.0
    nperseg = min(samples.size, max(int(fs * 1.0), 8))
    freqs, psd = welch(samples, fs=fs, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    band_freqs, band_psd = freqs[mask], psd[mask]
    if band_freqs.size < 2:
        return float(band_psd.sum())
    return float(np.sum((band_psd[1:] + band_psd[:-1]) * np.diff(band_freqs)) / 2.0)


def raw_faa(
    window: dict[str, np.ndarray],
    fs: float,
    channel_left: str = "F3",
    channel_right: str = "F4",
    band: tuple[float, float] = (8.0, 13.0),
    eps: float = 1e-12,
) -> float:
    """ln(power_right) - ln(power_left) over one window of samples per channel."""
    p_left = band_power(window[channel_left], fs, band) + eps
    p_right = band_power(window[channel_right], fs, band) + eps
    return float(np.log(p_right) - np.log(p_left))


@dataclass
class RunningBaseline:
    """Mean/std of raw FAA collected during the rest period, for z-scoring."""

    mean: float = 0.0
    std: float = 1.0
    n: int = 0

    def fit(self, samples: list[float]) -> None:
        arr = np.asarray(samples, dtype=float)
        self.mean = float(arr.mean())
        self.std = float(arr.std()) or 1.0
        self.n = len(samples)

    def z_score(self, value: float) -> float:
        return (value - self.mean) / self.std


class FAARewardComputer:
    """Sliding-window FAA -> baseline z-score -> clip to [-1, 1] = r(t).

    Feed it raw multi-channel samples as they arrive; call `update()` on the
    cadence you want reward readings (every ~250ms per the spec) and it slices
    the trailing `window_s` seconds out of its ring buffer.
    """

    def __init__(
        self,
        fs: float,
        channel_left: str = "F3",
        channel_right: str = "F4",
        band: tuple[float, float] = (8.0, 13.0),
        window_s: float = 2.0,
        clip: tuple[float, float] = (-1.0, 1.0),
    ):
        self.fs = fs
        self.channel_left = channel_left
        self.channel_right = channel_right
        self.band = band
        self.window_s = window_s
        self.clip = clip
        self._maxlen = int(fs * window_s) + 1
        self._buffers: dict[str, deque[float]] = {
            channel_left: deque(maxlen=self._maxlen),
            channel_right: deque(maxlen=self._maxlen),
        }
        self.baseline = RunningBaseline()

    def push_sample(self, channel_values: dict[str, float]) -> None:
        for ch in (self.channel_left, self.channel_right):
            if ch in channel_values:
                self._buffers[ch].append(channel_values[ch])

    def ready(self) -> bool:
        return len(self._buffers[self.channel_left]) >= self._maxlen - 1

    def raw_value(self) -> float | None:
        if not self.ready():
            return None
        window = {ch: np.asarray(buf) for ch, buf in self._buffers.items()}
        return raw_faa(window, self.fs, self.channel_left, self.channel_right, self.band)

    def reward(self) -> float | None:
        """Baseline-normalized r(t), or None if the buffer isn't full yet."""
        value = self.raw_value()
        if value is None:
            return None
        z = self.baseline.z_score(value)
        lo, hi = self.clip
        return float(np.clip(z, lo, hi))
