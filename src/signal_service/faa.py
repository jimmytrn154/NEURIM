"""Frontal alpha asymmetry: the entire "reward" signal.

FAA = ln(alpha_power(right)) - ln(alpha_power(left)), combined across frontal
mirror pairs with fixed pair weights, z-scored against a per-subject resting
baseline and clipped to [-1, 1]. Higher = more left-frontal activation =
approach motivation, per the standard EEG asymmetry literature. This module
only knows about numbers in, numbers out - it never sees the image generator,
the optimizer, or anything about images.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import welch


EPOC_X_POSITIONS: dict[str, tuple[float, float, float]] = {
    "AF3": (-0.42, 0.88, 0.22),
    "F7": (-0.86, 0.58, 0.04),
    "F3": (-0.46, 0.55, 0.36),
    "FC5": (-0.72, 0.22, 0.22),
    "T7": (-0.95, -0.08, 0.0),
    "P7": (-0.78, -0.58, 0.1),
    "O1": (-0.34, -0.9, 0.18),
    "O2": (0.34, -0.9, 0.18),
    "P8": (0.78, -0.58, 0.1),
    "T8": (0.95, -0.08, 0.0),
    "FC6": (0.72, 0.22, 0.22),
    "F4": (0.46, 0.55, 0.36),
    "F8": (0.86, 0.58, 0.04),
    "AF4": (0.42, 0.88, 0.22),
}

DEFAULT_FAA_PAIRS: tuple[tuple[str, str], ...] = (
    ("F7", "F8"),
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("FC5", "FC6"),
)

DEFAULT_FAA_PAIR_WEIGHTS: dict[str, float] = {
    "F3/F4": 1.0,
    "F7/F8": 0.75,
    "AF3/AF4": 0.5,
    "FC5/FC6": 0.5,
}


@dataclass
class PairFAAMetrics:
    left: str
    right: str
    power_left: float
    power_right: float
    raw_faa: float
    pair_weight: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "left": self.left,
            "right": self.right,
            "power_left": self.power_left,
            "power_right": self.power_right,
            "raw_faa": self.raw_faa,
            "pair_weight": self.pair_weight,
        }


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


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    if total <= 0:
        return None
    return float(np.average(values, weights=weights))


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
    """Sliding-window weighted FAA -> baseline z-score -> clip to [-1, 1] = r(t).

    Feed it raw multi-channel samples as they arrive; call `reward()` on the
    cadence you want reward readings and it slices the trailing `window_s`
    seconds out of its ring buffer.
    """

    def __init__(
        self,
        fs: float,
        channel_left: str = "F3",
        channel_right: str = "F4",
        band: tuple[float, float] = (8.0, 13.0),
        window_s: float = 3.0,
        clip: tuple[float, float] = (-1.0, 1.0),
        channels: list[str] | None = None,
        channel_pairs: list[list[str]] | list[tuple[str, str]] | None = None,
        pair_weights: dict[str, float] | None = None,
    ):
        self.fs = fs
        self.channel_left = channel_left
        self.channel_right = channel_right
        self.channel_pairs = self._normalize_pairs(channel_pairs)
        self.pair_weights = self._normalize_pair_weights(pair_weights)
        self.band = band
        self.window_s = window_s
        self.clip = clip
        self._maxlen = int(fs * window_s) + 1
        pair_channels = [ch for pair in self.channel_pairs for ch in pair]
        channel_names = list(dict.fromkeys([*(channels or []), *pair_channels]))
        self._buffers: dict[str, deque[float]] = {
            ch: deque(maxlen=self._maxlen) for ch in channel_names
        }
        self.baseline = RunningBaseline()

    def _normalize_pairs(
        self,
        channel_pairs: list[list[str]] | list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]]:
        pairs = channel_pairs or DEFAULT_FAA_PAIRS
        normalized = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"FAA channel pair must contain exactly 2 channels: {pair}")
            normalized.append((str(pair[0]), str(pair[1])))
        return normalized

    def _normalize_pair_weights(self, pair_weights: dict[str, float] | None) -> dict[str, float]:
        weights = dict(DEFAULT_FAA_PAIR_WEIGHTS)
        if pair_weights:
            for label, raw_weight in pair_weights.items():
                value = _safe_float(raw_weight)
                if value is None or value < 0:
                    raise ValueError(f"FAA pair weight must be a non-negative number: {label}")
                weights[str(label)] = value
        return weights

    def push_sample(self, channel_values: dict[str, Any]) -> None:
        for ch, value in channel_values.items():
            if not isinstance(value, int | float):
                continue
            if ch not in self._buffers:
                self._buffers[ch] = deque(maxlen=self._maxlen)
            self._buffers[ch].append(float(value))

    def ready(self) -> bool:
        return any(
            len(self._buffers.get(left, ())) >= self._maxlen - 1
            and len(self._buffers.get(right, ())) >= self._maxlen - 1
            for left, right in self.channel_pairs
        )

    def _calculate_pair_metrics(self) -> list[PairFAAMetrics]:
        metrics = []
        for left, right in self.channel_pairs:
            if len(self._buffers.get(left, ())) < self._maxlen - 1:
                continue
            if len(self._buffers.get(right, ())) < self._maxlen - 1:
                continue
            left_samples = np.asarray(self._buffers[left], dtype=float)
            right_samples = np.asarray(self._buffers[right], dtype=float)
            p_left = band_power(left_samples, self.fs, self.band)
            p_right = band_power(right_samples, self.fs, self.band)
            raw = float(np.log(p_right + 1e-12) - np.log(p_left + 1e-12))
            label = f"{left}/{right}"
            metrics.append(
                PairFAAMetrics(
                    left=left,
                    right=right,
                    power_left=p_left,
                    power_right=p_right,
                    raw_faa=raw,
                    pair_weight=self.pair_weights.get(label, 1.0),
                )
            )
        return metrics

    def _aggregate_pair_metrics(self, metrics: list[PairFAAMetrics]) -> float | None:
        if not metrics:
            return None
        values = np.asarray([metric.raw_faa for metric in metrics], dtype=float)
        weights = np.asarray([metric.pair_weight for metric in metrics], dtype=float)
        return _weighted_average(values, weights)

    def raw_value(self) -> float | None:
        if not self.ready():
            return None
        return self._aggregate_pair_metrics(self._calculate_pair_metrics())

    def pair_metrics(self) -> list[dict[str, float | str]]:
        """Pair-level alpha power and raw FAA values for diagnostics."""
        return [metric.as_dict() for metric in self._calculate_pair_metrics()]

    def eeg_features(self, reward: float | None = None, raw: float | None = None) -> dict | None:
        """Compact EEG visualization payload for the frontend.

        Uses alpha-band power over the same sliding window as FAA. This is
        intentionally low-rate derived telemetry, not raw high-frequency EEG.
        """
        if not self.ready():
            return None

        channels = []
        for name, buf in self._buffers.items():
            if not buf:
                continue
            arr = np.asarray(buf, dtype=float)
            alpha = band_power(arr, self.fs, self.band) if arr.size >= 8 else 0.0
            channels.append(
                {
                    "name": name,
                    "value": float(arr[-1]),
                    "alpha_power": float(alpha),
                    "quality": min(1.0, len(buf) / max(1, self._maxlen - 1)),
                    "position": list(EPOC_X_POSITIONS.get(name, (0.0, 0.0, 0.0))),
                }
            )

        return {
            "channels": channels,
            "faa": {
                "raw": raw if raw is not None else self.raw_value(),
                "reward": reward if reward is not None else self.reward(),
                "left_channel": self.channel_left,
                "right_channel": self.channel_right,
                "channel_pairs": [list(pair) for pair in self.channel_pairs],
                "pair_metrics": self.pair_metrics(),
            },
        }

    def reward(self) -> float | None:
        """Baseline-normalized r(t), or None if the buffer isn't full yet."""
        value = self.raw_value()
        if value is None:
            return None
        z = self.baseline.z_score(value)
        lo, hi = self.clip
        return float(np.clip(z, lo, hi))
