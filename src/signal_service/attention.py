"""Real-time attention / relaxation from EMOTIV Cortex performance metrics.

This is a *separate* path from the raw-EEG preference pipeline. Instead of band
power, it consumes Cortex's `met` (performance-metrics) stream - EMOTIV's own
on-device estimates of focus (attention), engagement, relaxation, stress, etc. -
gated by the `eq` (EEG-quality) stream. The question it answers: when the
subject sees an image they want vs. don't want, does measured attention/relaxation
move reliably?

Pipeline (matches the intended design):

    EPOC X -> Cortex `met` @ ~2 Hz -> check foc.isActive -> check `eq` quality
      -> 3 s median / EMA smoothing -> subject-specific normalization
      -> real-time concentration confidence in [0, 1]

`CortexMetricsSource` reuses `EmotivCortexSource`'s auth/session handshake and
just subscribes to `met` + `eq`. `MockMetricsSource` synthesizes a scripted
attention signal so the whole chain can be exercised without a headset.
`AttentionMonitor` is pure (no I/O): feed it `MetricSample`s, get `MetricReading`s.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from src.signal_service.eeg_sources import EmotivCortexSource

# Cortex `met` short codes -> human names. Attention is `foc` (focus); the
# emotional-state metrics carry a `<code>.isActive` reliability flag.
MET_ATTENTION = "foc"
MET_RELAXATION = "rel"


@dataclass
class MetricSample:
    """One parsed `met` reading with the latest known `eq` quality attached."""

    t: float
    attention: float | None
    relaxation: float | None
    attention_active: bool
    relaxation_active: bool
    quality_overall: float | None
    raw: dict = field(default_factory=dict)
    quality_raw: dict = field(default_factory=dict)


@dataclass
class MetricReading:
    """Processed output: smoothed, normalized, and turned into a confidence."""

    t: float
    reliable: bool
    quality_overall: float | None
    attention_raw: float | None
    relaxation_raw: float | None
    attention_smooth: float
    relaxation_smooth: float
    attention_z: float
    relaxation_z: float
    concentration_confidence: float  # in [0, 1], from normalized attention
    relaxation_confidence: float     # in [0, 1], from normalized relaxation


# ---------------------------------------------------------------------------
# Smoothing + normalization (small, pure helpers)
# ---------------------------------------------------------------------------
class _Smoother:
    """EMA or trailing-median smoother over a real-time metric stream."""

    def __init__(self, mode: str = "ema", tau_s: float = 3.0,
                 window_s: float = 3.0, dt_s: float = 0.5):
        if mode not in ("ema", "median"):
            raise ValueError(f"unknown smoothing mode: {mode}")
        self.mode = mode
        self.window_s = window_s
        # EMA weight for one step at the nominal sample period; slower tau = smoother.
        self._alpha = 1.0 - math.exp(-dt_s / max(tau_s, 1e-6))
        self._ema: float | None = None
        self._buf: deque[tuple[float, float]] = deque()

    def push(self, t: float, x: float) -> None:
        if self.mode == "ema":
            self._ema = x if self._ema is None else (
                self._alpha * x + (1.0 - self._alpha) * self._ema
            )
        else:
            self._buf.append((t, x))
            while self._buf and t - self._buf[0][0] > self.window_s:
                self._buf.popleft()

    def value(self, default: float = 0.0) -> float:
        if self.mode == "ema":
            return default if self._ema is None else float(self._ema)
        if not self._buf:
            return default
        return float(np.median([v for _t, v in self._buf]))


class _Normalizer:
    """Subject-specific z-scoring. Running Welford stats until a baseline is
    frozen; afterwards a fixed subject mean/std (the calibration reference)."""

    def __init__(self, std_floor: float = 1e-3):
        self.std_floor = std_floor
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._fixed: tuple[float, float] | None = None

    def observe(self, x: float) -> None:
        self._n += 1
        d = x - self._mean
        self._mean += d / self._n
        self._m2 += d * (x - self._mean)

    def _running_std(self) -> float:
        if self._n < 2:
            return self.std_floor
        return max(math.sqrt(self._m2 / (self._n - 1)), self.std_floor)

    def freeze_baseline(self) -> tuple[float, float]:
        self._fixed = (self._mean, self._running_std())
        return self._fixed

    @property
    def has_baseline(self) -> bool:
        return self._fixed is not None

    def z(self, x: float) -> float:
        if self._fixed is not None:
            mean, std = self._fixed
        else:
            if self._n < 2:
                return 0.0
            mean, std = self._mean, self._running_std()
        return (x - mean) / std


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class AttentionMonitor:
    """Gate -> smooth -> subject-normalize -> confidence, per metric.

    Call `update(sample)` for each `MetricSample`. Unreliable samples (metric
    inactive or poor EEG quality) don't move the smoother/normalizer; the last
    good values are carried so the confidence display stays steady, and the
    reading is flagged `reliable=False`.
    """

    def __init__(
        self,
        smoothing: str = "ema",
        tau_s: float = 3.0,
        median_window_s: float = 3.0,
        met_rate_hz: float = 2.0,
        quality_min: float = 2.0,
        confidence_gain: float = 1.0,
        online_adapt: bool = True,
    ):
        dt = 1.0 / max(met_rate_hz, 1e-6)
        self.quality_min = quality_min
        self.confidence_gain = confidence_gain
        self.online_adapt = online_adapt
        self._att_smooth = _Smoother(smoothing, tau_s, median_window_s, dt)
        self._rel_smooth = _Smoother(smoothing, tau_s, median_window_s, dt)
        self._att_norm = _Normalizer()
        self._rel_norm = _Normalizer()
        self._last_att = 0.0
        self._last_rel = 0.0

    def _quality_ok(self, sample: MetricSample) -> bool:
        if sample.quality_overall is None:
            return True  # no eq yet -> don't block; caller can require it separately
        return sample.quality_overall >= self.quality_min

    def freeze_baseline(self) -> None:
        """Lock in the subject's resting mean/std as the normalization reference."""
        self._att_norm.freeze_baseline()
        self._rel_norm.freeze_baseline()

    @property
    def has_baseline(self) -> bool:
        return self._att_norm.has_baseline

    def update(self, sample: MetricSample) -> MetricReading:
        quality_ok = self._quality_ok(sample)
        att_reliable = quality_ok and sample.attention_active and sample.attention is not None
        rel_reliable = quality_ok and sample.relaxation_active and sample.relaxation is not None

        if att_reliable:
            self._att_smooth.push(sample.t, float(sample.attention))
            if self.online_adapt or not self._att_norm.has_baseline:
                # Feed baseline stats only while unfrozen (calibration phase);
                # once frozen, online_adapt keeps a slow running estimate for z
                # only if no baseline was set.
                if not self._att_norm.has_baseline:
                    self._att_norm.observe(self._att_smooth.value())
        if rel_reliable:
            self._rel_smooth.push(sample.t, float(sample.relaxation))
            if not self._rel_norm.has_baseline:
                self._rel_norm.observe(self._rel_smooth.value())

        att_s = self._att_smooth.value(self._last_att)
        rel_s = self._rel_smooth.value(self._last_rel)
        self._last_att, self._last_rel = att_s, rel_s

        att_z = self._att_norm.z(att_s)
        rel_z = self._rel_norm.z(rel_s)
        return MetricReading(
            t=sample.t,
            reliable=att_reliable,
            quality_overall=sample.quality_overall,
            attention_raw=sample.attention,
            relaxation_raw=sample.relaxation,
            attention_smooth=att_s,
            relaxation_smooth=rel_s,
            attention_z=att_z,
            relaxation_z=rel_z,
            concentration_confidence=_sigmoid(self.confidence_gain * att_z),
            relaxation_confidence=_sigmoid(self.confidence_gain * rel_z),
        )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def _cols_to_dict(cols: list[str] | None, values: list) -> dict:
    if not cols:
        return {}
    return {c: v for c, v in zip(cols, values)}


def _eq_overall_from_dict(d: dict) -> float | None:
    """Overall EEG quality from a parsed `eq` message (0-4 scale in Cortex)."""
    for key in ("overall", "OVERALL", "Overall"):
        if key in d and isinstance(d[key], (int, float)):
            return float(d[key])
    # Fallback: mean of the per-channel quality columns (drop battery/sample-rate).
    skip = {"batteryPercent", "sampleRateQuality", "SampleRateQuality"}
    per_ch = [float(v) for k, v in d.items()
              if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool)]
    return float(np.mean(per_ch)) if per_ch else None


class CortexMetricsSource(EmotivCortexSource):
    """Subscribes to Cortex `met` + `eq` and yields `MetricSample`s.

    Reuses the parent's auth/headset/session handshake; only the subscription
    and message parsing differ. `met` arrives at ~2 Hz (sometimes slower); each
    `met` message is emitted with the most recent `eq` overall quality attached.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._met_cols: list[str] | None = None
        self._eq_cols: list[str] | None = None

    @staticmethod
    def _extract_cols(subscription_result: dict, stream: str) -> list[str] | None:
        for item in subscription_result.get("success", []):
            if item.get("streamName") == stream and item.get("cols"):
                return list(item["cols"])
        return None

    def connect(self) -> None:
        self._open_session()
        sub = self._subscribe(["met", "eq"])
        self._met_cols = self._extract_cols(sub, "met")
        self._eq_cols = self._extract_cols(sub, "eq")
        failures = [f for f in sub.get("failure", [])]
        if failures and self._met_cols is None:
            raise RuntimeError(f"Cortex did not accept the 'met' subscription: {failures}")

    def stream(self) -> Iterator[MetricSample]:  # type: ignore[override]
        import json

        assert self._ws is not None, "call connect() first"
        latest_quality: float | None = None
        latest_eq: dict = {}
        while True:
            msg = json.loads(self._ws.recv())
            if "eq" in msg:
                latest_eq = _cols_to_dict(self._eq_cols, msg["eq"])
                latest_quality = _eq_overall_from_dict(latest_eq)
                continue
            if "met" not in msg:
                continue
            d = _cols_to_dict(self._met_cols, msg["met"])
            t = msg.get("time", time.time())
            yield MetricSample(
                t=t,
                attention=_as_float(d.get(MET_ATTENTION)),
                relaxation=_as_float(d.get(MET_RELAXATION)),
                attention_active=bool(d.get(f"{MET_ATTENTION}.isActive", True)),
                relaxation_active=bool(d.get(f"{MET_RELAXATION}.isActive", True)),
                quality_overall=latest_quality,
                raw=d,
                quality_raw=latest_eq,
            )


def _as_float(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class MockMetricsSource:
    """Scripted attention/relaxation for offline testing (no headset).

    Emits a resting baseline, then alternating 'wanted' (high attention, lower
    relaxation) and 'unwanted' (low attention) blocks, so the monitor and its
    normalization can be validated at a known ground truth. A configurable early
    stretch has `attention_active=False` / poor quality to exercise the gate.
    """

    def __init__(
        self,
        rate_hz: float = 2.0,
        baseline_s: float = 10.0,
        block_s: float = 8.0,
        n_blocks: int = 6,
        noise: float = 0.05,
        bad_quality_until_s: float = 3.0,
        seed: int = 0,
        realtime: bool = False,
    ):
        self.rate_hz = rate_hz
        self.baseline_s = baseline_s
        self.block_s = block_s
        self.n_blocks = n_blocks
        self.noise = noise
        self.bad_quality_until_s = bad_quality_until_s
        self.realtime = realtime
        self._rng = np.random.default_rng(seed)

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def ground_truth(self, t: float) -> str:
        if t < self.baseline_s:
            return "baseline"
        k = int((t - self.baseline_s) // self.block_s)
        if k >= self.n_blocks:
            return "done"
        return "wanted" if k % 2 == 0 else "unwanted"

    def stream(self) -> Iterator[MetricSample]:
        dt = 1.0 / self.rate_hz
        total = self.baseline_s + self.block_s * self.n_blocks
        t = 0.0
        while t <= total:
            label = self.ground_truth(t)
            att_mean = {"baseline": 0.45, "wanted": 0.72, "unwanted": 0.28, "done": 0.45}[label]
            rel_mean = {"baseline": 0.55, "wanted": 0.40, "unwanted": 0.60, "done": 0.55}[label]
            att = float(np.clip(att_mean + self._rng.normal(0, self.noise), 0, 1))
            rel = float(np.clip(rel_mean + self._rng.normal(0, self.noise), 0, 1))
            bad = t < self.bad_quality_until_s
            yield MetricSample(
                t=t,
                attention=att,
                relaxation=rel,
                attention_active=not bad,
                relaxation_active=not bad,
                quality_overall=1.0 if bad else 4.0,
                raw={"label": label},
            )
            if self.realtime:
                time.sleep(dt)
            t += dt
