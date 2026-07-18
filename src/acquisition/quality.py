"""Signal-quality gate for an evaluation epoch (spec §12).

Returns one of `valid` / `retry` / `invalid`. The critical rule the whole design
rests on: **invalid EEG is missing evidence, not dissatisfaction** - it must never
become a negative label and must never update a model (spec constraints 5-6). So
the gate only classifies signal integrity; it says nothing about the subject's
state.

Distinction:
  - invalid : structurally unusable (wrong channels, most channels flat/absent,
              almost no samples). The epoch is discarded.
  - retry   : recoverable contamination (a few bad channels, head motion,
              borderline coverage/noise). The trial should be repeated.
  - valid   : usable for training/inference.

Thresholds are configurable with EPOC-X-oriented defaults; the raw measurements
are always returned so windows can be re-judged offline with different cutoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.acquisition.ring_buffer import Epoch

Status = Literal["valid", "retry", "invalid"]


@dataclass
class QualityThresholds:
    # Coverage / timing
    min_coverage_retry: float = 0.9      # below -> retry (some samples missing)
    min_coverage_invalid: float = 0.5    # below -> invalid (barely any data)
    max_gap_s: float = 0.1               # largest inter-sample gap tolerated
    # Per-channel integrity (data units = whatever the source provides, e.g. µV)
    flat_std: float = 0.5                # channel std below this -> flat/dead
    max_peak_to_peak: float = 1500.0     # above -> saturated/clipping
    max_drift: float = 800.0             # |first-half mean - second-half mean|
    # std(diff)/std ; above -> broadband HF/EMG. Band-limited EEG (~1-40 Hz at
    # 128 Hz) sits ~0.4-0.9; pure white/EMG noise approaches sqrt(2) ~ 1.41.
    hf_noise_ratio: float = 1.2
    # How many bad channels tip the whole epoch over
    max_bad_channel_frac_retry: float = 0.15
    max_bad_channel_frac_invalid: float = 0.5
    # Contact quality (0-4 EPOC scale), if a `dev`/`eq` reading is supplied
    min_contact_quality: float = 2.0
    min_good_contact_frac: float = 0.6
    # Head motion (RMS of `mot` magnitude), if supplied
    max_motion_rms: float = 1.5


@dataclass
class QualityResult:
    status: Status
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    bad_channels: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"


def _channel_flags(data: np.ndarray, th: QualityThresholds) -> dict[int, str]:
    """Map channel index -> failure reason for structurally bad channels."""
    flags: dict[int, str] = {}
    n = data.shape[1]
    for i in range(data.shape[0]):
        x = data[i]
        finite = np.isfinite(x)
        if finite.sum() < max(2, 0.5 * n):
            flags[i] = "absent"
            continue
        xf = x[finite]
        std = float(xf.std())
        if std < th.flat_std:
            flags[i] = "flat"
            continue
        p2p = float(xf.max() - xf.min())
        if p2p > th.max_peak_to_peak:
            flags[i] = "saturated"
            continue
        half = xf.size // 2
        if half >= 1:
            drift = abs(float(xf[:half].mean() - xf[half:].mean()))
            if drift > th.max_drift:
                flags[i] = "drift"
                continue
        hf = float(np.std(np.diff(xf))) / (std + 1e-9)
        if hf > th.hf_noise_ratio:
            flags[i] = "hf_noise"
    return flags


def evaluate_epoch(
    epoch: Epoch,
    expected_channels: list[str],
    thresholds: QualityThresholds | None = None,
    contact_quality: dict[str, float] | None = None,
    motion_rms: float | None = None,
) -> QualityResult:
    """Judge one evaluation epoch. `contact_quality` maps channel -> 0-4 quality
    from the `dev`/`eq` stream; `motion_rms` is the RMS of the `mot` magnitude
    over the epoch. Both optional - absent streams simply skip their checks."""
    th = thresholds or QualityThresholds()
    reasons: list[str] = []
    metrics: dict = {}

    # Structural: channel set + order must match exactly.
    if list(epoch.channels) != list(expected_channels):
        return QualityResult(
            status="invalid",
            reasons=["channel set/order mismatch"],
            metrics={"channels": list(epoch.channels)},
        )

    if epoch.n_samples < 2:
        return QualityResult("invalid", ["epoch has < 2 samples"], {"n_samples": epoch.n_samples})

    metrics["n_samples"] = epoch.n_samples
    metrics["coverage"] = round(epoch.coverage, 4)
    metrics["max_gap_s"] = round(epoch.max_gap_s, 4)

    flags = _channel_flags(epoch.data, th)
    bad_channels = [epoch.channels[i] for i in flags]
    bad_frac = len(flags) / len(epoch.channels)
    metrics["bad_channel_frac"] = round(bad_frac, 4)
    metrics["channel_flags"] = {epoch.channels[i]: r for i, r in flags.items()}

    # Contact quality from dev/eq, if provided.
    good_contact_frac = None
    if contact_quality:
        vals = [contact_quality.get(c) for c in expected_channels]
        present = [v for v in vals if isinstance(v, (int, float))]
        if present:
            good = sum(1 for v in present if v >= th.min_contact_quality)
            good_contact_frac = good / len(present)
            metrics["good_contact_frac"] = round(good_contact_frac, 4)
    if motion_rms is not None:
        metrics["motion_rms"] = round(float(motion_rms), 4)

    # --- invalid conditions (structurally unusable) -----------------------
    if epoch.coverage < th.min_coverage_invalid:
        reasons.append(f"coverage {epoch.coverage:.2f} < {th.min_coverage_invalid}")
        return QualityResult("invalid", reasons, metrics, bad_channels)
    if bad_frac >= th.max_bad_channel_frac_invalid:
        reasons.append(f"{len(flags)}/{len(epoch.channels)} channels bad")
        return QualityResult("invalid", reasons, metrics, bad_channels)
    if good_contact_frac is not None and good_contact_frac < th.min_good_contact_frac / 2:
        reasons.append(f"only {good_contact_frac:.2f} channels have good contact")
        return QualityResult("invalid", reasons, metrics, bad_channels)

    # --- retry conditions (recoverable) -----------------------------------
    if epoch.coverage < th.min_coverage_retry:
        reasons.append(f"coverage {epoch.coverage:.2f} < {th.min_coverage_retry}")
    if epoch.max_gap_s > th.max_gap_s:
        reasons.append(f"gap {epoch.max_gap_s:.3f}s > {th.max_gap_s}s")
    if bad_frac > th.max_bad_channel_frac_retry:
        reasons.append(f"{len(flags)} bad channels ({sorted(set(flags.values()))})")
    if good_contact_frac is not None and good_contact_frac < th.min_good_contact_frac:
        reasons.append(f"contact good on only {good_contact_frac:.2f} of channels")
    if motion_rms is not None and motion_rms > th.max_motion_rms:
        reasons.append(f"motion {motion_rms:.2f} > {th.max_motion_rms}")

    if reasons:
        return QualityResult("retry", reasons, metrics, bad_channels)
    return QualityResult("valid", [], metrics, bad_channels)
