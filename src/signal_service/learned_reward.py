"""Learned EEG *preference* features and sklearn model wrapper.

The reward is no longer an absolute per-image score. Each image presentation
yields a baseline-normalized feature vector; the model input is the CONTRAST
between the B window and the A window (`contrast_features`), and the model
predicts P(B preferred over A). Working in the A-vs-B difference cancels the
session/electrode drift that makes single-window EEG unusable across days.

Feature vector per window (all baseline-normalized against a within-session rest
baseline when one is supplied):
  - per-channel broadband std + log band power (theta/alpha/beta)
  - frontal mirror-pair asymmetries (FAA is now one feature among many)
  - aggregate frontal-midline theta
  - event-related (ERP) window features, if the window is stimulus-locked:
    a P300-like parietal/occipital positivity and a frontocentral
    reward-positivity/FRN component, both baseline-corrected within the epoch.

The live reward is `2 * p - 1` to match the optimizer's [-1, 1] convention.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.signal_service.faa import DEFAULT_FAA_PAIRS, band_power


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# ERP components measurable on the EPOC X montage. Each entry maps a component
# name to (t_start_s, t_end_s, channel group). The epoch is assumed to start at
# stimulus/settle onset; the recorder and live loop align the buffer that way.
DEFAULT_ERP_WINDOWS: dict[str, tuple[float, float, tuple[str, ...]]] = {
    "p300": (0.30, 0.50, ("P7", "P8", "O1", "O2")),
    "frn": (0.25, 0.35, ("FC5", "FC6", "F3", "F4")),
}
ERP_BASELINE_S = 0.1  # pre-response baseline for ERP baseline correction

FRONTAL_MIDLINE = ("F3", "F4", "FC5", "FC6", "AF3", "AF4")


def _safe_log(x: float) -> float:
    return float(np.log(max(float(x), 1e-12)))


@dataclass
class FeatureBaseline:
    """Per-feature mean/std collected on rest windows, for within-session z-scoring.

    Unlike faa.RunningBaseline (a single scalar), this normalizes the whole
    feature vector so absolute band-power levels - which drift session to session
    with impedance and arousal - become comparable.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    names: list[str] = field(default_factory=list)

    def fit(self, vectors: list[np.ndarray], names: list[str]) -> None:
        arr = np.asarray(vectors, dtype=float)
        self.mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std = std
        self.names = list(names)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return np.asarray(x, dtype=np.float32)
        return ((np.asarray(x, dtype=float) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "names": list(self.names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "FeatureBaseline | None":
        if not payload:
            return None
        mean = payload.get("mean")
        std = payload.get("std")
        return cls(
            mean=None if mean is None else np.asarray(mean, dtype=float),
            std=None if std is None else np.asarray(std, dtype=float),
            names=list(payload.get("names", [])),
        )


class EEGFeatureExtractor:
    """Sliding-window feature extractor for EMOTIV-style channel samples.

    Feed cleaned samples via `push_sample`; call `vector()` for the feature
    vector of the trailing `window_s` seconds. For ERP features to be meaningful
    the buffer must start at stimulus/settle onset (the recorder clears and
    refills it per epoch); on the free-running live path ERP features are still
    computed but are effectively broadband and get normalized away.
    """

    def __init__(
        self,
        fs: float,
        channels: list[str],
        window_s: float = 3.0,
        bands: dict[str, tuple[float, float]] | None = None,
        pairs: list[list[str]] | list[tuple[str, str]] | None = None,
        erp_windows: dict[str, tuple[float, float, tuple[str, ...]]] | None = None,
        baseline: FeatureBaseline | None = None,
    ):
        self.fs = fs
        self.channels = list(channels)
        self.window_s = window_s
        self.bands = bands or DEFAULT_BANDS
        self.pairs = [(str(a), str(b)) for a, b in (pairs or DEFAULT_FAA_PAIRS)]
        self.erp_windows = erp_windows or DEFAULT_ERP_WINDOWS
        self.baseline = baseline
        self._maxlen = int(fs * window_s) + 1
        self._buffers = {ch: deque(maxlen=self._maxlen) for ch in self.channels}

    def clear(self) -> None:
        for buf in self._buffers.values():
            buf.clear()

    def push_sample(self, sample: dict[str, Any]) -> None:
        for ch, value in sample.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if ch not in self._buffers:
                self._buffers[ch] = deque(maxlen=self._maxlen)
                self.channels.append(ch)
            self._buffers[ch].append(float(value))

    def ready(self) -> bool:
        required = max(8, self._maxlen - 1)
        return any(len(buf) >= required for buf in self._buffers.values())

    def _erp_amplitude(self, arr: np.ndarray, t0: float, t1: float) -> float:
        """Baseline-corrected mean amplitude in [t0, t1] s of a stimulus-locked epoch."""
        base_n = max(1, int(ERP_BASELINE_S * self.fs))
        i0, i1 = int(t0 * self.fs), int(t1 * self.fs)
        if arr.size <= i0 or i1 <= i0:
            return 0.0
        baseline = float(arr[: min(base_n, arr.size)].mean())
        seg = arr[i0 : min(i1, arr.size)]
        if seg.size == 0:
            return 0.0
        return float(seg.mean() - baseline)

    def vector(self) -> tuple[np.ndarray, list[str]] | None:
        if not self.ready():
            return None

        features: list[float] = []
        names: list[str] = []
        band_values: dict[tuple[str, str], float] = {}
        theta_by_channel: dict[str, float] = {}
        arrays: dict[str, np.ndarray] = {}

        for ch in sorted(self._buffers):
            arr = np.asarray(self._buffers[ch], dtype=float)
            if arr.size < 8:
                continue
            arrays[ch] = arr
            features.append(float(arr.std()))
            names.append(f"{ch}:std")
            for band_name, band in self.bands.items():
                power = band_power(arr, self.fs, band)
                band_values[(ch, band_name)] = power
                log_power = _safe_log(power)
                features.append(log_power)
                names.append(f"{ch}:{band_name}_log_power")
                if band_name == "theta":
                    theta_by_channel[ch] = log_power

        # Frontal mirror-pair asymmetries (FAA generalized across bands).
        for left, right in self.pairs:
            for band_name in self.bands:
                p_left = band_values.get((left, band_name))
                p_right = band_values.get((right, band_name))
                if p_left is None or p_right is None:
                    continue
                features.append(_safe_log(p_right) - _safe_log(p_left))
                names.append(f"{left}/{right}:{band_name}_asym")

        # Aggregate frontal-midline theta (engagement/effort).
        frontal_theta = [theta_by_channel[ch] for ch in FRONTAL_MIDLINE if ch in theta_by_channel]
        if frontal_theta:
            features.append(float(np.mean(frontal_theta)))
            names.append("frontal_midline_theta")

        # ERP-window features (assume buffer starts at settle onset).
        for comp_name, (t0, t1, group) in self.erp_windows.items():
            vals = [self._erp_amplitude(arrays[ch], t0, t1) for ch in group if ch in arrays]
            if vals:
                features.append(float(np.mean(vals)))
                names.append(f"erp_{comp_name}")

        if not features:
            return None
        vec = np.asarray(features, dtype=np.float32)
        if self.baseline is not None:
            vec = self.baseline.transform(vec)
        return vec, names


def contrast_features(
    feat_b: np.ndarray, feat_a: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Model input for pairwise preference: features(B) - features(A).

    Difference-only (not concatenated with the raw windows) by design: the whole
    point of the A-vs-B contrast is to cancel session/electrode drift, and the
    absolute windows would reintroduce it. Strict antisymmetry also makes the
    label-flip augmentation in training exact: contrast(A, B) == -contrast(B, A).
    """
    b = np.asarray(feat_b, dtype=np.float32)
    a = np.asarray(feat_a, dtype=np.float32)
    if b.shape != a.shape:
        raise ValueError(f"feature shape mismatch: {b.shape} vs {a.shape}")
    return (b - a), [f"{n}:dBA" for n in names]


@dataclass
class LearnedRewardModel:
    model: Any
    scaler: Any
    feature_names: list[str]
    model_type: str
    positive_label: int = 1
    feature_baseline: FeatureBaseline | None = None

    def predict_probability(self, x: np.ndarray) -> float:
        row = np.asarray(x, dtype=np.float32).reshape(1, -1)
        scaled = self.scaler.transform(row)
        if hasattr(self.model, "predict_proba"):
            classes = list(self.model.classes_)
            idx = classes.index(self.positive_label)
            return float(self.model.predict_proba(scaled)[0, idx])
        score = float(self.model.decision_function(scaled)[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    def reward(self, x: np.ndarray) -> float:
        return float(2.0 * self.predict_probability(x) - 1.0)


@dataclass
class PreferenceEnsemble:
    """Bagged pairwise-preference model with calibrated probability + uncertainty.

    `models` are fitted sklearn pipelines (each an internal StandardScaler +
    estimator) trained on bootstrap resamples; their spread is the epistemic
    uncertainty of the reward. `calibrator` maps the ensemble-mean probability to
    a calibrated one (fit on pooled leave-session-out predictions). Input is the
    contrast vector features(B) - features(A) in `feature_names` order.

    `feature_names` is always the full live extractor schema, so the schema check
    in the live loop stays a meaningful guard. `feature_mask` (a boolean over that
    schema, or None) selects the subset the members were actually trained on; it
    is applied internally at predict time, so callers still pass the full vector.
    """

    models: list[Any]
    feature_names: list[str]
    model_type: str
    calibrator: Any | None = None
    positive_label: int = 1
    feature_mask: np.ndarray | None = None

    def _member_prob(self, model: Any, row: np.ndarray) -> float:
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            idx = classes.index(self.positive_label)
            return float(model.predict_proba(row)[0, idx])
        score = float(model.decision_function(row)[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    def predict(self, contrast_vec: np.ndarray) -> tuple[float, float]:
        """Return (calibrated P(B preferred), ensemble std of raw member probs)."""
        row = np.asarray(contrast_vec, dtype=np.float32).reshape(1, -1)
        if self.feature_mask is not None:
            row = row[:, self.feature_mask]
        probs = np.asarray([self._member_prob(m, row) for m in self.models], dtype=float)
        mean = float(probs.mean())
        std = float(probs.std())
        if self.calibrator is not None:
            mean = float(self.calibrator.predict_proba([[mean]])[0, 1])
        return mean, std

    def reward(self, contrast_vec: np.ndarray) -> tuple[float, float]:
        """(reward in [-1, 1], ensemble std) for a single contrast vector."""
        p, std = self.predict(contrast_vec)
        return 2.0 * p - 1.0, std


def save_preference_ensemble(
    path: str | Path, ensemble: PreferenceEnsemble, metrics: dict[str, Any]
) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": ensemble.models,
            "feature_names": ensemble.feature_names,
            "model_type": ensemble.model_type,
            "calibrator": ensemble.calibrator,
            "positive_label": ensemble.positive_label,
            "feature_mask": None if ensemble.feature_mask is None
            else np.asarray(ensemble.feature_mask, dtype=bool).tolist(),
            "metrics": metrics,
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_preference_ensemble(path: str | Path) -> PreferenceEnsemble:
    import joblib

    payload = joblib.load(path)
    mask = payload.get("feature_mask")
    return PreferenceEnsemble(
        models=list(payload["models"]),
        feature_names=list(payload["feature_names"]),
        model_type=str(payload.get("model_type", "unknown")),
        calibrator=payload.get("calibrator"),
        positive_label=int(payload.get("positive_label", 1)),
        feature_mask=None if mask is None else np.asarray(mask, dtype=bool),
    )


def save_reward_model(path: str | Path, wrapper: LearnedRewardModel, metrics: dict[str, Any]) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": wrapper.model,
            "scaler": wrapper.scaler,
            "feature_names": wrapper.feature_names,
            "model_type": wrapper.model_type,
            "positive_label": wrapper.positive_label,
            "feature_baseline": None
            if wrapper.feature_baseline is None
            else wrapper.feature_baseline.to_dict(),
            "metrics": metrics,
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_reward_model(path: str | Path) -> LearnedRewardModel:
    import joblib

    payload = joblib.load(path)
    return LearnedRewardModel(
        model=payload["model"],
        scaler=payload["scaler"],
        feature_names=list(payload["feature_names"]),
        model_type=str(payload.get("model_type", "unknown")),
        positive_label=int(payload.get("positive_label", 1)),
        feature_baseline=FeatureBaseline.from_dict(payload.get("feature_baseline")),
    )
