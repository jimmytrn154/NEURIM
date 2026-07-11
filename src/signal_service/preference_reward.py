"""Pairwise-preference EEG reward adapter for the optimizer.

Turns the calibrated PreferenceEnsemble into the optimizer's Observation
contract. Each candidate is image B; it is contrasted against a FIXED per-session
anchor window A (captured on the first step / at calibration):

    reward_mean = 2 * P(B preferred over anchor A) - 1
    reward_variance grows with ensemble disagreement, window artifact fraction,
        and (inversely) a per-session reliability factor from catch trials.

Why a fixed anchor rather than the previous candidate: a rolling A makes the
reward a function of (z_current, z_previous), which is non-Markovian and breaks
the GP surrogate's assumption that reward = f(z) - the optimizer then cannot
localize good regions. A fixed within-session anchor keeps the drift-cancelling
contrast and in-distribution (image-vs-image) inputs while giving the GP a
consistent objective. The first step captures the anchor and returns a neutral,
high-variance observation that the trust-region GP effectively ignores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.optimizer.observation import Observation, window_statistics
from src.signal_service.learned_reward import (
    EEGFeatureExtractor,
    FeatureBaseline,
    contrast_features,
    load_preference_ensemble,
)


class LearnedPreferenceReward:
    def __init__(
        self,
        eeg_source,
        model_path: str | Path,
        sample_rate_hz: float,
        channels: list[str],
        pairs: list[list[str]],
        window_s: float,
        scoring_seconds: float | None = None,
        stride_s: float = 0.25,
        baseline_windows: int = 8,
        baseline_path: str | Path | None = None,
        reliability: float = 1.0,
        preprocessor=None,
        mock_source=None,
        preference_fn=None,
    ):
        self.eeg_source = eeg_source
        self.ensemble = load_preference_ensemble(model_path)
        self.fs = sample_rate_hz
        self.window_s = window_s
        # Emit several overlapping sub-window reward samples per candidate (like the
        # FAA path) so one candidate's reward averages out single-window EEG noise.
        self.scoring_seconds = window_s if scoring_seconds is None else scoring_seconds
        self.stride_s = stride_s
        self.baseline_windows = baseline_windows
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.reliability = max(1e-3, float(reliability))
        self.preprocessor = preprocessor
        self.mock_source = mock_source
        self.preference_fn = preference_fn
        self.extractor = EEGFeatureExtractor(fs=sample_rate_hz, channels=channels,
                                             window_s=window_s, pairs=pairs)
        self._stream = None
        self._anchor: np.ndarray | None = None

    # -- setup -------------------------------------------------------------
    def calibrate(self) -> None:
        self._stream = self.eeg_source.stream()
        if self.baseline_path and self.baseline_path.exists():
            import json

            baseline = FeatureBaseline.from_dict(json.loads(self.baseline_path.read_text()))
            self.extractor.baseline = baseline
            print(f"[reward] loaded session baseline from {self.baseline_path}")
        else:
            print(f"[reward] fitting rest baseline over {self.baseline_windows} windows; hold still")
            self.extractor.baseline = self._fit_baseline()
        # Verify the live feature schema matches what the model expects.
        _vec, names = self._capture_window(preference=0.0)
        if names != self.ensemble.feature_names:
            raise RuntimeError(
                "Preference model feature schema does not match live EEG features. "
                "Record/train with the same config and headset channels."
            )

    def set_anchor(self, z: np.ndarray) -> None:
        """Capture the fixed reference window A from a known reference image z.

        Call once before optimization with the neutral starting display, so the
        reward is P(candidate preferred over this fixed reference) - a stable
        objective the GP can climb, centered on the reference's quality.
        """
        preference = float(self.preference_fn(z)) if self.preference_fn is not None else None
        fb, _ = self._capture_window(preference=preference)
        self._anchor = fb

    def _fit_baseline(self) -> FeatureBaseline:
        if self.mock_source is not None:
            self.mock_source.set_preference(0.0)
        vectors, names = [], None
        for _ in range(self.baseline_windows):
            vec, names = self._capture_window(preference=0.0, normalize=False)
            vectors.append(vec)
        baseline = FeatureBaseline()
        baseline.fit(vectors, names)
        return baseline

    # -- capture -----------------------------------------------------------
    def _capture_window(self, preference: float | None = None, normalize: bool = True):
        if self.mock_source is not None and preference is not None:
            self.mock_source.set_preference(preference)
        # Temporarily bypass baseline normalization when fitting the baseline.
        saved_baseline = self.extractor.baseline
        if not normalize:
            self.extractor.baseline = None
        self.extractor.clear()  # epoch onset = settle -> ERP features stimulus-locked
        win: dict[str, list[float]] = {}
        need = int(self.fs * self.window_s)
        count = 0
        while count < need or not self.extractor.ready():
            _t, sample = next(self._stream)
            self.extractor.push_sample(sample)
            for ch, v in sample.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    win.setdefault(ch, []).append(float(v))
            count += 1
        result = self.extractor.vector()
        self.extractor.baseline = saved_baseline
        if result is None:
            raise RuntimeError("preference reward: feature extractor never became ready")
        vec, names = result
        self._last_window = {ch: np.asarray(v, dtype=float) for ch, v in win.items()}
        return vec, names

    def _artifact_fraction(self) -> float:
        if self.preprocessor is None or not getattr(self, "_last_window", None):
            return 0.0
        return self.preprocessor.window_artifact_fraction(self._last_window)

    def _reward_from_buffer(self) -> tuple[float, float]:
        """One preference reward sample from the extractor's current window."""
        result = self.extractor.vector()
        if result is None:
            raise RuntimeError("preference reward: feature extractor never became ready")
        vec, names = result
        contrast, _ = contrast_features(vec, self._anchor, self.ensemble.feature_names)
        return self.ensemble.reward(contrast)  # (reward_mean, ensemble_std)

    # -- reward ------------------------------------------------------------
    def observe(self, z: np.ndarray, t: int) -> Observation:
        preference = float(self.preference_fn(z)) if self.preference_fn is not None else None

        if self._anchor is None:
            # No anchor yet: capture one and return a neutral, high-variance
            # observation the trust-region GP effectively ignores.
            self._capture_window(preference=preference)
            self._anchor, _ = self.extractor.vector()
            return Observation(0.0, 1.0, 1.0, self._artifact_fraction(), t)

        if self.mock_source is not None and preference is not None:
            self.mock_source.set_preference(preference)

        # Fill the first window (epoch-aligned to settle), then slide it forward,
        # emitting a reward sample every stride so single-window EEG noise averages out.
        self.extractor.clear()
        win: dict[str, list[float]] = {}
        need = int(self.fs * self.window_s)
        count = 0
        while count < need or not self.extractor.ready():
            _tt, sample = next(self._stream)
            self.extractor.push_sample(sample)
            for ch, v in sample.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    win.setdefault(ch, []).append(float(v))
            count += 1

        rewards: list[float] = []
        stds: list[float] = []
        r0, s0 = self._reward_from_buffer()
        rewards.append(r0)
        stds.append(s0)
        stride = max(1, int(self.fs * self.stride_s))
        extra_needed = int(self.fs * self.scoring_seconds)
        emitted = 0
        while emitted < extra_needed:
            _tt, sample = next(self._stream)
            self.extractor.push_sample(sample)
            for ch, v in sample.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    win.setdefault(ch, []).append(float(v))
            emitted += 1
            if emitted % stride == 0:
                r, s = self._reward_from_buffer()
                rewards.append(r)
                stds.append(s)

        self._last_window = {ch: np.asarray(v, dtype=float) for ch, v in win.items()}
        artifact = self._artifact_fraction()
        obs = window_statistics(rewards, clip=(-1.0, 1.0), t=t, min_variance=1e-4)

        # Fold ensemble disagreement, artifacts, and low session reliability into
        # the observation variance so the GP down-weights uncertain readings.
        ens_var = (2.0 * float(np.mean(stds))) ** 2
        var = obs.reward_variance / self.reliability + ens_var + 0.5 * artifact
        var = float(max(var, 1e-4))
        print(f"[reward] preference={obs.reward_mean:+.3f} n={len(rewards)} "
              f"ens_std={np.mean(stds):.3f} artifact={artifact:.2f} var={var:.4f}")
        return Observation(reward_mean=obs.reward_mean, reward_variance=var,
                           effective_sample_count=obs.effective_sample_count,
                           artifact_fraction=artifact, t=t)
