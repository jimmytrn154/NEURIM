"""Streaming EEG signal conditioning: the stage that was missing entirely.

Raw EPOC X samples go straight from the headset into band-power/FAA today, with
no filtering, referencing, or artifact handling. Frontal channels
(AF3/AF4/F3/F4/F7/F8) are dominated by blink (EOG) and forehead (EMG) activity,
and blink rate is itself a state confound - so an unconditioned signal lets a
model cheat on artifacts instead of neural activity.

`Preprocessor` is a causal, per-sample streaming filter:

    raw sample dict -> common-average reference -> bandpass + mains notch -> clean sample dict

It keeps per-channel filter state so it can run online in the live loop, and it
exposes `window_artifact_fraction(...)` so a window's blink/EMG contamination can
flow into `Observation.artifact_fraction` (replacing the clip-saturation proxy in
optimizer/observation.py).

`PreprocessedSource` wraps any `EEGSource` and yields cleaned samples in the same
`(timestamp, {channel: value})` shape, so FAARewardComputer / EEGFeatureExtractor
consume conditioned EEG transparently.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

DEFAULT_EOG_CHANNELS: tuple[str, ...] = ("AF3", "AF4", "F7", "F8")


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score; robust to the very spikes we want to flag."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad  # MAD -> std for a normal distribution
    if scale <= 1e-9:
        std = float(x.std())
        scale = std if std > 1e-9 else 1.0
    return (x - med) / scale


class Preprocessor:
    """Stateful causal conditioning for streaming multi-channel EEG."""

    def __init__(
        self,
        fs: float,
        channels: list[str],
        bandpass_low_hz: float = 1.0,
        bandpass_high_hz: float = 40.0,
        notch_hz: float | None = 60.0,
        notch_q: float = 30.0,
        filter_order: int = 4,
        common_average: bool = True,
        eog_channels: list[str] | tuple[str, ...] = DEFAULT_EOG_CHANNELS,
        blink_threshold_mad: float = 5.0,
        emg_threshold_mad: float = 6.0,
    ):
        from scipy.signal import butter, iirnotch, sosfilt_zi, tf2sos

        self.fs = float(fs)
        self.channels = list(channels)
        self.common_average = common_average
        self.eog_channels = tuple(eog_channels)
        self.blink_threshold_mad = blink_threshold_mad
        self.emg_threshold_mad = emg_threshold_mad

        nyq = self.fs / 2.0
        high = min(bandpass_high_hz, nyq * 0.98)
        low = max(bandpass_low_hz, 1e-3)
        if not low < high:
            raise ValueError(f"invalid bandpass: low={low} high={high} (fs={fs})")

        sos = butter(filter_order, [low / nyq, high / nyq], btype="band", output="sos")
        if notch_hz is not None and 0 < notch_hz < nyq:
            b, a = iirnotch(w0=notch_hz / nyq, Q=notch_q)
            sos = np.vstack([sos, tf2sos(b, a)])
        self._sos = sos
        self._zi_template = sosfilt_zi(sos)  # (n_sections, 2)
        self._zi: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._zi = {}

    def _numeric_channels(self, sample: dict[str, Any]) -> list[str]:
        out = []
        for ch, value in sample.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            out.append(ch)
        return out

    def process_sample(
        self, sample: dict[str, Any], quality: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Reference + filter one raw sample. `quality` optionally maps channel ->
        contact quality in [0, 1]; channels below 0.5 are dropped from the common
        average and passed through unfiltered-but-flagged (excluded downstream by
        their absence). Returns cleaned numeric channels only."""
        from scipy.signal import sosfilt

        chans = self._numeric_channels(sample)
        good = [c for c in chans if (quality is None or quality.get(c, 1.0) >= 0.5)]

        ref = 0.0
        if self.common_average and good:
            ref = float(np.mean([float(sample[c]) for c in good]))

        cleaned: dict[str, float] = {}
        for ch in chans:
            x = float(sample[ch]) - ref
            zi = self._zi.get(ch)
            if zi is None:
                # Seed filter state with the first value so we don't ring on startup.
                zi = self._zi_template * x
            y, zf = sosfilt(self._sos, [x], zi=zi)
            self._zi[ch] = zf
            cleaned[ch] = float(y[0])
        return cleaned

    def window_artifact_fraction(self, window: dict[str, np.ndarray]) -> float:
        """Fraction of samples in a cleaned window flagged as blink or EMG.

        Blink: large deflection on the frontal EOG proxy (mean of frontal
        channels). EMG: large sample-to-sample jumps (high-frequency power) on
        any channel. Both use a robust median/MAD threshold so a few big spikes
        don't inflate their own baseline.
        """
        if not window:
            return 1.0
        lengths = [arr.size for arr in window.values() if arr.size]
        if not lengths:
            return 1.0
        n = min(lengths)
        if n < 4:
            return 1.0

        flagged = np.zeros(n, dtype=bool)

        eog_avail = [c for c in self.eog_channels if c in window and window[c].size >= n]
        if eog_avail:
            eog = np.mean([window[c][-n:] for c in eog_avail], axis=0)
            flagged |= np.abs(_robust_z(eog)) > self.blink_threshold_mad

        for arr in window.values():
            if arr.size < n + 1:
                a = arr[-n:]
            else:
                a = arr[-(n + 1):]
            diff = np.abs(np.diff(a))
            if diff.size < n:
                diff = np.concatenate([diff, diff[-1:]])
            emg_flag = np.abs(_robust_z(diff[-n:])) > self.emg_threshold_mad
            flagged |= emg_flag

        return float(np.mean(flagged))


class PreprocessedSource:
    """Wrap an EEGSource so downstream consumers get conditioned samples.

    Same (timestamp, {channel: value}) contract as the raw sources, so it is a
    drop-in for FAARewardComputer / EEGFeatureExtractor. Non-numeric columns
    (e.g. Cortex markers) are dropped, matching how the reward path already
    ignores them.
    """

    def __init__(self, source, preprocessor: Preprocessor):
        self._source = source
        self._pre = preprocessor

    def connect(self) -> None:
        self._pre.reset()
        self._source.connect()

    def stream(self, *args, **kwargs) -> Iterator[tuple[float, dict[str, float]]]:
        for t, sample in self._source.stream(*args, **kwargs):
            yield t, self._pre.process_sample(sample)

    def close(self) -> None:
        self._source.close()


def build_preprocessor(config) -> Preprocessor | None:
    """Construct a Preprocessor from a Config, or None if disabled."""
    pc = getattr(config, "preprocessing", None)
    if pc is None or not getattr(pc, "enabled", False):
        return None
    return Preprocessor(
        fs=config.eeg.sample_rate_hz,
        channels=list(config.eeg.channels),
        bandpass_low_hz=pc.bandpass_low_hz,
        bandpass_high_hz=pc.bandpass_high_hz,
        notch_hz=pc.notch_hz,
        notch_q=pc.notch_q,
        filter_order=pc.filter_order,
        common_average=pc.common_average,
        eog_channels=pc.eog_channels,
        blink_threshold_mad=pc.blink_threshold_mad,
        emg_threshold_mad=pc.emg_threshold_mad,
    )
