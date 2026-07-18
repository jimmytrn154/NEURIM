"""Signal pre-processing for the digit classifier (paper Sect. 3.2.1, Table 3).

The paper's chain is:  raw EEG
    -> 5th-order low-pass Butterworth (denoise, cut > 45 Hz)
    -> 50 Hz notch (power line)
    -> [optional] 6 Butterworth band-pass sub-bands
    -> [optional] EMD -> HHT -> IA / IP / IF features.

For **real-time** windowed inference we stop after the notch step: the denoised
14-channel signal, z-scored per channel, is the CNN input (fast, no iterative
EMD). The band-pass and EMD/HHT helpers reproduce the full offline pipeline and
are used for band-wise experiments; EMD/HHT is far too slow for live use and is
gated behind an optional PyEMD import.

All filters are applied to a whole epoch with zero-phase `filtfilt`, so an epoch
handed to `denoise` comes back the same length.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, iirnotch

from . import config


def _butter_lowpass(cutoff_hz: float, fs: float, order: int):
    return butter(order, cutoff_hz / (0.5 * fs), btype="low")


def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int):
    nyq = 0.5 * fs
    # Clamp the upper edge just under Nyquist so band-pass design stays stable.
    high = min(high_hz, nyq - 1e-3)
    return butter(order, [low_hz / nyq, high / nyq], btype="band")


def _safe_filtfilt(b, a, x: np.ndarray) -> np.ndarray:
    """filtfilt needs len(x) > 3*max(len(a),len(b)); short epochs fall back to
    a plain forward filter so callers never crash on a stub window."""
    from scipy.signal import lfilter

    padlen = 3 * max(len(a), len(b))
    if x.shape[-1] <= padlen:
        return lfilter(b, a, x, axis=-1)
    return filtfilt(b, a, x, axis=-1)


def denoise(epoch: np.ndarray, fs: float = config.SAMPLE_RATE_HZ) -> np.ndarray:
    """Low-pass (45 Hz) + 50 Hz notch. `epoch` is (channels, samples)."""
    b_lp, a_lp = _butter_lowpass(config.LOWPASS_HZ, fs, config.BUTTER_ORDER)
    out = _safe_filtfilt(b_lp, a_lp, np.asarray(epoch, dtype=np.float64))
    b_n, a_n = iirnotch(config.NOTCH_HZ, config.NOTCH_Q, fs)
    out = _safe_filtfilt(b_n, a_n, out)
    return out


def bandpass(epoch: np.ndarray, band: str, fs: float = config.SAMPLE_RATE_HZ) -> np.ndarray:
    """Isolate one of the six named sub-bands from a (channels, samples) epoch."""
    low, high = config.SUB_BANDS[band]
    b, a = _butter_bandpass(low, high, fs, config.BUTTER_ORDER)
    return _safe_filtfilt(b, a, np.asarray(epoch, dtype=np.float64))


def zscore(epoch: np.ndarray) -> np.ndarray:
    """Per-channel zero-mean / unit-variance normalisation (guards flat channels)."""
    x = np.asarray(epoch, dtype=np.float32)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + 1e-6)


def fit_window(epoch: np.ndarray, n: int = config.WINDOW_SAMPLES) -> np.ndarray:
    """Crop/pad an epoch to exactly `n` samples along time (MBD epochs vary
    ~250-260 samples; live windows are exactly n)."""
    x = np.asarray(epoch, dtype=np.float32)
    t = x.shape[-1]
    if t == n:
        return x
    if t > n:
        start = (t - n) // 2
        return x[..., start:start + n]
    pad = n - t
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad // 2, pad - pad // 2)], mode="edge")


def preprocess(
    epoch: np.ndarray,
    fs: float = config.SAMPLE_RATE_HZ,
    band: str | None = None,
) -> np.ndarray:
    """Full default feature: fit to window -> denoise (-> optional sub-band) ->
    z-score. Returns a (channels, WINDOW_SAMPLES) float32 array ready for the CNN.
    """
    x = fit_window(epoch, config.WINDOW_SAMPLES)
    x = denoise(x, fs)
    if band is not None:
        x = bandpass(x, band, fs)
    return zscore(x)


# ---------------------------------------------------------------------------
# Optional EMD + Hilbert-Huang features (paper-faithful, offline only).
# ---------------------------------------------------------------------------
def hht_features(signal_1d: np.ndarray, fs: float, max_imf: int = 10) -> dict[str, np.ndarray]:
    """EMD -> HHT of a single channel: return instantaneous amplitude (IA),
    phase (IP) and frequency (IF) of each IMF. Requires the optional `EMD-signal`
    package (PyEMD). Slow and iterative -- not for the live path."""
    try:
        from PyEMD import EMD
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "EMD/HHT features need the optional 'EMD-signal' package. "
            "Install with: pip install EMD-signal"
        ) from exc

    imfs = EMD().emd(np.asarray(signal_1d, dtype=np.float64), max_imf=max_imf)
    analytic = hilbert(imfs, axis=-1)
    ia = np.abs(analytic)                                   # instantaneous amplitude
    ip = np.unwrap(np.angle(analytic), axis=-1)            # instantaneous phase
    if_ = np.diff(ip, axis=-1) / (2.0 * np.pi) * fs         # instantaneous frequency
    return {"IA": ia, "IP": ip, "IF": if_}


def _fix_imf_time(arr: np.ndarray, n_imf: int, n_time: int) -> np.ndarray:
    """Force an (n_actual_imf, t) HHT array to exactly (n_imf, n_time): EMD emits
    a variable IMF count per signal (pad/truncate) and IF is one sample short of
    the window (edge-pad time)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    k, t = a.shape
    if k >= n_imf:
        a = a[:n_imf]
    else:
        a = np.pad(a, [(0, n_imf - k), (0, 0)])
    if t >= n_time:
        a = a[:, :n_time]
    else:
        a = np.pad(a, [(0, 0), (0, n_time - t)], mode="edge")
    return a


HHT_ATTRS = ("IA", "IF")  # default HHT attributes fed to the CNN


def epoch_hht_features(
    epoch: np.ndarray,
    fs: float = config.SAMPLE_RATE_HZ,
    n_imf: int = 6,
    attrs: tuple[str, ...] = HHT_ATTRS,
    band: str | None = None,
) -> np.ndarray:
    """Paper-faithful EMD+HHT feature map for one (channels, samples) epoch.

    For each EEG channel: fit-window -> denoise (-> optional sub-band) -> EMD ->
    HHT, then keep the first `n_imf` IMFs' `attrs` (IA/IP/IF) as time series.
    Returns a z-scored (channels * len(attrs) * n_imf, WINDOW_SAMPLES) float32
    array -- the CNN input for the offline HHT path. Slow (EMD); offline only.
    """
    x = fit_window(epoch, config.WINDOW_SAMPLES)
    x = denoise(x, fs)
    if band is not None:
        x = bandpass(x, band, fs)
    n_time = x.shape[-1]
    rows: list[np.ndarray] = []
    for ch in range(x.shape[0]):
        h = hht_features(x[ch], fs, max_imf=n_imf)
        for attr in attrs:
            rows.append(_fix_imf_time(h[attr], n_imf, n_time))  # (n_imf, n_time)
    out = np.concatenate(rows, axis=0)  # (channels*len(attrs)*n_imf, n_time)
    return zscore(out)


def hht_channels(n_imf: int = 6, attrs: tuple[str, ...] = HHT_ATTRS) -> int:
    """Number of CNN input channels the HHT feature map produces."""
    return config.NUM_CHANNELS * len(attrs) * n_imf
