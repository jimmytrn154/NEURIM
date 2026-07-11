import numpy as np

from src.signal_service.preprocessing import Preprocessor, PreprocessedSource
from src.signal_service.eeg_sources import MockEEGSource

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def _stream_window(pre: Preprocessor, signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(next(iter(signals.values())))
    out = {ch: [] for ch in signals}
    for i in range(n):
        cleaned = pre.process_sample({ch: float(vals[i]) for ch, vals in signals.items()})
        for ch in signals:
            out[ch].append(cleaned[ch])
    return {ch: np.asarray(v) for ch, v in out.items()}


def _alpha(fs, dur, amp=1.0, freq=10.0, seed=0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.05, n)


def test_bandpass_attenuates_dc_and_line_noise():
    fs = 128.0
    n = int(fs * 4)
    t = np.arange(n) / fs
    # 10 Hz alpha (in band) + 3.0 DC offset (out, below 1 Hz) + 60 Hz mains (notch).
    signals = {ch: np.sin(2 * np.pi * 10 * t) + 3.0 + 0.8 * np.sin(2 * np.pi * 60 * t) for ch in CHANNELS}
    pre = Preprocessor(fs, CHANNELS, common_average=False)
    out = _stream_window(pre, signals)
    settled = out["F3"][int(fs):]  # drop filter startup
    assert abs(settled.mean()) < 0.2          # DC removed
    assert settled.std() < np.std(signals["F3"])  # 60 Hz + DC energy removed


def test_blink_raises_artifact_fraction():
    fs = 128.0
    dur = 4.0
    n = int(fs * dur)
    pre = Preprocessor(fs, CHANNELS)

    clean = {ch: _alpha(fs, dur, amp=1.0, seed=i) for i, ch in enumerate(CHANNELS)}
    clean_out = _stream_window(pre, clean)
    clean_af = pre.window_artifact_fraction(clean_out)

    pre.reset()
    blinky = {ch: _alpha(fs, dur, amp=1.0, seed=i) for i, ch in enumerate(CHANNELS)}
    # Inject three large blink deflections on the frontal (EOG) channels.
    for center in (int(0.2 * n), int(0.5 * n), int(0.8 * n)):
        for ch in ("AF3", "AF4", "F7", "F8"):
            blinky[ch][center : center + 8] += 12.0
    blink_out = _stream_window(pre, blinky)
    blink_af = pre.window_artifact_fraction(blink_out)

    assert blink_af > clean_af
    assert clean_af < 0.15


def test_preprocessed_source_is_drop_in():
    fs = 128
    src = MockEEGSource(CHANNELS, sample_rate_hz=fs)
    pre = Preprocessor(fs, CHANNELS)
    wrapped = PreprocessedSource(src, pre)
    wrapped.connect()
    stream = wrapped.stream()
    for _ in range(300):
        t, sample = next(stream)
    assert set(sample) == set(CHANNELS)
    assert all(isinstance(v, float) for v in sample.values())
    wrapped.close()
