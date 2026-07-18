import numpy as np
import pytest

from src.acquisition.markers import (
    CANDIDATE_OFFSET,
    CANDIDATE_ONSET,
    MarkerLog,
)
from src.acquisition.quality import QualityThresholds, evaluate_epoch
from src.acquisition.ring_buffer import RingBuffer

CHANNELS = ["AF3", "AF4", "F3", "F4"]


def _eeg_value(ch: str, t: float, rng, amp: float) -> float:
    """EEG-like sample: theta+alpha+beta oscillations (autocorrelated) + light
    noise, so std(diff)/std stays in the clean-EEG range, unlike white noise."""
    phase = hash(ch) % 100 / 100.0 * 2 * np.pi
    sig = (np.sin(2 * np.pi * 5.0 * t + phase)
           + 0.8 * np.sin(2 * np.pi * 10.0 * t + phase)
           + 0.4 * np.sin(2 * np.pi * 18.0 * t + phase))
    return float(amp * sig + 0.15 * amp * rng.standard_normal())


def _fill(buf: RingBuffer, fs: float, seconds: float, amp: float = 20.0, t0: float = 0.0,
          flat: str | None = None):
    n = int(fs * seconds)
    rng = np.random.default_rng(0)
    for k in range(n):
        t = t0 + k / fs
        sample = {c: _eeg_value(c, t, rng, amp) for c in CHANNELS}
        if flat is not None:
            sample[flat] = 0.0
        buf.push(t, sample)


# ---- ring buffer ------------------------------------------------------------
def test_extract_returns_channel_by_samples_in_fixed_order():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    _fill(buf, 128, 4.0)
    ep = buf.extract_around(onset=1.0, start_offset=0.5, end_offset=2.5)  # 2s window
    assert ep.channels == CHANNELS
    assert ep.data.shape[0] == len(CHANNELS)
    assert abs(ep.data.shape[1] - int(128 * 2.0)) <= 2
    assert ep.coverage > 0.95
    assert ep.max_gap_s < 0.05


def test_extract_out_of_range_raises():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    _fill(buf, 128, 1.0)
    with pytest.raises(ValueError):
        buf.extract(100.0, 102.0)


def test_missing_channel_filled_with_nan_not_reordered():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=5)
    for k in range(200):
        buf.push(k / 128, {"AF3": 1.0, "F4": 2.0})  # AF4, F3 absent
    ep = buf.extract(0.1, 0.9)
    assert ep.channels == CHANNELS
    assert np.isnan(ep.data[CHANNELS.index("AF4")]).all()
    assert np.isnan(ep.data[CHANNELS.index("F3")]).all()
    assert np.allclose(ep.data[CHANNELS.index("AF3")], 1.0)


# ---- markers ----------------------------------------------------------------
def test_marker_sequence_and_context_inheritance():
    log = MarkerLog(clock=lambda: 0.0)
    log.context(subject_id="sub-001", session_id="ses-001", block_id=1, trial_id=3,
                candidate_id="c7")
    a = log.emit(CANDIDATE_ONSET, timestamp=10.0)
    b = log.emit(CANDIDATE_OFFSET, timestamp=13.0)
    assert (a.sequence, b.sequence) == (1, 2)
    assert a.subject_id == "sub-001" and a.trial_id == 3 and a.candidate_id == "c7"
    assert log.last(CANDIDATE_ONSET).timestamp == 10.0


def test_unknown_marker_rejected():
    log = MarkerLog()
    with pytest.raises(ValueError):
        log.emit("NOT_A_MARKER")


def test_markers_define_extractable_epoch():
    """The onset/offset markers should bound a real slice in the buffer."""
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    log = MarkerLog(clock=lambda: 0.0)
    _fill(buf, 128, 5.0)
    on = log.emit(CANDIDATE_ONSET, timestamp=1.0)
    off = log.emit(CANDIDATE_OFFSET, timestamp=4.0)
    ep = buf.extract(on.timestamp + 0.5, off.timestamp)  # 0.5..4.0 post-onset
    assert ep.duration_s == pytest.approx(2.5)
    assert ep.n_samples > 128


# ---- quality gate -----------------------------------------------------------
def _epoch(buf):
    return buf.extract(0.5, 2.5)


def test_clean_epoch_is_valid():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    _fill(buf, 128, 3.0, amp=20.0)
    res = evaluate_epoch(_epoch(buf), CHANNELS)
    assert res.status == "valid", res.reasons


def test_flat_channel_pushes_to_retry():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    _fill(buf, 128, 3.0, amp=20.0, flat="F3")  # one dead channel
    res = evaluate_epoch(_epoch(buf), CHANNELS)
    assert res.status == "retry"
    assert "F3" in res.bad_channels


def test_most_channels_flat_is_invalid_not_a_label():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    for k in range(400):
        buf.push(k / 128, {c: 0.0 for c in CHANNELS})  # all dead
    res = evaluate_epoch(_epoch(buf), CHANNELS)
    assert res.status == "invalid"
    assert not res.is_valid  # invalid must never be treated as usable/negative


def test_channel_mismatch_is_invalid():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    _fill(buf, 128, 3.0)
    res = evaluate_epoch(_epoch(buf), ["AF3", "AF4", "F3", "F4", "EXTRA"])
    assert res.status == "invalid"


def test_poor_contact_quality_flagged():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    _fill(buf, 128, 3.0, amp=20.0)
    cq = {"AF3": 0.0, "AF4": 1.0, "F3": 4.0, "F4": 4.0}  # half bad contact
    res = evaluate_epoch(_epoch(buf), CHANNELS, contact_quality=cq)
    assert res.status in ("retry", "invalid")
    assert any("contact" in r for r in res.reasons)


def test_high_motion_triggers_retry():
    buf = RingBuffer(CHANNELS, sample_rate_hz=128, capacity_s=10)
    _fill(buf, 128, 3.0, amp=20.0)
    res = evaluate_epoch(_epoch(buf), CHANNELS, motion_rms=5.0)
    assert res.status == "retry"
    assert any("motion" in r for r in res.reasons)
