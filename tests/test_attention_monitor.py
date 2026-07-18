import numpy as np

from src.signal_service.attention import (
    AttentionMonitor,
    MetricSample,
    MockMetricsSource,
)


def _run(monitor, source, baseline_s):
    """Drive monitor over the mock stream, freezing baseline at baseline_s."""
    readings = []
    t0 = None
    for s in source.stream():
        if t0 is None:
            t0 = s.t
        r = monitor.update(s)
        if not monitor.has_baseline and s.t - t0 >= baseline_s:
            monitor.freeze_baseline()
            continue
        if monitor.has_baseline:
            readings.append((source.ground_truth(s.t), r))
    return readings


def test_wanted_blocks_have_higher_concentration_than_unwanted():
    src = MockMetricsSource(rate_hz=2.0, baseline_s=10.0, block_s=8.0, n_blocks=6,
                            bad_quality_until_s=3.0, seed=1)
    mon = AttentionMonitor(smoothing="ema", tau_s=2.0, met_rate_hz=2.0, quality_min=2.0)
    readings = _run(mon, src, baseline_s=10.0)

    wanted = [r.concentration_confidence for gt, r in readings if gt == "wanted"]
    unwanted = [r.concentration_confidence for gt, r in readings if gt == "unwanted"]
    assert wanted and unwanted
    # High-attention ('wanted') blocks must read as clearly more concentrated.
    assert np.mean(wanted) > np.mean(unwanted) + 0.2
    # Baseline-relative: wanted sits above 0.5, unwanted below.
    assert np.mean(wanted) > 0.55
    assert np.mean(unwanted) < 0.45


def test_poor_quality_and_inactive_samples_are_flagged_unreliable():
    mon = AttentionMonitor(quality_min=2.0)
    # Good sample -> reliable.
    good = mon.update(MetricSample(0.0, 0.6, 0.4, True, True, 4.0))
    assert good.reliable
    # Low eq quality -> not reliable.
    low_q = mon.update(MetricSample(0.5, 0.6, 0.4, True, True, 1.0))
    assert not low_q.reliable
    # foc.isActive False -> not reliable.
    inactive = mon.update(MetricSample(1.0, 0.6, 0.4, False, True, 4.0))
    assert not inactive.reliable


def test_unreliable_samples_do_not_move_the_baseline():
    mon = AttentionMonitor(quality_min=2.0)
    # Feed only bad-quality samples during "baseline", then freeze.
    for i in range(10):
        mon.update(MetricSample(i * 0.5, 0.9, 0.1, True, True, 0.0))  # all low quality
    mon.freeze_baseline()
    # Nothing was observed, so the normalizer has no spread; a mid reading should
    # not explode into a huge z-score (std floored, mean ~0 -> guarded).
    r = mon.update(MetricSample(100.0, 0.6, 0.4, True, True, 4.0))
    assert np.isfinite(r.concentration_confidence)
    assert 0.0 <= r.concentration_confidence <= 1.0


def test_smoothing_reduces_variance():
    rng = np.random.default_rng(0)
    raw = 0.5 + rng.normal(0, 0.1, size=200)
    mon = AttentionMonitor(smoothing="ema", tau_s=3.0, met_rate_hz=2.0)
    smoothed = []
    for i, x in enumerate(raw):
        r = mon.update(MetricSample(i * 0.5, float(x), 0.5, True, True, 4.0))
        smoothed.append(r.attention_smooth)
    # EMA output must be less variable than the raw metric.
    assert np.std(smoothed[20:]) < np.std(raw[20:])


def test_median_smoothing_rejects_single_spike():
    mon = AttentionMonitor(smoothing="median", median_window_s=3.0, met_rate_hz=2.0)
    vals = [0.5, 0.5, 0.5, 0.99, 0.5, 0.5]  # one spike
    out = [mon.update(MetricSample(i * 0.5, v, 0.5, True, True, 4.0)).attention_smooth
           for i, v in enumerate(vals)]
    # The spike sample's median should stay near 0.5, not jump to 0.99.
    assert out[3] < 0.6
