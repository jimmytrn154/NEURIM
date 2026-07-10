import numpy as np

from src.signal_service.faa import FAARewardComputer, RunningBaseline, band_power


def _alpha_signal(fs, duration_s, amplitude=1.0, freq=10.0, seed=0):
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return amplitude * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.05, n)


def _clean_alpha_signal(fs, duration_s, amplitude=1.0, freq=10.0):
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


def _push_window(computer, signals):
    n = len(next(iter(signals.values())))
    for i in range(n):
        computer.push_sample({channel: values[i] for channel, values in signals.items()})


def test_band_power_higher_for_larger_amplitude():
    fs = 128.0
    quiet = _alpha_signal(fs, 2.0, amplitude=0.2)
    loud = _alpha_signal(fs, 2.0, amplitude=2.0)
    assert band_power(loud, fs, (8, 13)) > band_power(quiet, fs, (8, 13))


def test_running_baseline_z_score():
    baseline = RunningBaseline()
    baseline.fit([0.0, 0.1, -0.1, 0.05, -0.05])
    assert abs(baseline.z_score(baseline.mean)) < 1e-9
    assert baseline.z_score(baseline.mean + baseline.std) == 1.0


def test_faa_reward_computer_ready_and_clipped():
    fs = 128.0
    computer = FAARewardComputer(fs=fs, window_s=1.0, clip=(-1.0, 1.0))
    computer.baseline.fit([0.0])
    computer.baseline.std = 0.01  # tiny std -> any real signal saturates the clip

    assert not computer.ready()
    signals = {
        "F7": _alpha_signal(fs, 1.1, amplitude=0.1, seed=1),
        "F8": _alpha_signal(fs, 1.1, amplitude=2.0, seed=2),
        "AF3": _alpha_signal(fs, 1.1, amplitude=0.1, seed=3),
        "AF4": _alpha_signal(fs, 1.1, amplitude=2.0, seed=4),
        "F3": _alpha_signal(fs, 1.1, amplitude=0.1, seed=5),
        "F4": _alpha_signal(fs, 1.1, amplitude=2.0, seed=6),
        "FC5": _alpha_signal(fs, 1.1, amplitude=0.1, seed=7),
        "FC6": _alpha_signal(fs, 1.1, amplitude=2.0, seed=8),
    }
    _push_window(computer, signals)
    assert computer.ready()

    r = computer.reward()
    assert r is not None
    assert -1.0 <= r <= 1.0


def test_faa_reward_computer_pair_metrics():
    fs = 128.0
    pairs = [("F3", "F4"), ("AF3", "AF4")]
    computer = FAARewardComputer(fs=fs, window_s=1.0, channel_pairs=pairs)

    n_samples = int(fs * 1.0) + 5
    for i in range(n_samples):
        computer.push_sample(
            {
                "F3": 0.5 * np.sin(i),
                "F4": 2.0 * np.sin(i),
                "AF3": 0.7 * np.sin(i),
                "AF4": 1.5 * np.sin(i),
            }
        )

    powers = computer.pair_metrics()

    assert len(powers) == 2
    assert [(item["left"], item["right"]) for item in powers] == pairs
    assert all("raw_faa" in item for item in powers)


def test_faa_reward_computer_uses_configured_pair_weights():
    fs = 128.0
    pairs = [("F3", "F4"), ("AF3", "AF4")]
    base = _clean_alpha_signal(fs, 1.1)
    signals = {
        "F3": 0.5 * base,
        "F4": 2.0 * base,
        "AF3": 2.0 * base,
        "AF4": 0.5 * base,
    }

    f3_weighted = FAARewardComputer(
        fs=fs,
        window_s=1.0,
        channel_pairs=pairs,
        pair_weights={"F3/F4": 1.0, "AF3/AF4": 0.1},
    )
    _push_window(f3_weighted, signals)

    af3_weighted = FAARewardComputer(
        fs=fs,
        window_s=1.0,
        channel_pairs=pairs,
        pair_weights={"F3/F4": 0.1, "AF3/AF4": 1.0},
    )
    _push_window(af3_weighted, signals)

    f3_metrics = {
        f"{item['left']}/{item['right']}": item for item in f3_weighted.pair_metrics()
    }
    assert f3_weighted.raw_value() > 1.0
    assert af3_weighted.raw_value() < -1.0
    assert f3_metrics["F3/F4"]["pair_weight"] == 1.0
    assert f3_metrics["AF3/AF4"]["pair_weight"] == 0.1


def test_faa_reward_computer_uses_default_epoc_x_pair_weights():
    fs = 128.0
    computer = FAARewardComputer(fs=fs, window_s=1.0)
    base = _clean_alpha_signal(fs, 1.1)
    signals = {
        "F7": 0.5 * base,
        "F8": 2.0 * base,
        "AF3": 0.5 * base,
        "AF4": 2.0 * base,
        "F3": 0.5 * base,
        "F4": 2.0 * base,
        "FC5": 0.5 * base,
        "FC6": 2.0 * base,
    }

    _push_window(computer, signals)

    metrics = {f"{item['left']}/{item['right']}": item for item in computer.pair_metrics()}

    assert metrics["F3/F4"]["pair_weight"] == 1.0
    assert metrics["F7/F8"]["pair_weight"] == 0.75
    assert metrics["AF3/AF4"]["pair_weight"] == 0.5
    assert metrics["FC5/FC6"]["pair_weight"] == 0.5


def test_faa_eeg_features_include_all_configured_channels():
    fs = 128.0
    channels = ["AF3", "F3", "F4", "O1"]
    computer = FAARewardComputer(fs=fs, window_s=1.0, channels=channels)
    computer.baseline.fit([0.0])

    n_samples = int(fs * 1.0) + 5
    for i in range(n_samples):
        computer.push_sample({ch: np.sin(i / 3.0) for ch in channels})

    reward = computer.reward()
    raw = computer.raw_value()
    features = computer.eeg_features(reward=reward, raw=raw)

    assert features is not None
    assert features["faa"]["left_channel"] == "F3"
    assert features["faa"]["right_channel"] == "F4"
    assert features["faa"]["channel_pairs"] == [["F7", "F8"], ["AF3", "AF4"], ["F3", "F4"], ["FC5", "FC6"]]
    assert {item["name"] for item in features["channels"]} == set(channels)
    for item in features["channels"]:
        assert len(item["position"]) == 3
        assert "alpha_power" in item
