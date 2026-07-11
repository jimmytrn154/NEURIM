"""Headless tests for the pairwise preference reward + optimizer core.

Exercises the same reward/optimizer path as run_poodle_turbo_morph.py without the
OpenCV window or the diffusion model. Two things are checked:

1. The reward gradient is correct: a candidate closer to the target earns a
   higher reward than one further away (deterministic, the core guarantee).
2. Aggregate steering: with a decodable mock EEG signal the optimizer concentrates
   the search above chance and more reliably than with no signal.

Absolute closed-loop convergence is deliberately modest - single-window EEG at a
realistic SNR gives a noisy, saturating reward, so the honest, robust claim is
"reliably steers above chance", not "pins the exact target".
"""

import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimizer.latent_turbo import NoiseAwareLatentTuRBO
from src.signal_service.mock_preference import MockPreferenceEEGSource
from src.signal_service.preprocessing import Preprocessor, PreprocessedSource
from src.signal_service.preference_reward import LearnedPreferenceReward

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
PAIRS = [["F7", "F8"], ["AF3", "AF4"], ["F3", "F4"], ["FC5", "FC6"]]
FS = 128
N_BREEDS = 7
TARGET = 3
UNIFORM = 1.0 / N_BREEDS


def _softmax(z, temp=3.0):
    e = np.exp((z - z.max()) * temp)
    return e / e.sum()


def _target_pref(z):
    return 2.0 * float(_softmax(z)[TARGET]) - 1.0


def _make_reward(model_path, signal_gain, seed):
    src = MockPreferenceEEGSource(CHANNELS, FS, signal_gain=signal_gain, noise_std=0.3, seed=seed)
    pre = Preprocessor(FS, CHANNELS)
    wrapped = PreprocessedSource(src, pre)
    wrapped.connect()
    reward = LearnedPreferenceReward(
        eeg_source=wrapped, model_path=model_path, sample_rate_hz=FS, channels=CHANNELS,
        pairs=PAIRS, window_s=2.0, scoring_seconds=1.5, baseline_windows=6,
        preprocessor=pre, mock_source=src, preference_fn=_target_pref,
    )
    return reward, src


def _quiet(fn, *a, **k):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _run_loop(model_path, signal_gain, seed, steps=35):
    reward, _src = _make_reward(model_path, signal_gain, seed)
    _quiet(reward.calibrate)
    _quiet(reward.set_anchor, np.zeros(N_BREEDS))  # neutral reference
    opt = NoiseAwareLatentTuRBO(dims=N_BREEDS, bounds=1.0, min_obs=5, length_init=1.6,
                                rng=np.random.default_rng(seed + 17))
    candidate = opt.propose()
    concentration = []
    for step in range(1, steps + 1):
        obs = _quiet(reward.observe, candidate, step)
        opt.observe(candidate, obs)
        candidate = opt.propose()
        concentration.append(float(_softmax(candidate)[TARGET]))
    return float(np.mean(concentration[-10:]))


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("pref")
    csv = out / "trials.csv"
    model = out / "model.joblib"
    subprocess.run(
        [sys.executable, "scripts/record_reward_trials.py", "--mock", "--sessions", "3",
         "--trials", "90", "--signal-gain", "0.6", "--stimuli-dir", "scripts/data/stimuli",
         "--out", str(csv)],
        cwd=ROOT, check=True, capture_output=True,
    )
    subprocess.run(
        [sys.executable, "scripts/train_reward_model.py", "--data", str(csv),
         "--model", "lda", "--out", str(model)],
        cwd=ROOT, check=True, capture_output=True,
    )
    return model


def test_reward_gradient_points_toward_target(trained_model):
    """A near-target candidate must out-reward a far-from-target one (deterministic)."""
    reward, _ = _make_reward(trained_model, signal_gain=0.6, seed=5)
    _quiet(reward.calibrate)
    far = np.full(N_BREEDS, 0.0); far[TARGET] = -1.0
    near = np.full(N_BREEDS, 0.0); near[TARGET] = 2.0
    _quiet(reward.set_anchor, far)  # anchor on a poor reference
    r_far = np.mean([_quiet(reward.observe, far, 1).reward_mean for _ in range(3)])
    r_near = np.mean([_quiet(reward.observe, near, 1).reward_mean for _ in range(3)])
    assert r_near > r_far + 0.2, f"near={r_near:.3f} not clearly above far={r_far:.3f}"


@pytest.mark.slow
def test_signal_steers_search_above_chance(trained_model):
    seeds = range(6)
    signal = [_run_loop(trained_model, 0.6, s) for s in seeds]
    no_signal = [_run_loop(trained_model, 0.0, s) for s in seeds]
    med_sig = statistics.median(signal)
    med_nos = statistics.median(no_signal)
    # With signal the search reliably concentrates above uniform, and more so than
    # without (median is robust to the no-signal random-walk outliers).
    assert med_sig > UNIFORM, f"signal median {med_sig:.3f} not above uniform {UNIFORM:.3f}"
    assert med_sig > med_nos, f"signal median {med_sig:.3f} <= no-signal median {med_nos:.3f}"
