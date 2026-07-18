"""Fake reward sources with the exact same interface FAA reward has: a scalar
in [-1, 1], read a few times a second. This is what build-order step 1 uses -
prove the optimizer/generator loop converges before EEG ever touches it.

Downstream code (Optimizer, Orchestrator) only ever sees RewardMessage, so
swapping this out for real FAA later is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.common.messages import RewardMessage


class RewardSource:
    """Common interface: FAARewardComputer-backed or fake, doesn't matter."""

    def read_reward(self) -> RewardMessage | None:
        raise NotImplementedError


class KeyboardRewardSource(RewardSource):
    """Up/down arrow keys nudge reward; it decays toward 0 between presses.

    Uses `pynput` for global key capture (works without terminal focus, which
    matters once the frontend/pyramid window has focus instead). Falls back
    to raising ImportError with a clear message if pynput isn't installed or
    the OS denies the accessibility permission it needs on macOS.
    """

    def __init__(self, decay: float = 0.92, step: float = 0.15):
        from pynput import keyboard

        self._value = 0.0
        self._decay = decay
        self._step = step
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key) -> None:
        from pynput.keyboard import Key

        if key == Key.up:
            self._value = float(np.clip(self._value + self._step, -1.0, 1.0))
        elif key == Key.down:
            self._value = float(np.clip(self._value - self._step, -1.0, 1.0))

    def read_reward(self) -> RewardMessage:
        reading = self._value
        self._value *= self._decay
        return RewardMessage(r=reading, source="fake")

    def close(self) -> None:
        self._listener.stop()


@dataclass
class ScriptedRewardSource(RewardSource):
    """A synthetic reward that *looks like* an FAA brain signal: a hidden-target
    preference gradient buried under the kind of wandering, rhythmic fluctuation
    real frontal-alpha asymmetry shows. Used to drive the loop without any human
    (or EEG) in the loop - CI convergence checks and the live no-headset demo.

    Unlike a plain `1 - distance` hill, the reward here *alternates*: it drifts
    up and down on a slow "mood" timescale and carries a couple of slow rhythms,
    so the optimizer keeps moving and the on-screen morph stays alive instead of
    snapping to one image and freezing. The fluctuation anneals down over the
    session, so the signal eventually calms into a plateau and the search can
    still SETTLE on a good frame in bounded time.

    Leave `seed=None` (the default) for a fresh trajectory every session - each
    generation then wanders differently. Pass an int seed for a reproducible run
    (CLI, CI).

    `get_current_z` must return the *candidate currently being evaluated*
    (e.g. `OptimizerService.pending_candidate`), not the last accepted point -
    the latter is frozen for the whole reward window and would carry no
    signal about the thing actually on screen.
    """

    target: np.ndarray
    get_current_z: Callable[[], np.ndarray]
    noise_std: float = 0.05
    seed: int | None = None
    # How strongly the hidden target pulls the reward. Must stay dominant over
    # the fluctuation below or the windowed hill-climb can't tell candidates
    # apart and the search stalls (thrashes explore<->recover, never converges).
    preference_gain: float = 1.0
    # Ornstein-Uhlenbeck "mood" wander: slow, mean-reverting drift.
    wander_std: float = 0.05
    wander_theta: float = 0.12
    # A few slow sinusoids give the reward a rhythmic up/down alternation.
    osc_amplitude: float = 0.08
    n_oscillators: int = 2
    osc_period_range: tuple[float, float] = (30.0, 90.0)
    # Fluctuation amplitude decays from 1.0 toward `anneal_floor` with this
    # time constant (in reads), so early = lively/alternating, late = calm
    # enough to plateau and SETTLE.
    anneal_tau_reads: float = 180.0
    anneal_floor: float = 0.25

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._t = 0
        self._wander = float(self._rng.uniform(-0.2, 0.2))
        self._oscillators = [
            (
                float(self._rng.uniform(*self.osc_period_range)),
                float(self._rng.uniform(0.0, 2.0 * np.pi)),
            )
            for _ in range(max(0, self.n_oscillators))
        ]

    def _fluctuation_scale(self) -> float:
        decay = np.exp(-self._t / max(self.anneal_tau_reads, 1e-6))
        return float(self.anneal_floor + (1.0 - self.anneal_floor) * decay)

    def read_reward(self) -> RewardMessage:
        z = np.asarray(self.get_current_z())
        dist = np.linalg.norm(z - self.target[: len(z)])
        # Slow-moving preference: rewards getting close to the hidden target,
        # spanning ~[-1, 1] over the optimizer's bounds before the gain.
        preference = self.preference_gain * (1.0 - dist)

        scale = self._fluctuation_scale()

        # Ornstein-Uhlenbeck mean-reverting wander - the slow drift that makes
        # the signal alternate up and down across reward windows (a plain white
        # noise term averages out over a window and never moves the search).
        self._wander += -self.wander_theta * self._wander + self._rng.normal(0.0, self.wander_std)

        osc = 0.0
        for period, phase in self._oscillators:
            osc += np.sin(2.0 * np.pi * self._t / period + phase)
        if self._oscillators:
            osc *= self.osc_amplitude / len(self._oscillators)

        jitter = self._rng.normal(0.0, self.noise_std)

        raw = preference + scale * (self._wander + osc) + jitter
        self._t += 1
        r = float(np.clip(raw, -1.0, 1.0))
        return RewardMessage(r=r, raw_faa=float(raw), source="fake")
