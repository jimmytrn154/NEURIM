"""Synthetic EEG carrying a controllable, decodable *preference* signal.

MockEEGSource modulates only frontal alpha asymmetry, so it can validate the FAA
path but not a learned preference model. This source injects a preference signal
at a known signal-to-noise ratio so the whole pairwise pipeline (features ->
contrast -> leave-session-out validation) can be tested offline with ground truth:

  - a parietal/occipital component whose band power rises monotonically with the
    preference of the currently shown image (the signal a learned model can use
    beyond FAA), and
  - a weaker frontal alpha asymmetry (so FAA-alone gets *some*, but less, signal).

Both are monotonic in signed preference (amplitude scales as 1 + gain*p, so power
is monotonic in p rather than in |p|). At signal_gain == 0 - or preference 0 -
there is no preference information and leave-session-out AUC must collapse to 0.5,
which is the negative control the validation harness checks.

The recorder/live loop sets the current image's preference via `set_preference`
(or supply a `preference_fn(t)`); preference is the image's closeness to the
hidden target, in [-1, 1].
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

FRONTAL_RIGHT = {"F8", "AF4", "F4", "FC6"}
FRONTAL_LEFT = {"F7", "AF3", "F3", "FC5"}
PARIETAL = {"P7", "P8", "O1", "O2"}


class MockPreferenceEEGSource:
    """14-channel synthetic EEG with a tunable-SNR preference signal.

    SNR is governed by `signal_gain / noise_std`. `parietal_gain` is the depth of
    the learned-model signal; `faa_gain` is the (smaller) depth of the FAA signal.
    """

    def __init__(
        self,
        channels: list[str],
        sample_rate_hz: int = 128,
        preference_fn=None,
        signal_gain: float = 0.6,
        parietal_gain: float = 1.0,
        faa_gain: float = 0.35,
        noise_std: float = 0.3,
        seed: int = 0,
    ):
        self.channels = list(channels)
        self.fs = sample_rate_hz
        self._preference_fn = preference_fn
        self.preference = 0.0
        self.signal_gain = signal_gain
        self.parietal_gain = parietal_gain
        self.faa_gain = faa_gain
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._dt = 1.0 / sample_rate_hz

    def set_preference(self, value: float) -> None:
        self.preference = float(np.clip(value, -1.0, 1.0))

    def _current_preference(self, t: float) -> float:
        if self._preference_fn is not None:
            return float(np.clip(self._preference_fn(t), -1.0, 1.0))
        return self.preference

    def connect(self) -> None:
        self._t = 0.0

    def stream(self, *args, **kwargs) -> Iterator[tuple[float, dict[str, float]]]:
        while True:
            self._t += self._dt
            p = self._current_preference(self._t)
            g = self.signal_gain * p
            alpha = np.sin(2 * np.pi * 10.0 * self._t)
            theta = np.sin(2 * np.pi * 6.0 * self._t)
            sample = {}
            for ch in self.channels:
                noise = self._rng.normal(0, self.noise_std)
                value = alpha + noise
                if ch in PARIETAL:
                    # Learned-model signal: parietal theta amplitude rises with preference.
                    value += (1.0 + self.parietal_gain * g) * theta
                if ch in FRONTAL_RIGHT:
                    value *= 1.0 + self.faa_gain * g
                elif ch in FRONTAL_LEFT:
                    value *= 1.0 - self.faa_gain * g
                sample[ch] = float(value)
            yield self._t, sample

    def close(self) -> None:
        pass
