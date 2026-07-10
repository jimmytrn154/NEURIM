"""The dumbest thing that works: momentum hill-climbing on a noisy scalar.

Propose a small step in some direction; if the (windowed, noise-thresholded)
reward went up, keep stepping that way; if not, reverse or pick a new
direction. This is deliberately not clever - it's the thing you ship first,
per the spec, before reaching for (1+1)-ES or GP-BO.
"""

from __future__ import annotations

import numpy as np


class MomentumHillClimb:
    def __init__(
        self,
        dims: int,
        bounds: float = 1.0,
        step_size: float = 0.1,
        momentum: float = 0.5,
        noise_threshold: float = 0.08,
        rng: np.random.Generator | None = None,
    ):
        self.dims = dims
        self.bounds = bounds
        self.step_size = step_size
        self.momentum = momentum
        self.noise_threshold = noise_threshold
        self.rng = rng or np.random.default_rng()

        self.z = np.zeros(dims)
        self.velocity = self._random_direction() * step_size
        self.best_z = self.z.copy()
        self.best_reward = -np.inf

    def _random_direction(self) -> np.ndarray:
        d = self.rng.normal(size=self.dims)
        norm = np.linalg.norm(d)
        return d / norm if norm > 1e-9 else d

    def propose(self) -> np.ndarray:
        """The candidate z to show next; doesn't mutate state until update()."""
        candidate = self.z + self.velocity
        return np.clip(candidate, -self.bounds, self.bounds)

    def update(self, candidate: np.ndarray, reward_before: float, reward_after: float) -> bool:
        """Tell the optimizer what happened after showing `candidate` for a
        full window. Returns True if the step was accepted.
        """
        if reward_after > self.best_reward:
            self.best_reward = reward_after
            self.best_z = candidate.copy()

        diff = reward_after - reward_before
        if abs(diff) < self.noise_threshold:
            # Inside the noise band - don't trust it either way. Hold position,
            # try a fresh random direction next time.
            self.velocity = self._random_direction() * self.step_size
            return False

        # If reward increase, continue moving with some random in direction
        if diff > 0:
            self.z = candidate
            new_dir = self.momentum * self.velocity + (1 - self.momentum) * (
                self._random_direction() * self.step_size
            )
            norm = np.linalg.norm(new_dir)
            self.velocity = new_dir / norm * self.step_size if norm > 1e-9 else new_dir
            return True

        # Reward dropped clearly: reverse course, damped so the reversal shrinks
        # rather than bouncing back at full magnitude (avoids reversal oscillation
        # around the ridge; the next set_step_size renormalizes the direction).
        self.velocity = -self.velocity * 0.75
        return False

    def set_step_size(self, step_size: float) -> None:
        self.step_size = step_size

        norm = np.linalg.norm(self.velocity)
        self.velocity = self.velocity / norm * step_size if norm > 1e-9 else self.velocity

    def revert_to_best(self) -> None:
        self.z = self.best_z.copy()
        self.velocity = self._random_direction() * self.step_size
