"""Real-time latent morphing between a jumpy stream of target latents and a
smooth, bounded-speed path the generator can actually render frame by frame.

The core problem: the optimizer emits a new target z every ~1s, but a smooth
morph needs many small intermediate frames, and a *new* target often arrives
before the morph to the old one finishes. A fixed-endpoint morph(z_old, z_new)
therefore always lags or jumps.

The fix is a FOLLOWER, not a fixed morph. `LatentMorpher.step(target)` returns
the next z to render, moving at most `max_step` toward the latest target. That:
  - bounds the per-frame change (smoothness by construction),
  - decouples morph speed (`max_step`) from optimizer rate,
  - is preemptible: a new target just re-aims, no discontinuity.

Render each returned z img2img from the previous frame (StreamDiffusion-style)
and you get a genuine morph rather than a crossfade or a re-roll.

`morph_path(z_old, z_new, n)` is the simple fixed-endpoint helper for the
offline case where both endpoints are known and won't change.
"""

from __future__ import annotations

import numpy as np


class LatentMorpher:
    def __init__(
        self,
        z0: np.ndarray,
        max_step: float = 0.1,
        smoothing: float = 0.0,
    ):
        """
        max_step:  max Euclidean distance z may move per step() call (per
                   rendered frame). Smaller = smoother + slower. A full
                   anchor-to-anchor traversal of distance D takes ~D/max_step
                   frames, i.e. D/(max_step * server_fps) seconds - tune from that.
        smoothing: 0 = constant-speed (cruise at max_step, hard stop at target).
                   In (0, 1] adds exponential ease-in near the target: the step
                   is capped at smoothing * remaining_distance once that is below
                   max_step, so arrival is gentle instead of abrupt.
        """
        self.z = np.asarray(z0, dtype=float).copy()
        self.max_step = float(max_step)
        self.smoothing = float(smoothing)

    def step(self, target: np.ndarray) -> np.ndarray:
        """Advance toward `target` by at most max_step; return the new z."""
        target = np.asarray(target, dtype=float)
        delta = target - self.z
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-12:
            return self.z.copy()

        step_len = min(self.max_step, dist)
        if self.smoothing > 0.0:
            # Ease in near the target, but never stall (Zeno): keep a tiny floor.
            eased = self.smoothing * dist
            step_len = min(step_len, max(eased, min(dist, 1e-3)))

        self.z = self.z + delta / dist * step_len
        return self.z.copy()

    def at_target(self, target: np.ndarray, tol: float = 1e-3) -> bool:
        return bool(np.linalg.norm(np.asarray(target, dtype=float) - self.z) <= tol)

    def frames_to_target(self, target: np.ndarray) -> int:
        """Lower bound on frames to reach `target` at the current max_step -
        useful for reasoning about morph latency vs server fps."""
        dist = float(np.linalg.norm(np.asarray(target, dtype=float) - self.z))
        return int(np.ceil(dist / self.max_step)) if self.max_step > 0 else 0

    def reset(self, z0: np.ndarray) -> None:
        self.z = np.asarray(z0, dtype=float).copy()


def morph_path(z_old: np.ndarray, z_new: np.ndarray, n: int) -> np.ndarray:
    """Fixed-endpoint linear path of `n` intermediate latents (inclusive of
    z_new, exclusive of z_old). Use only when both endpoints are known and
    fixed - for the live loop use LatentMorpher instead."""
    z_old = np.asarray(z_old, dtype=float)
    z_new = np.asarray(z_new, dtype=float)
    alphas = np.linspace(0.0, 1.0, n + 1)[1:]  # skip alpha=0 (that's z_old, already shown)
    return z_old[None, :] + (z_new - z_old)[None, :] * alphas[:, None]
