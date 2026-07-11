"""Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the
noisy, nonstationary, delay-prone reward an FAA signal actually produces.

Components (see the design spec in the repo history):
  - Local GP surrogate inside a trust region (not a global GP over the whole box).
  - Heteroscedastic observation noise: each window's Observation carries its own
    variance, so a stable-EEG reading influences the GP more than a
    motion-artifact one. Fed to the GP as a per-point `alpha`.
  - Thompson-sampling candidate selection: draw one plausible reward function
    from the GP posterior, maximize it inside the trust region. Less jumpy than
    UCB and robust to a noisy "current best".
  - Adaptive trust region with a PROBABILISTIC success test: a step counts as a
    success only when P(f(new) > f(incumbent) + delta | D) > tau, not when the
    raw FAA average is merely higher. Grow on repeated credible improvement,
    shrink on repeated failure.
  - Recency via a sliding observation window (+ retained high-reward
    checkpoints), so slow drift in the user's preference doesn't get averaged
    against stale data.
  - Safe visual-motion constraint: consecutive displayed candidates can't jump
    more than `motion_limit` in normalized space, so the morph stays smooth.
  - Checkpoint-based recovery: when the trust region collapses, recenter on the
    best robust checkpoint and re-widen.

Coordinates are normalized to ~[-1, 1] internally (divide by `bounds`) so the
GP length scales and trust-region geometry are meaningful; the outer interface
still speaks raw z.

Interface: exposes propose()/update()/set_step_size()/revert_to_best() like the
other optimizers, plus a richer observe(candidate, Observation) that
OptimizerService calls when `wants_observation` is set - that's the path that
actually uses the uncertainty.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.optimizer.observation import Observation


class NoiseAwareLatentTuRBO:
    wants_observation = True  # tells OptimizerService to hand us Observations, not scalars

    def __init__(
        self,
        dims: int,
        bounds: float = 1.0,
        window_size: int = 50,
        min_obs: int = 5,
        length_init: float = 0.8,
        length_min: float = 0.05,
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol: int = 5,
        n_candidates: int = 256,
        improve_delta: float = 0.05,
        success_prob_tau: float = 0.6,
        motion_limit: float = 0.2,
        recency_halflife: float = 25.0,
        recovery_reward_floor: float = 0.2,
        rng: np.random.Generator | None = None,
    ):
        self.dims = dims
        self.bounds = bounds
        self.window_size = window_size
        self.min_obs = min_obs
        self.length = length_init
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.success_tol = success_tol
        self.failure_tol = failure_tol
        self.n_candidates = n_candidates
        self.improve_delta = improve_delta
        self.success_prob_tau = success_prob_tau
        self.motion_limit = motion_limit
        self.recency_halflife = recency_halflife
        self.recovery_reward_floor = recovery_reward_floor
        self.rng = rng or np.random.default_rng()

        # Normalized-space state (raw = norm * bounds).
        self._center = np.zeros(dims)          # trust-region center = best robust checkpoint
        self._display = np.zeros(dims)          # currently displayed candidate (for motion clamp)
        self._obs: deque[dict] = deque(maxlen=window_size)
        self._checkpoints: list[dict] = []      # retained high-reward robust points
        self._success = 0
        self._failure = 0
        self._t = 0
        self._last_norm = np.zeros(dims)        # the candidate propose() last handed out

        # Outer-interface state (raw coordinates).
        self.z = np.zeros(dims)
        self.best_z = np.zeros(dims)
        self.best_reward = -np.inf
        self._best_var = 1.0
        self.velocity = np.zeros(dims)

        self._gp = None

    # ---- coordinate helpers -------------------------------------------------
    def _to_norm(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(raw, dtype=float) / self.bounds, -1.0, 1.0)

    def _to_raw(self, norm: np.ndarray) -> np.ndarray:
        return np.clip(norm * self.bounds, -self.bounds, self.bounds)

    # ---- GP over the sliding window + checkpoints ---------------------------
    def _fit_gp(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        data = list(self._obs) + self._checkpoints
        X = np.array([d["z"] for d in data])
        y = np.array([d["y"] for d in data])
        # Heteroscedastic + recency-inflated noise: older and artifact-heavy
        # observations get larger alpha (less influence).
        ages = self._t - np.array([d["t"] for d in data])
        recency = np.exp(np.log(2.0) * ages / max(self.recency_halflife, 1e-6))  # 1 at now, grows with age
        alpha = np.array([d["var"] for d in data]) * recency
        alpha = np.maximum(alpha, 1e-6)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(self.dims), length_scale_bounds=(1e-2, 1e1), nu=2.5
        )
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=0)
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        # A local GP over a small window legitimately pins length scales to their
        # bounds; the warning is expected here and not actionable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y)
        return gp

    def _length_scales(self) -> np.ndarray:
        """Per-dim GP length scales (ARD), for shaping the trust region."""
        if self._gp is None:
            return np.ones(self.dims)
        try:
            ls = self._gp.kernel_.k2.length_scale
            ls = np.asarray(ls, dtype=float)
            if ls.ndim == 0:
                ls = np.full(self.dims, float(ls))
            return ls
        except Exception:  # noqa: BLE001 - kernel introspection is best-effort
            return np.ones(self.dims)

    def _trust_region_box(self) -> tuple[np.ndarray, np.ndarray]:
        """TuRBO box: side length `self.length` scaled per-dim by the ARD length
        scales (long length scale = flat direction = search wider there)."""
        ls = self._length_scales()
        weights = ls / np.exp(np.mean(np.log(np.maximum(ls, 1e-6))))  # geometric-mean-normalized
        half = 0.5 * self.length * weights
        lo = np.clip(self._center - half, -1.0, 1.0)
        hi = np.clip(self._center + half, -1.0, 1.0)
        return lo, hi

    # ---- propose ------------------------------------------------------------
    def propose(self) -> np.ndarray:
        if len(self._obs) + len(self._checkpoints) < self.min_obs or self.dims == 0:
            # Cold start: random point in the trust region around the center.
            lo, hi = self._center - 0.5 * self.length, self._center + 0.5 * self.length
            cand = self.rng.uniform(np.clip(lo, -1, 1), np.clip(hi, -1, 1))
        else:
            self._gp = self._fit_gp()
            lo, hi = self._trust_region_box()
            pool = self.rng.uniform(lo, hi, size=(self.n_candidates, self.dims))
            # Thompson sampling: one draw from the posterior over the pool, argmax.
            sample = self._gp.sample_y(pool, n_samples=1, random_state=int(self.rng.integers(1 << 31)))
            cand = pool[int(np.argmax(sample[:, 0]))]

        # Safe visual-motion constraint: don't let the displayed candidate jump
        # more than motion_limit from the current display, per dimension.
        cand = np.clip(cand, self._display - self.motion_limit, self._display + self.motion_limit)
        cand = np.clip(cand, -1.0, 1.0)
        self._last_norm = cand
        return self._to_raw(cand)

    # ---- observe (the uncertainty-aware update) -----------------------------
    def observe(self, candidate: np.ndarray, obs: Observation) -> bool:
        self._t += 1
        znorm = self._to_norm(candidate)
        self._display = znorm.copy()
        self._obs.append({"z": znorm, "y": obs.reward_mean, "var": obs.reward_variance, "t": self._t})

        # Probabilistic success test: credible improvement over the incumbent,
        # not just a numerically higher (possibly noise-spike) FAA average.
        if not np.isfinite(self.best_reward):
            # No incumbent yet: bootstrap it from the first observation.
            p_improve, credible = 1.0, True
        else:
            gap = obs.reward_mean - (self.best_reward + self.improve_delta)
            denom = float(np.sqrt(obs.reward_variance + self._best_var)) or 1e-6
            p_improve = 0.5 * (1.0 + _erf(gap / (np.sqrt(2.0) * denom)))
            credible = p_improve > self.success_prob_tau
        if credible:
            self._success += 1
            self._failure = 0
        else:
            self._failure += 1
            self._success = 0

        # Trust-region adaptation.
        if self._success >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            self._success = 0
        elif self._failure >= self.failure_tol:
            self.length = max(self.length * 0.5, self.length_min)
            self._failure = 0

        # Accept as new best/checkpoint on credible improvement.
        accepted = credible and obs.reward_mean > self.best_reward
        if accepted:
            prev = self.best_z.copy()
            self.best_reward = obs.reward_mean
            self._best_var = obs.reward_variance
            self.best_z = self._to_raw(znorm)
            self._center = znorm.copy()
            self.z = self.best_z.copy()
            self.velocity = self.best_z - prev
            self._checkpoints.append({"z": znorm.copy(), "y": obs.reward_mean, "var": obs.reward_variance, "t": self._t})
            self._checkpoints = sorted(self._checkpoints, key=lambda d: d["y"], reverse=True)[:3]
        else:
            # Motion signal for the settle check shrinks as the TR collapses.
            self.velocity = (self._to_raw(znorm) - self.z) * (self.length / self.length_init)

        # Checkpoint-based recovery: only RESTART (recenter + re-widen) if the
        # trust region collapsed onto a *bad* point - that's a stuck local
        # region worth escaping. If it collapsed onto a good point, that IS
        # convergence: stay tight so the state machine can SETTLE.
        collapsed = self.length <= self.length_min * 1.001
        if collapsed and not accepted and self.best_reward < self.recovery_reward_floor:
            self.revert_to_best()

        return accepted

    # ---- scalar fallback (if no Observation is available) -------------------
    def update(self, candidate: np.ndarray, reward_before: float, reward_after: float) -> bool:
        # Unknown per-window variance -> assume a moderate fixed noise.
        return self.observe(candidate, Observation(reward_after, 0.05, 1.0, 0.0, self._t + 1))

    def set_step_size(self, step_size: float) -> None:
        # TuRBO manages its own trust-region length; the state-machine schedule
        # only caps the max so REFINE can't keep exploring wide.
        self.length_max = float(np.clip(step_size * 2.0, self.length_min * 2, 1.6))
        self.length = min(self.length, self.length_max)

    def revert_to_best(self) -> None:
        if np.isfinite(self.best_reward):
            self._center = self._to_norm(self.best_z)
            self.z = self.best_z.copy()
        self.length = self.length_init
        self._success = 0
        self._failure = 0


def _erf(x: float) -> float:
    """math.erf on a scalar (avoids importing scipy just for the normal CDF)."""
    import math

    return math.erf(x)
