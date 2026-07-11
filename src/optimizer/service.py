"""The Optimizer service: reward in, latent stream out. ~150 lines including
the state machine wiring - this is the whole "brain" of the system.

Protocol: a candidate z is shown for `reward_window_steps` reward readings
(matching how long the morph is on screen); the windowed average is the
noise-robust signal the hill-climb accepts or rejects against.
"""

from __future__ import annotations

import numpy as np

from src.common.config import Config
from src.common.messages import LatentMessage
from src.optimizer.evolution import GPBanditOptimizer, OnePlusOneES
from src.optimizer.hill_climb import MomentumHillClimb
from src.optimizer.latent_turbo import NoiseAwareLatentTuRBO
from src.optimizer.observation import window_statistics
from src.optimizer.state_machine import StateMachine

_ALGORITHMS = {
    "hill_climb": MomentumHillClimb,
    "es_1p1": OnePlusOneES,
    "gp_bo": GPBanditOptimizer,
    "latent_turbo": NoiseAwareLatentTuRBO,
}


def _build_algorithm(config: Config):
    dims = config.optimizer.search_dims
    bounds = config.optimizer.bounds
    algo_cls = _ALGORITHMS[config.optimizer.algorithm]
    if algo_cls is MomentumHillClimb:
        return MomentumHillClimb(
            dims,
            bounds=bounds,
            step_size=config.optimizer.step_size_explore,
            momentum=config.optimizer.step_momentum,
            noise_threshold=config.optimizer.noise_threshold,
        )
    if algo_cls is OnePlusOneES:
        return OnePlusOneES(
            dims,
            bounds=bounds,
            sigma=config.optimizer.step_size_explore,
            noise_threshold=config.optimizer.noise_threshold,
        )
    if algo_cls is NoiseAwareLatentTuRBO:
        return NoiseAwareLatentTuRBO(dims, bounds=bounds)
    return GPBanditOptimizer(dims, bounds=bounds)


class OptimizerService:
    def __init__(self, config: Config):
        self.config = config
        self.optimizer = _build_algorithm(config)
        self.state_machine = StateMachine(config.state_machine, config.optimizer)
        self._reward_buffer: list[float] = []
        # -inf, not 0.0: the first window always beats it, so the search accepts
        # its first real move and adopts that reward as the baseline. Seeding 0.0
        # stranded the search whenever the whole neighborhood of the origin
        # scored below 0 (e.g. a distance-based reward), because nothing could
        # beat the optimistic 0.0 and every step was rejected.
        self._current_reward_estimate = float("-inf")
        self._step_index = 0
        self._candidate = self.optimizer.propose()

    def current_z(self) -> np.ndarray:
        """The last *accepted* latent - what should be on screen at rest."""
        return self.optimizer.z

    def pending_candidate(self) -> np.ndarray:
        """The candidate currently being shown, awaiting a verdict."""
        return self._candidate

    def notify_calibrated(self) -> None:
        self.state_machine.mark_calibrated()

    def observe_reward(self, r: float) -> LatentMessage | None:
        """Feed one reward reading. Returns a LatentMessage once a full
        window has accumulated and a step decision has been made; otherwise
        None (caller should keep interpolating toward the pending candidate).

        This is the simple per-window path (average the last N readings). The
        scheduled path - one clean Observation per candidate, gathered only
        during the scoring interval - calls `observe_observation` instead; see
        src/signal_service/presentation.py.
        """
        self._reward_buffer.append(r)
        if len(self._reward_buffer) < self.config.optimizer.reward_window_steps:
            return None
        observation = window_statistics(
            self._reward_buffer, clip=self.config.faa.clip, t=self._step_index
        )
        self._reward_buffer.clear()
        return self._apply_step(observation)

    def observe_observation(self, observation) -> LatentMessage:
        """Feed one fully-formed Observation (mean + variance + effective N +
        artifact fraction) for the current candidate and take exactly one
        optimizer step. Used by the presentation-schedule path, where the
        Observation is aggregated only from the scoring interval."""
        return self._apply_step(observation)

    def _apply_step(self, observation) -> LatentMessage:
        windowed_reward = float(observation.reward_mean)
        reward_before = self._current_reward_estimate
        candidate = self._candidate

        # Uncertainty-aware optimizers (Noise-Aware Latent TuRBO) consume the
        # whole Observation - so a stable reading weighs more than a
        # motion-artifact one. Everything else gets the plain scalar mean.
        if getattr(self.optimizer, "wants_observation", False):
            accepted = self.optimizer.observe(candidate, observation)
        else:
            accepted = self.optimizer.update(candidate, reward_before, windowed_reward)
        if accepted:
            self._current_reward_estimate = windowed_reward

        step_norm = float(np.linalg.norm(candidate - self.optimizer.z)) if not accepted else float(
            np.linalg.norm(getattr(self.optimizer, "velocity", candidate - self.optimizer.best_z))
        )
        state = self.state_machine.observe(windowed_reward, step_norm)

        if state == "recover":
            self.state_machine.blacklist.append(np.array(self.optimizer.z))
            self.optimizer.revert_to_best()
        else:
            self.optimizer.set_step_size(self.state_machine.step_size())

        self._step_index += 1
        self._candidate = self.optimizer.propose()

        return LatentMessage(
            z=self._candidate.tolist(),
            step_index=self._step_index,
            state=state,
            reward_estimate=windowed_reward,
        )
