"""CALIBRATE -> EXPLORE -> REFINE -> SETTLE, with a RECOVER escape hatch.

CALIBRATE: waiting on the Signal service's baseline capture.
EXPLORE: large steps, wide search.
REFINE: steps shrink as the reward trend climbs.
SETTLE: reward stays high and motion drops below threshold -> lock.
RECOVER: reward stays negative for several steps -> revert to the last
         high-reward checkpoint, widen the search, blacklist the region left.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.common.config import StateMachineConfig, OptimizerConfig
from src.common.messages import State


@dataclass
class StateMachine:
    sm_config: StateMachineConfig
    opt_config: OptimizerConfig
    state: State = "calibrate"
    step_index: int = 0
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    negative_streak: int = 0
    settle_streak: int = 0
    blacklist: list[np.ndarray] = field(default_factory=list)

    def mark_calibrated(self) -> None:
        if self.state == "calibrate":
            self.state = "explore"

    def _recent_average(self, n: int = 5) -> float:
        if not self.reward_history:
            return 0.0
        recent = list(self.reward_history)[-n:]
        return float(np.mean(recent))

    def observe(self, reward: float, step_norm: float) -> State:
        """Call once per optimizer step with the accepted/estimated reward and
        the norm of the last accepted step. Returns the (possibly new) state.
        """
        self.step_index += 1
        self.reward_history.append(reward)

        if self.state == "calibrate":
            return self.state

        if reward < 0:
            self.negative_streak += 1
        else:
            self.negative_streak = 0

        if self.negative_streak >= self.sm_config.recover_negative_streak:
            self.state = "recover"
            return self.state

        if self.state == "recover":
            # One recovery step taken (widen + revert handled by caller); go
            # back to exploring from the restored checkpoint.
            self.negative_streak = 0
            self.state = "explore"
            return self.state

        if reward >= self.sm_config.settle_reward_threshold and step_norm < self.sm_config.settle_motion_threshold:
            self.settle_streak += 1
        else:
            self.settle_streak = 0

        can_settle = self.step_index >= self.sm_config.min_steps_before_settle
        if can_settle and self.settle_streak >= self.sm_config.settle_patience_steps:
            self.state = "settle"
            return self.state

        avg_recent = self._recent_average()
        if self.state == "explore" and avg_recent >= self.sm_config.refine_entry_reward:
            self.state = "refine"
        elif self.state == "refine" and avg_recent < (
            self.sm_config.refine_entry_reward - self.sm_config.refine_exit_margin
        ):
            self.state = "explore"

        return self.state

    def step_size(self) -> float:
        """Interpolate step size: wide in EXPLORE, shrinking in REFINE as the
        reward trend climbs, and widened again after RECOVER.
        """
        explore = self.opt_config.step_size_explore
        refine_min = self.opt_config.step_size_refine_min

        if self.state in ("calibrate", "explore"):
            return explore
        if self.state == "recover":
            return explore * self.sm_config.recover_widen_factor
        if self.state == "settle":
            return refine_min

        # REFINE: shrink proportional to how high the recent average reward is
        # above the settle threshold (0 -> explore-sized steps, 1 -> refine_min).
        progress = np.clip(self._recent_average() / max(self.sm_config.settle_reward_threshold, 1e-6), 0.0, 1.0)
        return float(explore + progress * (refine_min - explore))

    def is_locked(self) -> bool:
        return self.state == "settle"

    def should_stop(self) -> bool:
        return self.is_locked() or self.step_index >= self.sm_config.max_steps
