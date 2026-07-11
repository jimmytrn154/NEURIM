"""Presentation schedule and scoring gate.

The reward-delay / credit-assignment fix. When the latent changes from A to B,
the 2s FAA sliding window still contains brain activity elicited by A, so the
first readings after a transition must NOT be attributed to B. Each candidate
gets three intervals:

    0 .. transition_s          morph A -> B          (do NOT score)
    transition_s .. +stabilize_s  hold B, FAA window fills with stable-B EEG (do NOT score)
    +score_s                   scoring window        (THIS FAA is the reward for B)

Critical constraint: stabilize_s must be >= faa.window_s, because FAA is a
trailing-window estimate - the whole scoring window's EEG has to post-date the
morph, or the reward is contaminated by the transition. `validate()` checks it.

`ScoringGate` accumulates only the scoring-interval readings for the current
candidate and, when the window closes, emits one Observation (mean + variance +
effective N + artifact fraction) for the optimizer - see
src/optimizer/observation.py and OptimizerService.observe_observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.optimizer.observation import Observation, window_statistics


@dataclass
class PresentationSchedule:
    transition_s: float = 1.5   # morph A -> B; not scored
    stabilize_s: float = 2.0    # hold B so the FAA window fills with stable EEG; not scored
    score_s: float = 1.5        # scoring window; the FAA here is the reward for B

    @property
    def scoring_start(self) -> float:
        return self.transition_s + self.stabilize_s

    @property
    def scoring_end(self) -> float:
        return self.transition_s + self.stabilize_s + self.score_s

    @property
    def total_s(self) -> float:
        return self.scoring_end

    def phase(self, elapsed: float) -> str:
        if elapsed < self.transition_s:
            return "transition"
        if elapsed < self.scoring_start:
            return "stabilize"
        if elapsed < self.scoring_end:
            return "score"
        return "done"

    def is_scoring(self, elapsed: float) -> bool:
        return self.scoring_start <= elapsed < self.scoring_end

    def morph_alpha(self, elapsed: float) -> float:
        """Interpolation fraction for the morph: reaches 1.0 by the end of the
        transition interval, then holds (so the image is stable while scored)."""
        if self.transition_s <= 0:
            return 1.0
        return min(1.0, elapsed / self.transition_s)

    def validate(self, faa_window_s: float) -> list[str]:
        """Return a list of human-readable warnings (empty if the schedule is sound)."""
        warnings = []
        if self.stabilize_s < faa_window_s:
            warnings.append(
                f"stabilize_s ({self.stabilize_s:.2f}s) < faa.window_s ({faa_window_s:.2f}s): "
                "the scoring window's FAA will still contain transition EEG. "
                f"Raise stabilize_s to >= {faa_window_s:.2f}s or shorten the FAA window."
            )
        if self.score_s <= 0:
            warnings.append("score_s must be > 0 (no scoring interval).")
        return warnings


class ScoringGate:
    """Collects only the scoring-interval reward readings for one candidate,
    then emits a single Observation when the presentation window closes."""

    def __init__(self, schedule: PresentationSchedule, clip: tuple[float, float] = (-1.0, 1.0)):
        self.schedule = schedule
        self.clip = clip
        self._samples: list[float] = []
        self._done = False
        self._step = 0

    def reset(self) -> None:
        """Start a fresh candidate (call when a new latent is presented)."""
        self._samples = []
        self._done = False
        self._step += 1

    def phase(self, elapsed: float) -> str:
        return self.schedule.phase(elapsed)

    def feed(self, elapsed: float, r: float) -> Observation | None:
        """Feed a reward reading at `elapsed` seconds since this candidate was
        presented. Accumulates it only if inside the scoring interval; returns
        an Observation exactly once, when the presentation window closes."""
        if self._done:
            return None
        if self.schedule.is_scoring(elapsed):
            self._samples.append(r)
        if elapsed >= self.schedule.total_s:
            self._done = True
            if not self._samples:
                # No scoring samples landed (e.g. reading cadence too coarse) -
                # emit a maximally-uncertain neutral observation so the loop
                # still advances rather than stalling.
                return Observation(0.0, 1.0, 0.0, 1.0, float(self._step))
            return window_statistics(self._samples, clip=self.clip, t=float(self._step))
        return None
