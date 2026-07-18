"""Brain-LLM Interface controller (replicates arXiv:2603.16897 in NEURIM).

The loop, from the paper's test-time-scaling design:

    user prompt
      -> generate a small candidate pool (LLM-refined prompts -> diffusion)
      -> score each candidate for prompt ALIGNMENT (VQAScore/CLIP proxy)
      -> rank (alignment is a hard floor; among aligned ones, predicted preference)
      -> select best, MORPH the display from the current image into it, hold static
      -> measure EEG SATISFACTION on the static hold (never during the morph)
      -> accept if confidently satisfied (2 consistent valid reads), else the LLM
         REFINES toward what satisfied the user and the pool regenerates
      -> keep BEST-SO-FAR; stop on acceptance or max iterations, return best.

Design constraints honored:
  - alignment is enforced (a high-satisfaction but off-brief image cannot win),
  - invalid-quality EEG is missing evidence, never dissatisfaction,
  - uncertain EEG does not refine the generator, and
  - best-so-far never regresses.

Components are Protocols so the same controller runs three ways: pure simulation
(instant, hardware-free, used by tests), mock-EEG (real acquisition + injected
signal), and real (SD-Turbo + CLIP + EPOC X). See scripts/run_brain_llm.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Candidate:
    id: str
    prompt: str
    feat: np.ndarray | None = None      # embedding used for alignment/preference/morph
    image: Any = None                   # PIL.Image or None in simulation
    alignment: float = 0.0              # in [0, 1]; VQAScore/CLIP proxy vs the brief
    pred_pref: float = 0.0              # surrogate preference (pre-EEG), in [0, 1]
    p_satisfied: float | None = None    # calibrated EEG P(satisfied), if measured
    quality: str = "valid"             # valid | retry | invalid


@dataclass
class IterationLog:
    iteration: int
    selected: str
    alignment: float
    p_satisfied: float | None
    quality: str
    decision: str                       # accept | refine | uncertain | invalid-skip
    best_so_far: str


class CandidateGenerator(Protocol):
    def generate(self, brief: str, feedback: "Feedback") -> list[Candidate]: ...


class AlignmentScorer(Protocol):
    def score(self, candidate: Candidate, brief: str) -> float: ...


class PreferenceSurrogate(Protocol):
    def predict(self, candidate: Candidate) -> float: ...


class SatisfactionSource(Protocol):
    """Show a candidate (after morphing to it) and return calibrated EEG feedback."""
    def evaluate(self, candidate: Candidate) -> tuple[float | None, str]: ...  # (p_satisfied, quality)


class Morpher(Protocol):
    def morph_to(self, candidate: Candidate) -> None: ...


@dataclass
class Feedback:
    """What the LLM/generator sees to refine: the accepted-so-far region and the
    per-candidate EEG verdicts gathered so far (satisfied / dissatisfied)."""
    best: Candidate | None = None
    satisfied: list[Candidate] = field(default_factory=list)
    dissatisfied: list[Candidate] = field(default_factory=list)


@dataclass
class ControllerConfig:
    max_iterations: int = 12
    n_candidates: int = 8
    min_alignment: float = 0.5          # hard floor: below this can never be selected
    accept_threshold: float = 0.70      # P(satisfied) to count as an accept
    reject_threshold: float = 0.35      # P(satisfied) below this refines the generator
    consecutive_accepts: int = 2        # accepts required to stop (paper: consistency)
    w_alignment: float = 0.5            # ranking weight for pre-EEG selection
    w_preference: float = 0.5


class BrainLLMController:
    def __init__(
        self,
        generator: CandidateGenerator,
        aligner: AlignmentScorer,
        surrogate: PreferenceSurrogate,
        satisfaction: SatisfactionSource,
        morpher: Morpher,
        config: ControllerConfig | None = None,
    ):
        self.generator = generator
        self.aligner = aligner
        self.surrogate = surrogate
        self.satisfaction = satisfaction
        self.morpher = morpher
        self.cfg = config or ControllerConfig()
        self.history: list[IterationLog] = []
        self.best: Candidate | None = None
        self._best_score: float = -np.inf

    # -- ranking -----------------------------------------------------------
    def _rank(self, pool: list[Candidate]) -> list[Candidate]:
        """Alignment is a hard constraint; among aligned candidates rank by a
        weighted alignment+preference score. If none clear the floor, fall back
        to the single most-aligned so the loop still advances on-brief."""
        aligned = [c for c in pool if c.alignment >= self.cfg.min_alignment]
        pool_to_rank = aligned or [max(pool, key=lambda c: c.alignment)]

        def score(c: Candidate) -> float:
            return self.cfg.w_alignment * c.alignment + self.cfg.w_preference * c.pred_pref

        return sorted(pool_to_rank, key=score, reverse=True)

    def _combined_objective(self, c: Candidate) -> float:
        """Best-so-far objective: alignment gated, then EEG satisfaction if we have
        it, else the surrogate preference. Never lets an off-brief image be 'best'."""
        if c.alignment < self.cfg.min_alignment:
            return -1.0
        sat = c.p_satisfied if c.p_satisfied is not None else c.pred_pref
        return 0.5 * c.alignment + 0.5 * float(sat)

    def _update_best(self, c: Candidate) -> None:
        s = self._combined_objective(c)
        if s > self._best_score:
            self._best_score = s
            self.best = c

    # -- main loop ---------------------------------------------------------
    def run(self, brief: str) -> Candidate:
        feedback = Feedback()
        consecutive = 0

        for it in range(1, self.cfg.max_iterations + 1):
            pool = self.generator.generate(brief, feedback)
            for c in pool:
                c.alignment = self.aligner.score(c, brief)
                c.pred_pref = self.surrogate.predict(c)
            ranked = self._rank(pool)
            selected = ranked[0]

            # Morph the display into the selected candidate, then measure EEG on
            # the STATIC hold (satisfaction epoch), never during the transition.
            self.morpher.morph_to(selected)
            p_sat, quality = self.satisfaction.evaluate(selected)
            selected.p_satisfied = p_sat
            selected.quality = quality

            decision = self._decide(selected, feedback, consecutive)
            # best-so-far only from valid evidence (or the on-brief surrogate).
            if quality != "invalid":
                self._update_best(selected)

            self.history.append(IterationLog(
                iteration=it, selected=selected.id, alignment=round(selected.alignment, 3),
                p_satisfied=None if p_sat is None else round(p_sat, 3),
                quality=quality, decision=decision,
                best_so_far=self.best.id if self.best else "-",
            ))

            if decision == "accept":
                consecutive += 1
                if consecutive >= self.cfg.consecutive_accepts:
                    return self.best or selected
            else:
                consecutive = 0

        return self.best or selected

    def _decide(self, c: Candidate, feedback: Feedback, consecutive: int) -> str:
        if c.quality == "invalid":
            return "invalid-skip"        # missing evidence: not dissatisfaction
        p = c.p_satisfied
        if p is None:
            return "uncertain"
        if p >= self.cfg.accept_threshold:
            feedback.satisfied.append(c)
            if self._combined_objective(c) >= self._best_score:
                feedback.best = c
            return "accept"
        if p <= self.cfg.reject_threshold:
            feedback.dissatisfied.append(c)   # confident dissatisfaction -> refine
            return "refine"
        return "uncertain"                    # uncertain EEG does not refine
