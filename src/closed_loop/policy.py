"""Closed-loop decision policy and candidate ranking for NEURIM.

This module is intentionally model-agnostic. The existing diffusion/latent
pipeline owns image generation and morphing; this layer only decides what to
present next and how to interpret coarse EEG/manual feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


FeedbackState = Literal["satisfied", "dissatisfied", "uncertain", "invalid"]
ControlMode = Literal["manual", "shadow", "eeg_assisted", "eeg_autonomous"]


@dataclass(frozen=True)
class DecisionThresholds:
    satisfied: float = 0.70
    dissatisfied: float = 0.35
    pseudo_positive: float = 0.75
    pseudo_negative: float = 0.25


@dataclass(frozen=True)
class FeedbackDecision:
    state: FeedbackState
    controls_refinement: bool
    update_model: bool
    label: int | None
    reason: str


def classify_satisfaction(
    p_satisfied: float | None,
    quality: str,
    thresholds: DecisionThresholds | None = None,
) -> FeedbackState:
    """Map EEG probability + quality gate to the coarse product states.

    Invalid signal is missing evidence, never dissatisfaction.
    """
    th = thresholds or DecisionThresholds()
    if quality == "invalid":
        return "invalid"
    if p_satisfied is None or quality == "retry":
        return "uncertain"
    if p_satisfied >= th.satisfied:
        return "satisfied"
    if p_satisfied <= th.dissatisfied:
        return "dissatisfied"
    return "uncertain"


def decide_feedback(
    *,
    mode: ControlMode,
    p_satisfied: float | None,
    quality: str,
    manual_label: FeedbackState | None = None,
    thresholds: DecisionThresholds | None = None,
) -> FeedbackDecision:
    """Resolve manual/EEG feedback for the four closed-loop modes.

    - manual: manual label controls refinement and model updates.
    - shadow: EEG is logged, manual label controls refinement.
    - eeg_assisted: confident valid EEG controls; uncertain falls back to manual.
    - eeg_autonomous: confident valid EEG controls; uncertain/invalid does nothing.
    """
    th = thresholds or DecisionThresholds()
    eeg_state = classify_satisfaction(p_satisfied, quality, th)

    def from_state(state: FeedbackState, controls: bool, source: str) -> FeedbackDecision:
        if state == "satisfied":
            label = 1
            update = True
        elif state == "dissatisfied":
            label = 0
            update = True
        else:
            label = None
            update = False
        return FeedbackDecision(
            state=state,
            controls_refinement=controls and state in ("satisfied", "dissatisfied"),
            update_model=update and state in ("satisfied", "dissatisfied"),
            label=label,
            reason=source,
        )

    if mode == "manual":
        return from_state(manual_label or "uncertain", True, "manual")
    if mode == "shadow":
        return from_state(manual_label or "uncertain", True, f"shadow_eeg={eeg_state}")
    if mode == "eeg_assisted":
        if eeg_state in ("satisfied", "dissatisfied"):
            return from_state(eeg_state, True, "eeg_confident")
        return from_state(manual_label or "uncertain", True, f"manual_fallback_eeg={eeg_state}")
    if mode == "eeg_autonomous":
        return from_state(eeg_state, True, "eeg_autonomous")
    raise ValueError(f"unknown control mode: {mode!r}")


@dataclass
class CandidateScore:
    candidate_id: str
    latent: np.ndarray
    prompt: str
    alignment: float
    predicted_preference: float
    visual_quality: float = 0.5
    diversity: float = 0.0
    critical_violation: bool = False
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RankingWeights:
    alignment: float = 0.45
    preference: float = 0.35
    quality: float = 0.15
    diversity: float = 0.05
    min_alignment: float = 0.55


def rank_candidates(
    candidates: list[CandidateScore],
    weights: RankingWeights | None = None,
) -> list[tuple[CandidateScore, float]]:
    """Rank candidates while preserving objective prompt requirements.

    Critical prompt violations are rejected. If every candidate misses the
    alignment floor, return the best aligned non-critical candidate so the loop
    can still make progress instead of stalling.
    """
    if not candidates:
        raise ValueError("rank_candidates needs at least one candidate")
    w = weights or RankingWeights()
    eligible = [c for c in candidates if not c.critical_violation and c.alignment >= w.min_alignment]
    if not eligible:
        eligible = sorted(
            [c for c in candidates if not c.critical_violation],
            key=lambda c: c.alignment,
            reverse=True,
        )[:1]
    if not eligible:
        return []

    def score(c: CandidateScore) -> float:
        return float(
            w.alignment * c.alignment
            + w.preference * c.predicted_preference
            + w.quality * c.visual_quality
            + w.diversity * c.diversity
        )

    return sorted(((c, score(c)) for c in eligible), key=lambda item: item[1], reverse=True)


@dataclass
class BestSoFar:
    candidate: CandidateScore | None = None
    score: float = -np.inf

    def update(self, candidate: CandidateScore, score: float, valid_evidence: bool = True) -> bool:
        """Keep the best candidate without regressing on invalid EEG evidence."""
        if not valid_evidence:
            return False
        if candidate.critical_violation:
            return False
        if score > self.score:
            self.candidate = candidate
            self.score = float(score)
            return True
        return False
