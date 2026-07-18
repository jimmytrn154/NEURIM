import numpy as np

from src.closed_loop.policy import (
    BestSoFar,
    CandidateScore,
    DecisionThresholds,
    classify_satisfaction,
    decide_feedback,
    rank_candidates,
)


def _candidate(cid, alignment, pref, violation=False):
    return CandidateScore(
        candidate_id=cid,
        latent=np.zeros(3),
        prompt=cid,
        alignment=alignment,
        predicted_preference=pref,
        visual_quality=0.5,
        diversity=0.0,
        critical_violation=violation,
    )


def test_invalid_or_uncertain_eeg_never_becomes_negative_label():
    assert classify_satisfaction(0.02, "invalid") == "invalid"
    assert classify_satisfaction(None, "valid") == "uncertain"
    invalid = decide_feedback(mode="eeg_autonomous", p_satisfied=0.02, quality="invalid")
    uncertain = decide_feedback(mode="eeg_autonomous", p_satisfied=0.5, quality="valid")
    assert invalid.label is None and not invalid.update_model
    assert uncertain.label is None and not uncertain.update_model


def test_threshold_policy_is_configurable():
    th = DecisionThresholds(satisfied=0.8, dissatisfied=0.2)
    assert classify_satisfaction(0.75, "valid", th) == "uncertain"
    assert classify_satisfaction(0.81, "valid", th) == "satisfied"
    assert classify_satisfaction(0.19, "valid", th) == "dissatisfied"


def test_shadow_mode_logs_eeg_but_manual_controls_refinement():
    decision = decide_feedback(
        mode="shadow",
        p_satisfied=0.9,
        quality="valid",
        manual_label="dissatisfied",
    )
    assert decision.state == "dissatisfied"
    assert decision.controls_refinement
    assert decision.label == 0
    assert "shadow_eeg=satisfied" in decision.reason


def test_eeg_assisted_uses_manual_fallback_for_uncertain_eeg():
    decision = decide_feedback(
        mode="eeg_assisted",
        p_satisfied=0.55,
        quality="valid",
        manual_label="satisfied",
    )
    assert decision.state == "satisfied"
    assert decision.reason.startswith("manual_fallback")


def test_candidate_ranking_rejects_critical_prompt_violations():
    ranked = rank_candidates([
        _candidate("off_prompt_but_liked", 0.99, 1.0, violation=True),
        _candidate("aligned", 0.7, 0.6),
        _candidate("weak", 0.4, 1.0),
    ])
    assert ranked[0][0].candidate_id == "aligned"
    assert all(not c.critical_violation for c, _ in ranked)


def test_ranking_falls_back_to_most_aligned_when_floor_not_met():
    ranked = rank_candidates([
        _candidate("low", 0.2, 1.0),
        _candidate("less_low", 0.4, 0.1),
    ])
    assert [c.candidate_id for c, _ in ranked] == ["less_low"]


def test_best_so_far_never_regresses_on_invalid_or_lower_score():
    best = BestSoFar()
    a = _candidate("a", 0.8, 0.8)
    b = _candidate("b", 0.9, 0.9)
    assert best.update(a, 0.7)
    assert not best.update(b, 0.95, valid_evidence=False)
    assert best.candidate == a
    assert not best.update(_candidate("c", 0.8, 0.1), 0.3)
    assert best.candidate == a
