import numpy as np

from src.common.config import Config, OptimizerConfig, StateMachineConfig
from src.optimizer.observation import Observation
from src.optimizer.service import OptimizerService
from src.signal_service.presentation import PresentationSchedule, ScoringGate


def test_schedule_phases():
    s = PresentationSchedule(transition_s=1.5, stabilize_s=2.0, score_s=1.5)
    assert s.phase(0.0) == "transition"
    assert s.phase(1.0) == "transition"
    assert s.phase(2.0) == "stabilize"
    assert s.phase(3.5) == "score"
    assert s.phase(4.9) == "score"
    assert s.phase(5.0) == "done"
    assert s.total_s == 5.0
    assert s.is_scoring(4.0) and not s.is_scoring(2.0)


def test_morph_alpha_completes_by_transition_then_holds():
    s = PresentationSchedule(transition_s=1.5, stabilize_s=2.0, score_s=1.5)
    assert s.morph_alpha(0.0) == 0.0
    assert s.morph_alpha(0.75) == 0.5
    assert s.morph_alpha(1.5) == 1.0
    assert s.morph_alpha(4.0) == 1.0  # holds at the target during stabilize/score


def test_validate_flags_short_stabilize():
    s = PresentationSchedule(transition_s=1.0, stabilize_s=1.0, score_s=1.0)
    warnings = s.validate(faa_window_s=2.0)  # stabilize < window -> contaminated
    assert any("stabilize_s" in w for w in warnings)
    assert PresentationSchedule(stabilize_s=2.0).validate(faa_window_s=2.0) == []


def test_scoring_gate_only_accumulates_scoring_interval():
    s = PresentationSchedule(transition_s=1.0, stabilize_s=2.0, score_s=1.0)  # score in [3, 4)
    gate = ScoringGate(s, clip=(-1, 1))
    # Feed readings across the whole presentation; only the scoring-interval
    # values (0.8) should shape the emitted observation, not the transition (-0.9).
    obs = None
    for elapsed, r in [(0.5, -0.9), (1.5, -0.9), (2.5, -0.9), (3.2, 0.8), (3.6, 0.8), (4.0, 0.8)]:
        out = gate.feed(elapsed, r)
        if out is not None:
            obs = out
    assert obs is not None
    assert obs.reward_mean > 0.5  # transition's -0.9 readings were excluded


def test_scoring_gate_emits_once_then_none():
    s = PresentationSchedule(transition_s=0.5, stabilize_s=1.0, score_s=0.5)  # total 2.0
    gate = ScoringGate(s)
    assert gate.feed(1.6, 0.5) is None       # in scoring interval, accumulating
    first = gate.feed(2.0, 0.5)              # window closes -> emit
    assert first is not None
    assert gate.feed(2.5, 0.5) is None       # already done, no double-emit


def test_observe_observation_takes_one_step():
    cfg = Config(optimizer=OptimizerConfig(search_dims=4, algorithm="hill_climb"),
                 state_machine=StateMachineConfig())
    svc = OptimizerService(cfg)
    svc.notify_calibrated()
    before = svc.state_machine.step_index
    result = svc.observe_observation(Observation(0.6, 0.01, 4.0, 0.0, 1.0))
    assert result is not None
    assert svc.state_machine.step_index == before + 1
    assert len(result.z) == 4
