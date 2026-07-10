from src.common.config import OptimizerConfig, StateMachineConfig
from src.optimizer.state_machine import StateMachine


def _sm(**overrides):
    defaults = dict(
        settle_reward_threshold=0.6,
        settle_motion_threshold=0.05,
        settle_patience_steps=3,
        recover_negative_streak=3,
        max_steps=100,
    )
    defaults.update(overrides)
    return StateMachine(StateMachineConfig(**defaults), OptimizerConfig())


def test_starts_in_calibrate_and_needs_explicit_transition():
    sm = _sm()
    assert sm.state == "calibrate"
    sm.observe(reward=0.9, step_norm=0.01)
    assert sm.state == "calibrate"  # ignores everything until calibrated
    sm.mark_calibrated()
    assert sm.state == "explore"


def test_explore_moves_to_refine_on_climbing_trend():
    sm = _sm()
    sm.mark_calibrated()
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        sm.observe(reward=r, step_norm=0.2)
    assert sm.state == "refine"


def test_settle_after_sustained_high_reward_low_motion():
    sm = _sm()
    sm.mark_calibrated()
    for _ in range(5):
        sm.observe(reward=0.8, step_norm=0.01)
    assert sm.is_locked()
    assert sm.state == "settle"


def test_min_steps_before_settle_prevents_immediate_lock():
    sm = _sm(min_steps_before_settle=5)
    sm.mark_calibrated()
    for _ in range(4):
        sm.observe(reward=0.8, step_norm=0.01)
    assert not sm.is_locked()
    sm.observe(reward=0.8, step_norm=0.01)
    assert sm.is_locked()


def test_recover_after_negative_streak():
    sm = _sm(recover_negative_streak=3)
    sm.mark_calibrated()
    sm.observe(reward=-0.2, step_norm=0.2)
    sm.observe(reward=-0.3, step_norm=0.2)
    state = sm.observe(reward=-0.4, step_norm=0.2)
    assert state == "recover"


def test_recover_returns_to_explore_next_step():
    sm = _sm(recover_negative_streak=2)
    sm.mark_calibrated()
    sm.observe(reward=-0.5, step_norm=0.2)
    assert sm.observe(reward=-0.5, step_norm=0.2) == "recover"
    assert sm.observe(reward=0.1, step_norm=0.2) == "explore"


def test_step_size_shrinks_from_explore_to_refine():
    sm = _sm()
    sm.mark_calibrated()
    explore_step = sm.step_size()
    for r in [0.1, 0.2, 0.3, 0.4, 0.55]:
        sm.observe(reward=r, step_norm=0.2)
    assert sm.state == "refine"
    refine_step = sm.step_size()
    assert refine_step < explore_step
