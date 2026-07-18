"""Trial protocols (spec §7) and a deterministic driver.

Defines the phase timelines for the absolute-satisfaction and pairwise-preference
protocols as data, plus a `Presenter` interface and a pump-based driver that runs
one trial against an `Acquisition`: it emits the right markers, advances the EEG
stream through each phase, and extracts the evaluation epoch(s) *after* the static
candidate hold - never during a transition (constraints 1-4).

The driver is timebase-driven via `acq.pump_seconds(...)`, so it is fully
deterministic and hardware-free for tests and mock collection. A live driver
(real-time sleeps + a background acquisition thread) is a thin variant that swaps
`pump_seconds` for real sleeps; kept out of here to keep this testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.acquisition import markers as M
from src.acquisition.acquisition import Acquisition
from src.acquisition.ring_buffer import Epoch
from src.acquisition.quality import QualityResult


@dataclass
class PhaseSpec:
    marker: str
    screen: str            # "fixation" | "brief" | "blank" | "candidate" | "pair_a" | "pair_b"
    lo_s: float
    hi_s: float | None = None   # None -> fixed duration lo_s

    def duration(self, rng=None) -> float:
        if self.hi_s is None or rng is None:
            return self.lo_s if self.hi_s is None else 0.5 * (self.lo_s + self.hi_s)
        return float(rng.uniform(self.lo_s, self.hi_s))


@dataclass
class ProtocolConfig:
    fixation_lo_s: float = 1.0
    fixation_hi_s: float = 1.5
    brief_s: float = 3.0
    blank_s: float = 0.5
    candidate_s: float = 3.0
    pair_s: float = 2.5
    eval_start_s: float = 0.5   # epoch start after CANDIDATE/PAIR onset
    eval_end_s: float = 2.5     # epoch end after onset


def satisfaction_phases(cfg: ProtocolConfig) -> list[PhaseSpec]:
    return [
        PhaseSpec(M.FIXATION_ONSET, "fixation", cfg.fixation_lo_s, cfg.fixation_hi_s),
        PhaseSpec(M.BRIEF_ONSET, "brief", cfg.brief_s),
        PhaseSpec(M.FIXATION_ONSET, "blank", cfg.blank_s),
        PhaseSpec(M.CANDIDATE_ONSET, "candidate", cfg.candidate_s),
    ]


def pairwise_phases(cfg: ProtocolConfig) -> list[PhaseSpec]:
    return [
        PhaseSpec(M.BRIEF_ONSET, "brief", cfg.brief_s),
        PhaseSpec(M.FIXATION_ONSET, "fixation", cfg.fixation_lo_s, cfg.fixation_hi_s),
        PhaseSpec(M.PAIR_A_ONSET, "pair_a", cfg.pair_s),
        PhaseSpec(M.FIXATION_ONSET, "blank", cfg.blank_s),
        PhaseSpec(M.PAIR_B_ONSET, "pair_b", cfg.pair_s),
        PhaseSpec(M.FIXATION_ONSET, "blank", cfg.blank_s),
    ]


class Presenter(Protocol):
    def show(self, screen: str, payload: dict | None = None) -> None: ...


class HeadlessPresenter:
    """No-op presenter that records what it was asked to show (for tests/mock)."""

    def __init__(self) -> None:
        self.shown: list[tuple[str, dict | None]] = []

    def show(self, screen: str, payload: dict | None = None) -> None:
        self.shown.append((screen, payload))


@dataclass
class TrialResult:
    epochs: dict[str, Epoch]
    quality: dict[str, QualityResult]
    onsets: dict[str, float]
    markers: list = field(default_factory=list)


def _run_phases(
    acq: Acquisition,
    presenter: Presenter,
    phases: list[PhaseSpec],
    cfg: ProtocolConfig,
    onset_screens: dict[str, str],
    satisfaction_by_screen: dict[str, float] | None,
    rng,
) -> TrialResult:
    onsets: dict[str, float] = {}
    backend = getattr(acq, "backend", None)
    for ph in phases:
        # Inject the trial's satisfaction signal only during evaluation screens.
        if satisfaction_by_screen is not None and hasattr(backend, "set_satisfaction"):
            backend.set_satisfaction(satisfaction_by_screen.get(ph.screen, 0.0))
        presenter.show(ph.screen)
        m = acq.markers.emit(ph.marker, timestamp=acq.now())
        if ph.screen in onset_screens:
            onsets[onset_screens[ph.screen]] = m.timestamp
        acq.pump_seconds(ph.duration(rng))

    epochs: dict[str, Epoch] = {}
    quality: dict[str, QualityResult] = {}
    for name, onset in onsets.items():
        ep = acq.extract_epoch(onset, cfg.eval_start_s, cfg.eval_end_s)
        epochs[name] = ep
        quality[name] = acq.evaluate(ep)
    return TrialResult(epochs=epochs, quality=quality, onsets=onsets, markers=acq.markers.markers())


def run_satisfaction_trial(
    acq: Acquisition,
    presenter: Presenter,
    cfg: ProtocolConfig,
    true_satisfaction: float = 0.0,
    rng=None,
) -> TrialResult:
    """One absolute-satisfaction trial. `true_satisfaction` in [-1,1] drives the
    injected mock EEG; ignored on real hardware."""
    return _run_phases(
        acq, presenter, satisfaction_phases(cfg), cfg,
        onset_screens={"candidate": "candidate"},
        satisfaction_by_screen={"candidate": true_satisfaction},
        rng=rng,
    )


def run_pairwise_trial(
    acq: Acquisition,
    presenter: Presenter,
    cfg: ProtocolConfig,
    sat_a: float = 0.0,
    sat_b: float = 0.0,
    rng=None,
) -> TrialResult:
    """One pairwise trial: two evaluation epochs, A and B."""
    return _run_phases(
        acq, presenter, pairwise_phases(cfg), cfg,
        onset_screens={"pair_a": "A", "pair_b": "B"},
        satisfaction_by_screen={"pair_a": sat_a, "pair_b": sat_b},
        rng=rng,
    )
