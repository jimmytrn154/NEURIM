"""Loads config/config.yaml into plain dataclasses.

Secrets (EMOTIV_CLIENT_ID/SECRET) are read from the environment, never from the
YAML file, so the config can be committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass
class FAAConfig:
    channel_left: str = "F3"
    channel_right: str = "F4"
    channel_pairs: list[list[str]] = field(
        default_factory=lambda: [["F7", "F8"], ["AF3", "AF4"], ["F3", "F4"], ["FC5", "FC6"]]
    )
    pair_weights: dict[str, float] = field(
        default_factory=lambda: {
            "F3/F4": 1.0,
            "F7/F8": 0.75,
            "AF3/AF4": 0.5,
            "FC5/FC6": 0.5,
        }
    )
    band_hz: tuple[float, float] = (8.0, 13.0)
    window_s: float = 3.0
    update_interval_s: float = 0.25
    baseline_duration_s: float = 30.0
    clip: tuple[float, float] = (-1.0, 1.0)


@dataclass
class EEGConfig:
    device: str = "EPOC_X"
    channels: list[str] = field(default_factory=list)
    sample_rate_hz: int = 128


@dataclass
class PreprocessingConfig:
    # Streaming signal conditioning (src/signal_service/preprocessing.py).
    # enabled=False keeps the legacy raw-EEG path; enabled=True inserts
    # bandpass + mains notch + common-average reference before band power.
    enabled: bool = True
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    notch_hz: float = 60.0        # mains frequency; set 50.0 outside NA
    notch_q: float = 30.0
    filter_order: int = 4
    common_average: bool = True
    eog_channels: list[str] = field(default_factory=lambda: ["AF3", "AF4", "F7", "F8"])
    blink_threshold_mad: float = 5.0
    emg_threshold_mad: float = 6.0


@dataclass
class OptimizerConfig:
    # 8 dims (the favorable end of the spec's 8-16 range) converges
    # noticeably more reliably than 12+ within a demo-length step budget -
    # see tests/test_optimizer.py and the tuning notes in README.md.
    search_dims: int = 8
    step_size_explore: float = 0.5
    step_size_refine_min: float = 0.03
    step_momentum: float = 0.6
    noise_threshold: float = 0.05
    reward_window_steps: int = 4
    bounds: float = 1.0
    algorithm: str = "hill_climb"  # "hill_climb" | "es_1p1" | "gp_bo" | "latent_turbo"


@dataclass
class StateMachineConfig:
    calibrate_duration_s: float = 30.0
    settle_reward_threshold: float = 0.30        # realistic plateau level for a clipped FAA z-score
    settle_reward_std_threshold: float = 0.15    # plateau tightness: recent reward std must be below this
    settle_motion_threshold: float = 0.1
    settle_patience_steps: int = 3
    min_steps_before_settle: int = 0
    recover_negative_streak: int = 4
    recover_reward_margin: float = -0.25         # RECOVER only counts rewards clearly below 0, not any dip
    recover_widen_factor: float = 1.5
    # Escape a low-variance plateau that is too low to SETTLE: after this many
    # stagnant steps, kick like RECOVER (revert to best + widen). Decouples
    # convergence from absolute-reward RECOVER; never fires on a high-variance
    # signal (e.g. FAA baseline noise) because that is not a plateau.
    stagnation_patience_steps: int = 6
    max_steps: int = 100
    # EXPLORE -> REFINE once the recent average reward climbs above this
    # level (not a slope check - near convergence the trend flattens out
    # from noise, so gating on slope alone can strand the search in EXPLORE
    # forever). REFINE falls back to EXPLORE if reward drops below
    # refine_entry_reward - refine_exit_margin (hysteresis, avoids flapping).
    refine_entry_reward: float = 0.05
    refine_exit_margin: float = 0.1


@dataclass
class GeneratorConfig:
    backend: str = "procedural"  # "procedural" | "diffusion" | "remote_diffusion" | "openai"
    diffusion_model_id: str = "stabilityai/sdxl-turbo"
    diffusion_steps: int = 2
    remote_diffusion_url: str = "http://localhost:8766"
    remote_diffusion_timeout_s: float = 30.0
    # Pinned diffusion seed: keeps consecutive morph keyframes visually close so
    # the client crossfade reads as a smooth morph, not a dissolve.
    remote_diffusion_seed: int = 0
    # Min seconds between server keyframe renders (also the crossfade window).
    # ~0.2s ≈ 5 keyframes/s, about what remote SDXL-Turbo sustains over a tunnel.
    remote_diffusion_keyframe_interval_s: float = 0.2
    openai_image_model: str = "gpt-image-2"
    openai_text_model: str = "gpt-5-mini"
    openai_image_size: str = "1024x1024"
    openai_image_quality: str = "low"
    openai_image_output_format: str = "png"
    # After the session's last morphed frame is rendered, run it back through
    # OpenAI's image-edit API to resolve morph artifacts into a clean image
    # before handing it to the frontend (src/generator/image_finalize.py).
    finalize_enabled: bool = True
    target_fps: int = 30
    frame_size: int = 512
    anchor_prompts: list[str] = field(default_factory=list)


@dataclass
class LoopConfig:
    target_latency_ms: int = 250
    optimizer_step_interval_s: float = 1.5


@dataclass
class PresentationConfig:
    # Per-candidate presentation schedule (see src/signal_service/presentation.py).
    # enabled=False keeps the simple per-window path (average the last N FAA
    # readings); enabled=True scores only the scoring interval, fixing the
    # reward-delay/credit-assignment problem.
    enabled: bool = False
    transition_s: float = 1.5   # morph A -> B; not scored
    stabilize_s: float = 2.0    # hold B; FAA window fills with stable EEG; MUST be >= faa.window_s
    score_s: float = 1.5        # scoring window; the FAA here is the reward


@dataclass
class Config:
    eeg: EEGConfig = field(default_factory=EEGConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    faa: FAAConfig = field(default_factory=FAAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    presentation: PresentationConfig = field(default_factory=PresentationConfig)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        def section(name: str, dc_type):
            return dc_type(**raw.get(name, {}))

        return cls(
            eeg=section("eeg", EEGConfig),
            preprocessing=section("preprocessing", PreprocessingConfig),
            faa=FAAConfig(**{**raw.get("faa", {}), **_tuple_fields(raw.get("faa", {}))}),
            optimizer=section("optimizer", OptimizerConfig),
            state_machine=section("state_machine", StateMachineConfig),
            generator=section("generator", GeneratorConfig),
            loop=section("loop", LoopConfig),
            presentation=section("presentation", PresentationConfig),
        )


def _tuple_fields(raw: dict) -> dict:
    """YAML lists need to become tuples for band_hz/clip fields."""
    out = {}
    if "band_hz" in raw:
        out["band_hz"] = tuple(raw["band_hz"])
    if "clip" in raw:
        out["clip"] = tuple(raw["clip"])
    return out


def emotiv_credentials() -> tuple[str | None, str | None]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("EMOTIV_CLIENT_ID"), os.environ.get("EMOTIV_CLIENT_SECRET")
