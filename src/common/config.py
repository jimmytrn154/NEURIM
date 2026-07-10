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
    band_hz: tuple[float, float] = (8.0, 13.0)
    window_s: float = 2.0
    update_interval_s: float = 0.25
    baseline_duration_s: float = 30.0
    clip: tuple[float, float] = (-1.0, 1.0)


@dataclass
class EEGConfig:
    device: str = "EPOC_X"
    channels: list[str] = field(default_factory=list)
    sample_rate_hz: int = 128


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
    algorithm: str = "hill_climb"  # "hill_climb" | "es_1p1" | "gp_bo"


@dataclass
class StateMachineConfig:
    calibrate_duration_s: float = 30.0
    settle_reward_threshold: float = 0.55
    settle_motion_threshold: float = 0.1
    settle_patience_steps: int = 3
    min_steps_before_settle: int = 0
    recover_negative_streak: int = 4
    recover_widen_factor: float = 1.5
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
    openai_image_model: str = "gpt-image-2"
    openai_image_size: str = "1024x1024"
    openai_image_quality: str = "low"
    openai_image_output_format: str = "png"
    target_fps: int = 30
    frame_size: int = 512
    anchor_prompts: list[str] = field(default_factory=list)


@dataclass
class LoopConfig:
    target_latency_ms: int = 250
    optimizer_step_interval_s: float = 1.5


@dataclass
class Config:
    eeg: EEGConfig = field(default_factory=EEGConfig)
    faa: FAAConfig = field(default_factory=FAAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        def section(name: str, dc_type):
            return dc_type(**raw.get(name, {}))

        return cls(
            eeg=section("eeg", EEGConfig),
            faa=FAAConfig(**{**raw.get("faa", {}), **_tuple_fields(raw.get("faa", {}))}),
            optimizer=section("optimizer", OptimizerConfig),
            state_machine=section("state_machine", StateMachineConfig),
            generator=section("generator", GeneratorConfig),
            loop=section("loop", LoopConfig),
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
