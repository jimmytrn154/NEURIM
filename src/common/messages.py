"""Wire format for the websocket messages passed between services.

Signal -> Orchestrator -> Optimizer -> Orchestrator -> Generator -> Orchestrator.
Every message is a flat dict of JSON-safe types so any service can be swapped
for a different language later without touching the others.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

State = Literal["calibrate", "explore", "refine", "settle", "recover"]


@dataclass
class RewardMessage:
    """Signal service -> Orchestrator. One scalar reward reading."""

    r: float
    t: float = field(default_factory=time.time)
    raw_faa: float | None = None
    source: str = "eeg"  # "eeg" | "fake"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "RewardMessage":
        return RewardMessage(**json.loads(s))


@dataclass
class LatentMessage:
    """Optimizer service -> Orchestrator. Next point in the low-dim search space."""

    z: list[float]
    step_index: int
    state: State
    reward_estimate: float
    t: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "LatentMessage":
        return LatentMessage(**json.loads(s))


@dataclass
class FrameMessage:
    """Generator service -> Orchestrator. One rendered frame, ready to display.

    Extra fields (state, reward_estimate) are forwarded from the optimizer so
    display clients can update their UI from a single message without a
    separate status channel.
    """

    frame_b64: str
    z: list[float]
    step_index: int
    t: float = field(default_factory=time.time)
    format: str = "jpeg"
    state: str = "explore"
    reward_estimate: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "FrameMessage":
        return FrameMessage(**json.loads(s))


@dataclass
class ControlMessage:
    """Orchestrator -> any service. Session control (start/stop/reset/calibrate)."""

    command: str
    args: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "ControlMessage":
        return ControlMessage(**json.loads(s))
