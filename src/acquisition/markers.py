"""Structured experimental event markers with monotonic sequencing.

Every epoch the decoder ever trains or infers on is defined by a pair of markers
(e.g. CANDIDATE_ONSET .. CANDIDATE_OFFSET). Markers carry the full trial identity
so raw EEG and metadata can be relinked later by stable IDs (spec §10, §11), and
a per-emitter sequence number so dropped/reordered events are detectable.

This is deliberately transport-agnostic: `MarkerLog` records markers with the
wall/stream clock the presentation loop uses, and can serialize to JSONL. Cortex
`injectMarker` (hardware marker) can be wired on top later; the local timestamp
is always recorded so analysis never depends on the headset clock.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Canonical marker vocabulary (spec §10). Kept as plain strings so they serialize
# cleanly and match across processes without importing an enum everywhere.
SESSION_START = "SESSION_START"
SESSION_END = "SESSION_END"
BLOCK_START = "BLOCK_START"
BLOCK_END = "BLOCK_END"
TRIAL_START = "TRIAL_START"
BRIEF_ONSET = "BRIEF_ONSET"
FIXATION_ONSET = "FIXATION_ONSET"
MORPH_START = "MORPH_START"
MORPH_END = "MORPH_END"
CANDIDATE_ONSET = "CANDIDATE_ONSET"
CANDIDATE_OFFSET = "CANDIDATE_OFFSET"
RESPONSE_SCREEN_ONSET = "RESPONSE_SCREEN_ONSET"
RESPONSE_KEEP = "RESPONSE_KEEP"
RESPONSE_REFINE = "RESPONSE_REFINE"
RESPONSE_UNSURE = "RESPONSE_UNSURE"
PAIR_A_ONSET = "PAIR_A_ONSET"
PAIR_A_OFFSET = "PAIR_A_OFFSET"
PAIR_B_ONSET = "PAIR_B_ONSET"
PAIR_B_OFFSET = "PAIR_B_OFFSET"
PAIR_RESPONSE_A = "PAIR_RESPONSE_A"
PAIR_RESPONSE_B = "PAIR_RESPONSE_B"
PAIR_RESPONSE_EQUAL = "PAIR_RESPONSE_EQUAL"
TRIAL_END = "TRIAL_END"

MARKER_NAMES = frozenset({
    SESSION_START, SESSION_END, BLOCK_START, BLOCK_END, TRIAL_START, BRIEF_ONSET,
    FIXATION_ONSET, MORPH_START, MORPH_END, CANDIDATE_ONSET, CANDIDATE_OFFSET,
    RESPONSE_SCREEN_ONSET, RESPONSE_KEEP, RESPONSE_REFINE, RESPONSE_UNSURE,
    PAIR_A_ONSET, PAIR_A_OFFSET, PAIR_B_ONSET, PAIR_B_OFFSET,
    PAIR_RESPONSE_A, PAIR_RESPONSE_B, PAIR_RESPONSE_EQUAL, TRIAL_END,
})


@dataclass
class Marker:
    name: str
    timestamp: float
    sequence: int
    subject_id: str | None = None
    session_id: str | None = None
    block_id: int | None = None
    trial_id: int | None = None
    candidate_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarkerLog:
    """Append-only marker log with an auto-incrementing sequence number.

    Bind the current trial context with `context(...)` so every emitted marker
    inherits subject/session/block/trial/candidate without repeating them.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._seq = 0
        self._markers: list[Marker] = []
        self._ctx: dict[str, Any] = {}

    def context(self, **fields: Any) -> None:
        """Set/update the trial context inherited by subsequent markers.

        Pass a field explicitly as None to clear it (e.g. candidate_id=None at
        trial end). Unmentioned fields are left unchanged.
        """
        self._ctx.update(fields)

    def emit(self, name: str, timestamp: float | None = None, **extra: Any) -> Marker:
        if name not in MARKER_NAMES:
            raise ValueError(f"unknown marker name: {name!r}")
        self._seq += 1
        m = Marker(
            name=name,
            timestamp=self._clock() if timestamp is None else float(timestamp),
            sequence=self._seq,
            subject_id=self._ctx.get("subject_id"),
            session_id=self._ctx.get("session_id"),
            block_id=self._ctx.get("block_id"),
            trial_id=self._ctx.get("trial_id"),
            candidate_id=self._ctx.get("candidate_id"),
            extra=dict(extra),
        )
        self._markers.append(m)
        return m

    def markers(self) -> list[Marker]:
        return list(self._markers)

    def last(self, name: str) -> Marker | None:
        for m in reversed(self._markers):
            if m.name == name:
                return m
        return None

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for m in self._markers:
                f.write(json.dumps(m.to_dict()) + "\n")
