"""Atomic, resumable per-trial storage: raw EEG arrays + linked metadata.

The existing recorder (`scripts/record_reward_trials.py`) saves *feature* CSV rows
- it throws the raw signal away, so preprocessing can never be changed later.
Spec §11 / constraint 14 require the raw `[channels, samples]` array plus complete
trial metadata, saved atomically and linked by stable IDs.

Layout (pseudonymous subject id):

    <root>/<subject_id>/<session_id>/
        session.json          # session-level manifest (subject, config, git, channels)
        manifest.jsonl        # one JSON line per saved trial (metadata + npz path)
        trials/trial_0001.npz # raw arrays: eeg [channels, samples], times, per-epoch extras

Writes are atomic (temp file + os.replace) so a crash mid-write never corrupts an
existing trial, and `next_trial_id()` reads the manifest so an interrupted session
resumes without overwriting.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.acquisition.ring_buffer import Epoch


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=2)
        return out.stdout.strip() or None
    except Exception:
        return None


def _atomic_write_bytes(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class TrialStore:
    def __init__(self, root: str | Path, subject_id: str, session_id: str):
        self.subject_id = subject_id
        self.session_id = session_id
        self.dir = Path(root) / subject_id / session_id
        self.trials_dir = self.dir / "trials"
        self.manifest_path = self.dir / "manifest.jsonl"
        self.session_path = self.dir / "session.json"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

    # -- session ------------------------------------------------------------
    def write_session(self, meta: dict[str, Any]) -> None:
        payload = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "git_commit": _git_commit(),
            **meta,
        }
        _atomic_write_bytes(
            self.session_path,
            lambda f: f.write(json.dumps(payload, indent=2, default=str).encode()),
        )

    # -- trials -------------------------------------------------------------
    def load_manifest(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        rows = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def next_trial_id(self) -> int:
        rows = self.load_manifest()
        return (max((r["trial_id"] for r in rows), default=0)) + 1

    def save_trial(
        self,
        trial_id: int,
        meta: dict[str, Any],
        epochs: dict[str, Epoch],
        extra_arrays: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """Atomically save one trial's raw epoch arrays + append its manifest row.

        `epochs` maps a name (e.g. "candidate", or "A"/"B" for pairwise) to an
        Epoch; each is stored as `<name>_eeg` [channels, samples] + `<name>_times`.
        """
        npz_path = self.trials_dir / f"trial_{trial_id:04d}.npz"
        arrays: dict[str, np.ndarray] = {}
        epoch_index: dict[str, dict] = {}
        channels: list[str] | None = None
        fs: float | None = None
        for name, ep in epochs.items():
            arrays[f"{name}_eeg"] = ep.data
            arrays[f"{name}_times"] = ep.times
            channels = ep.channels
            fs = ep.fs
            epoch_index[name] = {
                "shape": list(ep.data.shape),
                "t0": ep.t0, "t1": ep.t1,
                "coverage": ep.coverage, "max_gap_s": ep.max_gap_s,
            }
        if extra_arrays:
            arrays.update(extra_arrays)
        arrays["channels"] = np.asarray(channels or [], dtype=object)

        _atomic_write_bytes(npz_path, lambda f: np.savez_compressed(f, **arrays))

        row = {
            "trial_id": trial_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "npz": str(npz_path.relative_to(self.dir)),
            "channels": channels,
            "sample_rate_hz": fs,
            "epochs": epoch_index,
            **meta,
        }
        # Append the manifest line (single write; append is atomic enough on local fs).
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return npz_path

    def load_trial_arrays(self, trial_id: int) -> dict[str, np.ndarray]:
        npz_path = self.trials_dir / f"trial_{trial_id:04d}.npz"
        with np.load(npz_path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
