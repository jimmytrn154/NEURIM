#!/usr/bin/env python3
"""Validate a raw NEURIM EEG recording session.

Checks the resumable TrialStore layout produced by scripts/collect_eeg_data.py:
manifest consistency, NPZ existence, epoch array shapes, channel order, usable
label counts, and quality-status distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_dir", type=Path, help="data/eeg/<subject>/<session>")
    p.add_argument("--min-usable", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = args.session_dir
    session_path = session_dir / "session.json"
    manifest_path = session_dir / "manifest.jsonl"
    if not session_path.exists():
        sys.exit(f"missing session.json: {session_path}")
    if not manifest_path.exists():
        sys.exit(f"missing manifest.jsonl: {manifest_path}")

    session = json.loads(session_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        sys.exit("manifest has no trials")

    status = Counter()
    usable = 0
    shapes: Counter[str] = Counter()
    errors: list[str] = []
    expected_channels = session.get("channels")

    for row in rows:
        npz_path = session_dir / row["npz"]
        if not npz_path.exists():
            errors.append(f"trial {row.get('trial_id')}: missing {npz_path}")
            continue
        if row.get("usable_label"):
            usable += 1
        q = row.get("quality_status", "unknown")
        if isinstance(q, dict):
            for value in q.values():
                status[str(value)] += 1
        else:
            status[str(q)] += 1
        with np.load(npz_path, allow_pickle=True) as payload:
            channels = list(payload["channels"]) if "channels" in payload else []
            if expected_channels and channels != expected_channels:
                errors.append(f"trial {row.get('trial_id')}: channel order mismatch")
            for name in payload.files:
                if name.endswith("_eeg"):
                    arr = payload[name]
                    shapes[f"{name}:{tuple(arr.shape)}"] += 1
                    if arr.ndim != 2:
                        errors.append(f"trial {row.get('trial_id')}: {name} not 2D")
                    if expected_channels and arr.shape[0] != len(expected_channels):
                        errors.append(f"trial {row.get('trial_id')}: {name} channel count mismatch")

    print(f"[validate] session={session.get('subject_id')}/{session.get('session_id')}")
    print(f"[validate] trials={len(rows)} usable={usable} quality={dict(status)}")
    print(f"[validate] shapes={dict(shapes)}")
    if usable < args.min_usable:
        errors.append(f"usable trials {usable} < --min-usable {args.min_usable}")
    if errors:
        for err in errors:
            print(f"[validate] ERROR {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
