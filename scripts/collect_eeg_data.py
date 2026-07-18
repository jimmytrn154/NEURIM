#!/usr/bin/env python3
"""Collect satisfaction / pairwise EEG trials, saving raw arrays + metadata.

Runs the §7 protocols against an `Acquisition` (mock by default, EPOC X with
--no-mock) and writes each trial atomically via `TrialStore`: the raw
`[channels, samples]` epoch plus complete metadata, so preprocessing can be
changed later (constraint 14). Sessions resume without overwriting.

Invalid-quality trials are still saved (raw + quality flags) but marked so
training never treats them as labels (constraints 5-6).

Examples:
    python scripts/collect_eeg_data.py --subject sub-001 --protocol satisfaction --trials 60
    python scripts/collect_eeg_data.py --subject sub-001 --protocol pairwise --trials 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.acquisition.acquisition import Acquisition, MockBackend
from src.common.config import Config
from src.experiment.protocol import (
    HeadlessPresenter,
    ProtocolConfig,
    run_pairwise_trial,
    run_satisfaction_trial,
)
from src.experiment.trial_store import TrialStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, help="pseudonymous id, e.g. sub-001")
    p.add_argument("--session", default=None, help="session id (default: ses-NNN auto)")
    p.add_argument("--protocol", choices=["satisfaction", "pairwise"], default="satisfaction")
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--brief", default="a cozy watercolor of a red bicycle by a canal")
    p.add_argument("--root", type=Path, default=Path("data/eeg"))
    p.add_argument("--no-mock", action="store_true", help="use real EPOC X via Cortex")
    p.add_argument("--mock-signal-gain", type=float, default=0.6)
    p.add_argument("--bad-contact-every", type=int, default=13,
                   help="mock: force a poor-contact (retry/invalid) trial every N")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_acquisition(args, config) -> Acquisition:
    channels = list(config.eeg.channels)
    fs = config.eeg.sample_rate_hz
    if args.no_mock:
        from src.acquisition.acquisition import CortexBackend
        from src.common.config import emotiv_credentials

        cid, secret = emotiv_credentials()
        backend = CortexBackend(cid, secret, channels)
    else:
        backend = MockBackend(channels, fs, signal_gain=args.mock_signal_gain, seed=args.seed)
    return Acquisition(backend, channels, fs)


def _auto_session(store_root: Path, subject: str) -> str:
    base = store_root / subject
    existing = [d.name for d in base.glob("ses-*")] if base.exists() else []
    return f"ses-{len(existing) + 1:03d}"


def main() -> int:
    args = parse_args()
    config = Config.load()
    session = args.session or _auto_session(args.root, args.subject)
    store = TrialStore(args.root, args.subject, session)
    store.write_session({
        "protocol": args.protocol,
        "brief": args.brief,
        "channels": list(config.eeg.channels),
        "sample_rate_hz": config.eeg.sample_rate_hz,
        "control_mode": "manual",
        "mock": not args.no_mock,
    })

    acq = build_acquisition(args, config)
    acq.connect()
    acq.pump_seconds(2.0)  # pre-roll so the ring buffer is warm
    cfg = ProtocolConfig()
    rng = np.random.default_rng(args.seed)

    start_id = store.next_trial_id()
    if start_id > 1:
        print(f"[collect] resuming {args.subject}/{session} at trial {start_id}")
    print(f"[collect] {args.protocol} | subject={args.subject} session={session} "
          f"| {args.trials} trials, blocks of {args.block_size}")

    n_valid = 0
    for i in range(args.trials):
        trial_id = start_id + i
        if i > 0 and i % args.block_size == 0:
            print(f"[collect] --- block break after {i} trials (rest, then continue) ---")

        backend = acq.backend
        force_bad = (not args.no_mock and args.bad_contact_every > 0
                     and (i + 1) % args.bad_contact_every == 0)
        if hasattr(backend, "inject_bad_contact"):
            backend.inject_bad_contact = set(list(config.eeg.channels)[:9]) if force_bad else set()

        if args.protocol == "satisfaction":
            satisfied = bool(rng.integers(0, 2))  # balanced labels
            true_s = 0.8 if satisfied else -0.8
            res = run_satisfaction_trial(acq, HeadlessPresenter(), cfg, true_s, rng)
            q = res.quality["candidate"]
            meta = {
                "protocol": "satisfaction",
                "brief": args.brief,
                "candidate_id": f"c{trial_id}",
                "true_label": "satisfied" if satisfied else "dissatisfied",
                "satisfaction_rating": 5 if satisfied else 1,
                "manual_response": "keep" if satisfied else "refine",
                "label_source": "manual",
                "quality_status": q.status,
                "quality_reasons": q.reasons,
                "quality_metrics": q.metrics,
                "usable_label": q.is_valid,  # invalid EEG is never a label
            }
            store.save_trial(trial_id, meta, res.epochs)
            n_valid += int(q.is_valid)
            print(f"[trial {trial_id:04d}] {meta['true_label']:12} quality={q.status:7} "
                  f"contact_ok={acq.latest_contact() and 'y' or '-'} reasons={q.reasons}")
        else:
            a_better = bool(rng.integers(0, 2))
            sat_a, sat_b = (0.8, -0.8) if a_better else (-0.8, 0.8)
            res = run_pairwise_trial(acq, HeadlessPresenter(), cfg, sat_a, sat_b, rng)
            qa, qb = res.quality["A"], res.quality["B"]
            usable = qa.is_valid and qb.is_valid
            meta = {
                "protocol": "pairwise",
                "brief": args.brief,
                "candidate_a_id": f"c{trial_id}a",
                "candidate_b_id": f"c{trial_id}b",
                "preferred": "A" if a_better else "B",
                "label_source": "manual",
                "quality_status": {"A": qa.status, "B": qb.status},
                "usable_label": usable,
            }
            store.save_trial(trial_id, meta, res.epochs)
            n_valid += int(usable)
            print(f"[trial {trial_id:04d}] prefer={meta['preferred']} "
                  f"qA={qa.status} qB={qb.status}")

    acq.close()
    print(f"[collect] done: {args.trials} trials saved to {store.dir} "
          f"({n_valid} usable / {args.trials})")
    print(f"[collect] manifest: {store.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
