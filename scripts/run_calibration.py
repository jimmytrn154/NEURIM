#!/usr/bin/env python3
"""Phase 2: per-subject baseline. Sit still and rest for
config.faa.baseline_duration_s; this fits the mean/std that every later FAA
reading gets z-scored against. Saves the fitted baseline so a session can be
resumed without re-calibrating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config, emotiv_credentials
from src.signal_service.baseline import calibrate_baseline
from src.signal_service.eeg_sources import EmotivCortexSource, MockEEGSource
from src.signal_service.faa import FAARewardComputer

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="subject01")
    parser.add_argument("--mock", action="store_true", help="use synthetic EEG instead of real hardware")
    args = parser.parse_args()

    config = Config.load()

    if args.mock:
        eeg_source = MockEEGSource(config.eeg.channels, config.eeg.sample_rate_hz)
    else:
        client_id, client_secret = emotiv_credentials()
        eeg_source = EmotivCortexSource(client_id, client_secret)

    eeg_source.connect()

    computer = FAARewardComputer(
        fs=config.eeg.sample_rate_hz,
        channel_left=config.faa.channel_left,
        channel_right=config.faa.channel_right,
        band=config.faa.band_hz,
        window_s=config.faa.window_s,
        clip=config.faa.clip,
        channels=config.eeg.channels,
        channel_pairs=config.faa.channel_pairs,
        pair_weights=config.faa.pair_weights,
    )

    try:
        print(f"[calibration] rest for {config.faa.baseline_duration_s:.0f}s ...")
        stream = (
            eeg_source.stream(config.eeg.channels)
            if isinstance(eeg_source, EmotivCortexSource)
            else eeg_source.stream()
        )
        baseline = calibrate_baseline(computer, stream, duration_s=config.faa.baseline_duration_s)
    finally:
        # Always release the Cortex session, even on Ctrl+C or a mid-run
        # error - an unclosed session blocks the next connection attempt.
        eeg_source.close()

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIBRATION_DIR / f"{args.subject}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(asdict(baseline), f, indent=2)

    print(f"[calibration] mean={baseline.mean:.4f} std={baseline.std:.4f} n={baseline.n}")
    print(f"[calibration] saved to {out_path}")


if __name__ == "__main__":
    main()
