#!/usr/bin/env python3
"""Live FAA diagnostic: stream EEG and print, a few times a second, weighted
frontal alpha asymmetry, pair-level FAA values, and the mapped reward r(t).

Read-only sanity check on the Signal service - no optimizer, no generator.
Use it to confirm the headset gives a sane, responsive FAA before wiring the
signal into the loop, and to watch how leaning-in / pulling-away moves the
number.

    python scripts/test_faa_stream.py                # real EPOC X via Cortex
    python scripts/test_faa_stream.py --mock         # synthetic signal, no hardware
    python scripts/test_faa_stream.py --baseline 0   # skip baseline (prints raw FAA as reward)
    python scripts/test_faa_stream.py --baseline 15  # shorter 15s rest calibration

Column meanings:
    FAAcomp           weighted composite ln(P_right) - ln(P_left) across pairs
    pair FAA values   per-pair raw FAA, useful for spotting one bad electrode pair
    reward            FAA z-scored against your resting baseline, clipped to [-1, 1]
    cue               lean-in / neutral / pull-away label from current reward
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config, emotiv_credentials
from src.signal_service.baseline import calibrate_baseline
from src.signal_service.eeg_sources import EmotivCortexSource, MockEEGSource, wall_clock_pace
from src.signal_service.faa import FAARewardComputer


def _pair_label(pair: list[str] | tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _pair_value(
    raw_by_pair: dict[str, float], pair: list[str] | tuple[str, str], width: int
) -> str:
    value = raw_by_pair.get(_pair_label(pair))
    return f"{'n/a':<{width}}" if value is None else f"{value:+.2f}".ljust(width)


def _cue_label(reward: float, neutral_threshold: float = 0.20) -> str:
    if reward > neutral_threshold:
        return "lean-in"
    if reward < -neutral_threshold:
        return "pull-away"
    return "neutral"


def _no_reward_reason(computer: FAARewardComputer) -> str:
    if not computer.ready():
        return f"filling {computer.window_s:.1f}s FAA window"

    metrics = computer.pair_metrics()
    if not metrics:
        return "no configured channel pair has a full window"
    return f"{len(metrics)} configured pair(s) available, but reward unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mock", action="store_true", help="synthetic EEG instead of hardware")
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="rest-calibration seconds (default: config faa.baseline_duration_s; 0 to skip)",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        help="seconds between no-reward diagnostic messages",
    )
    parser.add_argument(
        "--cue-threshold",
        type=float,
        default=0.20,
        help="absolute reward value below which the cue is neutral",
    )
    args = parser.parse_args()

    config = Config.load()
    fs = config.eeg.sample_rate_hz
    baseline_s = config.faa.baseline_duration_s if args.baseline is None else args.baseline

    if args.mock:
        # Slow oscillating bias so F3/F4 asymmetry visibly swings, exercising
        # the whole path without a headset.
        eeg_source = MockEEGSource(
            config.eeg.channels, fs, bias_fn=lambda t: math.sin(2 * math.pi * t / 20.0)
        )
    else:
        client_id, client_secret = emotiv_credentials()
        eeg_source = EmotivCortexSource(client_id, client_secret)

    print(f"[faa] connecting ({'mock' if args.mock else 'EPOC X via Cortex'}) ...")
    eeg_source.connect()

    computer = FAARewardComputer(
        fs=fs,
        channel_left=config.faa.channel_left,
        channel_right=config.faa.channel_right,
        band=config.faa.band_hz,
        window_s=config.faa.window_s,
        clip=config.faa.clip,
        channels=config.eeg.channels,
        channel_pairs=config.faa.channel_pairs,
        pair_weights=config.faa.pair_weights,
    )
    pair_labels = [_pair_label(pair) for pair in config.faa.channel_pairs]
    print(
        f"[faa] pairs={', '.join(pair_labels)}  "
        f"band={config.faa.band_hz[0]:.0f}-{config.faa.band_hz[1]:.0f}Hz  "
        f"window={config.faa.window_s:.1f}s"
    )

    # A single generator instance drives both the baseline and live phases.
    sample_iter = (
        wall_clock_pace(eeg_source.stream(), fs) if args.mock else eeg_source.stream()
    )

    try:
        if baseline_s > 0:
            print(f"[faa] hold still and rest for {baseline_s:.0f}s to fit the baseline ...")
            baseline = calibrate_baseline(computer, sample_iter, duration_s=baseline_s)
            print(f"[faa] baseline fitted: mean={baseline.mean:+.4f} std={baseline.std:.4f} n={baseline.n}")
        else:
            print("[faa] --baseline 0: no calibration, reward = clipped raw FAA (uncentered)")

        emit_every = max(1, int(fs * config.faa.update_interval_s))
        status_every = max(1, int(fs * args.status_interval))
        print()
        column_widths = [8, 8, *(max(8, len(label)) for label in pair_labels), 10]
        headers = ["FAAcomp", "reward", *pair_labels, "cue"]
        header = " ".join(
            f"{name:<{width}}" for name, width in zip(headers, column_widths, strict=True)
        )
        print(header)
        print("-" * len(header))

        i = 0
        for t, sample in sample_iter:
            computer.push_sample(sample)
            i += 1
            if i % emit_every != 0:
                continue
            raw = computer.raw_value()
            reward = computer.reward()
            if raw is None or reward is None:
                if i % status_every == 0:
                    print(f"[faa] no reward: {_no_reward_reason(computer)}")
                continue  # window not full yet
            raw_by_pair = {
                f"{item['left']}/{item['right']}": float(item["raw_faa"])
                for item in computer.pair_metrics()
            }
            values = [
                f"{raw:+.3f}".ljust(column_widths[0]),
                f"{reward:+.2f}".ljust(column_widths[1]),
                *(
                    _pair_value(raw_by_pair, pair, width)
                    for pair, width in zip(
                        config.faa.channel_pairs, column_widths[2:-1], strict=True
                    )
                ),
                _cue_label(reward, args.cue_threshold).ljust(column_widths[-1]),
            ]
            print(" ".join(values))
    except KeyboardInterrupt:
        print("\n[faa] stopped.")
    finally:
        eeg_source.close()


if __name__ == "__main__":
    main()
