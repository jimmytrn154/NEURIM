#!/usr/bin/env python3
"""Real-time attention / relaxation from EMOTIV Cortex performance metrics.

Answers: when the subject sees an image they want vs. don't want, does measured
attention move reliably? Uses Cortex's `met` (focus/relaxation) stream gated by
`eq` (EEG quality) - not the raw-EEG preference model.

    EPOC X -> Cortex `met` @ ~2 Hz -> foc.isActive gate -> eq quality gate
      -> EMA / 3 s median smoothing -> subject baseline normalization
      -> real-time concentration confidence in [0, 1]

Flow: a short baseline (rest / neutral viewing) fixes the subject's mean/std,
then the live readout shows normalized concentration + relaxation confidence.

Examples:
    # Offline: scripted attention signal, no headset
    python scripts/measure_attention.py --mock

    # Real EPOC X (needs EMOTIV Launcher running + EMOTIV_CLIENT_ID/SECRET)
    python scripts/measure_attention.py --baseline-seconds 30 --smoothing median
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signal_service.attention import (
    AttentionMonitor,
    CortexMetricsSource,
    MockMetricsSource,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true",
                   help="use a scripted synthetic metric stream (no headset)")
    p.add_argument("--baseline-seconds", type=float, default=20.0,
                   help="rest/neutral seconds used to fix the subject's mean/std")
    p.add_argument("--smoothing", choices=["ema", "median"], default="ema")
    p.add_argument("--tau-seconds", type=float, default=3.0,
                   help="EMA time constant (only for --smoothing ema)")
    p.add_argument("--median-seconds", type=float, default=3.0,
                   help="trailing-median window (only for --smoothing median)")
    p.add_argument("--met-rate-hz", type=float, default=2.0,
                   help="expected met rate; sets the EMA step size")
    p.add_argument("--quality-min", type=float, default=2.0,
                   help="minimum eq overall quality (0-4) to trust a reading")
    p.add_argument("--confidence-gain", type=float, default=1.0,
                   help="logistic slope mapping normalized attention -> confidence")
    p.add_argument("--duration", type=float, default=0.0,
                   help="live seconds after baseline (0 = until Ctrl-C)")
    p.add_argument("--headset-id", default=None)
    p.add_argument("--debug", action="store_true",
                   help="dump the raw Cortex met flags + per-channel eq quality for "
                        "the first samples (diagnose an inactive metric / poor contact)")
    return p.parse_args()


def _bar(x: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "#" * n + "-" * (width - n)


def build_source(args):
    if args.mock:
        # realtime=False streams as fast as possible; flip to True to watch it live.
        return MockMetricsSource(rate_hz=args.met_rate_hz, baseline_s=args.baseline_seconds,
                                 realtime=False)
    from src.common.config import emotiv_credentials

    client_id, client_secret = emotiv_credentials()
    return CortexMetricsSource(client_id, client_secret, headset_id=args.headset_id)


def main() -> int:
    args = parse_args()
    monitor = AttentionMonitor(
        smoothing=args.smoothing,
        tau_s=args.tau_seconds,
        median_window_s=args.median_seconds,
        met_rate_hz=args.met_rate_hz,
        quality_min=args.quality_min,
        confidence_gain=args.confidence_gain,
    )
    source = build_source(args)
    print("[attention] connecting ...")
    source.connect()
    stream = source.stream()

    t0 = None
    n_baseline = 0
    n_debug = 0
    print(f"[attention] baseline: hold a neutral/rest state for "
          f"{args.baseline_seconds:.0f}s while I learn your resting level ...")
    try:
        for sample in stream:
            if t0 is None:
                t0 = sample.t
            elapsed = sample.t - t0
            reading = monitor.update(sample)

            if args.debug and n_debug < 8:
                n_debug += 1
                active = {k: v for k, v in sample.raw.items() if k.endswith(".isActive")}
                print(f"\n[debug] met.foc={sample.attention} foc.isActive={sample.attention_active} "
                      f"met.rel={sample.relaxation} rel.isActive={sample.relaxation_active}")
                print(f"[debug] all isActive flags: {active}")
                print(f"[debug] eq overall={sample.quality_overall} per-channel={sample.quality_raw}")

            in_baseline = not monitor.has_baseline
            if in_baseline:
                if reading.reliable:
                    n_baseline += 1
                if elapsed >= args.baseline_seconds:
                    monitor.freeze_baseline()
                    print(f"[attention] baseline locked over {n_baseline} good samples; "
                          f"live readout starting.\n")
                    continue
                q = "  --" if sample.quality_overall is None else f"{sample.quality_overall:.1f}"
                flag = "ok " if reading.reliable else "LOW"
                print(f"  [baseline {elapsed:4.1f}s] att={_fmt(sample.attention)} "
                      f"quality={q} {flag}", end="\r")
                continue

            gt = sample.raw.get("label")
            gt_s = f" truth={gt}" if gt is not None else ""
            rel_flag = "" if reading.reliable else "  (signal LOW - carrying last)"
            print(
                f"t={elapsed:6.1f}s  CONCENTRATION {reading.concentration_confidence:4.2f} "
                f"[{_bar(reading.concentration_confidence)}]  "
                f"relax {reading.relaxation_confidence:4.2f}  "
                f"att_z={reading.attention_z:+5.2f}{gt_s}{rel_flag}"
            )
            if args.duration and elapsed - args.baseline_seconds >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n[attention] stopped.")
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()
    return 0


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:4.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
