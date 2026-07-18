"""Real-time EEG-to-digit classifier for the EMOTIV EPOC X.

Streams samples from the headset (via the EMOTIV Cortex API, reusing NEURIM's
`EmotivCortexSource`), keeps a sliding 2 s window of the 14 EEG channels, and
runs the trained BrainDigiCNN every `--stride` seconds -- printing the decoded
digit and its confidence.

"Real-time" here is the standard windowed-BCI sense: classify the most recent
2 s window on a fixed cadence. The heavy EMD/HHT feature path from the paper is
offline-only; the live path uses the denoised-signal features (see
preprocessing.py), which the same trained model consumes.

Examples
--------
    # No headset -- drive it with a synthetic stream to see it work end to end:
    python -m demo.classifier.realtime --source synthetic --model demo/classifier/artifacts/braindigicnn.pt

    # Live EPOC X (needs EMOTIV Launcher running + EMOTIV_CLIENT_ID/SECRET):
    python -m demo.classifier.realtime --source cortex
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np

# Allow `from src...` imports when run as a script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo.classifier import config
from demo.classifier.infer import DigitClassifier


class SlidingWindow:
    """Fixed-length per-channel ring of the latest `size` samples."""

    def __init__(self, channels: list[str], size: int):
        self.channels = channels
        self.size = size
        self._buf: dict[str, deque[float]] = {c: deque(maxlen=size) for c in channels}

    def push(self, sample: dict[str, float]) -> None:
        for c in self.channels:
            if c in sample:
                self._buf[c].append(float(sample[c]))

    def full(self) -> bool:
        return all(len(self._buf[c]) == self.size for c in self.channels)

    def epoch(self) -> np.ndarray:
        return np.stack([np.asarray(self._buf[c], dtype=np.float32) for c in self.channels], axis=0)


def _cortex_stream() -> Iterator[tuple[float, dict[str, float]]]:
    from src.signal_service.eeg_sources import EmotivCortexSource

    src = EmotivCortexSource(
        client_id=os.environ.get("EMOTIV_CLIENT_ID"),
        client_secret=os.environ.get("EMOTIV_CLIENT_SECRET"),
        headset_id=os.environ.get("EMOTIV_HEADSET_ID"),
    )
    src.connect()
    print("[cortex] streaming EPOC X ...")
    try:
        yield from src.stream(channels=config.EPOC_CHANNELS)
    finally:
        src.close()


def _synthetic_stream(seed: int = 0) -> Iterator[tuple[float, dict[str, float]]]:
    """Live synthetic EPOC stream that cycles the imagined digit every ~3 s, using
    the same per-digit signatures as the training fixture -- so a model trained
    with `--synthetic` visibly tracks the changing digit."""
    from demo.classifier.braindigidata import config as C

    rng = np.random.default_rng(seed)
    fs = config.SAMPLE_RATE_HZ
    dt = 1.0 / fs
    t = 0.0
    k = 0
    while True:
        digit = (k // int(fs * 3)) % config.NUM_CLASSES  # switch digit every 3 s
        base_f = 6.0 + 1.8 * digit
        active = set(np.roll(np.arange(C.NUM_CHANNELS), digit)[: 4 + digit % 3].tolist())
        sample = {}
        for ci, ch in enumerate(config.EPOC_CHANNELS):
            gain = 1.8 if ci in active else 0.6
            val = gain * np.sin(2 * np.pi * base_f * t) + 0.5 * np.sin(2 * np.pi * (base_f / 2) * t)
            val += rng.normal(0, 0.7)
            sample[ch] = 20.0 * val
        yield t, sample
        t += dt
        k += 1


def run(source: str, model_path: Path, stride_s: float, pace: bool, seed: int = 0) -> None:
    clf = DigitClassifier(model_path)
    if clf.feature == "hht":
        raise SystemExit(
            f"Model {model_path} was trained with EMD+HHT features, which are "
            "offline-only (EMD is too slow for the live loop). Train a "
            "'--feature denoised' model for real-time use."
        )
    print(f"loaded model {model_path} (feature={clf.feature}, band={clf.band or 'full'})")

    win = SlidingWindow(config.EPOC_CHANNELS, config.WINDOW_SAMPLES)
    stream = _cortex_stream() if source == "cortex" else _synthetic_stream(seed)
    stride_samples = max(1, int(round(stride_s * config.SAMPLE_RATE_HZ)))

    n = 0
    last_wall = time.monotonic()
    for _t, sample in stream:
        win.push(sample)
        n += 1
        if pace and source != "cortex":
            # Emit at ~real time so the synthetic demo isn't instantaneous.
            now = time.monotonic()
            sleep = (1.0 / config.SAMPLE_RATE_HZ) - (now - last_wall)
            if sleep > 0:
                time.sleep(sleep)
            last_wall = time.monotonic()
        if win.full() and n % stride_samples == 0:
            digit, probs = clf.predict(win.epoch())
            top3 = np.argsort(probs)[::-1][:3]
            bars = "  ".join(f"{d}:{probs[d]:.2f}" for d in top3)
            print(f"[{time.strftime('%H:%M:%S')}] digit = {digit}   (conf {probs[digit]:.2f})   {bars}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time EPOC X digit classifier")
    ap.add_argument("--source", choices=["cortex", "synthetic"], default="synthetic")
    ap.add_argument("--model", default=str(config.DEFAULT_MODEL_PATH))
    ap.add_argument("--stride", type=float, default=0.5, help="seconds between predictions")
    ap.add_argument("--no-pace", action="store_true", help="don't throttle the synthetic stream to real time")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        run(args.source, Path(args.model), args.stride, pace=not args.no_pace, seed=args.seed)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
