"""Loader for the public MindBigData "EPOC" digit dataset.

Source: http://mindbigdata.com/opendb/  (the "EP1.01" text dump, EMOTIV EPOC).

File format: one signal per line, TAB-separated::

    [id]  [event]  [device]  [channel]  [code]  [size]  [data]

  id      unique row id
  event   groups the 14 channel-rows that belong to one 2 s capture
  device  "EP" for EPOC
  channel one of AF3, F7, ... AF4
  code    the digit shown (0-9), or -1 for the random/blank stimulus
  size    number of samples in `data`
  data    comma-separated signal values

We regroup rows by `event` into (14, ~256) epochs labelled by `code`, keeping
only complete 14-channel captures with a digit label in 0-9.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config


def _iter_rows(path: Path) -> Iterator[tuple[int, str, int, np.ndarray]]:
    """Yield (event, channel, code, samples) for each valid EPOC line."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            _id, event, device, channel, code, _size, data = parts[:7]
            if device != "EP" or channel not in config.EPOC_CHANNELS:
                continue
            try:
                code_i = int(code)
                samples = np.fromstring(data, sep=",", dtype=np.float32)
            except ValueError:
                continue
            if samples.size == 0:
                continue
            yield int(event), channel, code_i, samples


def load_epochs(
    path: str | Path = config.DEFAULT_DATA_PATH,
    max_events: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X is (N, 14, WINDOW_SAMPLES) raw epochs (channel order per
    `config.EPOC_CHANNELS`), y is (N,) digit labels 0-9. Random-stimulus (-1)
    captures and incomplete events are dropped.
    """
    from .preprocessing import fit_window

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MindBigData EPOC file not found at {path}. Download 'EP1.01' from "
            "http://mindbigdata.com/opendb/ and place it there (or pass --data)."
        )

    # event -> {channel -> samples}, and event -> code
    by_event: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    codes: dict[int, int] = {}
    for event, channel, code, samples in _iter_rows(path):
        by_event[event][channel] = samples
        codes[event] = code

    X: list[np.ndarray] = []
    y: list[int] = []
    for event, chans in by_event.items():
        code = codes[event]
        if code < 0 or code > 9:
            continue
        if any(ch not in chans for ch in config.EPOC_CHANNELS):
            continue  # incomplete capture
        rows = [fit_window(chans[ch][None, :], config.WINDOW_SAMPLES)[0]
                for ch in config.EPOC_CHANNELS]
        X.append(np.stack(rows, axis=0))
        y.append(code)
        if max_events is not None and len(X) >= max_events:
            break

    if not X:
        raise ValueError(f"No complete labelled EPOC epochs parsed from {path}")
    return np.stack(X, axis=0).astype(np.float32), np.asarray(y, dtype=np.int64)


def synthetic_epochs(
    n_per_class: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-separable synthetic EPOC epochs so the whole pipeline (train + infer
    + real-time) is runnable without downloading the 2 GB MindBigData file or
    owning a headset. Each digit gets a distinct band-power/topography signature;
    it is NOT a claim about real neural decodability, just a smoke-test fixture.
    """
    rng = np.random.default_rng(seed)
    fs = config.SAMPLE_RATE_HZ
    t = np.arange(config.WINDOW_SAMPLES) / fs
    X: list[np.ndarray] = []
    y: list[int] = []
    for digit in range(config.NUM_CLASSES):
        # A per-digit frequency and a per-digit "active" channel subset.
        base_f = 6.0 + 1.8 * digit
        active = set(np.roll(np.arange(config.NUM_CHANNELS), digit)[: 4 + digit % 3].tolist())
        for _ in range(n_per_class):
            epoch = np.zeros((config.NUM_CHANNELS, config.WINDOW_SAMPLES), dtype=np.float32)
            phase = rng.uniform(0, 2 * np.pi)
            for c in range(config.NUM_CHANNELS):
                gain = 1.8 if c in active else 0.6
                sig = gain * np.sin(2 * np.pi * base_f * t + phase)
                sig += 0.5 * np.sin(2 * np.pi * (base_f / 2) * t)
                sig += rng.normal(0, 0.7, size=config.WINDOW_SAMPLES)
                epoch[c] = 20.0 * sig  # microvolt-ish scale
            X.append(epoch)
            y.append(digit)
    order = rng.permutation(len(X))
    return np.stack(X)[order].astype(np.float32), np.asarray(y)[order]
