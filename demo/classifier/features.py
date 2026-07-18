"""Build CNN input feature matrices from raw epochs, with parallelism + caching.

Two feature modes:
  - "denoised": low-pass + notch + z-score -> (N, 14, 256). Cheap; also the
    representation the real-time path uses.
  - "hht": paper-faithful EMD + Hilbert-Huang (IA/IF of the first n_imf IMFs per
    channel) -> (N, 14*len(attrs)*n_imf, 256). EMD is ~30 ms/epoch, so this is
    computed in a process pool and cached to disk keyed by the feature params.

Offline only for "hht": a model trained on HHT features cannot be driven by the
live denoised stream (different channel count), so realtime.py rejects it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config
from .preprocessing import epoch_hht_features, hht_channels, preprocess

CACHE_DIR = config.HERE / "artifacts" / "feature_cache"


# Module-level workers so multiprocessing can pickle them.
def _denoised_one(args) -> np.ndarray:
    epoch, band = args
    return preprocess(epoch, config.SAMPLE_RATE_HZ, band)


def _hht_one(args) -> np.ndarray:
    epoch, band, n_imf, attrs = args
    return epoch_hht_features(epoch, config.SAMPLE_RATE_HZ, n_imf, attrs, band)


def _cache_key(mode, band, n_imf, attrs, n, extra) -> str:
    h = hashlib.sha1(
        f"{mode}|{band}|{n_imf}|{attrs}|{n}|{extra}".encode()
    ).hexdigest()[:16]
    return f"{mode}_{band or 'full'}_{h}.npy"


def build_features(
    X: np.ndarray,
    mode: str = "denoised",
    band: str | None = None,
    n_imf: int = 6,
    attrs: tuple[str, ...] = ("IA", "IF"),
    jobs: int = 1,
    cache: bool = True,
    cache_tag: str = "",
) -> np.ndarray:
    """Return the (N, C_feat, WINDOW_SAMPLES) feature matrix for raw epochs `X`.

    `mode="hht"` is cached to artifacts/feature_cache and computed with `jobs`
    worker processes. `cache_tag` (e.g. the data path + max_events) disambiguates
    caches built from different source subsets.
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(X)

    cache_path = CACHE_DIR / _cache_key(mode, band, n_imf, attrs, n, cache_tag)
    if cache and mode == "hht" and cache_path.exists():
        print(f"[features] loading cached HHT features <- {cache_path.name}")
        return np.load(cache_path)

    if mode == "denoised":
        feats = np.stack([preprocess(e, config.SAMPLE_RATE_HZ, band) for e in X])
    elif mode == "hht":
        print(f"[features] extracting EMD+HHT for {n} epochs "
              f"(n_imf={n_imf}, attrs={attrs}, jobs={jobs}) ...")
        payload = [(e, band, n_imf, attrs) for e in X]
        if jobs and jobs > 1:
            from multiprocessing import Pool
            with Pool(jobs) as pool:
                out = pool.map(_hht_one, payload, chunksize=16)
        else:
            out = [_hht_one(p) for p in payload]
        feats = np.stack(out)
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, feats)
            print(f"[features] cached -> {cache_path.name}")
    else:
        raise ValueError(f"unknown feature mode {mode!r}")

    return feats.astype(np.float32)


def feature_channels(mode: str, n_imf: int = 6, attrs: tuple[str, ...] = ("IA", "IF")) -> int:
    return hht_channels(n_imf, attrs) if mode == "hht" else config.NUM_CHANNELS
