"""Torch Dataset that turns raw MBD epochs into preprocessed CNN tensors."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .preprocessing import preprocess


class DigitEpochs(Dataset):
    """Wraps (N, 14, samples) raw epochs + (N,) labels. Preprocessing (denoise +
    optional sub-band + z-score) runs once at construction so training epochs are
    cheap. Set `band` to train a band-wise model (paper Sect. 4)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, band: str | None = None):
        proc = np.stack([preprocess(e, config.SAMPLE_RATE_HZ, band) for e in X])
        self.X = torch.from_numpy(proc.astype(np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]
