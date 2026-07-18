"""Single-epoch inference: raw (14, samples) epoch -> predicted digit + probs."""

from __future__ import annotations

import numpy as np
import torch

from . import config
from .model import BrainDigiCNN, load
from .preprocessing import preprocess


class DigitClassifier:
    """Loads a trained BrainDigiCNN and classifies raw EPOC epochs. Applies the
    same denoise/(band)/z-score preprocessing used in training."""

    def __init__(self, model_path=config.DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.model, self.meta = load(model_path, map_location=device)
        self.model.to(device).eval()
        self.device = device
        self.band = self.meta.get("band")

    @classmethod
    def from_model(cls, model: BrainDigiCNN, band: str | None = None, device: str = "cpu"):
        obj = cls.__new__(cls)
        obj.model = model.to(device).eval()
        obj.meta = {"band": band}
        obj.device = device
        obj.band = band
        return obj

    def predict(self, epoch: np.ndarray) -> tuple[int, np.ndarray]:
        """`epoch` is (14, samples) raw EEG. Returns (digit, probability vector)."""
        x = preprocess(np.asarray(epoch, dtype=np.float32), config.SAMPLE_RATE_HZ, self.band)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x[None]).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return int(probs.argmax()), probs
