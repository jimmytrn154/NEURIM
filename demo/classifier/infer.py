"""Single-epoch inference: raw (14, samples) epoch -> predicted digit + probs."""

from __future__ import annotations

import numpy as np
import torch

from . import config
from .model import BrainDigiCNN, load
from .preprocessing import epoch_hht_features, preprocess


class DigitClassifier:
    """Loads a trained BrainDigiCNN and classifies raw EPOC epochs. Applies the
    same feature transform (denoised or EMD+HHT) used in training, read from the
    checkpoint metadata."""

    def __init__(self, model_path=config.DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.model, self.meta = load(model_path, map_location=device)
        self.model.to(device).eval()
        self.device = device
        self.band = self.meta.get("band")
        self.feature = self.meta.get("feature", "denoised")
        self.n_imf = self.meta.get("n_imf", 6)

    @classmethod
    def from_model(cls, model: BrainDigiCNN, band: str | None = None,
                   feature: str = "denoised", device: str = "cpu"):
        obj = cls.__new__(cls)
        obj.model = model.to(device).eval()
        obj.meta = {"band": band, "feature": feature}
        obj.device = device
        obj.band = band
        obj.feature = feature
        obj.n_imf = 6
        return obj

    def _featurize(self, epoch: np.ndarray) -> np.ndarray:
        x = np.asarray(epoch, dtype=np.float32)
        if self.feature == "hht":
            return epoch_hht_features(x, config.SAMPLE_RATE_HZ, self.n_imf, band=self.band)
        return preprocess(x, config.SAMPLE_RATE_HZ, self.band)

    def predict(self, epoch: np.ndarray) -> tuple[int, np.ndarray]:
        """`epoch` is (14, samples) raw EEG. Returns (digit, probability vector)."""
        x = self._featurize(epoch)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x[None]).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return int(probs.argmax()), probs
