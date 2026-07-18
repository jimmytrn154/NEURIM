"""BrainDigiCNN: the 1D CNN from Tiwari et al. (2023), Table 4.

Architecture (per the paper), 4 conv blocks then 3 dense layers::

    Conv1D(256, k=7) -> BN -> ReLU -> MaxPool(2)
    Conv1D(128, k=7) -> BN -> ReLU -> MaxPool(2)
    Conv1D( 64, k=7) -> BN -> ReLU -> MaxPool(2)
    Conv1D( 32, k=7) -> BN -> ReLU -> MaxPool(2)
    Flatten -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(10, Softmax)

The paper builds it in Keras; this is the faithful PyTorch equivalent (Softmax
is folded into CrossEntropyLoss during training). Input is (batch, channels,
samples) = (B, 14, 256) by default -- the 14 EEG sensors are the conv input
channels and convolution runs along the 256-sample time axis.
"""

from __future__ import annotations

import torch
from torch import nn

from . import config


class BrainDigiCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = config.NUM_CHANNELS,
        seq_len: int = config.WINDOW_SAMPLES,
        num_classes: int = config.NUM_CLASSES,
        filters: tuple[int, ...] = config.CONV_FILTERS,
        kernel_size: int = config.CONV_KERNEL,
        dense_units: tuple[int, ...] = config.DENSE_UNITS,
    ):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding so only the pooling halves length
        blocks: list[nn.Module] = []
        c_in = in_channels
        length = seq_len
        for c_out in filters:
            blocks += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=pad),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
            length //= 2
        self.features = nn.Sequential(*blocks)

        flat = filters[-1] * max(length, 1)
        dense: list[nn.Module] = []
        d_in = flat
        for d_out in dense_units:
            dense += [nn.Linear(d_in, d_out), nn.ReLU(inplace=True)]
            d_in = d_out
        dense.append(nn.Linear(d_in, num_classes))
        self.classifier = nn.Sequential(*dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # logits; apply softmax for probabilities


def save(model: BrainDigiCNN, path, meta: dict | None = None) -> None:
    import pathlib

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load(path, map_location: str = "cpu") -> tuple[BrainDigiCNN, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt.get("meta", {})
    model = BrainDigiCNN(
        in_channels=meta.get("in_channels", config.NUM_CHANNELS),
        seq_len=meta.get("seq_len", config.WINDOW_SAMPLES),
        num_classes=meta.get("num_classes", config.NUM_CLASSES),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta
