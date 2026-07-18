"""Smoke + shape tests for the digit classifier pipeline (no hardware/data needed)."""

from __future__ import annotations

import numpy as np
import torch

from demo.classifier import config
from demo.classifier.braindigidata import synthetic_epochs
from demo.classifier.infer import DigitClassifier
from demo.classifier.model import BrainDigiCNN
from demo.classifier.preprocessing import denoise, fit_window, preprocess
from demo.classifier.realtime import SlidingWindow


def test_preprocess_shape_and_finiteness():
    raw = np.random.randn(config.NUM_CHANNELS, 251).astype(np.float32) * 30
    x = preprocess(raw)
    assert x.shape == (config.NUM_CHANNELS, config.WINDOW_SAMPLES)
    assert np.isfinite(x).all()


def test_fit_window_crop_and_pad():
    assert fit_window(np.zeros((14, 300)))[0].shape[0] == config.WINDOW_SAMPLES
    assert fit_window(np.zeros((14, 100)))[0].shape[0] == config.WINDOW_SAMPLES


def test_denoise_preserves_length():
    raw = np.random.randn(config.NUM_CHANNELS, config.WINDOW_SAMPLES).astype(np.float32)
    assert denoise(raw).shape == raw.shape


def test_model_forward_shape():
    model = BrainDigiCNN()
    out = model(torch.randn(4, config.NUM_CHANNELS, config.WINDOW_SAMPLES))
    assert out.shape == (4, config.NUM_CLASSES)


def test_model_learns_synthetic_fixture():
    """One short training run should beat chance (0.1) on the separable fixture."""
    X, y = synthetic_epochs(n_per_class=30, seed=1)
    from demo.classifier.dataset import DigitEpochs
    from torch.utils.data import DataLoader

    ds = DigitEpochs(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = BrainDigiCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(6):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(ds.X).argmax(1) == ds.y).float().mean().item()
    assert acc > 0.5  # well above 10% chance


def test_sliding_window_and_classifier_roundtrip():
    win = SlidingWindow(config.EPOC_CHANNELS, config.WINDOW_SAMPLES)
    assert not win.full()
    for _ in range(config.WINDOW_SAMPLES):
        win.push({c: np.random.randn() for c in config.EPOC_CHANNELS})
    assert win.full()
    clf = DigitClassifier.from_model(BrainDigiCNN())
    digit, probs = clf.predict(win.epoch())
    assert 0 <= digit < config.NUM_CLASSES
    assert probs.shape == (config.NUM_CLASSES,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)
