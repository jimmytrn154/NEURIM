"""Train BrainDigiCNN on the MindBigData EPOC digit dataset.

Examples
--------
    # Real data (download EP1.01 from mindbigdata.com/opendb first):
    python -m demo.classifier.train --data demo/classifier/data/EP1.01.txt

    # No dataset / no headset -- train on the synthetic fixture to smoke-test:
    python -m demo.classifier.train --synthetic --epochs 8

    # Band-wise model (paper Sect. 4), e.g. the delta band:
    python -m demo.classifier.train --data ... --band delta
"""

from __future__ import annotations

# Allow running directly (`python3 train.py`) from inside the folder, not just
# as a module (`python -m demo.classifier.train`) from the repo root.
if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "demo.classifier"

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from . import config
from .braindigidata import load_epochs, synthetic_epochs
from .dataset import DigitEpochs
from .model import BrainDigiCNN, save


def _evaluate(model: BrainDigiCNN, loader: DataLoader, device: str) -> tuple[float, list, list]:
    model.eval()
    correct = total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum())
            total += len(yb)
            y_true += yb.cpu().tolist()
            y_pred += pred.cpu().tolist()
    return correct / max(total, 1), y_true, y_pred


def train(
    X: np.ndarray,
    y: np.ndarray,
    band: str | None = None,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    out_path=config.DEFAULT_MODEL_PATH,
    seed: int = 0,
) -> BrainDigiCNN:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 70/30 split (paper), stratified by digit.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    train_ds = DigitEpochs(Xtr, ytr, band=band)
    test_ds = DigitEpochs(Xte, yte, band=band)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = BrainDigiCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss) * len(yb)
        acc, _, _ = _evaluate(model, test_dl, device)
        print(f"epoch {ep:2d}/{epochs}  train_loss={running / len(train_ds):.4f}  test_acc={acc:.3f}")

    acc, yt, yp = _evaluate(model, test_dl, device)
    print(f"\nfinal test accuracy: {acc:.3f}  (band={band or 'denoised'})\n")
    print(classification_report(yt, yp, digits=3, zero_division=0))

    save(model, out_path, meta={
        "in_channels": config.NUM_CHANNELS,
        "seq_len": config.WINDOW_SAMPLES,
        "num_classes": config.NUM_CLASSES,
        "band": band,
        "test_accuracy": acc,
        "channels": config.EPOC_CHANNELS,
    })
    print(f"saved model -> {out_path}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train BrainDigiCNN on MindBigData EPOC digits")
    ap.add_argument("--data", default=str(config.DEFAULT_DATA_PATH), help="MindBigData EP1.01 file")
    ap.add_argument("--synthetic", action="store_true", help="train on the built-in synthetic fixture")
    ap.add_argument("--band", choices=list(config.SUB_BANDS), default=None,
                    help="train a band-wise model instead of the denoised signal")
    ap.add_argument("--max-events", type=int, default=None, help="cap epochs parsed (for quick runs)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--out", default=str(config.DEFAULT_MODEL_PATH))
    args = ap.parse_args()

    if args.synthetic:
        print("loading synthetic fixture ...")
        X, y = synthetic_epochs()
    else:
        print(f"loading MindBigData from {args.data} ...")
        X, y = load_epochs(args.data, max_events=args.max_events)
    print(f"{len(X)} epochs, shape {X.shape[1:]}, class counts {np.bincount(y).tolist()}")

    train(X, y, band=args.band, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, out_path=args.out)


if __name__ == "__main__":
    main()
