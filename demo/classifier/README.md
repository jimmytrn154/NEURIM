# EEG → Digit (0–9) Classifier — EMOTIV EPOC X

A faithful re-implementation of **BrainDigiCNN**, the 1D-CNN from:

> S. Tiwari, S. Goel, A. Bhardwaj, *"EEG Signals to Digit Classification Using
> Deep Learning-Based One-Dimensional Convolutional Neural Network,"* Arabian
> Journal for Science and Engineering, 48:9675–9691 (2023).

It trains on the public **MindBigData** "EPOC" digit dataset and runs **windowed
real-time inference** on an EMOTIV EPOC X via the Cortex API. Runs fully offline
on a synthetic fixture too, so you can see the whole pipeline work without a
headset or the 2 GB dataset.

## What matches the paper

| Paper | Here |
|---|---|
| EMOTIV EPOC, 14 ch, 128 Hz, 2 s epoch (256 samples) | [config.py](config.py) |
| 5th-order low-pass Butterworth (45 Hz) + 50 Hz notch | `preprocessing.denoise` |
| 6 Butterworth band-pass sub-bands (δ/θ/α/β-low/β-high/γ) | `preprocessing.bandpass`, `--band` |
| EMD → Hilbert-Huang (IA/IP/IF) features | `preprocessing.hht_features` (optional, offline) |
| 4× Conv1D(256/128/64/32, k=7)→BN→ReLU→MaxPool2 → Dense 128→64→10 | [model.py](model.py) `BrainDigiCNN` |
| Adam, lr 1e-3, categorical cross-entropy, batch 32, 70/30 split | [train.py](train.py) |

**Two deliberate engineering choices** (documented, not hidden):

1. **PyTorch, not Keras.** The rest of NEURIM is PyTorch; the architecture,
   layer sizes, and hyperparameters are identical to the paper's Table 4/5.
   (Softmax is folded into `CrossEntropyLoss`.)
2. **The live path uses the denoised signal, not EMD/HHT.** Empirical Mode
   Decomposition is iterative and takes seconds per channel — impractical for
   real-time. The default feature is the low-pass+notch denoised, z-scored
   14-channel window, which the same architecture consumes. The full EMD/HHT
   feature extractor is implemented (`preprocessing.hht_features`) for offline /
   band-wise experiments and is gated behind the optional `EMD-signal` package.

## Install

Dependencies are already in the NEURIM `.venv` (torch, scipy, numpy, sklearn,
websocket-client). Otherwise:

```bash
pip install -r demo/classifier/requirements.txt
```

## Quick start (no headset, no dataset)

```bash
# 1. Train on the built-in synthetic fixture (writes artifacts/braindigicnn.pt)
python -m demo.classifier.train --synthetic --epochs 8

# 2. Classify a synthetic live stream (imagined digit cycles every ~3 s)
python -m demo.classifier.realtime --source synthetic --stride 0.5
```

## Train on real MindBigData

Download the `EP1.01` EPOC text dump from <http://mindbigdata.com/opendb/> and
place it at `demo/classifier/data/EP1.01.txt`, then:

```bash
python -m demo.classifier.train --data demo/classifier/data/EP1.01.txt
python -m demo.classifier.train --data ... --band delta   # band-wise model
```

## Real-time on the EPOC X

Requires EMOTIV Launcher running with the headset paired, and
`EMOTIV_CLIENT_ID` / `EMOTIV_CLIENT_SECRET` (optionally `EMOTIV_HEADSET_ID`) in
the environment — the same setup the rest of NEURIM uses.

```bash
python -m demo.classifier.realtime --source cortex --stride 0.5
```

It keeps a sliding 2 s window of the 14 EEG channels and prints the decoded
digit + confidence every `--stride` seconds.

## Files

- `config.py` — channels, rates, filter/band and training hyperparameters
- `preprocessing.py` — Butterworth denoise, notch, sub-bands, optional EMD/HHT
- `braindigidata.py` — MindBigData EPOC parser + synthetic fixture
- `model.py` — `BrainDigiCNN` + save/load
- `dataset.py` / `train.py` — torch dataset and training loop
- `infer.py` — `DigitClassifier`: raw epoch → digit + probabilities
- `realtime.py` — sliding-window live classifier (Cortex + synthetic sources)
- `tests/test_pipeline.py` — shape/smoke tests (`pytest demo/classifier/tests`)

## Caveats

Digit-from-EEG is a genuinely hard problem — the paper's ~98 % is on a single
device/lab; independent replications on MindBigData report far lower numbers.
The **synthetic** fixture is class-separable by construction (a smoke-test
harness), so its 100 % accuracy says nothing about real neural decodability.
Expect real-data accuracy to be modest and subject-dependent.
