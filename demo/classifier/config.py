"""Configuration for the EEG-to-digit classifier.

Constants follow the BrainDigiCNN paper (Tiwari, Goel, Bhardwaj, "EEG Signals to
Digit Classification Using Deep Learning-Based One-Dimensional Convolutional
Neural Network", Arabian J. Sci. Eng. 2023) and the EMOTIV EPOC device geometry
used by the public MindBigData (MBD) "EPOC" subset that we train on.
"""

from __future__ import annotations

from pathlib import Path

# --- EPOC hardware / dataset geometry --------------------------------------
# 14 EEG sensors in the exact order MindBigData emits them for the "EP" device.
# This is also the channel order the EMOTIV Cortex API reports for an EPOC X.
EPOC_CHANNELS: list[str] = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
NUM_CHANNELS = len(EPOC_CHANNELS)

SAMPLE_RATE_HZ = 128.0        # EPOC/EPOC X effective sampling rate
WINDOW_SECONDS = 2.0          # paper records 2 s of "thinking" per digit
WINDOW_SAMPLES = int(round(SAMPLE_RATE_HZ * WINDOW_SECONDS))  # 256
NUM_CLASSES = 10              # digits 0-9

# --- signal-processing parameters (paper Table 3) --------------------------
LOWPASS_HZ = 45.0             # 5th-order Butterworth denoise cutoff
BUTTER_ORDER = 5
NOTCH_HZ = 50.0              # power-line notch
NOTCH_Q = 30.0

# Six Butterworth band-pass sub-bands (Hz). Used by the optional band-wise and
# EMD/HHT feature paths; the default real-time path uses the denoised signal.
SUB_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta_low": (12.0, 16.0),
    "beta_high": (16.0, 25.0),
    "gamma": (25.0, 45.0),
}

# --- training hyperparameters (paper Table 5) ------------------------------
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 20
CONV_KERNEL = 7
CONV_FILTERS = (256, 128, 64, 32)
DENSE_UNITS = (128, 64)

# --- paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "artifacts" / "braindigicnn.pt"
DEFAULT_DATA_PATH = HERE / "data" / "EP1.01.txt"  # MindBigData EPOC dump
