#!/usr/bin/env python3
"""Standalone sanity check for the diffusion generator backend.

Needs the heavy deps: pip install -r requirements-diffusion.txt
First run downloads the SDXL-Turbo weights from Hugging Face.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

PROMPTS = [
    ("test_dog", "a photo of a dog"),
    ("test_cat", "a photo of a cat"),
]


def main() -> None:
    from src.generator.diffusion_pipeline import DiffusionGenerator

    print("[test-diffusion] loading SDXL-Turbo...")
    t0 = time.monotonic()
    generator = DiffusionGenerator(num_inference_steps=4)
    print(f"[test-diffusion] device={generator.device} dtype={generator.dtype} "
          f"loaded in {time.monotonic() - t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, prompt in PROMPTS:
        t0 = time.monotonic()
        image = generator.render_from_prompt(prompt)
        elapsed = time.monotonic() - t0
        out_path = OUT_DIR / f"{name}.png"
        image.save(out_path)
        print(f"[test-diffusion] '{prompt}' -> {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
