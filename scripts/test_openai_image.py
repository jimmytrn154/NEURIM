#!/usr/bin/env python3
"""Standalone sanity check for the OpenAI image generator backend.

Needs OPENAI_API_KEY in the environment or .env.
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
    from src.common.config import Config
    from src.generator.openai_image import OpenAIImageGenerator

    config = Config.load()
    generator = OpenAIImageGenerator(
        model=config.generator.openai_image_model,
        size=config.generator.openai_image_size,
        quality=config.generator.openai_image_quality,
        output_format=config.generator.openai_image_output_format,
        frame_size=config.generator.frame_size,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, prompt in PROMPTS:
        t0 = time.monotonic()
        image = generator.render_prompt(prompt)
        elapsed = time.monotonic() - t0
        out_path = OUT_DIR / f"{name}.png"
        image.save(out_path)
        print(f"[test-openai-image] '{prompt}' -> {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
