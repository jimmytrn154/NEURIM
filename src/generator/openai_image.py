"""OpenAI Image API renderer.

This backend turns the optimizer's selected anchor prompt into a generated
image through OpenAI's Images API. Generated images are cached by prompt because
the orchestrator render loop may request the same prompt many times per second.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image


class OpenAIImageGenerator:
    def __init__(
        self,
        model: str = "gpt-image-2",
        size: str = "1024x1024",
        quality: str = "low",
        output_format: str = "png",
        frame_size: int = 512,
        client: Any | None = None,
    ):
        if client is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            from openai import OpenAI

            client = OpenAI()

        self.client = client
        self.model = model
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.frame_size = frame_size
        self._cache: dict[str, Image.Image] = {}

    def render_prompt(self, prompt: str) -> Image.Image:
        if prompt in self._cache:
            return self._cache[prompt].copy()

        result = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            quality=self.quality,
            output_format=self.output_format,
        )
        image_b64 = result.data[0].b64_json
        if not image_b64:
            raise RuntimeError("OpenAI image generation response did not include b64_json")

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.size != (self.frame_size, self.frame_size):
            image = image.resize((self.frame_size, self.frame_size), Image.Resampling.LANCZOS)

        self._cache[prompt] = image
        return image.copy()
