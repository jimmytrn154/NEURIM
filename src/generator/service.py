"""The Generator service: z in, rendered pyramid frame out.

Backend is picked by config.generator.backend ("procedural" for the
GPU-free fallback / fake-reward loop, "diffusion" for SDXL-Turbo, "openai"
for the OpenAI Image API). The
Optimizer only emits one z every 1-2s, so the Orchestrator interpolates
between accepted latents and calls render() at 30-60fps for a smooth morph -
see Interpolator below.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from src.common.config import Config
from src.common.messages import FrameMessage
from src.generator.procedural import ProceduralRenderer
from src.generator.to_3d import mirrored_quadrants


def _encode_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_jpeg(image: Image.Image, quality: int = 85) -> str:
    """JPEG is 5-10x smaller than PNG and decodes faster in the browser.
    Forces RGB conversion to handle any RGBA input (e.g. pyramid mode).
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Interpolator:
    """ -- Morphing Process (animation)
    Linear interpolation between the last two accepted latents, sampled
    continuously so the morph renders at target_fps between optimizer steps
    that only arrive every 1-2s.
    """
    def __init__(self):
        self.z_start = None
        self.z_end = None

    def set_target(self, z_end: np.ndarray) -> None:
        self.z_start = self.z_end if self.z_end is not None else z_end
        self.z_end = z_end

    def sample(self, alpha: float) -> np.ndarray:
        if self.z_end is None:
            raise RuntimeError("call set_target() before sample()")
        if self.z_start is None:
            return self.z_end
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return self.z_start + (self.z_end - self.z_start) * alpha


class GeneratorService:
    def __init__(self, config: Config, projector=None):
        self.config = config
        self.backend = config.generator.backend
        self.projector = projector
        self.anchor_prompts = config.generator.anchor_prompts
        if self.backend == "diffusion":
            from src.generator.diffusion_pipeline import DiffusionGenerator

            self._diffusion = DiffusionGenerator(
                config.generator.diffusion_model_id,
                num_inference_steps=config.generator.diffusion_steps,
            )
        elif self.backend == "openai":
            from src.generator.openai_image import OpenAIImageGenerator

            self._openai = OpenAIImageGenerator(
                model=config.generator.openai_image_model,
                size=config.generator.openai_image_size,
                quality=config.generator.openai_image_quality,
                output_format=config.generator.openai_image_output_format,
                frame_size=config.generator.frame_size,
            )
        else:
            self._procedural = ProceduralRenderer()

    def _anchor_prompt_for(self, z: np.ndarray) -> str:
        prompts = self.anchor_prompts or ["a little brown puppy"]
        weights = np.asarray(z[: len(prompts)], dtype=float)
        if weights.size == 0:
            return prompts[0]
        return prompts[int(np.argmax(weights))]

    def render_image(self, z: np.ndarray) -> Image.Image:
        if self.backend == "diffusion":
            if self.projector is not None:
                embedding = self.projector.to_embedding(z)
                return self._diffusion.render(embedding)
            return self._diffusion.render_prompt(self._anchor_prompt_for(z))
        if self.backend == "openai":
            return self._openai.render_prompt(self._anchor_prompt_for(z))
        return self._procedural.render(z, size=self.config.generator.frame_size)

    def render(
        self,
        z: np.ndarray,
        step_index: int,
        as_pyramid: bool = False,
        state: str = "explore",
        reward_estimate: float = 0.0,
    ) -> FrameMessage:
        image = self.render_image(z)
        if as_pyramid:
            image = mirrored_quadrants(image, self.config.generator.frame_size)
        return FrameMessage(
            frame_b64=_encode_jpeg(image),
            z=list(map(float, z)),
            step_index=step_index,
            format="jpeg",
            state=state,
            reward_estimate=reward_estimate,
        )
