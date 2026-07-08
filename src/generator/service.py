"""The Generator service: z in, rendered pyramid frame out.

Backend is picked by config.generator.backend ("procedural" for the
GPU-free fallback / fake-reward loop, "diffusion" for SDXL-Turbo). The
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
    """Linear interpolation between the last two accepted latents, sampled
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
        if self.backend == "diffusion":
            from src.generator.diffusion_pipeline import DiffusionGenerator

            self._diffusion = DiffusionGenerator(
                config.generator.diffusion_model_id,
                num_inference_steps=config.generator.diffusion_steps,
            )
        else:
            self._procedural = ProceduralRenderer()

    def render_image(self, z: np.ndarray) -> Image.Image:
        if self.backend == "diffusion":
            assert self.projector is not None, "diffusion backend needs a projector"
            embedding = self.projector.to_embedding(z)
            return self._diffusion.render(embedding)
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
