"""Manifest-driven diffusion render server."""

from .http import DiffusionServer, make_handler
from .renderer import (
    AnchorMorphRenderer,
    blend_noise_latents,
    blend_prompt_embeds,
    softmax_weights,
    top_anchors,
)

__all__ = [
    "AnchorMorphRenderer",
    "DiffusionServer",
    "blend_noise_latents",
    "blend_prompt_embeds",
    "make_handler",
    "softmax_weights",
    "top_anchors",
]
