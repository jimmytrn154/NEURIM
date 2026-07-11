"""Diffusion pipeline loading and manifest-anchor rendering."""

from __future__ import annotations

import threading
from io import BytesIO
from typing import Sequence

import numpy as np


def encode_anchor_prompts(pipe, prompts: Sequence[str], device: str):
    import torch

    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=list(prompts),
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return prompt_embeds


def random_latent(generator, device: str, dtype, size: int, channels: int, vae_scale_factor: int):
    import torch

    shape = (1, channels, size // vae_scale_factor, size // vae_scale_factor)
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def make_anchor_latents(
    n: int,
    generator,
    device: str,
    dtype,
    size: int,
    channels: int,
    vae_scale_factor: int,
):
    import torch

    return torch.cat(
        [random_latent(generator, device, dtype, size, channels, vae_scale_factor) for _ in range(n)],
        dim=0,
    )


def make_cpu_generator(seed: int):
    import torch

    return torch.Generator(device="cpu").manual_seed(seed)


def softmax_weights(z: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    values = np.asarray(z, dtype=float) * temperature
    values = values - values.max()
    weights = np.exp(values)
    return weights / max(float(weights.sum()), 1e-12)


def blend_prompt_embeds(anchor_embeds, weights: np.ndarray):
    import torch

    tensor_weights = torch.tensor(weights, device=anchor_embeds.device, dtype=torch.float32)
    blended = torch.sum(anchor_embeds.float() * tensor_weights[:, None, None], dim=0, keepdim=True)
    return blended.to(dtype=anchor_embeds.dtype)


def blend_noise_latents(anchor_latents, weights: np.ndarray):
    import torch

    tensor_weights = torch.tensor(weights, device=anchor_latents.device, dtype=torch.float32)
    blended = torch.sum(
        anchor_latents.float() * tensor_weights[:, None, None, None], dim=0, keepdim=True
    )
    target_norm = torch.sum(
        torch.linalg.vector_norm(anchor_latents.float().reshape(anchor_latents.shape[0], -1), dim=1)
        * tensor_weights
    )
    current_norm = torch.linalg.vector_norm(blended.reshape(1, -1), dim=1).clamp_min(1e-8)
    blended = blended * (target_norm / current_norm).reshape(1, 1, 1, 1)
    return blended.to(dtype=anchor_latents.dtype)


def top_anchors(anchor_labels: Sequence[str], weights: np.ndarray, n: int = 3) -> str:
    order = np.argsort(weights)[::-1][:n]
    return " | ".join(f"{anchor_labels[index]} {weights[index]:.2f}" for index in order)


def load_pipeline(model_id: str, device: str | None = None):
    import torch
    from diffusers import StableDiffusionPipeline

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print(f"[anchor-morph-server] loading {model_id} on {device} ({dtype})...")
    kwargs: dict = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except Exception:
        kwargs.pop("variant", None)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe, device, dtype


class AnchorMorphRenderer:
    def __init__(
        self,
        pipe,
        anchor_labels: list[str],
        anchor_prompts: list[str],
        anchor_embeds,
        anchor_latents,
        frame_size: int,
        device: str,
        dtype,
        num_inference_steps: int,
        guidance_scale: float,
        softmax_temperature: float,
        log_weights_every: int,
        target_anchor: str | None,
        manifest: dict | None = None,
    ) -> None:
        self.pipe = pipe
        self.anchor_labels = anchor_labels
        self.anchor_prompts = anchor_prompts
        self.target_anchor = target_anchor
        self.target_index = anchor_labels.index(target_anchor) if target_anchor is not None else None
        self.anchor_embeds = anchor_embeds
        self.anchor_latents = anchor_latents
        self.frame_size = frame_size
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.softmax_temperature = softmax_temperature
        self.log_weights_every = max(0, log_weights_every)
        self.manifest = manifest or {
            "anchor_count": len(anchor_labels),
            "anchor_labels": list(anchor_labels),
        }
        self._render_count = 0
        self._lock = threading.Lock()

    def render_png(self, payload: dict) -> bytes:
        requested_size = int(payload.get("frame_size", self.frame_size))
        z = payload.get("z")
        if z is None:
            raise RuntimeError("payload needs a z vector (a len(anchors)-dim anchor-weight vector)")
        z_array = np.asarray(z, dtype=float)
        if z_array.shape != (len(self.anchor_labels),):
            raise RuntimeError(
                f"z has shape {z_array.shape}, expected ({len(self.anchor_labels)},) - one weight per anchor. "
                f"Set config.optimizer.search_dims == {len(self.anchor_labels)} to match the session manifest."
            )

        import torch

        with self._lock:
            weights = softmax_weights(z_array, temperature=self.softmax_temperature)
            self._render_count += 1
            self._log_weights(weights)
            prompt_embeds = blend_prompt_embeds(self.anchor_embeds, weights)
            latents = blend_noise_latents(self.anchor_latents, weights)
            with torch.inference_mode():
                image = self.pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    latents=latents,
                    height=self.frame_size,
                    width=self.frame_size,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    output_type="pil",
                ).images[0]

        if image.size != (requested_size, requested_size):
            image = image.resize((requested_size, requested_size))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _log_weights(self, weights: np.ndarray) -> None:
        if not self.log_weights_every or self._render_count % self.log_weights_every != 0:
            return
        target = ""
        if self.target_index is not None and self.target_anchor is not None:
            target = f" target={self.target_anchor} {weights[self.target_index]:.2f}"
        print(
            f"[anchor-morph-server] render={self._render_count} "
            f"top={top_anchors(self.anchor_labels, weights)}{target}"
        )


# Compatibility name for integrations that imported the old script class.
AnchorMorphRenderServer = AnchorMorphRenderer
