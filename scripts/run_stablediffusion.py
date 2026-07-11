#!/usr/bin/env python3
"""Real-time breed-mixture morph server - the render backend for a real-EEG
NoiseAwareLatentTuRBO session, using the EXACT mechanism proven in
scripts/run_poodle_turbo_morph.py (not the PCA-projection mechanism of
scripts/run_streamdiffusion_server.py).

Why this exists (see PLAN_run_stablediffusion.md for the full rationale):
  run_streamdiffusion_server.py parameterizes the morph as
  z -> PCAProjector.to_embedding(z), a projection fit from config.yaml's
  anchor_prompts. That path is more fragile than what the reference scripts
  actually do (stray-anchor bugs, n_anchors-1 rank limits, PCA directions that
  don't map cleanly to one attribute each). This server instead uses the
  run_poodle_turbo_morph.py mechanism directly: z is a len(breeds)-dimensional
  breed-WEIGHT vector, softmax'd, used to simultaneously blend both the prompt
  embeddings AND per-breed fixed noise latents, then a single plain-diffusers
  pipe(prompt_embeds=..., latents=...) call renders the frame. No PCA anywhere.

Wire-compatible with the existing client (scripts/run_real_eeg_optimizer.py):
  same POST /render contract - {"z": [...], "frame_size": ...} in, PNG out.
  The ONLY difference from run_streamdiffusion_server.py is the meaning of z:
  a len(breeds)-dim breed-weight vector here, not an 8-dim PCA point. So set
  config.optimizer.search_dims == len(breeds) before driving this server.

There is deliberately NO /anchors endpoint: there's no PCA projector to re-fit.
To change the breed list, restart with a different --breeds (the same
restart-required rule as every other backend in this codebase).

Pure helper functions below (encode_breed_prompts / make_breed_latents /
softmax_weights / blend_prompt_embeds / blend_noise_latents) are lifted
near-verbatim from run_poodle_turbo_morph.py, where they're already correct and
exercised standalone.

Run on the GPU server:

    python scripts/run_stablediffusion.py \\
        --breeds "Golden Retriever" "German Shepherd" "Siberian Husky" \\
                 "Pembroke Welsh Corgi" "Shiba Inu" "Dalmatian" "Standard Poodle" \\
        --port 8766
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config

# Matches run_poodle_turbo_morph.py's DEFAULT_BREEDS / PROMPT_TEMPLATE exactly,
# so this server's default behavior is validatable against that proven script.
DEFAULT_BREEDS = [
    "Golden Retriever",
    "German Shepherd",
    "Siberian Husky",
    "Pembroke Welsh Corgi",
    "Shiba Inu",
    "Dalmatian",
    "Standard Poodle",
]

PROMPT_TEMPLATE = (
    "centered studio portrait photograph of a {breed} dog, "
    "head and shoulders, looking directly at the camera, "
    "same neutral gray background, soft even lighting, "
    "symmetrical composition, highly detailed realistic fur"
)


# --------------------------------------------------------------------------- #
# Pure rendering helpers - lifted from run_poodle_turbo_morph.py.
# --------------------------------------------------------------------------- #
def encode_breed_prompts(pipe, breeds: Sequence[str], template: str, device: str):
    import torch

    prompts = [template.format(breed=breed) for breed in breeds]
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return prompt_embeds


def random_latent(generator, device: str, dtype, size: int, channels: int, vae_scale_factor: int):
    import torch

    shape = (1, channels, size // vae_scale_factor, size // vae_scale_factor)
    # Generate on CPU for deterministic behavior across cuda/mps, then move.
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def make_breed_latents(n: int, generator, device: str, dtype, size: int, channels: int, vae_scale_factor: int):
    """One fixed seed-derived noise tensor per breed, kept fixed for the session."""
    import torch

    return torch.cat(
        [random_latent(generator, device, dtype, size, channels, vae_scale_factor) for _ in range(n)],
        dim=0,
    )


def softmax_weights(z: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    x = np.asarray(z, dtype=float) * temperature
    x = x - x.max()
    w = np.exp(x)
    return w / max(float(w.sum()), 1e-12)


def blend_prompt_embeds(breed_embeds, weights: np.ndarray):
    import torch

    w = torch.tensor(weights, device=breed_embeds.device, dtype=torch.float32)
    blended = torch.sum(breed_embeds.float() * w[:, None, None], dim=0, keepdim=True)
    return blended.to(dtype=breed_embeds.dtype)


def blend_noise_latents(breed_latents, weights: np.ndarray):
    """Weighted latent blend, renormalized to preserve a plausible noise norm."""
    import torch

    w = torch.tensor(weights, device=breed_latents.device, dtype=torch.float32)
    blended = torch.sum(breed_latents.float() * w[:, None, None, None], dim=0, keepdim=True)
    target_norm = torch.sum(
        torch.linalg.vector_norm(breed_latents.float().reshape(breed_latents.shape[0], -1), dim=1) * w
    )
    current_norm = torch.linalg.vector_norm(blended.reshape(1, -1), dim=1).clamp_min(1e-8)
    blended = blended * (target_norm / current_norm).reshape(1, 1, 1, 1)
    return blended.to(dtype=breed_latents.dtype)


def top_breeds(breeds: Sequence[str], weights: np.ndarray, n: int = 3) -> str:
    order = np.argsort(weights)[::-1][:n]
    return " | ".join(f"{breeds[i]} {weights[i]:.2f}" for i in order)


# --------------------------------------------------------------------------- #
# Server.
# --------------------------------------------------------------------------- #
class BreedMorphRenderServer:
    def __init__(
        self,
        pipe,
        breeds: list[str],
        breed_embeds,
        breed_latents,
        frame_size: int,
        device: str,
        dtype,
        num_inference_steps: int,
        guidance_scale: float,
        softmax_temperature: float,
        log_weights_every: int,
    ):
        self.pipe = pipe
        self.breeds = breeds
        self.breed_embeds = breed_embeds
        self.breed_latents = breed_latents
        self.frame_size = frame_size
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.softmax_temperature = softmax_temperature
        self.log_weights_every = max(0, log_weights_every)
        self._render_count = 0
        self.lock = threading.Lock()

    def render_png(self, payload: dict) -> bytes:
        import torch

        frame_size = int(payload.get("frame_size", self.frame_size))
        z = payload.get("z")
        if z is None:
            raise RuntimeError("payload needs a z vector (a len(breeds)-dim breed-weight vector)")
        z = np.asarray(z, dtype=float)
        if z.shape != (len(self.breeds),):
            raise RuntimeError(
                f"z has shape {z.shape}, expected ({len(self.breeds)},) - one weight per breed. "
                f"Set config.optimizer.search_dims == {len(self.breeds)} to match --breeds."
            )

        with self.lock:
            weights = softmax_weights(z, temperature=self.softmax_temperature)
            self._render_count += 1
            if self.log_weights_every and self._render_count % self.log_weights_every == 0:
                print(f"[breed-morph-server] render={self._render_count} top={top_breeds(self.breeds, weights)}")
            prompt_embeds = blend_prompt_embeds(self.breed_embeds, weights)
            latents = blend_noise_latents(self.breed_latents, weights)
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

        if image.size != (frame_size, frame_size):
            image = image.resize((frame_size, frame_size))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def make_handler(render_server: BreedMorphRenderServer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/render":
                self.send_error(404, "expected POST /render (this server has no /anchors endpoint)")
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                png = render_server.render_png(payload)
            except Exception as exc:  # noqa: BLE001 - report server-side failures to the client.
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

            self.send_response(200)
            self.send_header("content-type", "image/png")
            self.send_header("content-length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[breed-morph-server] {self.address_string()} - {fmt % args}")

    return Handler


def load_pipeline(model_id: str, device: str | None = None):
    """Load a plain diffusers.StableDiffusionPipeline. Mirrors
    run_streamdiffusion_server.py's load_pipeline() (minus the fixed latent -
    this server builds per-breed latents in make_breed_latents instead).
    Returns (pipe, device, dtype).
    """
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

    print(f"[breed-morph-server] loading {model_id} on {device} ({dtype})...")
    kwargs: dict = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except Exception:
        # sd-turbo (and some other repos) don't ship an fp16 variant under that name.
        kwargs.pop("variant", None)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe, device, dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", default="stabilityai/sd-turbo",
                        help="e.g. stabilityai/sd-turbo (default, 1 step, no CFG), or a plain SD "
                             "checkpoint - pair a non-turbo model with --steps 20-50 / --guidance-scale 7.5")
    parser.add_argument("--breeds", nargs="+", default=DEFAULT_BREEDS,
                        help="breed list - REPLACES config.yaml anchor_prompts entirely. "
                             "config.optimizer.search_dims MUST equal len(breeds).")
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE,
                        help="template with a single {breed} placeholder, applied to each --breeds value")
    parser.add_argument("--size", type=int, default=None,
                        help="square render size, divisible by 8 (default: config.generator.frame_size)")
    parser.add_argument("--steps", type=int, default=1,
                        help="denoising steps - 1-4 for sd-turbo, 20-50 for a plain SD checkpoint")
    parser.add_argument("--guidance-scale", type=float, default=0.0,
                        help="0.0 (default, no CFG) for turbo/LCM. ~7-8 for a plain SD checkpoint.")
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="softmax temperature for z -> breed weights. Higher values make the same "
                             "optimizer movement snap more strongly toward one breed/species; lower values "
                             "produce smoother but more uniform blends.")
    parser.add_argument("--log-weights-every", type=int, default=12,
                        help="print the top breed weights every N renders; 0 disables logging.")
    parser.add_argument("--seed", type=int, default=None,
                        help="per-breed latent seed (default: config.generator.remote_diffusion_seed)")
    args = parser.parse_args()

    if "{breed}" not in args.prompt_template:
        sys.exit("[breed-morph-server] --prompt-template must contain a literal {breed} placeholder")
    if len(args.breeds) < 2:
        sys.exit("[breed-morph-server] provide at least two --breeds")

    config = Config.load()
    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed
    frame_size = args.size if args.size is not None else config.generator.frame_size
    if frame_size < 256 or frame_size % 8 != 0:
        sys.exit("[breed-morph-server] --size must be at least 256 and divisible by 8")

    if config.optimizer.search_dims != len(args.breeds):
        print(f"[breed-morph-server] WARNING: config.optimizer.search_dims="
              f"{config.optimizer.search_dims} but len(breeds)={len(args.breeds)}. "
              f"Set optimizer.search_dims: {len(args.breeds)} in config.yaml, or the client "
              f"will POST a wrongly-sized z and /render will reject it.")

    import torch

    pipe, device, dtype = load_pipeline(args.model, device=None)

    print(f"[breed-morph-server] encoding {len(args.breeds)} breed anchor(s) as one batch...")
    breed_embeds = encode_breed_prompts(pipe, args.breeds, args.prompt_template, device)

    latent_channels = int(pipe.unet.config.in_channels)
    vae_scale_factor = int(pipe.vae_scale_factor)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    breed_latents = make_breed_latents(
        len(args.breeds), generator, device, dtype, frame_size, latent_channels, vae_scale_factor
    )

    render_server = BreedMorphRenderServer(
        pipe, list(args.breeds), breed_embeds, breed_latents, frame_size,
        device, dtype, args.steps, args.guidance_scale, args.temperature, args.log_weights_every,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(render_server))
    print(f"[breed-morph-server] listening on http://{args.host}:{args.port} "
          f"(breeds={len(args.breeds)}, seed={seed}, size={frame_size}, "
          f"steps={args.steps}, guidance_scale={args.guidance_scale}, "
          f"temperature={args.temperature})")
    print(f"[breed-morph-server] breeds: {', '.join(args.breeds)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
