#!/usr/bin/env python3
"""Real-time morph server: plain diffusers.StableDiffusionPipeline, a fresh
render every frame from a FIXED seeded noise latent + an interpolated text
embedding. No StreamDiffusion, no img2img, no undocumented internals.

HISTORY: this file used to wrap cumulo-autumn/StreamDiffusion (overwriting its
internal stream.prompt_embeds attribute, running an img2img feedback loop
against the previous frame, and requiring a t_index_list tuned by trial and
error - see README.md's Milestone section and KNOWN_ISSUES.md items 3/7 for
that whole saga). Replaced after scripts/continuous_dog_latent_morph.py
demonstrated a simpler, OFFICIALLY-SUPPORTED technique that needs none of it:
pass an explicit `latents=` tensor and `prompt_embeds=` directly to a plain
diffusers pipeline, fresh every frame - no image-to-image feedback at all.

Why this is both smoother and simpler:
  - No img2img restyling-strength tuning (no t_index_list): every frame is an
    independent, from-scratch denoise, so there's no "how much of the previous
    frame should survive" question to tune at all.
  - No random-noise flicker: the noise latent is generated ONCE (seeded) at
    startup and reused unchanged for every frame; only the text embedding -
    from the existing PCAProjector/anchor-prompt machinery, completely
    unchanged - varies frame to frame. A smoothly-varying embedding into a
    FIXED noise latent is enough for smooth output; we don't need to also
    SLERP the noise the way continuous_dog_latent_morph.py does, because our
    axis of variation already comes from the optimizer's z, not a scripted
    breed-to-breed sequence with no other control signal.
  - Uses diffusers' OFFICIAL `latents=`/`prompt_embeds=` API - no undocumented
    internal attributes, no utils/wrapper.py sys.path hack, and no separate
    isolated venv beyond what run_diffusion_server.py already needs (plain
    torch+diffusers+transformers - this can run in torch-env; the old
    streamdiffusion-env's StreamDiffusion-specific pins are no longer needed).

Wire-compatible with the existing client: same POST /render contract (z in,
PNG out) as before - src/generator/remote_diffusion.py needs no changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config
from src.optimizer.projection import PCAProjector


class LatentWalkRenderServer:
    def __init__(
        self,
        pipe,
        projector: PCAProjector,
        prompt_embed_shape: tuple[int, int],
        frame_size: int,
        dims: int,
        fixed_latent,
        device: str,
        dtype,
        num_inference_steps: int,
        guidance_scale: float,
    ):
        self.pipe = pipe
        self.projector = projector
        self.seq_len, self.hidden = prompt_embed_shape
        self.frame_size = frame_size
        self.dims = dims
        self.anchor_prompts: list[str] = []
        self.lock = threading.Lock()
        self.fixed_latent = fixed_latent
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def _embedding_to_tensor(self, z: np.ndarray):
        import torch

        embedding = self.projector.to_embedding(z)
        vec = torch.tensor(embedding, dtype=self.dtype, device=self.device)
        return vec.reshape(1, self.seq_len, self.hidden)

    def render_png(self, payload: dict) -> bytes:
        frame_size = int(payload.get("frame_size", self.frame_size))
        z = payload.get("z")
        prompt = payload.get("prompt")
        with self.lock:
            if z is not None:
                prompt_embeds = self._embedding_to_tensor(np.asarray(z, dtype=float))
                image = self.pipe(
                    prompt_embeds=prompt_embeds,
                    latents=self.fixed_latent,
                    height=self.frame_size,
                    width=self.frame_size,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    output_type="pil",
                ).images[0]
            elif prompt is not None:
                image = self.pipe(
                    prompt=prompt,
                    latents=self.fixed_latent,
                    height=self.frame_size,
                    width=self.frame_size,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    output_type="pil",
                ).images[0]
            else:
                raise RuntimeError("payload needs either a z vector or a prompt string")
        if image.size != (frame_size, frame_size):
            image = image.resize((frame_size, frame_size))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def set_anchor_prompts(self, prompts: list[str]) -> None:
        cleaned = [prompt.strip() for prompt in prompts if prompt.strip()]
        if len(cleaned) < 2:
            raise ValueError("this server requires at least two anchor prompts")
        with self.lock:
            self.projector, shape = _fit_projector(self.pipe, cleaned, self.dims, self.device)
            self.seq_len, self.hidden = shape
            self.anchor_prompts = cleaned


def make_handler(render_server: LatentWalkRenderServer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path not in {"/render", "/anchors"}:
                self.send_error(404, "expected POST /render or POST /anchors")
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/anchors":
                    prompts = payload.get("anchor_prompts", [])
                    if not isinstance(prompts, list):
                        raise ValueError("anchor_prompts must be a list")
                    render_server.set_anchor_prompts([str(prompt) for prompt in prompts])
                    self._send_json({"ok": True, "count": len(prompts)})
                    return
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
            print(f"[latent-walk-server] {self.address_string()} - {fmt % args}")

    return Handler


def load_pipeline(model_id: str, seed: int, frame_size: int, device: str | None = None):
    """Load a plain diffusers pipeline plus a fixed, seeded noise latent sized
    for frame_size. Mirrors scripts/continuous_dog_latent_morph.py's
    select_device()/load_pipeline()/random_latent() helpers.
    Returns (pipe, fixed_latent, device, dtype).
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

    print(f"[latent-walk-server] loading {model_id} on {device} ({dtype})...")
    kwargs: dict = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except Exception:
        # Some compatible repos (e.g. sd-turbo itself, in some snapshots) don't
        # ship an fp16 variant under that exact name.
        kwargs.pop("variant", None)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    latent_channels = int(pipe.unet.config.in_channels)
    vae_scale_factor = int(pipe.vae_scale_factor)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, latent_channels, frame_size // vae_scale_factor, frame_size // vae_scale_factor)
    # Generate on CPU for deterministic behavior across cuda/mps, then move -
    # same pattern as continuous_dog_latent_morph.py's random_latent().
    fixed_latent = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    return pipe, fixed_latent, device, dtype


def _fit_projector(pipe, anchor_prompts: list[str], dims: int, device: str) -> tuple[PCAProjector, tuple[int, int]]:
    if len(anchor_prompts) < 2:
        print(f"[latent-walk-server] WARNING: only {len(anchor_prompts)} anchor prompt(s) - "
              "the morph will be STATIC (projector subspace is degenerate).")
    import torch

    embeddings = []
    shape = None
    print(f"[latent-walk-server] encoding {len(anchor_prompts)} anchor prompt(s) and fitting projector...")
    with torch.no_grad():
        for prompt in anchor_prompts:
            out = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                      do_classifier_free_guidance=False)
            prompt_embeds = out[0] if isinstance(out, tuple) else out
            shape = (int(prompt_embeds.shape[-2]), int(prompt_embeds.shape[-1]))
            embeddings.append(prompt_embeds.flatten().float().cpu().numpy())
    embeddings = np.stack(embeddings).astype(np.float32)
    projector = PCAProjector(dims=dims).fit(embeddings)
    print(f"[latent-walk-server] projector fit: {dims}-dim search space over "
          f"{embeddings.shape[0]} prompts (per-prompt shape {shape})")
    return projector, shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model-id", default="stabilityai/sd-turbo",
                         help="e.g. stabilityai/sd-turbo (default, 1 step, no CFG), or a plain SD "
                              "checkpoint like runwayml/stable-diffusion-v1-5 - pair a non-turbo "
                              "model with --steps 20-50 and --guidance-scale 7.5")
    parser.add_argument("--steps", type=int, default=1,
                         help="denoising steps - 1-4 for sd-turbo, 20-50 for a plain SD checkpoint")
    parser.add_argument("--guidance-scale", type=float, default=0.0,
                         help="0.0 (default, no CFG) for turbo/LCM. ~7-8 for a plain SD checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="defaults to config.generator.remote_diffusion_seed")
    args = parser.parse_args()

    config = Config.load()
    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed
    frame_size = config.generator.frame_size

    pipe, fixed_latent, device, dtype = load_pipeline(args.model_id, seed, frame_size)
    projector, embed_shape = _fit_projector(pipe, config.generator.anchor_prompts, config.optimizer.search_dims, device)
    render_server = LatentWalkRenderServer(
        pipe, projector, embed_shape, frame_size, config.optimizer.search_dims,
        fixed_latent, device, dtype, args.steps, args.guidance_scale,
    )
    render_server.anchor_prompts = list(config.generator.anchor_prompts)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(render_server))
    print(f"[latent-walk-server] listening on http://{args.host}:{args.port} "
          f"(seed={seed}, steps={args.steps}, guidance_scale={args.guidance_scale})")
    server.serve_forever()


if __name__ == "__main__":
    main()
