#!/usr/bin/env python3
"""Real-time generalized anchor-mixture morph server.

This is the manifest-driven counterpart to scripts/run_stablediffusion.py.
Instead of hardcoded dog breeds, it consumes a curated 7-anchor prompt-session
manifest from scripts/run_prompt_curation.py and interprets incoming z as one
logit per anchor. The server stays wire-compatible with the existing optimizer
clients: POST /render still accepts {"z": [...], "frame_size": ...} and returns
PNG bytes.

Typical flow:

    python scripts/run_prompt_curation.py \
        --user-prompt "world cup players" \
        --out data/processed/prompt_sessions/world_cup_players.json

    python scripts/run_general_stable_diffusion.py \
        --session-manifest data/processed/prompt_sessions/world_cup_players.json \
        --port 8766

    python scripts/run_real_eeg_optimizer.py --server-url http://localhost:8766
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
from src.generator.anchor_session import load_prompt_session_manifest, manifest_metadata


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


def make_anchor_latents(n: int, generator, device: str, dtype, size: int, channels: int, vae_scale_factor: int):
    import torch

    return torch.cat(
        [random_latent(generator, device, dtype, size, channels, vae_scale_factor) for _ in range(n)],
        dim=0,
    )


def make_cpu_generator(seed: int):
    import torch

    return torch.Generator(device="cpu").manual_seed(seed)


def softmax_weights(z: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    x = np.asarray(z, dtype=float) * temperature
    x = x - x.max()
    w = np.exp(x)
    return w / max(float(w.sum()), 1e-12)


def blend_prompt_embeds(anchor_embeds, weights: np.ndarray):
    import torch

    w = torch.tensor(weights, device=anchor_embeds.device, dtype=torch.float32)
    blended = torch.sum(anchor_embeds.float() * w[:, None, None], dim=0, keepdim=True)
    return blended.to(dtype=anchor_embeds.dtype)


def blend_noise_latents(anchor_latents, weights: np.ndarray):
    import torch

    w = torch.tensor(weights, device=anchor_latents.device, dtype=torch.float32)
    blended = torch.sum(anchor_latents.float() * w[:, None, None, None], dim=0, keepdim=True)
    target_norm = torch.sum(
        torch.linalg.vector_norm(anchor_latents.float().reshape(anchor_latents.shape[0], -1), dim=1) * w
    )
    current_norm = torch.linalg.vector_norm(blended.reshape(1, -1), dim=1).clamp_min(1e-8)
    blended = blended * (target_norm / current_norm).reshape(1, 1, 1, 1)
    return blended.to(dtype=anchor_latents.dtype)


def top_anchors(anchor_labels: Sequence[str], weights: np.ndarray, n: int = 3) -> str:
    order = np.argsort(weights)[::-1][:n]
    return " | ".join(f"{anchor_labels[i]} {weights[i]:.2f}" for i in order)


class AnchorMorphRenderServer:
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
    ):
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
        self.lock = threading.Lock()

    def render_png(self, payload: dict) -> bytes:
        frame_size = int(payload.get("frame_size", self.frame_size))
        z = payload.get("z")
        if z is None:
            raise RuntimeError("payload needs a z vector (a len(anchors)-dim anchor-weight vector)")
        z = np.asarray(z, dtype=float)
        if z.shape != (len(self.anchor_labels),):
            raise RuntimeError(
                f"z has shape {z.shape}, expected ({len(self.anchor_labels)},) - one weight per anchor. "
                f"Set config.optimizer.search_dims == {len(self.anchor_labels)} to match the session manifest."
            )

        import torch

        with self.lock:
            weights = softmax_weights(z, temperature=self.softmax_temperature)
            self._render_count += 1
            if self.log_weights_every and self._render_count % self.log_weights_every == 0:
                target = ""
                if self.target_index is not None and self.target_anchor is not None:
                    target = f" target={self.target_anchor} {weights[self.target_index]:.2f}"
                print(
                    f"[anchor-morph-server] render={self._render_count} "
                    f"top={top_anchors(self.anchor_labels, weights)}{target}"
                )
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

        if image.size != (frame_size, frame_size):
            image = image.resize((frame_size, frame_size))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def make_handler(render_server: AnchorMorphRenderServer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/manifest":
                self.send_error(404, "expected GET /manifest")
                return
            self._send_json(
                {
                    "ok": True,
                    **render_server.manifest,
                    "server": {
                        "kind": "general_stable_diffusion",
                        "render_endpoint": "/render",
                    },
                }
            )

        def do_POST(self) -> None:
            if self.path != "/render":
                self.send_error(404, "expected POST /render (this server has no /anchors endpoint)")
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                png = render_server.render_png(payload)
            except Exception as exc:  # noqa: BLE001
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
            print(f"[anchor-morph-server] {self.address_string()} - {fmt % args}")

    return Handler


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-manifest", required=True, help="path to a curated prompt-session manifest JSON")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", default="stabilityai/sd-turbo",
                        help="e.g. stabilityai/sd-turbo (default, 1 step, no CFG), or a plain SD "
                             "checkpoint - pair a non-turbo model with --steps 20-50 / --guidance-scale 7.5")
    parser.add_argument("--target-anchor", default=None,
                        help="optional diagnostic label from the manifest's anchor_labels; logs its current "
                             "softmax weight but does not steer optimization")
    parser.add_argument("--size", type=int, default=None,
                        help="square render size, divisible by 8 (default: config.generator.frame_size)")
    parser.add_argument("--steps", type=int, default=1,
                        help="denoising steps - 1-4 for sd-turbo, 20-50 for a plain SD checkpoint")
    parser.add_argument("--guidance-scale", type=float, default=0.0,
                        help="0.0 (default, no CFG) for turbo/LCM. ~7-8 for a plain SD checkpoint.")
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="softmax temperature for z -> anchor weights. Higher values make the same "
                             "optimizer movement snap more strongly toward one anchor; lower values "
                             "produce smoother but more uniform blends.")
    parser.add_argument("--log-weights-every", type=int, default=12,
                        help="print the top anchor weights every N renders; 0 disables logging.")
    parser.add_argument("--seed", type=int, default=None,
                        help="per-anchor latent seed (default: config.generator.remote_diffusion_seed)")
    args = parser.parse_args(argv)

    manifest = load_prompt_session_manifest(args.session_manifest)
    if args.target_anchor is not None and args.target_anchor not in manifest.anchor_labels:
        sys.exit(
            f"[anchor-morph-server] --target-anchor must be one of session anchor_labels; got {args.target_anchor!r}"
        )

    config = Config.load()
    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed
    frame_size = args.size if args.size is not None else config.generator.frame_size
    if frame_size < 256 or frame_size % 8 != 0:
        sys.exit("[anchor-morph-server] --size must be at least 256 and divisible by 8")

    if config.optimizer.search_dims != manifest.anchor_count:
        print(
            f"[anchor-morph-server] WARNING: config.optimizer.search_dims={config.optimizer.search_dims} "
            f"but session anchor_count={manifest.anchor_count}. Set optimizer.search_dims: "
            f"{manifest.anchor_count} in config.yaml, or the client will POST a wrongly-sized z and "
            f"/render will reject it."
        )

    pipe, device, dtype = load_pipeline(args.model, device=None)

    print(f"[anchor-morph-server] encoding {manifest.anchor_count} anchor prompt(s) as one batch...")
    anchor_embeds = encode_anchor_prompts(pipe, manifest.realized_prompts, device)

    latent_channels = int(pipe.unet.config.in_channels)
    vae_scale_factor = int(pipe.vae_scale_factor)
    generator = make_cpu_generator(seed)
    anchor_latents = make_anchor_latents(
        manifest.anchor_count, generator, device, dtype, frame_size, latent_channels, vae_scale_factor
    )

    render_server = AnchorMorphRenderServer(
        pipe,
        list(manifest.anchor_labels),
        list(manifest.realized_prompts),
        anchor_embeds,
        anchor_latents,
        frame_size,
        device,
        dtype,
        args.steps,
        args.guidance_scale,
        args.temperature,
        args.log_weights_every,
        args.target_anchor,
        manifest=manifest_metadata(manifest),
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(render_server))
    print(f"[anchor-morph-server] listening on http://{args.host}:{args.port} "
          f"(anchors={manifest.anchor_count}, seed={seed}, size={frame_size}, "
          f"steps={args.steps}, guidance_scale={args.guidance_scale}, "
          f"temperature={args.temperature})")
    print(f"[anchor-morph-server] session manifest: {args.session_manifest}")
    print(f"[anchor-morph-server] user prompt: {manifest.user_prompt}")
    print(f"[anchor-morph-server] anchors: {', '.join(manifest.anchor_labels)}")
    if args.target_anchor is not None:
        print(f"[anchor-morph-server] target anchor diagnostic: {args.target_anchor}")
    server.serve_forever()


if __name__ == "__main__":
    main()
