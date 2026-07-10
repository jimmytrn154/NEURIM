#!/usr/bin/env python3
"""Real-time morph server using StreamDiffusion (cumulo-autumn/StreamDiffusion)
instead of the from-scratch SDXL-Turbo path in diffusion_pipeline.py.

Why this exists: run_diffusion_server.py re-renders each keyframe independently
at ~5fps, and the client crossfades between them to fake 30fps. StreamDiffusion
is built for genuinely continuous generation - it keeps a rolling image/latent
buffer across calls and reaches ~30-100fps on an RTX 4090 for a 1-step SD-Turbo
img2img stream - so the crossfade hack becomes unnecessary once render itself is
fast enough.

Wire-compatible with the existing client: this exposes the SAME POST /render
contract as run_diffusion_server.py (z in, PNG out), so src/generator/remote_diffusion.py
and GeneratorService need NO changes - only remote_diffusion_url points here
instead, and remote_diffusion_keyframe_interval_s should be lowered (see
config.yaml) since real renders no longer take ~200ms.

ARCHITECTURE MISMATCH, read before running:
  - StreamDiffusion's public API only supports SD1.x-family models
    (encoder_hidden_states only - no pooled_prompt_embeds/add_text_embeds/
    add_time_ids). It CANNOT load "stabilityai/sdxl-turbo". This server uses
    "stabilityai/sd-turbo" instead - a different checkpoint, different visual
    style than the SDXL-Turbo path.
  - There is no official API to feed a continuous embedding instead of a text
    prompt. update_prompt(text) internally just does:
        self.prompt_embeds = self.pipe.encode_prompt(text, ...)[0].repeat(batch_size, 1, 1)
    This server does the SAME encode_prompt() call ourselves for each anchor
    prompt (to fit a projector), then on every /render request builds our own
    z -> embedding -> repeat(batch_size, 1, 1) tensor and assigns it directly to
    stream.prompt_embeds, bypassing update_prompt() entirely. This relies on an
    UNDOCUMENTED internal attribute and could break on a future StreamDiffusion
    release - it is not a supported integration point.

ISOLATED VENV REQUIRED - do not install into torch-env or diffmorpher-env:
    conda create -n streamdiffusion-env python=3.10 -y
    conda activate streamdiffusion-env
    pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
    pip install streamdiffusion
    # optional, faster, needs an engine-build step:
    # pip install "streamdiffusion[tensorrt]" && python -m streamdiffusion.tools.install-tensorrt
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config
from src.optimizer.projection import PCAProjector


class StreamDiffusionRenderServer:
    def __init__(self, wrapper, projector: PCAProjector, prompt_embed_shape: tuple[int, int], frame_size: int):
        self.wrapper = wrapper
        self.projector = projector
        self.seq_len, self.hidden = prompt_embed_shape
        self.frame_size = frame_size
        self.lock = threading.Lock()
        # StreamDiffusion's img2img mode conditions on the previous frame - this
        # IS the "stream" (the state that makes consecutive frames coherent).
        # It can only lightly restyle its input, though, so it CANNOT bootstrap a
        # coherent image from flat grey - the stream must start from a real,
        # prompt-conditioned frame or every output stays grey mush. Generate that
        # starting frame once via the SD-Turbo txt2img path (fresh per-frame noise
        # is irrelevant for a one-shot), conditioned on z=0 (the mean anchor embed).
        self._prev_image = self._bootstrap_frame()

    def _bootstrap_frame(self) -> Image.Image:
        try:
            self._inject_embedding(np.zeros(self.projector.dims))
            image = self.wrapper.txt2img()
            if isinstance(image, list):
                image = image[0]
            if image.size != (self.frame_size, self.frame_size):
                image = image.resize((self.frame_size, self.frame_size))
            print("[streamdiffusion-server] bootstrapped starting frame via txt2img")
            return image
        except Exception as exc:  # noqa: BLE001
            print(f"[streamdiffusion-server] WARNING: txt2img bootstrap failed ({exc}); "
                  "falling back to grey - the morph likely won't form a coherent image")
            return Image.new("RGB", (self.frame_size, self.frame_size), (128, 128, 128))

    def _inject_embedding(self, z: np.ndarray) -> None:
        import torch  # lazy: keep torch out of this module's import path for non-GPU callers

        stream = self.wrapper.stream
        embedding = self.projector.to_embedding(z)
        vec = torch.tensor(embedding, dtype=stream.dtype, device=stream.device)
        vec = vec.reshape(1, self.seq_len, self.hidden)
        # Mirrors update_prompt()'s own repeat() exactly - same shape contract,
        # just sourced from our projector instead of a fresh encode_prompt() call.
        stream.prompt_embeds = vec.repeat(stream.batch_size, 1, 1)

    def render_png(self, payload: dict) -> bytes:
        frame_size = int(payload.get("frame_size", self.frame_size))
        z = payload.get("z")
        with self.lock:
            if z is None:
                raise RuntimeError("stream_diffusion backend requires a z vector, not a prompt string")
            self._inject_embedding(np.asarray(z, dtype=float))
            image = self.wrapper(image=self._prev_image)
            if isinstance(image, list):
                image = image[0]
            self._prev_image = image
        if image.size != (frame_size, frame_size):
            image = image.resize((frame_size, frame_size))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def make_handler(render_server: StreamDiffusionRenderServer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/render":
                self.send_error(404, "expected POST /render")
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                png = render_server.render_png(payload)
            except Exception as exc:  # noqa: BLE001 - report server-side failures to the client.
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(200)
            self.send_header("content-type", "image/png")
            self.send_header("content-length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[streamdiffusion-server] {self.address_string()} - {fmt % args}")

    return Handler


def build_wrapper(config, streamdiffusion_repo, model_id, t_index_list, acceleration, seed, frame_size):
    """Construct + prepare a StreamDiffusionWrapper. Shared by the server and
    scripts/test_streamdiffusion.py so the diagnostic exercises the exact setup.
    """
    repo_path = str(Path(streamdiffusion_repo).resolve())
    if not (Path(repo_path) / "utils" / "wrapper.py").exists():
        sys.exit(f"[streamdiffusion-server] {repo_path}/utils/wrapper.py not found - "
                  "is --streamdiffusion-repo a real StreamDiffusion checkout?")
    sys.path.insert(0, repo_path)

    import torch
    from utils.wrapper import StreamDiffusionWrapper  # only importable with repo_path on sys.path

    print(f"[streamdiffusion-server] loading {model_id} (t_index_list={t_index_list}, "
          f"acceleration={acceleration})...")
    wrapper = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        t_index_list=t_index_list,
        mode="img2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=1,
        width=frame_size,
        height=frame_size,
        acceleration=acceleration,
        use_lcm_lora=False,   # this is a true SD-Turbo run, not LCM-LoRA-on-SD1.5
        use_tiny_vae=True,
        use_denoising_batch=True,
        cfg_type="none",      # avoids uncond-embeds concatenation, since we inject
                               # prompt_embeds directly rather than via update_prompt()
        seed=seed,
    )
    # Any placeholder prompt works here - prepare() just initializes internal state
    # (batch_size, schedules); real conditioning comes from overwriting prompt_embeds.
    wrapper.prepare(prompt=config.generator.anchor_prompts[0] if config.generator.anchor_prompts else "a photo",
                     guidance_scale=1.0)
    return wrapper


def _fit_projector(wrapper, anchor_prompts: list[str], dims: int) -> tuple[PCAProjector, tuple[int, int]]:
    if len(anchor_prompts) < 2:
        print(f"[streamdiffusion-server] WARNING: only {len(anchor_prompts)} anchor prompt(s) - "
              "the morph will be STATIC (projector subspace is degenerate).")
    import torch

    pipe = wrapper.stream.pipe
    device = wrapper.stream.device
    embeddings = []
    shape = None
    print(f"[streamdiffusion-server] encoding {len(anchor_prompts)} anchor prompt(s) and fitting projector...")
    with torch.no_grad():
        for prompt in anchor_prompts:
            out = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                      do_classifier_free_guidance=False)
            prompt_embeds = out[0]
            shape = (int(prompt_embeds.shape[-2]), int(prompt_embeds.shape[-1]))
            embeddings.append(prompt_embeds.flatten().float().cpu().numpy())
    embeddings = np.stack(embeddings).astype(np.float32)
    projector = PCAProjector(dims=dims).fit(embeddings)
    print(f"[streamdiffusion-server] projector fit: {dims}-dim search space over "
          f"{embeddings.shape[0]} prompts (per-prompt shape {shape})")
    return projector, shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--streamdiffusion-repo", required=True,
                         help="path to a cloned cumulo-autumn/StreamDiffusion checkout - StreamDiffusionWrapper "
                              "lives in utils/wrapper.py at the repo root, which setup.py does NOT package, so "
                              "'pip install streamdiffusion' alone does not make it importable")
    parser.add_argument("--model-id", default="stabilityai/sd-turbo")
    parser.add_argument("--t-index-list", default="0,16", help="comma-separated denoising step indices")
    parser.add_argument("--acceleration", default="xformers", choices=["none", "xformers", "tensorrt"])
    parser.add_argument("--seed", type=int, default=None, help="defaults to config.generator.remote_diffusion_seed")
    args = parser.parse_args()

    config = Config.load()
    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed
    t_index_list = [int(x) for x in args.t_index_list.split(",")]
    frame_size = config.generator.frame_size

    wrapper = build_wrapper(config, args.streamdiffusion_repo, args.model_id, t_index_list,
                            args.acceleration, seed, frame_size)
    projector, embed_shape = _fit_projector(wrapper, config.generator.anchor_prompts, config.optimizer.search_dims)
    render_server = StreamDiffusionRenderServer(wrapper, projector, embed_shape, frame_size)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(render_server))
    print(f"[streamdiffusion-server] listening on http://{args.host}:{args.port} (seed={seed})")
    server.serve_forever()


if __name__ == "__main__":
    main()
