"""CLI composition root for the manifest-driven diffusion server."""

from __future__ import annotations

import argparse
import sys

from src.common.config import Config
from src.generator.anchor_session import load_prompt_session_manifest, manifest_metadata

from .http import DiffusionServer
from .renderer import (
    AnchorMorphRenderer,
    encode_anchor_prompts,
    load_pipeline,
    make_anchor_latents,
    make_cpu_generator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-manifest", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--target-anchor", default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--log-weights-every", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def create_renderer(args, config: Config) -> tuple[AnchorMorphRenderer, object]:
    manifest = load_prompt_session_manifest(args.session_manifest)
    if args.target_anchor is not None and args.target_anchor not in manifest.anchor_labels:
        raise ValueError(
            f"--target-anchor must be one of session anchor_labels; got {args.target_anchor!r}"
        )

    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed
    frame_size = args.size if args.size is not None else config.generator.frame_size
    if frame_size < 256 or frame_size % 8 != 0:
        raise ValueError("--size must be at least 256 and divisible by 8")
    if config.optimizer.search_dims != manifest.anchor_count:
        print(
            f"[anchor-morph-server] WARNING: config.optimizer.search_dims={config.optimizer.search_dims} "
            f"but session anchor_count={manifest.anchor_count}. Set optimizer.search_dims: "
            f"{manifest.anchor_count} in config.yaml."
        )

    pipe, device, dtype = load_pipeline(args.model, device=None)
    print(f"[anchor-morph-server] encoding {manifest.anchor_count} anchor prompt(s) as one batch...")
    anchor_embeds = encode_anchor_prompts(pipe, manifest.realized_prompts, device)
    anchor_latents = make_anchor_latents(
        manifest.anchor_count,
        make_cpu_generator(seed),
        device,
        dtype,
        frame_size,
        int(pipe.unet.config.in_channels),
        int(pipe.vae_scale_factor),
    )
    renderer = AnchorMorphRenderer(
        pipe=pipe,
        anchor_labels=list(manifest.anchor_labels),
        anchor_prompts=list(manifest.realized_prompts),
        anchor_embeds=anchor_embeds,
        anchor_latents=anchor_latents,
        frame_size=frame_size,
        device=device,
        dtype=dtype,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        softmax_temperature=args.temperature,
        log_weights_every=args.log_weights_every,
        target_anchor=args.target_anchor,
        manifest=manifest_metadata(manifest),
    )
    return renderer, manifest


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = Config.load()
    try:
        renderer, manifest = create_renderer(args, config)
    except ValueError as exc:
        sys.exit(f"[anchor-morph-server] {exc}")

    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed
    server = DiffusionServer(renderer, args.host, args.port)
    print(
        f"[anchor-morph-server] listening on http://{args.host}:{args.port} "
        f"(anchors={manifest.anchor_count}, seed={seed}, size={renderer.frame_size}, "
        f"steps={args.steps}, guidance_scale={args.guidance_scale}, temperature={args.temperature})"
    )
    print(f"[anchor-morph-server] session manifest: {args.session_manifest}")
    print(f"[anchor-morph-server] user prompt: {manifest.user_prompt}")
    print(f"[anchor-morph-server] anchors: {', '.join(manifest.anchor_labels)}")
    server.serve_forever()
