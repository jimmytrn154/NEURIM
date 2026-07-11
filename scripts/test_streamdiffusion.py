#!/usr/bin/env python3
"""Diagnostic for the latent-walk backend (run_streamdiffusion_server.py) -
run it on the GPU server to localize WHERE the pipeline is failing, since
different failure modes look similar in the live demo but have different fixes:

  Test A (official API, no z-injection): does the model + steps/guidance
     config generate a coherent image AT ALL from a plain text prompt? If
     this is garbage, the problem is model/config, not our z-injection.

  Test B (our path): z -> PCAProjector.to_embedding(z) -> prompt_embeds,
     rendered fresh (fixed latent, no img2img) at each point in a z-sweep.
     Flip through the frames: should look like the anchor prompts and change
     smoothly. If Test A is fine but this is bad, the problem is the
     projector/anchor-prompt fit, not the model/generation config.

Writes PNGs to data/processed/streamdiffusion_test/. Eyeball them.

    python scripts/test_streamdiffusion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "streamdiffusion_test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="stabilityai/sd-turbo")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--sweep-frames", type=int, default=12, help="frames in the Test B z-sweep")
    parser.add_argument("--sweep-dim", type=int, default=0, help="which z dimension to sweep in Test B")
    args = parser.parse_args()

    # Imported here (not at module level) so any import-time errors are clearly
    # attributed to the server module, not this script.
    from run_streamdiffusion_server import LatentWalkRenderServer, _fit_projector, load_pipeline

    config = Config.load()
    seed = config.generator.remote_diffusion_seed
    frame_size = config.generator.frame_size
    # Tag the output folder by steps/guidance so re-runs with different values
    # don't overwrite each other, e.g. .../streamdiffusion_test/s1_g0.0/A_official.png
    out_dir = OUT_DIR / f"s{args.steps}_g{args.guidance_scale}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[test] writing to {out_dir}")

    pipe, fixed_latent, device, dtype = load_pipeline(args.model_id, seed, frame_size)

    # --- Test A: raw generation via the official API, no z-injection ----------
    anchor = config.generator.anchor_prompts[0] if config.generator.anchor_prompts else "a golden retriever puppy"
    print(f"[test] A: official generation for prompt: {anchor!r}")
    image = pipe(
        prompt=anchor,
        latents=fixed_latent,
        height=frame_size,
        width=frame_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        output_type="pil",
    ).images[0]
    image.save(out_dir / "A_official.png")
    print(f"[test] A: saved {out_dir / 'A_official.png'} - if this isn't a coherent image, "
          "the problem is model/steps/guidance_scale, not our z-injection")

    # --- Test B: our injected z -> embedding -> fresh render, swept ----------
    projector, embed_shape = _fit_projector(pipe, config.generator.anchor_prompts, config.optimizer.search_dims, device)
    server = LatentWalkRenderServer(
        pipe, projector, embed_shape, frame_size, config.optimizer.search_dims,
        fixed_latent, device, dtype, args.steps, args.guidance_scale,
    )

    dim = min(args.sweep_dim, projector.dims - 1)
    print(f"[test] B: sweeping z[{dim}] from -1 to +1 over {args.sweep_frames} frames via our render path")
    for i in range(args.sweep_frames):
        z = np.zeros(projector.dims)
        z[dim] = -1.0 + 2.0 * i / max(args.sweep_frames - 1, 1)
        png = server.render_png({"z": z.tolist()})
        (out_dir / f"B_sweep_{i:02d}.png").write_bytes(png)
    print(f"[test] B: saved B_sweep_00..{args.sweep_frames - 1:02d}.png in {out_dir} - "
          "flip through them: should look like the anchor prompts and change smoothly")


if __name__ == "__main__":
    main()
