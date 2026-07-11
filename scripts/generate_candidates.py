#!/usr/bin/env python3
"""Generate SD-Turbo A/B candidate images for EEG calibration, per target.

The calibration task needs candidates that form a graded near->far gradient
around each target - not random unrelated images. This produces them with the
SAME Stable Diffusion pipeline the live loop uses (no calibration-vs-inference
distribution gap), building the gradient by interpolating in prompt-embedding
space between the target's prompt and a distractor prompt (the same anchor-blend
mechanism as run_poodle_turbo_morph.py).

Output layout (consumed by record_reward_trials.py --candidate-dir):
    <out-dir>/<name>/L0_0.png ... Lk_j.png      # one subfolder per target

The subfolder name matches the target's real photo stem, so the recorder pairs
each real target with its own generated candidates. Closeness (and thus the
label) is still measured by the recorder via CLIP against the real target photo;
this script only has to produce a diverse spread from very-on-target to clearly-off.

Manifest (--targets targets.json), one entry per target:
    [
      {"name": "poodle", "prompt": "a white standard poodle, studio photo",
       "distractors": ["a black cat", "a red sports car"]},
      ...
    ]
`distractors` is optional; if omitted, the other targets' prompts are used. The
`name` should match the real target photo filename stem in --target-dir.

Example:
    python scripts/generate_candidates.py --targets data/targets.json \
        --out-dir data/candidates_ai --levels 5 --per-level 6 --model stabilityai/sd-turbo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_poodle_turbo_morph import load_pipeline, select_device  # reuse SD helpers


def load_targets(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path} must be a non-empty JSON list of target entries")
    for entry in data:
        if "name" not in entry or "prompt" not in entry:
            raise SystemExit("each target needs 'name' and 'prompt'")
    return data


def encode(pipe, prompt: str, device):
    import torch

    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return prompt_embeds  # (1, seq, hidden)


def generate_for_target(pipe, device, entry, distractor_prompts, args, out_root):
    import torch
    from PIL import Image

    name = entry["name"]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_t = encode(pipe, entry["prompt"], device)
    distractors = entry.get("distractors") or distractor_prompts
    if not distractors:
        distractors = ["a plain gray background"]
    emb_ds = [encode(pipe, d, device) for d in distractors]

    n = 0
    for level in range(args.levels):
        # alpha 0 -> on-target; alpha -> max_alpha -> morphed toward a distractor.
        alpha = (level / max(args.levels - 1, 1)) * args.max_alpha
        for j in range(args.per_level):
            emb_d = emb_ds[(level * args.per_level + j) % len(emb_ds)]
            emb = (1.0 - alpha) * emb_t + alpha * emb_d
            seed = args.seed + 1000 * level + j
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                image = pipe(
                    prompt_embeds=emb.to(dtype=pipe.unet.dtype),
                    num_inference_steps=args.steps,
                    guidance_scale=0.0,
                    height=args.size, width=args.size,
                    generator=gen,
                ).images[0]
            image.save(out_dir / f"L{level}_{j}.png")
            n += 1
    print(f"[gen] {name}: wrote {n} candidates -> {out_dir}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", type=Path, required=True, help="JSON manifest of targets")
    parser.add_argument("--out-dir", type=Path, default=Path("data/candidates_ai"))
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--levels", type=int, default=5, help="near->far gradient steps")
    parser.add_argument("--per-level", type=int, default=6, help="images per gradient step")
    parser.add_argument("--max-alpha", type=float, default=0.9,
                        help="how far toward the distractor the farthest level goes (0-1)")
    parser.add_argument("--steps", type=int, default=2, help="SD-Turbo denoising steps (1-4)")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    targets = load_targets(args.targets)
    all_prompts = [t["prompt"] for t in targets]

    device, dtype = select_device()
    print(f"[gen] device={device.type} dtype={dtype} model={args.model}")
    pipe = load_pipeline(args.model, device, dtype)

    total = 0
    for entry in targets:
        # Default distractors = the other targets' prompts (morph away from this one).
        others = [p for p in all_prompts if p != entry["prompt"]]
        total += generate_for_target(pipe, device, entry, others, args, args.out_dir)
    print(f"[gen] done: {total} candidate images across {len(targets)} targets -> {args.out_dir}")


if __name__ == "__main__":
    main()
