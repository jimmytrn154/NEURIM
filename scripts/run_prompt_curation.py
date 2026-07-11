#!/usr/bin/env python3
"""Prototype: turn one user prompt into a curated anchor-session manifest.

Calls the live OpenAI API once and returns a JSON-first prompt pack suitable
for the future session-based diffusion system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config
from src.generator.prompt_curation import (
    DEFAULT_TEXT_MODEL,
    curate_prompt_manifest,
    format_manifest_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--user-prompt", required=True, help="simple user prompt to curate")
    parser.add_argument("--out", default=None, help="optional path to write the JSON manifest")
    parser.add_argument("--model", default=None, help="override the OpenAI text model used for curation")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print a short human-readable summary to stderr alongside the JSON output",
    )
    args = parser.parse_args(argv)

    config = Config.load()
    model = args.model or config.generator.openai_text_model or DEFAULT_TEXT_MODEL
    manifest = curate_prompt_manifest(args.user_prompt, model=model)
    payload = manifest.to_dict()
    rendered = json.dumps(payload, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")

    if args.verbose:
        print(format_manifest_summary(manifest), file=sys.stderr)

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
