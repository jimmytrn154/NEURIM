"""Prompt-manifest curation service and CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.common.config import Config
from src.generator.prompt_curation import (
    DEFAULT_TEXT_MODEL,
    PromptCurationManifest,
    curate_prompt_manifest,
    format_manifest_summary,
)


class PromptCurationService:
    def __init__(self, model: str, curator=curate_prompt_manifest) -> None:
        self.model = model
        self.curator = curator

    def curate(self, user_prompt: str) -> PromptCurationManifest:
        return self.curator(user_prompt, model=self.model)

    @staticmethod
    def write(manifest: PromptCurationManifest, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate a user prompt into a diffusion manifest.")
    parser.add_argument("--user-prompt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    model = args.model or config.generator.openai_text_model or DEFAULT_TEXT_MODEL
    service = PromptCurationService(model)
    manifest = service.curate(args.user_prompt)
    if args.out:
        service.write(manifest, Path(args.out))
    if args.verbose:
        print(format_manifest_summary(manifest), file=sys.stderr)
    print(json.dumps(manifest.to_dict(), indent=2))
    return 0
