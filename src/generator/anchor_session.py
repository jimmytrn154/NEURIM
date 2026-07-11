"""Helpers for manifest-backed anchor sessions used by the generalized server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.generator.prompt_curation import (
    DEFAULT_ANCHOR_COUNT,
    PROMPT_CURATION_VERSION,
    PromptCurationManifest,
    validate_prompt_curation_payload,
)


def load_prompt_session_manifest(path: str | Path) -> PromptCurationManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise RuntimeError(f"Prompt session manifest does not exist: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Prompt session manifest contained malformed JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Prompt session manifest must be a JSON object")

    version = payload.get("version")
    if version != PROMPT_CURATION_VERSION:
        raise RuntimeError(
            f"Prompt session manifest version must be {PROMPT_CURATION_VERSION}, got {version!r}"
        )

    anchor_count = payload.get("anchor_count")
    if anchor_count != DEFAULT_ANCHOR_COUNT:
        raise RuntimeError(
            f"Prompt session manifest anchor_count must be {DEFAULT_ANCHOR_COUNT}, got {anchor_count!r}"
        )

    user_prompt = str(payload.get("user_prompt", "")).strip()
    if not user_prompt:
        raise RuntimeError("Prompt session manifest is missing a non-empty user_prompt")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("Prompt session manifest model must be an object")
    provider = str(model.get("provider", "")).strip()
    model_name = str(model.get("name", "")).strip()
    if provider != "openai":
        raise RuntimeError(f"Prompt session manifest model.provider must be 'openai', got {provider!r}")
    if not model_name:
        raise RuntimeError("Prompt session manifest model.name must be non-empty")

    manifest = validate_prompt_curation_payload(
        payload,
        user_prompt=user_prompt,
        model=model_name,
        anchor_count=DEFAULT_ANCHOR_COUNT,
    )
    return PromptCurationManifest(
        version=PROMPT_CURATION_VERSION,
        user_prompt=manifest.user_prompt,
        anchor_count=manifest.anchor_count,
        scaffold=manifest.scaffold,
        prompt_template=manifest.prompt_template,
        anchor_labels=manifest.anchor_labels,
        realized_prompts=manifest.realized_prompts,
        notes=manifest.notes,
        model={"provider": provider, "name": model_name},
    )


def manifest_metadata(manifest: PromptCurationManifest) -> dict[str, Any]:
    return {
        "user_prompt": manifest.user_prompt,
        "anchor_count": manifest.anchor_count,
        "anchor_labels": list(manifest.anchor_labels),
        "realized_prompts": list(manifest.realized_prompts),
        "model": dict(manifest.model),
    }
