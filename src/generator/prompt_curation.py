"""OpenAI-backed prompt curation for session-oriented diffusion runs.

This stage turns one simple user prompt into a structured prompt pack:
one shared scaffold/template, a fixed set of anchor labels, and one full
realized prompt per anchor. The output is intentionally JSON-first so later
session setup can consume it directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


PROMPT_CURATION_VERSION = 1
DEFAULT_ANCHOR_COUNT = 7
DEFAULT_TEXT_MODEL = "gpt-5-mini"

PROMPT_CURATION_META_PROMPT = """You are generating a structured prompt pack for a live image-morph system.

Your job is to convert one simple user prompt into:
1. one shared scaffold
2. one prompt template
3. exactly {anchor_count} anchor labels
4. exactly {anchor_count} realized per-anchor prompts

Requirements:
- Preserve the user's core subject, intent, and visual domain.
- The scaffold must describe the shared subject identity, composition, medium,
  lighting, and style constraints that all anchors inherit.
- The prompt_template must contain the literal placeholder {{anchor}} exactly
  once and be reusable for future anchor substitutions.
- Anchor labels must be short, distinct, and meaningful visual axes or
  compatible variations. They must not contradict the user's subject.
- Realized prompts must be full prompts ready to send to a diffusion model.
- Each realized prompt must correspond positionally to the matching anchor
  label and remain compatible with the shared scaffold.
- Avoid contradictory or incompatible anchors.
- Return JSON only. No markdown, no prose outside JSON.

Return exactly this shape:
{{
  "scaffold": "...",
  "prompt_template": "... {{anchor}} ...",
  "anchor_labels": ["...", "..."],
  "realized_prompts": ["...", "..."],
  "notes": "..."
}}
"""


@dataclass(frozen=True)
class PromptCurationManifest:
    version: int
    user_prompt: str
    anchor_count: int
    scaffold: str
    prompt_template: str
    anchor_labels: list[str]
    realized_prompts: list[str]
    notes: str
    model: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_default_client():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from openai import OpenAI

    return OpenAI()


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    texts.append(text)
        if texts:
            return "\n".join(texts)

    raise RuntimeError("OpenAI response did not include readable text output")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_prompt_curation_json(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 and end == -1:
            raise RuntimeError("Prompt curation response did not contain valid JSON") from None
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Prompt curation response contained malformed JSON: {exc}") from exc
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Prompt curation response contained malformed JSON: {exc}") from exc


def _clean_string_list(value: Any, field_name: str, expected_count: int) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{field_name} must be a list of exactly {expected_count} non-empty strings")
    items = [str(item).strip() for item in value]
    if len(items) != expected_count:
        raise RuntimeError(f"{field_name} must contain exactly {expected_count} items")
    if any(not item for item in items):
        raise RuntimeError(f"{field_name} must not contain empty strings")
    return items


def validate_prompt_curation_payload(
    payload: dict[str, Any],
    user_prompt: str,
    model: str,
    anchor_count: int = DEFAULT_ANCHOR_COUNT,
) -> PromptCurationManifest:
    scaffold = str(payload.get("scaffold", "")).strip()
    if not scaffold:
        raise RuntimeError("Prompt curation payload is missing a non-empty scaffold")

    prompt_template = str(payload.get("prompt_template", "")).strip()
    if not prompt_template:
        raise RuntimeError("Prompt curation payload is missing a non-empty prompt_template")
    if "{anchor}" not in prompt_template:
        raise RuntimeError("prompt_template must contain the literal {anchor} placeholder")

    anchor_labels = _clean_string_list(payload.get("anchor_labels"), "anchor_labels", anchor_count)
    if len(set(anchor_labels)) != len(anchor_labels):
        raise RuntimeError("anchor_labels must be unique")

    realized_prompts = _clean_string_list(
        payload.get("realized_prompts"),
        "realized_prompts",
        anchor_count,
    )

    notes = str(payload.get("notes", "")).strip()

    return PromptCurationManifest(
        version=PROMPT_CURATION_VERSION,
        user_prompt=user_prompt.strip(),
        anchor_count=anchor_count,
        scaffold=scaffold,
        prompt_template=prompt_template,
        anchor_labels=anchor_labels,
        realized_prompts=realized_prompts,
        notes=notes,
        model={"provider": "openai", "name": model},
    )


def curate_prompt_manifest(
    user_prompt: str,
    model: str = DEFAULT_TEXT_MODEL,
    anchor_count: int = DEFAULT_ANCHOR_COUNT,
    client: Any | None = None,
) -> PromptCurationManifest:
    prompt = user_prompt.strip()
    if not prompt:
        raise RuntimeError("user_prompt must be non-empty")
    if anchor_count != DEFAULT_ANCHOR_COUNT:
        raise RuntimeError(f"anchor_count must be fixed to {DEFAULT_ANCHOR_COUNT} for this prototype")

    if client is None:
        client = _load_default_client()

    response = client.responses.create(
        model=model,
        instructions=PROMPT_CURATION_META_PROMPT.format(anchor_count=anchor_count),
        input=prompt,
    )
    raw_text = _extract_response_text(response)
    payload = parse_prompt_curation_json(raw_text)
    return validate_prompt_curation_payload(payload, user_prompt=prompt, model=model, anchor_count=anchor_count)


def format_manifest_summary(manifest: PromptCurationManifest) -> str:
    lines = [
        f"user_prompt: {manifest.user_prompt}",
        f"model: {manifest.model['name']}",
        f"anchor_count: {manifest.anchor_count}",
        f"scaffold: {manifest.scaffold}",
        f"prompt_template: {manifest.prompt_template}",
    ]
    for idx, (label, prompt) in enumerate(
        zip(manifest.anchor_labels, manifest.realized_prompts, strict=True),
        start=1,
    ):
        lines.append(f"{idx}. {label}: {prompt}")
    if manifest.notes:
        lines.append(f"notes: {manifest.notes}")
    return "\n".join(lines)
