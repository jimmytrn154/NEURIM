import json

import pytest

from src.generator.prompt_curation import (
    DEFAULT_ANCHOR_COUNT,
    PromptCurationManifest,
    curate_prompt_manifest,
    format_manifest_summary,
    parse_prompt_curation_json,
    validate_morph_compatibility,
    validate_prompt_curation_payload,
)


def _payload():
    labels = [f"axis_{i}" for i in range(DEFAULT_ANCHOR_COUNT)]
    return {
        "scaffold": "centered product photograph, neutral backdrop, soft studio light",
        "prompt_template": "centered product photograph of {anchor}, neutral backdrop, soft studio light",
        "anchor_labels": labels,
        "realized_prompts": [
            f"centered product photograph of {label}, neutral backdrop, soft studio light"
            for label in labels
        ],
        "notes": "Keep all anchors compatible with the same subject.",
    }


def test_validate_prompt_curation_payload_accepts_valid_shape():
    manifest = validate_prompt_curation_payload(_payload(), user_prompt="red sneakers", model="gpt-test")

    assert isinstance(manifest, PromptCurationManifest)
    assert manifest.anchor_count == DEFAULT_ANCHOR_COUNT
    assert manifest.model == {"provider": "openai", "name": "gpt-test"}
    assert len(manifest.anchor_labels) == DEFAULT_ANCHOR_COUNT
    assert len(manifest.realized_prompts) == DEFAULT_ANCHOR_COUNT


def test_validate_prompt_curation_payload_rejects_missing_scaffold():
    payload = _payload()
    payload["scaffold"] = ""

    with pytest.raises(RuntimeError, match="scaffold"):
        validate_prompt_curation_payload(payload, user_prompt="red sneakers", model="gpt-test")


def test_validate_prompt_curation_payload_rejects_duplicate_anchor_labels():
    payload = _payload()
    payload["anchor_labels"][1] = payload["anchor_labels"][0]

    with pytest.raises(RuntimeError, match="unique"):
        validate_prompt_curation_payload(payload, user_prompt="red sneakers", model="gpt-test")


def test_validate_prompt_curation_payload_rejects_empty_anchor_labels():
    payload = _payload()
    payload["anchor_labels"][3] = "  "

    with pytest.raises(RuntimeError, match="empty strings"):
        validate_prompt_curation_payload(payload, user_prompt="red sneakers", model="gpt-test")


def test_validate_prompt_curation_payload_rejects_wrong_realized_prompt_count():
    payload = _payload()
    payload["realized_prompts"] = payload["realized_prompts"][:-1]

    with pytest.raises(RuntimeError, match="exactly 7"):
        validate_prompt_curation_payload(payload, user_prompt="red sneakers", model="gpt-test")


def test_validate_prompt_curation_payload_rejects_template_without_placeholder():
    payload = _payload()
    payload["prompt_template"] = "plain string without placeholder"

    with pytest.raises(RuntimeError, match="\\{anchor\\}"):
        validate_prompt_curation_payload(payload, user_prompt="red sneakers", model="gpt-test")


def test_validate_morph_compatibility_accepts_exact_template_expansion():
    manifest = validate_prompt_curation_payload(_payload(), user_prompt="red sneakers", model="gpt-test")

    assert validate_morph_compatibility(manifest) is manifest


def test_validate_morph_compatibility_rejects_multiple_placeholders():
    payload = _payload()
    payload["prompt_template"] = "{anchor} centered portrait of {anchor}, neutral backdrop"
    payload["realized_prompts"] = [
        payload["prompt_template"].format(anchor=label) for label in payload["anchor_labels"]
    ]
    manifest = validate_prompt_curation_payload(payload, user_prompt="red sneakers", model="gpt-test")

    with pytest.raises(RuntimeError, match="exactly once"):
        validate_morph_compatibility(manifest)


def test_validate_morph_compatibility_rejects_per_anchor_scene_details():
    payload = _payload()
    payload["realized_prompts"][2] += ", photographed from a low camera angle"
    manifest = validate_prompt_curation_payload(payload, user_prompt="red sneakers", model="gpt-test")

    with pytest.raises(RuntimeError, match="must exactly equal"):
        validate_morph_compatibility(manifest)


def test_validate_morph_compatibility_rejects_heterogeneous_mammoth_prompts():
    labels = [
        "grazing adult",
        "mother and calf",
        "migrating herd",
        "charging bull",
        "cliff sentinel",
        "river drinking",
        "close-up portrait",
    ]
    payload = {
        "scaffold": "Photorealistic woolly mammoths in a Pleistocene tundra.",
        "prompt_template": "{anchor} woolly mammoth in a fixed tundra composition",
        "anchor_labels": labels,
        "realized_prompts": [
            "A grazing adult woolly mammoth in a full-body tundra scene",
            "A mother and calf in an intimate close composition",
            "A migrating herd in a panoramic composition",
            "A charging bull photographed from a low angle",
            "A mammoth standing on a cliff overlooking a valley",
            "A mammoth drinking beside a frozen river",
            "An extreme close-up portrait of a mammoth eye",
        ],
        "notes": "Heterogeneous legacy-style output.",
    }
    manifest = validate_prompt_curation_payload(payload, user_prompt="woolly mammoths", model="gpt-test")

    with pytest.raises(RuntimeError, match="not morph-compatible"):
        validate_morph_compatibility(manifest)


def test_parse_prompt_curation_json_handles_code_fences():
    text = "```json\n" + json.dumps(_payload(), indent=2) + "\n```"

    parsed = parse_prompt_curation_json(text)

    assert parsed["anchor_labels"][0] == "axis_0"


def test_parse_prompt_curation_json_rejects_malformed_payload():
    with pytest.raises(RuntimeError, match="malformed JSON"):
        parse_prompt_curation_json('{"scaffold": "x",')


def test_curate_prompt_manifest_uses_openai_client_response_text():
    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-test"
            assert kwargs["input"] == "red sneakers"
            assert "Select exactly one semantic visual axis" in kwargs["instructions"]
            assert "Fix composition, pose, framing" in kwargs["instructions"]
            return type("Response", (), {"output_text": json.dumps(_payload())})()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    manifest = curate_prompt_manifest("red sneakers", model="gpt-test", client=FakeClient())

    assert manifest.user_prompt == "red sneakers"
    assert manifest.anchor_labels[-1] == "axis_6"


def test_format_manifest_summary_mentions_anchor_labels():
    manifest = validate_prompt_curation_payload(_payload(), user_prompt="red sneakers", model="gpt-test")

    summary = format_manifest_summary(manifest)

    assert "user_prompt: red sneakers" in summary
    assert "1. axis_0:" in summary


def test_cli_writes_json_manifest(tmp_path, capsys, monkeypatch):
    from src.session import curation as cli

    class FakeResponses:
        def create(self, **kwargs):
            return type("Response", (), {"output_text": json.dumps(_payload())})()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    def fake_curate(user_prompt: str, model: str, client=None):
        return curate_prompt_manifest(user_prompt, model=model, client=FakeClient())

    out_path = tmp_path / "manifest.json"
    monkeypatch.setattr(cli.PromptCurationService, "curate", lambda self, prompt: fake_curate(prompt, self.model))
    exit_code = cli.main(["--user-prompt", "red sneakers", "--out", str(out_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["user_prompt"] == "red sneakers"
    assert out_path.exists()
