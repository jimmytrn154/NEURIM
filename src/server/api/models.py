"""Request models for the local frontend API."""

from pydantic import BaseModel, Field, field_validator


class StartSessionRequest(BaseModel):
    prompt: str | None = None
    mock: bool = False
    baseline_seconds: float = Field(default=30.0, ge=0)
    server_url: str = "http://localhost:8766"

    @field_validator("server_url")
    @classmethod
    def validate_server_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("server_url must start with http:// or https://")
        return cleaned

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None
