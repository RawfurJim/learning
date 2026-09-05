"""The single LLM entry point: `generate_json(prompt, schema, *, case)`.

Modes (from `LLM_MODE`):
- `replay` (test default): read `tests/recordings/<agent>/<case>.json`, validate with the schema.
- `record`: call the provider, validate, write the recording.
- `live`: call the provider, write nothing.

Invalid JSON from the provider is retried once, then `LLMOutputError` is raised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from resume_tailor.settings import Settings

DEFAULT_AGENT = "llm"
MAX_ATTEMPTS = 2


class RecordingMissing(FileNotFoundError):
    """Replay mode found no recording for the requested case."""


class LLMOutputError(RuntimeError):
    """The provider returned output that did not validate against the schema, twice."""


class ProviderConfigError(RuntimeError):
    """The configured provider cannot be built (missing key, unknown provider)."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    model: str


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: Usage


class Provider(Protocol):
    def generate(self, prompt: str, schema: type[BaseModel], *, model: str) -> ProviderResponse: ...


class GeminiProvider:
    """Gemini via `google-genai`, JSON mode with `response_schema`."""

    def __init__(self, api_key: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, schema: type[BaseModel], *, model: str) -> ProviderResponse:
        from google.genai import types

        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        meta = response.usage_metadata
        usage = Usage(
            input_tokens=(meta.prompt_token_count if meta else None) or 0,
            output_tokens=(meta.candidates_token_count if meta else None) or 0,
            model=model,
        )
        return ProviderResponse(text=response.text or "", usage=usage)


_provider: Provider | None = None
_last_usage: Usage | None = None


def set_provider(provider: Provider | None) -> None:
    """Install a provider (tests use a stub). `None` rebuilds from settings on next call."""
    global _provider
    _provider = provider


def reset_usage() -> None:
    global _last_usage
    _last_usage = None


def last_usage() -> Usage | None:
    """Usage of the most recent `generate_json` call, or None if none has run."""
    return _last_usage


def recording_path(case: str, recordings_dir: Path | None = None) -> Path:
    """`tests/recordings/<agent>/<case>.json`; a bare case name uses agent `llm`."""
    if recordings_dir is None:
        recordings_dir = Settings.from_env().recordings_dir
    agent, _, name = case.rpartition("/")
    return recordings_dir / (agent or DEFAULT_AGENT) / f"{name}.json"


def generate_json[T: BaseModel](prompt: str, schema: type[T], *, case: str) -> T:
    settings = Settings.from_env()
    if settings.llm_mode == "replay":
        return _replay(schema, recording_path(case, settings.recordings_dir))

    provider = _get_provider(settings)
    result, text = _call_with_retry(provider, prompt, schema, settings.llm_model)
    if settings.llm_mode == "record":
        path = recording_path(case, settings.recordings_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json.loads(text), indent=2) + "\n", encoding="utf-8")
    return result


def _replay[T: BaseModel](schema: type[T], path: Path) -> T:
    global _last_usage
    if not path.exists():
        raise RecordingMissing(
            f"No recording at {path}. Produce it with LLM_MODE=record (needs GEMINI_API_KEY)."
        )
    try:
        result = schema.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise LLMOutputError(f"Recording {path} does not match {schema.__name__}: {exc}") from exc
    _last_usage = Usage(input_tokens=0, output_tokens=0, model="replay")
    return result


def _call_with_retry[T: BaseModel](
    provider: Provider, prompt: str, schema: type[T], model: str
) -> tuple[T, str]:
    global _last_usage
    last_error: Exception | None = None
    for _ in range(MAX_ATTEMPTS):
        response = provider.generate(prompt, schema, model=model)
        _last_usage = response.usage
        try:
            return schema.model_validate_json(response.text), response.text
        except ValidationError as exc:
            last_error = exc
    raise LLMOutputError(
        f"Provider returned invalid {schema.__name__} JSON after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def _get_provider(settings: Settings) -> Provider:
    global _provider
    if _provider is None:
        if settings.llm_provider != "gemini":
            raise ProviderConfigError(
                f"LLM_PROVIDER={settings.llm_provider!r} arrives in SCRUM-15; only 'gemini' is available."
            )
        if not settings.gemini_api_key:
            raise ProviderConfigError("GEMINI_API_KEY is not set; needed for LLM_MODE=record/live.")
        _provider = GeminiProvider(settings.gemini_api_key)
    return _provider
