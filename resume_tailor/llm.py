"""The single LLM entry point: `generate_json(prompt, schema, *, case)`.

Modes (from `LLM_MODE`):
- `replay` (test default): read `tests/recordings/<agent>/<case>.json`, validate with the schema.
- `record`: call the provider, validate, write the recording.
- `live`: call the provider, write nothing.

Providers (from `LLM_PROVIDER`, built by `build_provider`):
- `gemini`: `google-genai`, JSON mode with `response_schema`. Needs `GEMINI_API_KEY`.
- `groq`: OpenAI-compatible chat completions over plain HTTPS, `response_format=json_object`.
  Needs `GROQ_API_KEY`.
- `ollama`: local `/api/chat` with `format=json` (`OLLAMA_BASE_URL`, default localhost:11434).
Any other value -> `ConfigError`.

Invalid JSON from the provider is retried once, then `LLMOutputError` is raised.
`estimate_cost(usage)` prices a `Usage`/`TokenUsage` with the GBP-per-1M-token table below.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from resume_tailor.settings import PROVIDERS, Settings

DEFAULT_AGENT = "llm"
MAX_ATTEMPTS = 2
HTTP_TIMEOUT_S = 120

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class RecordingMissing(FileNotFoundError):
    """Replay mode found no recording for the requested case."""


class LLMOutputError(RuntimeError):
    """The provider returned output that did not validate against the schema, twice."""


class ConfigError(RuntimeError):
    """The configured provider cannot be built (unknown LLM_PROVIDER, missing API key)."""


ProviderConfigError = ConfigError  # SCRUM-8 name; `ConfigError` is canonical since SCRUM-15.


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


# ---------------------------------------------------------------------------------------
# Price table
# ---------------------------------------------------------------------------------------

# 1 USD in GBP. open.er-api.com, 2026-09-05 00:02 UTC. Update when the pound moves a lot.
USD_TO_GBP = 0.7396


@dataclass(frozen=True)
class ModelPrice:
    input_gbp_per_1m: float
    output_gbp_per_1m: float


def _usd(input_usd: float, output_usd: float) -> ModelPrice:
    return ModelPrice(input_usd * USD_TO_GBP, output_usd * USD_TO_GBP)


# GBP per 1M tokens (input, output), paid tier, text prompts. Keys are matched exactly first,
# then as the longest prefix of the model name, so `gemini-3.6-flash-preview` prices as
# `gemini-3.6-flash`.
#
# Gemini: https://ai.google.dev/gemini-api/docs/pricing, read 2026-09-05. The 3.x Flash
# prices ($0.75 / $3.75) are promotional through 2026-12-31 and double from 2027-01-01.
# Groq: list prices from groq.com/pricing as remembered, not fetched. TODO(Jim): confirm.
# Ollama runs locally, so every model served by it costs 0.
PRICES_GBP_PER_1M: dict[str, ModelPrice] = {
    "gemini-3.8-flash": _usd(0.75, 3.75),
    "gemini-3.7-flash": _usd(0.75, 3.75),
    "gemini-3.6-flash": _usd(0.75, 3.75),
    "gemini-3.5-flash": _usd(1.50, 9.00),
    "gemini-2.5-pro": _usd(1.25, 10.00),  # prompts <= 200k tokens
    "gemini-2.5-flash-lite": _usd(0.10, 0.40),
    "gemini-2.5-flash": _usd(0.30, 2.50),
    "llama-3.3-70b-versatile": _usd(0.59, 0.79),  # TODO(Jim): confirm Groq price
    "llama-3.1-8b-instant": _usd(0.05, 0.08),  # TODO(Jim): confirm Groq price
    "openai/gpt-oss-120b": _usd(0.15, 0.60),  # TODO(Jim): confirm Groq price
    "openai/gpt-oss-20b": _usd(0.075, 0.30),  # TODO(Jim): confirm Groq price
    # Ollama (local, free). Add any other local model here with ModelPrice(0, 0).
    "llama3.1": ModelPrice(0.0, 0.0),
    "llama3.2": ModelPrice(0.0, 0.0),
    "gemma3": ModelPrice(0.0, 0.0),
    "qwen3": ModelPrice(0.0, 0.0),
    "mistral": ModelPrice(0.0, 0.0),
}


def price_for(model: str) -> ModelPrice | None:
    """Price of `model`, or None when it is not in the table (cost is then reported as 0)."""
    if model in PRICES_GBP_PER_1M:
        return PRICES_GBP_PER_1M[model]
    for key in sorted(PRICES_GBP_PER_1M, key=len, reverse=True):
        if model.startswith(key):
            return PRICES_GBP_PER_1M[key]
    return None


def estimate_cost(usage: object, model: str | None = None) -> float:
    """GBP for `usage` (`Usage` or `TokenUsage`: needs `input_tokens`, `output_tokens`, `model`).

    Unknown models (and local Ollama models) cost 0.0; `price_for` tells the two apart.
    """
    model = model or getattr(usage, "model", "") or ""
    price = price_for(model)
    if price is None:
        return 0.0
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return (input_tokens * price.input_gbp_per_1m + output_tokens * price.output_gbp_per_1m) / 1_000_000


# ---------------------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------------------


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


PostJson = Callable[[str, dict, dict], dict]
"""`post(url, payload, headers) -> parsed JSON body`; providers accept a fake for tests."""


def post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST `payload` as JSON with the standard library; HTTP errors carry the response body."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310 (fixed https/localhost)
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc


def json_instructions(schema: type[BaseModel]) -> str:
    """Prompt suffix for providers without a schema-constrained JSON mode."""
    return (
        "\n\nRespond with a single JSON object and nothing else (no code fences, no commentary). "
        "It must validate against this JSON schema:\n" + json.dumps(schema.model_json_schema())
    )


class GroqProvider:
    """Groq's OpenAI-compatible chat completions endpoint, JSON object mode, plain HTTPS (no SDK)."""

    def __init__(self, api_key: str, *, url: str = GROQ_URL, post: PostJson | None = None) -> None:
        self._api_key = api_key
        self._url = url
        self._post = post or post_json

    def generate(self, prompt: str, schema: type[BaseModel], *, model: str) -> ProviderResponse:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt + json_instructions(schema)}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        data = self._post(self._url, payload, {"Authorization": f"Bearer {self._api_key}"})
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return ProviderResponse(
            text=text,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                model=model,
            ),
        )


class OllamaProvider:
    """A local Ollama server: `POST <base_url>/api/chat` with `format=json`, plain HTTP (no SDK)."""

    def __init__(self, base_url: str, *, post: PostJson | None = None) -> None:
        self._url = base_url.rstrip("/") + "/api/chat"
        self._post = post or post_json

    def generate(self, prompt: str, schema: type[BaseModel], *, model: str) -> ProviderResponse:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt + json_instructions(schema)}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        data = self._post(self._url, payload, {})
        text = (data.get("message") or {}).get("content") or ""
        return ProviderResponse(
            text=text,
            usage=Usage(
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
                model=model,
            ),
        )


def build_provider(settings: Settings) -> Provider:
    """The provider named by `settings.llm_provider`, or `ConfigError`. Makes no network call."""
    name = settings.llm_provider.strip().lower()
    if name == "gemini":
        if not settings.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set; needed for LLM_PROVIDER=gemini in LLM_MODE=record/live.")
        return GeminiProvider(settings.gemini_api_key)
    if name == "groq":
        if not settings.groq_api_key:
            raise ConfigError("GROQ_API_KEY is not set; needed for LLM_PROVIDER=groq.")
        return GroqProvider(settings.groq_api_key)
    if name == "ollama":
        return OllamaProvider(settings.ollama_base_url)
    raise ConfigError(f"Unknown LLM_PROVIDER={settings.llm_provider!r}; choose one of {', '.join(PROVIDERS)}.")


# ---------------------------------------------------------------------------------------
# generate_json, usage tracking, record/replay
# ---------------------------------------------------------------------------------------

_provider: Provider | None = None
_provider_lock = threading.Lock()
# Per thread, so agents running in parallel (pipeline ThreadPoolExecutor) each read
# the usage of their own last call.
_state = threading.local()


def set_provider(provider: Provider | None) -> None:
    """Install a provider (tests use a stub). `None` rebuilds from settings on next call."""
    global _provider
    with _provider_lock:
        _provider = provider


def reset_usage() -> None:
    _state.last_usage = None


def record_usage(usage: Usage | None) -> None:
    """Set what `last_usage()` returns in this thread (providers do this; tests stamp fake usage)."""
    _state.last_usage = usage


def last_usage() -> Usage | None:
    """Usage of the most recent `generate_json` call *in this thread*, or None if none has run."""
    return getattr(_state, "last_usage", None)


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
    if not path.exists():
        raise RecordingMissing(
            f"No recording at {path}. Produce it with LLM_MODE=record (needs GEMINI_API_KEY)."
        )
    try:
        result = schema.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise LLMOutputError(f"Recording {path} does not match {schema.__name__}: {exc}") from exc
    record_usage(Usage(input_tokens=0, output_tokens=0, model="replay"))
    return result


def _call_with_retry[T: BaseModel](
    provider: Provider, prompt: str, schema: type[T], model: str
) -> tuple[T, str]:
    last_error: Exception | None = None
    for _ in range(MAX_ATTEMPTS):
        response = provider.generate(prompt, schema, model=model)
        record_usage(response.usage)
        try:
            return schema.model_validate_json(response.text), response.text
        except ValidationError as exc:
            last_error = exc
    raise LLMOutputError(
        f"Provider returned invalid {schema.__name__} JSON after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def _get_provider(settings: Settings) -> Provider:
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = build_provider(settings)
        return _provider
