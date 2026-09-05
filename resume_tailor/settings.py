"""Runtime settings loaded from `.env` and the process environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_RECORDINGS_DIR = REPO_ROOT / "tests" / "recordings"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache"

LLMProvider = Literal["gemini", "groq", "ollama"]
PROVIDERS: tuple[str, ...] = ("gemini", "groq", "ollama")
# The model used when the provider is switched without naming a model (sidebar, .env).
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.6-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
}
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
LLMMode = Literal["replay", "record", "live"]

# env var name -> Settings field
_ENV_FIELDS = {
    "GEMINI_API_KEY": "gemini_api_key",
    "GROQ_API_KEY": "groq_api_key",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "LLM_PROVIDER": "llm_provider",
    "LLM_MODEL": "llm_model",
    "LLM_MODE": "llm_mode",
    "LLM_RECORDINGS_DIR": "recordings_dir",
    "CACHE_DIR": "cache_dir",
    "REVIEWER_SEMANTIC": "reviewer_semantic",
}


class Settings(BaseModel):
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    # Kept as `str` (not `LLMProvider`) so an unknown value surfaces as `llm.ConfigError`
    # when the provider is built, not as a validation error while reading `.env`.
    llm_provider: str = "gemini"
    llm_model: str = DEFAULT_MODELS["gemini"]
    llm_mode: LLMMode = "replay"
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    # Agent 6's semantic-drift call. Its deterministic checks always run; set
    # REVIEWER_SEMANTIC=0 to skip the one extra LLM call per rewritten paragraph.
    reviewer_semantic: bool = True

    @classmethod
    def from_env(
        cls,
        env_file: Path | None = DEFAULT_ENV_FILE,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Build settings from `env_file` (if it exists), overridden by `environ`.

        `environ` defaults to `os.environ`; pass `{}` to read the file alone.
        """
        raw: dict[str, str] = {}
        if env_file is not None and env_file.exists():
            raw.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        raw.update(os.environ if environ is None else environ)

        values: dict[str, object] = {}
        for env_name, field in _ENV_FIELDS.items():
            value = raw.get(env_name)
            if value is None or value.strip() == "":
                continue  # blank means "unset", e.g. GEMINI_API_KEY= in .env.example
            values[field] = value.strip()
        return cls.model_validate(values)
