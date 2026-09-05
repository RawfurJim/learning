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

LLMProvider = Literal["gemini", "groq", "ollama"]
LLMMode = Literal["replay", "record", "live"]

# env var name -> Settings field
_ENV_FIELDS = {
    "GEMINI_API_KEY": "gemini_api_key",
    "LLM_PROVIDER": "llm_provider",
    "LLM_MODEL": "llm_model",
    "LLM_MODE": "llm_mode",
    "LLM_RECORDINGS_DIR": "recordings_dir",
}


class Settings(BaseModel):
    gemini_api_key: str | None = None
    llm_provider: LLMProvider = "gemini"
    llm_model: str = "gemini-3.6-flash"
    llm_mode: LLMMode = "replay"
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR

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
