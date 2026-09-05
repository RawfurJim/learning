"""Pydantic models for every LLM output. Agent schemas are added by their tickets."""

from __future__ import annotations

from pydantic import BaseModel


class Ping(BaseModel):
    """Smallest possible schema, used by the LLM smoke test and harness tests."""

    ok: bool
