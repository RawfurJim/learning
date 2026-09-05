"""LLM agents. Each module exposes one `run(...)` and keeps its prompt as a module constant."""

from __future__ import annotations

import hashlib


def default_case(jd_text: str) -> str:
    """Recording case name for a JD with no explicit name: a short content hash."""
    return hashlib.sha256(jd_text.strip().encode("utf-8")).hexdigest()[:12]
