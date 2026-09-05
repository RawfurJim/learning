"""Pydantic models for every LLM output. Agent schemas are added by their tickets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, computed_field


class Ping(BaseModel):
    """Smallest possible schema, used by the LLM smoke test and harness tests."""

    ok: bool


class ProjectFact(BaseModel):
    """One `## <Project name>` block from the knowledge base (`knowledge/projects.md`)."""

    name: str
    category: Literal["work", "personal"]
    employer: str = ""
    period: str = ""
    problem: str = ""
    built: str = ""
    stack: list[str] = []
    metrics: list[str] = []
    keywords: list[str] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def professional(self) -> bool:
        """Work (JudgeService) evidence outranks personal projects everywhere."""
        return self.category == "work"
