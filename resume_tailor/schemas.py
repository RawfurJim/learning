"""Pydantic models for every LLM output. Agent schemas are added by their tickets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


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


Seniority = Literal["junior", "mid", "senior", "lead"]


class JDIntent(BaseModel):
    """Agent 1: what the job description really wants."""

    role_summary: str
    seniority: Seniority
    tone: str
    top_priorities: list[str] = Field(min_length=5, max_length=5)
    domain: str


class JDKeywords(BaseModel):
    """Agent 2: skills and phrases an ATS will scan for, bucketed by how hard the JD requires them."""

    mandatory: list[str]
    nice_to_have: list[str]
    ats_keywords: list[str]
    responsibilities: list[str]
