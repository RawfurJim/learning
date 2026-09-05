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


class SkillMatch(BaseModel):
    """What Jim can truthfully claim for a JD (code, not an LLM output).

    Every JD skill lands in exactly one bucket, spelled as the JD spelled it.
    `adjacent` skills are suggestions only: they move to `matched` when the user
    approves them (`approved_adjacent`), never on their own. `aliases` records the
    inventory term that justified a match or suggestion (`{"Flask": "FastAPI"}`).
    """

    matched: list[str] = []
    adjacent: list[str] = []
    missing: list[str] = []
    aliases: dict[str, str] = {}


AliasRelation = Literal["same", "adjacent", "none"]


class AliasMapping(BaseModel):
    """One JD skill resolved against the inventory by the alias call."""

    jd_skill: str
    relation: AliasRelation
    inventory_skill: str = ""
    reason: str = ""


class AliasResolution(BaseModel):
    """Alias / adjacency call output: one mapping per unmatched JD skill."""

    mappings: list[AliasMapping]


class ProjectScore(BaseModel):
    """Relevance of one knowledge-base project to a JD, 0 (irrelevant) to 1 (central)."""

    name: str
    score: float
    reason: str = ""


class ProjectRanking(BaseModel):
    """Project-ranking call output: a raw score per project (professional weighting is applied in code)."""

    scores: list[ProjectScore]
