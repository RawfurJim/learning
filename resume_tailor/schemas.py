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


class SummarySkillsRewrite(BaseModel):
    """Agent 4 raw output: the rewritten professional summary and the new skills-line entries."""

    summary: str
    skills: list[str]


class SummarySkillsResult(BaseModel):
    """Agent 4 after the Python checks (vocabulary, numbers, length, ordering, budget).

    `summary_reverted` / `skills_reverted` say that the model's output still broke a rule
    after one retry and the original text (skills: original entries, mandatory-first)
    was used instead. `notes` explains every correction in plain English.
    """

    summary: str
    skills: list[str]
    summary_reverted: bool = False
    skills_reverted: bool = False
    notes: list[str] = []
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class BulletRewrite(BaseModel):
    """Agent 3: one rewritten experience bullet.

    `used_kb_metrics` lists every knowledge-base metric (verbatim from the project's
    `metrics`) the bullet now mentions; `jd_keywords_used` lists the JD keywords woven in.
    """

    para_id: str
    text: str
    used_kb_metrics: list[str] = []
    jd_keywords_used: list[str] = []


class BulletGroupRewrite(BaseModel):
    """Agent 3 raw output for one project group: one entry per bullet, same order."""

    bullets: list[BulletRewrite]


class ExperienceResult(BaseModel):
    """Agent 3 after the Python checks: one checked `BulletRewrite` per experience bullet.

    A bullet that still broke a rule after one retry carries its original text and is
    listed in `reverted`; `projects` maps each bullet to the knowledge-base project it
    was grounded in (`""` when none matched). `notes` explains every correction.
    """

    bullets: list[BulletRewrite]
    originals: dict[str, str]
    projects: dict[str, str] = {}
    reverted: list[str] = []
    notes: list[str] = []
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def changed(self) -> dict[str, str]:
        """`{para_id: new text}` for the bullets whose text differs from the original."""
        return {b.para_id: b.text for b in self.bullets if b.text != self.originals.get(b.para_id)}


class Verdict(BaseModel):
    """Agent 6's answer for one rewritten paragraph: may it replace the original?

    `reason` is written for Jim, not for the model: "term not in CV/KB: React",
    "metric missing: 93%", "length: 34 vs budget 20-24" or the semantic check's
    explanation. It is filled in for an accepted rewrite too, so the UI can show why.
    """

    accept: bool
    reason: str = ""


class DriftVerdict(BaseModel):
    """The reviewer's one LLM call: is every claim in the rewrite supported by its sources?"""

    supported: bool
    reason: str = ""


class Revert(BaseModel):
    """A rewrite the reviewer refused: the paragraph keeps `original` and carries `reason`."""

    para_id: str
    section: str = ""
    reason: str = ""
    original: str = ""
    rejected: str = ""


class TokenUsage(BaseModel):
    """Tokens summed over every LLM call of a pipeline run (0 in replay mode) and their price.

    `estimated_cost_gbp` comes from `llm.estimate_cost` and the per-model price table; it is
    0 for unknown models and for local (Ollama) models. `total_*` are the ticket's names for
    the same sums and are included in `model_dump()`.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    model: str = ""
    estimated_cost_gbp: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_output_tokens(self) -> int:
        return self.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
