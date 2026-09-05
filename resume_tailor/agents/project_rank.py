"""Project relevance scoring used by `resume_tailor.matching.rank_projects`.

The model scores every knowledge-base project 0-1 for relevance to the JD intent. It
is deliberately not told which projects are professional: the x0.5 personal-project
weighting and the final ordering are applied in Python.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from resume_tailor import llm
from resume_tailor.schemas import JDIntent, ProjectFact, ProjectRanking

AGENT = "project_rank"

PROMPT = """You are helping a candidate decide which of their past projects to emphasise for a specific job. Score how relevant each project is to the role described below.

Role (from the job description):
- Summary: {role_summary}
- Seniority: {seniority}
- Domain: {domain}
- Top priorities, most important first:
{priorities}

Scoring rules:
- score is a number from 0 to 1. 1.0 = the project is direct evidence for the role's core priorities (same kind of system, same techniques, production use); 0.5 = partially related (shares some techniques or domain); 0.0 = unrelated.
- Judge only from the project descriptions given. Do not assume anything that is not written.
- Weigh production, scale, evaluation rigour and the specific technologies the role names. A project that uses the role's named tools for the role's actual task scores higher than one that only shares a general field.
- Use the full range; do not give every project a similar score.

Return a JSON object with one field, "scores": a list with exactly one entry per project, each with fields name (copied exactly from the heading), score and a one-sentence reason.

Projects:
{projects}
"""


def _project_block(fact: ProjectFact) -> str:
    lines = [f"### {fact.name}"]
    if fact.problem:
        lines.append(f"problem: {fact.problem}")
    if fact.built:
        lines.append(f"built: {fact.built}")
    if fact.stack:
        lines.append(f"stack: {', '.join(fact.stack)}")
    if fact.metrics:
        lines.append(f"metrics: {', '.join(fact.metrics)}")
    if fact.keywords:
        lines.append(f"keywords: {', '.join(fact.keywords)}")
    return "\n".join(lines)


def build_prompt(intent: JDIntent, facts: Iterable[ProjectFact]) -> str:
    return PROMPT.format(
        role_summary=intent.role_summary.strip(),
        seniority=intent.seniority,
        domain=intent.domain.strip(),
        priorities="\n".join(f"  {i}. {p}" for i, p in enumerate(intent.top_priorities, 1)),
        projects="\n\n".join(_project_block(fact) for fact in facts),
    )


def default_case(intent: JDIntent, facts: Iterable[ProjectFact]) -> str:
    """Recording case name derived from the call's inputs."""
    payload = json.dumps([intent.model_dump(), [f.name for f in facts]], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run(intent: JDIntent, facts: Iterable[ProjectFact], *, case: str | None = None) -> dict[str, float]:
    """Raw relevance score per project name (0-1, clamped). Unscored projects get 0.0.

    `case` names the recording (`tests/recordings/project_rank/<case>.json`).
    """
    facts = list(facts)
    if not facts:
        return {}
    case = case or default_case(intent, facts)
    raw = llm.generate_json(build_prompt(intent, facts), ProjectRanking, case=f"{AGENT}/{case}")
    by_lower = {entry.name.strip().lower(): entry.score for entry in raw.scores}
    return {fact.name: min(1.0, max(0.0, by_lower.get(fact.name.lower(), 0.0))) for fact in facts}
