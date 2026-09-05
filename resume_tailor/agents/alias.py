"""Alias / adjacency resolution used by `resume_tailor.matching`.

Only the JD skills that exact and fuzzy matching could not place reach this call. For
each one the model says whether Jim's inventory already contains the *same* thing under
another name (Torch -> PyTorch), a directly *adjacent* tool from the same family
(Flask -> FastAPI), or nothing (`none`). Python then checks that every named inventory
skill really is in the inventory; anything else is demoted to `none`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from resume_tailor import llm
from resume_tailor.schemas import AliasMapping, AliasResolution, ProjectFact

AGENT = "alias"

PROMPT = """You are checking a candidate's CV against a job description (JD) for an honest recruiter. The candidate must never claim a skill they do not have.

Below is the candidate's INVENTORY: every tool, technology and skill that appears in their CV or in their own project notes. Below that is a list of JD SKILLS that did not match the inventory by name.

For each JD skill decide exactly one relation:
- "same": the JD skill is the same thing as an inventory item under a different name, spelling, abbreviation or casing (for example Torch = PyTorch, K8s = Kubernetes, Amazon Web Services = AWS, sklearn = scikit-learn, Postgres = PostgreSQL, GH Actions = GitHub Actions). The two names must refer to one and the same tool or concept.
- "adjacent": a different tool from the same narrow family that does the same job, where hands-on experience with the inventory item transfers almost directly (for example Flask <-> FastAPI, TensorFlow <-> PyTorch, Azure <-> AWS, Pinecone <-> Chroma, GitLab CI <-> GitHub Actions). Do not use "adjacent" for tools that merely work together or are commonly used side by side.
- "none": the inventory has nothing that is the same or directly adjacent.

Be strict:
- Docker is not adjacent to Kubernetes (containers vs orchestration). A back-end framework is not adjacent to a front-end one. A cloud service is not "same" as the cloud provider. A programming language is not adjacent to a different language.
- A broad category in the inventory (for example "NLP" or "Cloud Platforms") does not make a specific tool "same"; that is "none" unless the specific tool itself is in the inventory.
- Skills that are experience levels, soft skills, domains or generic phrases with no inventory equivalent are "none".
- inventory_skill must be copied exactly from the INVENTORY list for "same" and "adjacent", and must be an empty string for "none".

Return a JSON object with one field, "mappings": a list with exactly one entry per JD skill, in the given order, each with fields jd_skill (copied exactly), relation, inventory_skill and a short reason.

INVENTORY:
{inventory}

How the candidate used these tools (for context only; do not add anything from here that is not in INVENTORY):
{context}

JD SKILLS:
{skills}
"""

_MAX_CONTEXT_CHARS = 220


def _context(facts: Iterable[ProjectFact]) -> str:
    lines = []
    for fact in facts:
        built = fact.built.strip()
        if len(built) > _MAX_CONTEXT_CHARS:
            built = built[: _MAX_CONTEXT_CHARS - 3].rstrip() + "..."
        lines.append(f"- {fact.name}: {built}" if built else f"- {fact.name}")
    return "\n".join(lines) or "- (no project notes)"


def build_prompt(skills: list[str], inventory: Iterable[str], facts: Iterable[ProjectFact] = ()) -> str:
    return PROMPT.format(
        inventory="\n".join(f"- {item}" for item in sorted(set(inventory), key=str.lower)),
        context=_context(facts),
        skills="\n".join(f"- {skill}" for skill in skills),
    )


def default_case(skills: list[str], inventory: Iterable[str]) -> str:
    """Recording case name derived from the call's inputs."""
    payload = json.dumps([skills, sorted({s.lower() for s in inventory})], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _validate(raw: AliasResolution, skills: list[str], inventory: Iterable[str]) -> AliasResolution:
    """One mapping per requested skill; a relation without a real inventory term becomes `none`."""
    by_lower = {item.lower(): item for item in inventory}
    by_skill = {m.jd_skill.strip().lower(): m for m in raw.mappings}
    mappings: list[AliasMapping] = []
    for skill in skills:
        found = by_skill.get(skill.lower())
        inventory_skill = by_lower.get(found.inventory_skill.strip().lower()) if found else None
        if found is None or found.relation == "none" or inventory_skill is None:
            reason = found.reason if found else "not resolved by the model"
            mappings.append(AliasMapping(jd_skill=skill, relation="none", reason=reason))
        else:
            mappings.append(
                AliasMapping(jd_skill=skill, relation=found.relation, inventory_skill=inventory_skill, reason=found.reason)
            )
    return AliasResolution(mappings=mappings)


def run(
    skills: list[str],
    inventory: Iterable[str],
    facts: Iterable[ProjectFact] = (),
    *,
    case: str | None = None,
) -> AliasResolution:
    """Resolve `skills` against `inventory`. `case` names the recording (`tests/recordings/alias/<case>.json`)."""
    inventory = list(inventory)
    if not skills:
        return AliasResolution(mappings=[])
    case = case or default_case(skills, inventory)
    raw = llm.generate_json(build_prompt(skills, inventory, facts), AliasResolution, case=f"{AGENT}/{case}")
    return _validate(raw, skills, inventory)
