"""Skill matching and project ranking: what Jim can truthfully claim for a JD.

Deterministic first. `build_inventory` collects every skill Jim actually has (the CV
skills line plus every knowledge-base project's stack and keywords). `match` places
each JD skill by exact or fuzzy name match (rapidfuzz ratio >= 90) in Python and sends
only the leftovers to the recorded alias call, which may return `same` (matched),
`adjacent` (suggestion, opt-in) or `none` (missing). Adjacent skills move into
`matched` only when listed in `approved_adjacent`. `rank_projects` asks the model for a
raw relevance score per project and applies the x0.5 personal-project weighting and the
ordering here, so professional evidence always outranks personal work at equal relevance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from rapidfuzz import fuzz

from resume_tailor.agents import alias, project_rank
from resume_tailor.docx_io import Para
from resume_tailor.schemas import JDIntent, JDKeywords, ProjectFact, SkillMatch
from resume_tailor.sections import classify

FUZZY_THRESHOLD = 90
PERSONAL_WEIGHT = 0.5

_PARENS = re.compile(r"\(([^()]*)\)")
_SPLIT = re.compile(r"[|,;]")
_WS = re.compile(r"\s+")


def _clean(item: str) -> str:
    return _WS.sub(" ", item).strip(" .")


def skills_line_terms(text: str) -> list[str]:
    """Terms of a `|`-delimited skills line; parenthesised lists are split out too.

    "Python (NumPy, Pandas)" -> ["Python", "NumPy", "Pandas"].
    """
    terms: list[str] = []
    for chunk in text.split("|"):
        inner = " , ".join(_PARENS.findall(chunk))
        outer = _PARENS.sub(" ", chunk)
        for part in _SPLIT.split(outer + " , " + inner):
            term = _clean(part)
            if term:
                terms.append(term)
    return terms


def build_inventory(cv_paras: Sequence[Para], facts: Iterable[ProjectFact]) -> set[str]:
    """Every skill Jim can claim: the CV skills line plus all facts' stack and keywords."""
    sections = classify(list(cv_paras))
    inventory: dict[str, str] = {}

    def add(term: str) -> None:
        term = _clean(term)
        if term:
            inventory.setdefault(term.lower(), term)

    for para in cv_paras:
        if sections.get(para.id) == "skills":
            for term in skills_line_terms(para.text):
                add(term)
    for fact in facts:
        for term in [*fact.stack, *fact.keywords]:
            add(term)
    return set(inventory.values())


def _jd_skills(keywords: JDKeywords) -> list[str]:
    seen: set[str] = set()
    skills: list[str] = []
    for skill in [*keywords.mandatory, *keywords.nice_to_have]:
        skill = _clean(skill)
        if skill and skill not in seen:
            seen.add(skill)
            skills.append(skill)
    return skills


def find_in_inventory(skill: str, inventory: Iterable[str]) -> str | None:
    """The inventory term equal to `skill` ignoring case, else the best fuzzy hit at ratio >= 90."""
    wanted = skill.lower()
    best: tuple[float, str] | None = None
    for item in inventory:
        if item.lower() == wanted:
            return item
        score = fuzz.ratio(wanted, item.lower())
        if score >= FUZZY_THRESHOLD and (best is None or score > best[0]):
            best = (score, item)
    return best[1] if best else None


def apply_approvals(result: SkillMatch, approved_adjacent: Iterable[str]) -> SkillMatch:
    """Move the approved adjacent skills into `matched`; everything else is unchanged."""
    approved = {_clean(s).lower() for s in approved_adjacent}
    promoted = [s for s in result.adjacent if s.lower() in approved]
    return SkillMatch(
        matched=[*result.matched, *promoted],
        adjacent=[s for s in result.adjacent if s.lower() not in approved],
        missing=list(result.missing),
        aliases=dict(result.aliases),
    )


def match(
    keywords: JDKeywords,
    inventory: Iterable[str],
    facts: Iterable[ProjectFact],
    *,
    approved_adjacent: Iterable[str] = (),
    case: str | None = None,
) -> SkillMatch:
    """Bucket every JD skill (mandatory + nice-to-have) into matched / adjacent / missing.

    `case` names the alias recording; it is only used when some skills need the model.
    """
    inventory = list(inventory)
    facts = list(facts)
    matched: list[str] = []
    aliases: dict[str, str] = {}
    unresolved: list[str] = []

    for skill in _jd_skills(keywords):
        hit = find_in_inventory(skill, inventory)
        if hit is None:
            unresolved.append(skill)
            continue
        matched.append(skill)
        if hit.lower() != skill.lower():
            aliases[skill] = hit

    adjacent: list[str] = []
    missing: list[str] = []
    if unresolved:
        resolution = alias.run(unresolved, inventory, facts, case=case)
        for mapping in resolution.mappings:
            if mapping.relation == "same":
                matched.append(mapping.jd_skill)
                aliases[mapping.jd_skill] = mapping.inventory_skill
            elif mapping.relation == "adjacent":
                adjacent.append(mapping.jd_skill)
                aliases[mapping.jd_skill] = mapping.inventory_skill
            else:
                missing.append(mapping.jd_skill)

    result = SkillMatch(matched=matched, adjacent=adjacent, missing=missing, aliases=aliases)
    return apply_approvals(result, approved_adjacent)


def weighted_ranking(facts: Sequence[ProjectFact], raw_scores: dict[str, float]) -> list[tuple[ProjectFact, float]]:
    """Apply the professional weighting and sort: score desc, professional first, then KB order."""
    by_lower = {name.lower(): score for name, score in raw_scores.items()}
    scored: list[tuple[int, ProjectFact, float]] = []
    for index, fact in enumerate(facts):
        score = min(1.0, max(0.0, float(by_lower.get(fact.name.lower(), 0.0))))
        if not fact.professional:
            score *= PERSONAL_WEIGHT
        scored.append((index, fact, round(score, 4)))
    scored.sort(key=lambda item: (-item[2], not item[1].professional, item[0]))
    return [(fact, score) for _, fact, score in scored]


def rank_projects(
    intent: JDIntent, facts: Iterable[ProjectFact], *, case: str | None = None
) -> list[tuple[ProjectFact, float]]:
    """Projects most relevant to the JD first; personal projects are weighted x0.5.

    `case` names the recording (`tests/recordings/project_rank/<case>.json`).
    """
    facts = list(facts)
    return weighted_ranking(facts, project_rank.run(intent, facts, case=case))
