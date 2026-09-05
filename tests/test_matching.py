"""SCRUM-12: skill matching (matched / adjacent / missing) and project relevance ranking.

Exact and fuzzy matching are pure Python; only the leftovers reach the recorded alias
call. Recordings: `tests/recordings/alias/<case>.json`, `tests/recordings/project_rank/<case>.json`
(re)written by `LLM_MODE=record uv run pytest -m live` (see tests/test_matching_live.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_tailor import matching
from resume_tailor.agents import jd_intent
from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.knowledge import load_projects
from resume_tailor.schemas import JDIntent, JDKeywords, ProjectFact, SkillMatch

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_SAMPLE = FIXTURES_DIR / "kb_sample.md"

# Case names shared with tests/test_matching_live.py, which records them.
CASE_TORCH = "pytorch_aliases"
CASE_FLASK = "flask_vs_fastapi"
CASE_MISSING = "react_kubernetes"
CASE_RANK = "senior_ai_engineer"

SMALL_INVENTORY = {"Python", "FastAPI", "Docker", "PyTorch", "AWS"}


def _keywords(mandatory: list[str], nice_to_have: list[str] = ()) -> JDKeywords:
    return JDKeywords(
        mandatory=list(mandatory),
        nice_to_have=list(nice_to_have),
        ats_keywords=[*mandatory, *nice_to_have],
        responsibilities=[],
    )


@pytest.fixture(scope="module")
def facts() -> list[ProjectFact]:
    return load_projects(KB_SAMPLE)


@pytest.fixture(scope="module")
def inventory(facts: list[ProjectFact]) -> set[str]:
    paras = iter_paragraphs(load(FIXTURES_DIR / "sample_cv.docx"))
    return matching.build_inventory(paras, facts)


def _lower(items) -> set[str]:
    return {item.lower() for item in items}


def assert_buckets_partition(result: SkillMatch, keywords: JDKeywords) -> None:
    """Every JD skill lands in exactly one bucket."""
    buckets = [*result.matched, *result.adjacent, *result.missing]
    assert len(buckets) == len(set(buckets)), buckets
    assert set(buckets) == set(keywords.mandatory) | set(keywords.nice_to_have)


def assert_torch_all_matched(result: SkillMatch) -> None:
    assert set(result.matched) == {"PyTorch", "Pytorch", "Torch"}, result
    assert result.adjacent == [] and result.missing == []


def assert_flask_adjacent(result: SkillMatch) -> None:
    assert "Flask" in result.adjacent, result
    assert "Flask" not in result.matched
    assert result.aliases.get("Flask", "").lower() == "fastapi", result.aliases
    assert set(result.matched) == {"Python", "Docker"}


def assert_react_kubernetes_missing(result: SkillMatch) -> None:
    assert {"React", "Kubernetes"} <= set(result.missing), result
    assert not {"React", "Kubernetes"} & (set(result.matched) | set(result.adjacent))
    assert "Python" in result.matched


def test_build_inventory_has_cv_skills_and_fact_stacks(inventory: set[str]) -> None:
    for skill in ("PyTorch", "Python", "NumPy", "AWS", "FastAPI", "Unsloth", "Chroma", "HAProxy"):
        assert skill.lower() in _lower(inventory), skill
    assert not {"react", "kubernetes", "terraform"} & _lower(inventory)


def test_exact_and_fuzzy_match(llm_replay, inventory: set[str], facts: list[ProjectFact]) -> None:
    keywords = _keywords(["PyTorch", "Pytorch", "Torch"])
    result = matching.match(keywords, inventory, facts, case=CASE_TORCH)
    assert_torch_all_matched(result)  # Torch via the recorded alias call
    assert_buckets_partition(result, keywords)


def test_fuzzy_threshold_is_python_only(llm_replay) -> None:
    """Exact and >= 90 ratio hits never reach the LLM (no recording exists for this case)."""
    keywords = _keywords(["pytorch", "Vector databases", "Github Actions"])
    inventory = {"PyTorch", "Vector Databases", "GitHub Actions"}
    result = matching.match(keywords, inventory, [], case="no_recording_needed")
    assert result.matched == ["pytorch", "Vector databases", "Github Actions"]
    assert result.adjacent == [] and result.missing == []


def test_adjacent_is_suggested_not_matched(llm_replay) -> None:
    keywords = _keywords(["Python", "Flask"], ["Docker"])
    result = matching.match(keywords, SMALL_INVENTORY, [], case=CASE_FLASK)
    assert_flask_adjacent(result)
    assert_buckets_partition(result, keywords)

    # approved_adjacent defaults to empty; approving moves the skill into matched.
    approved = matching.match(keywords, SMALL_INVENTORY, [], approved_adjacent=["flask"], case=CASE_FLASK)
    assert "Flask" in approved.matched and "Flask" not in approved.adjacent
    assert matching.apply_approvals(result, ["Flask"]) == approved


def test_missing_stays_missing(llm_replay, inventory: set[str], facts: list[ProjectFact]) -> None:
    keywords = _keywords(["Python", "React"], ["Kubernetes"])
    result = matching.match(keywords, inventory, facts, case=CASE_MISSING)
    assert_react_kubernetes_missing(result)
    assert_buckets_partition(result, keywords)


def assert_top_ranked_are_professional(ranked: list[tuple[ProjectFact, float]], facts: list[ProjectFact]) -> None:
    assert len(ranked) == len(facts)
    assert all(fact.professional for fact, _ in ranked[:3]), [(f.name, s) for f, s in ranked[:3]]
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_rank_prefers_professional(llm_replay, jd, facts: list[ProjectFact]) -> None:
    intent = jd_intent.run(jd("senior_ai_engineer"), case="senior_ai_engineer")
    ranked = matching.rank_projects(intent, facts, case=CASE_RANK)
    assert_top_ranked_are_professional(ranked, facts)

    # Identical raw scores: every work project ranks above every personal one (x0.5).
    raw = {fact.name: 0.8 for fact in facts}
    weighted = matching.weighted_ranking(facts, raw)
    categories = [fact.category for fact, _ in weighted]
    assert categories == ["work"] * 3 + ["personal"] * 6
    assert {score for fact, score in weighted if fact.professional} == {0.8}
    assert {score for fact, score in weighted if not fact.professional} == {0.4}


def test_rank_is_deterministic_in_replay(llm_replay, jd, facts: list[ProjectFact]) -> None:
    intent = jd_intent.run(jd("senior_ai_engineer"), case="senior_ai_engineer")
    first = matching.rank_projects(intent, facts, case=CASE_RANK)
    second = matching.rank_projects(intent, facts, case=CASE_RANK)
    assert [(f.name, s) for f, s in first] == [(f.name, s) for f, s in second]


def test_weighted_ranking_handles_unknown_and_out_of_range(facts: list[ProjectFact]) -> None:
    raw = {facts[0].name: 7.0, facts[-1].name: -1.0}  # everything else unscored -> 0
    ranked = matching.weighted_ranking(facts, raw)
    assert ranked[0][0].name == facts[0].name and ranked[0][1] == 1.0
    assert all(score == 0.0 for _, score in ranked[1:])
    # Ties are broken by professional first, then knowledge-base order.
    tail = [fact.name for fact, _ in ranked[1:]]
    assert tail == [f.name for f in facts[1:] if f.professional] + [f.name for f in facts[1:] if not f.professional]
