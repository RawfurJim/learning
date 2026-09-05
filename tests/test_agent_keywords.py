"""Agent 2 (Keyword Extractor) contract tests (names from SCRUM-11). Replays recorded Gemini output."""

from __future__ import annotations

import pytest

from resume_tailor.agents import keywords
from resume_tailor.schemas import JDKeywords

CASES = ["senior_ai_engineer", "ds_nlp", "ml_platform"]


def _lowered(items: list[str]) -> set[str]:
    return {item.lower() for item in items}


def assert_senior_ai_engineer_keywords(result: JDKeywords) -> None:
    """Shared with the live test: what a correct extraction from senior_ai_engineer.txt looks like."""
    assert {"Python", "PyTorch", "RAG", "AWS"} <= set(result.mandatory), result.mandatory
    assert "Kubernetes" in result.nice_to_have, result.nice_to_have
    assert "React" in result.nice_to_have, result.nice_to_have
    assert not _lowered(result.mandatory) & _lowered(result.nice_to_have)
    assert set(result.mandatory) <= set(result.ats_keywords)
    assert result.responsibilities


def test_keywords_mandatory_vs_nice(llm_replay, jd) -> None:
    result = keywords.run(jd("senior_ai_engineer"), case="senior_ai_engineer")
    assert {"Python", "PyTorch", "RAG", "AWS"} <= set(result.mandatory), result.mandatory
    assert "Kubernetes" in result.nice_to_have, result.nice_to_have
    assert "React" in result.nice_to_have, result.nice_to_have
    assert not _lowered(result.mandatory) & _lowered(result.nice_to_have)


@pytest.mark.parametrize("case", CASES)
def test_keywords_deduplicated_and_canonical(llm_replay, jd, case: str) -> None:
    result = keywords.run(jd(case), case=case)
    for bucket in (result.mandatory, result.nice_to_have, result.ats_keywords, result.responsibilities):
        assert bucket, case
        lowered = [item.lower() for item in bucket]
        assert len(lowered) == len(set(lowered)), bucket
        for item in bucket:
            assert item == item.strip() and item, repr(item)
            assert "  " not in item, repr(item)


@pytest.mark.parametrize("case", CASES)
def test_ats_keywords_superset(llm_replay, jd, case: str) -> None:
    result = keywords.run(jd(case), case=case)
    assert set(result.mandatory) <= set(result.ats_keywords), (
        set(result.mandatory) - set(result.ats_keywords)
    )


def test_normalise_merges_case_duplicates_and_prefers_jd_casing() -> None:
    """The deterministic post-processing that makes the recording tests hold regardless of the model."""
    raw = JDKeywords(
        mandatory=["python", " Python ", "pytorch", "Amazon  Web Services", "rag"],
        nice_to_have=["Kubernetes", "PYTHON", "react"],
        ats_keywords=["Docker", "kubernetes"],
        responsibilities=["Build RAG pipelines", "build rag pipelines "],
    )
    jd_text = "Must have: Python, PyTorch, RAG on AWS (Amazon Web Services). Nice: Kubernetes, React."
    result = keywords.normalise(raw, jd_text)
    assert result.mandatory == ["Python", "PyTorch", "Amazon Web Services", "RAG"]
    assert result.nice_to_have == ["Kubernetes", "React"]  # PYTHON dropped: already mandatory
    assert result.ats_keywords[:2] == ["Docker", "Kubernetes"]
    assert set(result.mandatory) <= set(result.ats_keywords)
    assert result.responsibilities == ["Build RAG pipelines"]
