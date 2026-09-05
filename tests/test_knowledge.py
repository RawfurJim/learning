"""SCRUM-10: knowledge base loader, allowed vocabulary and number extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_tailor.knowledge import allowed_vocabulary, extract_numbers, load_projects
from resume_tailor.schemas import ProjectFact

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_SAMPLE = FIXTURES_DIR / "kb_sample.md"
CV_TEXT = FIXTURES_DIR / "cv_text.txt"


@pytest.fixture(scope="module")
def facts() -> list[ProjectFact]:
    return load_projects(KB_SAMPLE)


@pytest.fixture(scope="module")
def vocab(facts: list[ProjectFact]) -> set[str]:
    return allowed_vocabulary(CV_TEXT.read_text(encoding="utf-8"), facts)


def _lower(terms: set[str]) -> set[str]:
    return {t.lower() for t in terms}


def test_parse_counts(facts: list[ProjectFact]) -> None:
    work = [f for f in facts if f.category == "work"]
    personal = [f for f in facts if f.category == "personal"]
    assert len(work) == 3
    assert len(personal) == 6
    assert len(facts) == 9


def test_work_projects_are_professional(facts: list[ProjectFact]) -> None:
    for fact in facts:
        if fact.category == "work":
            assert fact.professional is True
        else:
            assert fact.professional is False


def test_vocabulary_contains_kb_and_cv_terms(vocab: set[str]) -> None:
    expected = {"HAProxy", "Gemma", "WhisperX", "Unsloth", "FastAPI", "Chroma", "XGBRegressor"}
    assert _lower(expected) <= _lower(vocab)
    assert "react" not in _lower(vocab)
    assert "kubernetes" not in _lower(vocab)


def test_vocabulary_case_insensitive(vocab: set[str]) -> None:
    assert "pytorch" in vocab
    assert "PyTorch" in vocab


def test_metrics_extracted(facts: list[ProjectFact]) -> None:
    joined = " ".join(metric for fact in facts for metric in fact.metrics)
    for token in ["93%", "100k", "0.78", "0.96", "0.98", "20%", "90%", "2x", "48 GB", "2,000-character"]:
        assert token in joined, token


def test_extract_numbers() -> None:
    text = "Raised recall from 0.78 to 0.96 (F1 0.98) on 1,500 reviews"
    assert extract_numbers(text) == ["0.78", "0.96", "0.98", "1,500"]


def test_missing_kb_file_gives_empty_facts_not_crash(tmp_path: Path) -> None:
    assert load_projects(tmp_path / "does_not_exist.md") == []
