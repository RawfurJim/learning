"""SCRUM-16: Agent 6 (Reviewer / Guard) and the assembler's layout checks.

The three deterministic checks need no model at all; the semantic-drift check replays
`tests/recordings/reviewer/{semantic_drift_team,clean_rewrite}.json`, recorded with
`LLM_MODE=record uv run pytest -m live -k reviewer_live`. Every test runs under the
`llm_replay` fixture, so an unexpected LLM call fails with `RecordingMissing` instead of
quietly calling Gemini.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_tailor import assembler
from resume_tailor.agents import reviewer
from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.knowledge import allowed_vocabulary, load_projects
from resume_tailor.schemas import ProjectFact
from resume_tailor.sections import classify

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_SAMPLE = FIXTURES_DIR / "kb_sample.md"
SAMPLE_CV = FIXTURES_DIR / "sample_cv.docx"

DRIFT_CASE = "semantic_drift_team"
CLEAN_CASE = "clean_rewrite"

ORIGINAL = (
    "Built an LLM service that drafts three reply options for every customer review "
    "(detailed, professional, short), with sentiment-conditioned prompts and per-dealership "
    "rules baked in; the dealer picks one and publishes. Roughly 93% of drafts go out without "
    "an edit, across over 100k reviews a month."
)


def cv_bytes() -> bytes:
    return SAMPLE_CV.read_bytes()


def cv_text() -> str:
    return "\n".join(p.full_text for p in iter_paragraphs(load(cv_bytes())))


@pytest.fixture
def vocab() -> set[str]:
    """The vocabulary the reviewer allows: the fixture CV plus the sample knowledge base."""
    return allowed_vocabulary(cv_text(), load_projects(KB_SAMPLE))


@pytest.fixture
def fact() -> ProjectFact:
    facts = {f.name: f for f in load_projects(KB_SAMPLE)}
    return facts["AI-Powered Review Response System"]


def test_reviewer_reverts_invented_term(llm_replay, vocab: set[str], fact: ProjectFact) -> None:
    new = ORIGINAL.replace("with sentiment-conditioned prompts", "with a React dashboard and sentiment-conditioned prompts")
    verdict = reviewer.review(ORIGINAL, new, vocab, fact)
    assert verdict.accept is False
    assert "not in CV/KB" in verdict.reason and "React" in verdict.reason


def test_reviewer_reverts_missing_metric(llm_replay, vocab: set[str], fact: ProjectFact) -> None:
    new = ORIGINAL.replace("Roughly 93% of drafts", "Most drafts")
    verdict = reviewer.review(ORIGINAL, new, vocab, fact)
    assert verdict.accept is False
    assert verdict.reason == "metric missing: 93%"


def test_reviewer_reverts_length(llm_replay, vocab: set[str], fact: ProjectFact) -> None:
    filler = " ".join(["and the dealer publishes it"] * 6)  # ~+40% words, no new terms, no lost numbers
    verdict = reviewer.review(ORIGINAL, f"{ORIGINAL} {filler}", vocab, fact)
    assert verdict.accept is False
    assert verdict.reason.startswith("length"), verdict.reason


def test_reviewer_reverts_semantic_drift(llm_replay, vocab: set[str], fact: ProjectFact) -> None:
    """Nothing deterministic is wrong with this text: only the model can catch the claim."""
    new = (
        "Led a team of six engineers building an LLM service that drafts three reply options for "
        "every customer review, with sentiment-conditioned prompts and per-dealership rules; the "
        "dealer publishes one. Roughly 93% of drafts go out unedited, across over 100k reviews a month."
    )
    assert reviewer.invented_terms(new, ORIGINAL, vocab) == []
    assert reviewer.review(ORIGINAL, new, vocab, fact, semantic=False).accept is True

    verdict = reviewer.review(ORIGINAL, new, vocab, fact, case=DRIFT_CASE)
    assert verdict.accept is False
    assert verdict.reason.startswith("unsupported claim:"), verdict.reason


def test_reviewer_accepts_clean_rewrite(llm_replay, vocab: set[str], fact: ProjectFact) -> None:
    new = (
        "Built and shipped an LLM service that drafts three reply options for every customer "
        "review (detailed, professional, short), applying sentiment-conditioned prompts and "
        "per-dealership rules; the dealer picks one and publishes it. Roughly 93% of drafts go "
        "out unedited, across over 100k reviews a month."
    )
    verdict = reviewer.review(ORIGINAL, new, vocab, fact, case=CLEAN_CASE)
    assert verdict.accept is True, verdict.reason


def test_reviewer_accepts_unchanged_text_without_an_llm_call(llm_replay, vocab: set[str]) -> None:
    verdict = reviewer.review(ORIGINAL, f"  {ORIGINAL}  ", vocab)
    assert verdict.accept is True and verdict.reason == reviewer.UNCHANGED


def test_reviewer_allows_a_term_the_original_already_used(llm_replay) -> None:
    """The guard blocks inventions, not the CV's own words: an empty vocabulary is not a veto."""
    assert reviewer.invented_terms("FastAPI service on AWS", "FastAPI service on AWS", set()) == []
    assert reviewer.invented_terms("FastAPI service on Azure", "FastAPI service on AWS", set()) == ["Azure"]


def test_layout_check_fonts_and_page_count(llm_replay) -> None:
    """A rewrite applied by the assembler changes text only: no font, style, numbering or page diff."""
    original = cv_bytes()
    paras = iter_paragraphs(load(original))
    sections = classify(paras)
    bullet = next(p for p in paras if sections[p.id] == "exp_bullet")
    shortened = " ".join(bullet.text.split()[:-1])  # one word out: same fonts, same page

    output = assembler.apply(original, {bullet.id: shortened})
    assert assembler.check_layout(original, output) == []
    assert assembler.check_layout(original, output, {bullet.id}) == []
    assert [d.para_id for d in assembler.text_diffs(original, output)] == [bullet.id]

    if assembler.soffice() is None:
        pytest.skip("LibreOffice (soffice) is not installed; page count cannot be checked")
    before = assembler.page_count(original)
    assert before is not None and before >= 1
    assert assembler.page_count(output) == before
    assert assembler.check_layout(original, output, pages=True) == []


def test_layout_check_reports_a_changed_font(llm_replay) -> None:
    original = cv_bytes()
    doc = load(original)
    paragraph = next(p for p in doc.paragraphs if p.runs)
    paragraph.runs[0].font.name = "Comic Sans MS"
    problems = assembler.check_layout(original, assembler.save(doc))
    assert problems and any("font" in p for p in problems), problems


def test_layout_check_reports_an_oversized_document(llm_replay) -> None:
    original = cv_bytes()
    paras = iter_paragraphs(load(original))
    sections = classify(paras)
    summary = next(p for p in paras if sections[p.id] == "summary")
    output = assembler.apply(original, {summary.id: summary.text + " " + summary.text})
    problems = assembler.check_layout(original, output, {summary.id})
    assert problems and any("document length changed" in p for p in problems), problems
