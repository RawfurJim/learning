"""SCRUM-13: first end-to-end result, JD text + CV bytes -> tailored .docx (replayed recordings)."""

from __future__ import annotations

from collections import Counter

import pytest
from test_agent_summary_skills import CASES, KB_SAMPLE, STAGES, cv_bytes, run_case

from resume_tailor import assembler, pipeline
from resume_tailor.docx_io import iter_paragraphs, load, structure_diff
from resume_tailor.sections import classify


def assert_valid_tailored_docx(result: pipeline.RunResult, original: bytes) -> None:
    """Shared with the live test: the output is a docx whose only changes are the two rewritten texts."""
    doc = load(result.output)  # raises if the bytes are not a .docx
    paras = iter_paragraphs(doc)
    counts = Counter(classify(paras).values())
    assert counts["summary"] == 1 and counts["skills"] == 1, counts
    assert counts["exp_bullet"] == 12 and counts["education"] == 2, counts

    diffs = structure_diff(original, result.output)
    assert {d.kind for d in diffs} == {"text"}, diffs
    assert {d.para_id for d in diffs} == set(result.rewrites), (diffs, result.rewrites)
    assert set(result.rewrites) <= set(result.sections)
    assert assembler.check_layout(original, result.output, result.rewrites) == []
    assert abs(assembler.char_change(original, result.output)) <= assembler.CHAR_TOLERANCE

    by_id = {p.id: p for p in paras}
    for para_id, text in result.rewrites.items():
        assert by_id[para_id].text == text
    assert result.stages == STAGES


@pytest.mark.parametrize("case", CASES)
def test_e2e_summary_skills_docx(llm_replay, case: str) -> None:
    original = cv_bytes()
    result = run_case(case)
    assert_valid_tailored_docx(result, original)
    assert result.rewrites, result.notes
    assert result.coverage_after >= result.coverage_before, (result.coverage_before, result.coverage_after)
    assert result.usage.calls >= 1 and result.usage.model_dump()  # replay: tokens are 0, calls are counted


def test_pipeline_without_stages_returns_original(llm_replay) -> None:
    original = cv_bytes()
    jd_text = (KB_SAMPLE.parent / "jds" / "senior_ai_engineer.txt").read_text(encoding="utf-8")
    result = pipeline.run(original, jd_text, KB_SAMPLE, set(), [], case="senior_ai_engineer")
    assert result.output == original
    assert result.rewrites == {} and result.writer is None
    assert structure_diff(original, result.output) == []


def test_pipeline_reuses_analysis(llm_replay) -> None:
    original = cv_bytes()
    jd_text = (KB_SAMPLE.parent / "jds" / "senior_ai_engineer.txt").read_text(encoding="utf-8")
    analysis = pipeline.analyse(original, jd_text, KB_SAMPLE, case="senior_ai_engineer")
    assert "React" in analysis.match.missing
    result = pipeline.run(original, jd_text, KB_SAMPLE, set(), STAGES, analysis=analysis)
    assert result.case == "senior_ai_engineer"
    assert result.match.matched == analysis.match.matched  # nothing approved
    # Compare content, not bytes: zip entry timestamps in a .docx have minute resolution.
    texts = lambda b: [p.full_text for p in iter_paragraphs(load(b))]  # noqa: E731
    assert texts(result.output) == texts(run_case("senior_ai_engineer").output)
