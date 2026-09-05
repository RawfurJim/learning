"""SCRUM-16: the three-JD regression suite. The invariants that must hold forever.

The full pipeline (Agents 1-6, all three rewrite stages) runs against every fixture JD from
the committed recordings, and the assembled document is checked against the PRD's hard
rules: no invented term, no lost number, +/-10% words per paragraph, +/-3% characters for
the document, an unchanged page count where LibreOffice can measure it, read-only sections
untouched and ATS coverage that rises.

Everything here is deterministic: no API key, no network. Recordings are regenerated with
`LLM_MODE=record uv run pytest -m live` (the agents) and
`uv run python tests/fixtures/record_reviewer.py` (Agent 6's verdicts).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from resume_tailor import assembler, pipeline
from resume_tailor.agents import reviewer
from resume_tailor.agents.summary_skills_writer import within_word_budget, word_bounds, words
from resume_tailor.ats_score import coverage, present_keywords
from resume_tailor.docx_io import Para, iter_paragraphs, load
from resume_tailor.knowledge import allowed_vocabulary, extract_numbers, load_projects
from resume_tailor.schemas import SummarySkillsRewrite
from resume_tailor.sections import classify
from resume_tailor.settings import DEFAULT_RECORDINGS_DIR

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_SAMPLE = FIXTURES_DIR / "kb_sample.md"
SAMPLE_CV = FIXTURES_DIR / "sample_cv.docx"

CASES = ["senior_ai_engineer", "ds_nlp", "ml_platform"]
# ATS coverage must never fall on any JD. It must *rise* on the two JDs Jim genuinely
# matches. ml_platform is a Kubernetes / Terraform / Kubeflow platform role: almost every
# keyword the CV lacks is a tool he has never used, so honest tailoring has nothing to add
# and coverage is flat by design. Raising it there would mean inventing skills, which is the
# one thing this tool must never do - so the suite asserts "never falls" everywhere and
# "rises" where rising is honestly possible.
COVERAGE_MUST_RISE = ("senior_ai_engineer", "ds_nlp")
STAGES = ["summary", "skills", "experience"]
READ_ONLY = ("header", "exp_meta", "education", "other")  # never touched by any agent


def cv_bytes() -> bytes:
    return SAMPLE_CV.read_bytes()


def paragraphs(docx_bytes: bytes) -> list[Para]:
    return iter_paragraphs(load(docx_bytes))


@pytest.fixture(scope="module")
def replay_module():
    """Module-scoped replay mode: the three runs are shared by every assertion below."""
    from resume_tailor import llm

    mp = pytest.MonkeyPatch()
    mp.setenv("LLM_MODE", "replay")
    mp.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    yield
    mp.undo()
    llm.set_provider(None)


@pytest.fixture(scope="module")
def runs(replay_module, tmp_path_factory: pytest.TempPathFactory) -> dict[str, pipeline.RunResult]:
    cache_dir = tmp_path_factory.mktemp("regression-cache")
    out: dict[str, pipeline.RunResult] = {}
    for case in CASES:
        jd_text = (FIXTURES_DIR / "jds" / f"{case}.txt").read_text(encoding="utf-8")
        out[case] = pipeline.run(
            cv_bytes(), jd_text, KB_SAMPLE, [], STAGES, case=case, use_cache=False, cache_dir=cache_dir
        )
    return out


@pytest.mark.parametrize("case", CASES)
def test_regression_three_jds(runs: dict[str, pipeline.RunResult], case: str) -> None:
    """Every hard rule of the PRD, on one JD, end to end."""
    result = runs[case]
    original = cv_bytes()
    before = {para.id: para for para in paragraphs(original)}
    after = {para.id: para for para in paragraphs(result.output)}
    sections = classify(list(before.values()))
    facts = load_projects(KB_SAMPLE)
    vocab = allowed_vocabulary("\n".join(p.full_text for p in before.values()), facts)

    assert result.rewrites, result.notes
    assert set(result.rewrites) <= set(result.sections)

    # 1. no invented term anywhere in the tailored document
    for para_id, text in result.rewrites.items():
        assert reviewer.invented_terms(text, before[para_id].text, vocab) == [], (para_id, text)

    # 2. every number of every original paragraph survives
    for para_id, para in before.items():
        kept = set(extract_numbers(after[para_id].full_text))
        assert set(extract_numbers(para.full_text)) <= kept, (para_id, para.full_text)

    # 3. every rewritten paragraph stays within +/-10% words
    for para_id, text in result.rewrites.items():
        low, high = word_bounds(words(before[para_id].text))
        assert within_word_budget(text, before[para_id].text), (para_id, words(text), low, high)

    # 4. the document as a whole: text-only diffs, +/-3% characters, same page count
    assert assembler.check_layout(original, result.output, result.rewrites, pages=True) == []
    assert abs(assembler.char_change(original, result.output)) <= assembler.CHAR_TOLERANCE

    # 5. read-only sections are byte-identical
    for para_id, section in sections.items():
        if section in READ_ONLY or para_id not in result.rewrites:
            assert after[para_id].full_text == before[para_id].full_text, (para_id, section)

    # 6. the point of the exercise: the JD's keywords are better covered than before
    in_text = "\n".join(p.full_text for p in before.values())
    out_text = "\n".join(p.full_text for p in after.values())
    ats = result.keywords.ats_keywords
    assert result.coverage_after == pytest.approx(coverage(out_text, ats))
    # The skills line has a fixed length, so a rewrite may trade a keyword away (ds_nlp drops
    # "Git" for more relevant terms); what it may never do is cover the JD less well overall.
    assert len(present_keywords(out_text, ats)) >= len(present_keywords(in_text, ats))
    assert result.coverage_after >= result.coverage_before, (result.coverage_before, result.coverage_after)
    if case in COVERAGE_MUST_RISE:
        assert result.coverage_after > result.coverage_before, (result.coverage_before, result.coverage_after)
    else:
        assert result.match.missing, "coverage may only stay flat when the JD asks for skills Jim lacks"


@pytest.mark.parametrize("case", CASES)
def test_regression_reverts_carry_a_reason(runs: dict[str, pipeline.RunResult], case: str) -> None:
    """Whatever Agent 6 blocked is out of the document, kept its original text and says why."""
    result = runs[case]
    originals = {para.id: para.text for para in paragraphs(cv_bytes())}
    after = {para.id: para for para in paragraphs(result.output)}
    for revert in result.reverts:
        assert revert.para_id not in result.rewrites
        assert after[revert.para_id].text == originals[revert.para_id]
        assert revert.reason and revert.section
        assert revert.rejected and revert.rejected != revert.original


@pytest.mark.parametrize("case", CASES)
def test_regression_no_forbidden_skill_reaches_the_document(runs: dict[str, pipeline.RunResult], case: str) -> None:
    """Missing and unapproved adjacent skills (React, Kubernetes, ...) never appear in a rewrite."""
    from resume_tailor.ats_score import contains

    result = runs[case]
    originals = {para.id: para.text for para in paragraphs(cv_bytes())}
    forbidden = [*result.match.missing, *result.match.adjacent]
    for para_id, text in result.rewrites.items():
        added = [skill for skill in forbidden if contains(text, skill) and not contains(originals[para_id], skill)]
        assert added == [], (para_id, added)


# ---- the poisoned recording -------------------------------------------------------------------------

POISON_CASE = "senior_ai_engineer"
# The writer's own checks cannot catch this: every word is in the CV's vocabulary, every
# number of the original survives and the length is inside the budget. Only an independent
# reviewer, asking whether the claim is *supported*, can stop "led a team of ten engineers"
# reaching the document.
POISON_FROM = "AI Engineer with years building"
POISON_TO = "Led a team of ten engineers building"


def poisoned_recordings(destination: Path) -> Path:
    """A full copy of `tests/recordings/` whose summary for POISON_CASE claims a team lead."""
    scratch = destination / "recordings"
    shutil.copytree(DEFAULT_RECORDINGS_DIR, scratch)
    for name in (f"{POISON_CASE}.json", f"{POISON_CASE}_retry.json"):
        path = scratch / "summary_skills" / name
        if not path.exists():
            continue
        recorded = SummarySkillsRewrite.model_validate_json(path.read_text(encoding="utf-8"))
        assert POISON_FROM in recorded.summary, recorded.summary
        poisoned = recorded.model_copy(update={"summary": recorded.summary.replace(POISON_FROM, POISON_TO, 1)})
        path.write_text(json.dumps(poisoned.model_dump(), indent=2) + "\n", encoding="utf-8")
    return scratch


def test_reviewer_catches_a_poisoned_recording(llm_replay, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The acceptance criterion: a poisoned writer answer is reverted, with a reason to show."""
    monkeypatch.setenv("LLM_RECORDINGS_DIR", str(poisoned_recordings(tmp_path)))
    jd_text = (FIXTURES_DIR / "jds" / f"{POISON_CASE}.txt").read_text(encoding="utf-8")
    original = cv_bytes()
    result = pipeline.run(
        original, jd_text, KB_SAMPLE, [], ["summary", "skills"], case=POISON_CASE, use_cache=False
    )

    summary_id = next(pid for pid, section in result.sections.items() if section == "summary")
    assert result.writer is not None and not result.writer.summary_reverted  # the writer was happy
    assert POISON_TO in result.writer.summary
    blocked = {revert.para_id: revert for revert in result.reverts}
    assert summary_id in blocked, result.reverts
    assert blocked[summary_id].reason.startswith("unsupported claim:")
    assert "team" in blocked[summary_id].reason.lower()

    assert summary_id not in result.rewrites
    after = {para.id: para for para in paragraphs(result.output)}
    assert after[summary_id].text == originals_of(original)[summary_id]
    assert POISON_TO not in "\n".join(p.full_text for p in after.values())


def originals_of(docx_bytes: bytes) -> dict[str, str]:
    return {para.id: para.text for para in paragraphs(docx_bytes)}


def test_app_shows_the_reverted_paragraph(llm_replay, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_cv_path) -> None:
    """...and the app puts that reason in front of Jim instead of silently keeping the original."""
    from streamlit.testing.v1 import AppTest

    from resume_tailor.settings import REPO_ROOT

    monkeypatch.setenv("LLM_RECORDINGS_DIR", str(poisoned_recordings(tmp_path)))
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.session_state["llm_case"] = POISON_CASE
    at.session_state["cv_bytes"] = sample_cv_path.read_bytes()
    at.session_state["kb_path"] = str(KB_SAMPLE)
    at.run()
    at.text_area(key="jd_text").input((FIXTURES_DIR / "jds" / f"{POISON_CASE}.txt").read_text(encoding="utf-8")).run()
    at.button(key="analyse").click().run()
    at.button(key="rewrite").click().run()
    assert not at.exception, [e.value for e in at.exception]

    result: pipeline.RunResult = at.session_state["run_result"]
    assert result.reverts, result.notes
    warnings = " ".join(w.value for w in at.warning)
    assert "Reviewer reverted" in " ".join(sh.value for sh in at.subheader)
    assert any(revert.reason in warnings for revert in result.reverts), warnings
