"""SCRUM-13: Agent 4 (Summary & Skills Writer) contract tests. Replays recorded Gemini output.

Recordings (`LLM_MODE=record uv run pytest -m live -k e2e_live` rewrites them):
  tests/recordings/summary_skills/{senior_ai_engineer,ds_nlp,ml_platform}.json
      the fixture CV + tests/fixtures/kb_sample.md, no approvals (pipeline.run)
  tests/recordings/summary_skills/senior_ai_engineer_cv_only.json
  tests/recordings/summary_skills/senior_ai_engineer_cv_only__flask.json
      the fixture CV alone (no knowledge base, so Flask is *not* in the inventory), the
      senior JD with Flask as an adjacent nice-to-have, without / with approval.
A `<case>_retry.json` exists only when the first answer broke a rule during recording.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_tailor import matching, pipeline
from resume_tailor.agents import jd_intent, keywords, summary_skills_writer
from resume_tailor.agents.summary_skills_writer import skills_entries, suspicious_terms, words
from resume_tailor.docx_io import Para, iter_paragraphs, load
from resume_tailor.knowledge import allowed_vocabulary, extract_numbers, load_projects, raw_tokens
from resume_tailor.matching import skills_line_terms
from resume_tailor.schemas import SkillMatch
from resume_tailor.sections import classify

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_SAMPLE = FIXTURES_DIR / "kb_sample.md"
NO_KB = FIXTURES_DIR / "does_not_exist.md"  # load_projects -> [] : CV-only inventory
SAMPLE_CV = FIXTURES_DIR / "sample_cv.docx"

CASES = ["senior_ai_engineer", "ds_nlp", "ml_platform"]
FLASK_CASE = "senior_ai_engineer_cv_only"
STAGES = ["summary", "skills"]
FIXTURE_NUMBERS = ("93%", "100k", "0.98", "20%")


def cv_bytes() -> bytes:
    return SAMPLE_CV.read_bytes()


def editable_paras(cv: bytes) -> dict[str, Para]:
    paras = iter_paragraphs(load(cv))
    sections = classify(paras)
    return {sections[p.id]: p for p in paras if sections[p.id] in ("summary", "skills")}


def run_case(case: str, approved: set[str] = frozenset()) -> pipeline.RunResult:
    """The full pipeline on the fixture CV + kb_sample for one JD (all calls replayed or recorded)."""
    jd_text = (FIXTURES_DIR / "jds" / f"{case}.txt").read_text(encoding="utf-8")
    return pipeline.run(cv_bytes(), jd_text, KB_SAMPLE, approved, STAGES, case=case)


def flask_analysis(cv: bytes) -> pipeline.Analysis:
    """Senior JD + Flask as a nice-to-have, CV-only inventory, Flask adjacent to FastAPI.

    Built without the alias call so the test does not depend on a second model judgement:
    exact/fuzzy hits are matched, Flask is adjacent, everything else missing.
    """
    jd_text = (FIXTURES_DIR / "jds" / "senior_ai_engineer.txt").read_text(encoding="utf-8")
    intent = jd_intent.run(jd_text, case="senior_ai_engineer")
    base = keywords.run(jd_text, case="senior_ai_engineer")
    kw = base.model_copy(
        update={"nice_to_have": [*base.nice_to_have, "Flask"], "ats_keywords": [*base.ats_keywords, "Flask"]}
    )
    paras = iter_paragraphs(load(cv))
    inventory = matching.build_inventory(paras, [])
    assert "flask" not in {t.lower() for t in inventory}
    jd_skills = [*kw.mandatory, *kw.nice_to_have]
    matched = [s for s in jd_skills if matching.find_in_inventory(s, inventory)]
    missing = [s for s in jd_skills if s not in matched and s != "Flask"]
    return pipeline.Analysis(
        case=FLASK_CASE,
        intent=intent,
        keywords=kw,
        match=SkillMatch(matched=matched, adjacent=["Flask"], missing=missing, aliases={"Flask": "FastAPI"}),
        ranked=[],
        inventory=sorted(inventory, key=str.lower),
        cv_text="\n".join(p.full_text for p in paras),
    )


def run_flask_case(approved: set[str]) -> pipeline.RunResult:
    cv = cv_bytes()
    jd_text = (FIXTURES_DIR / "jds" / "senior_ai_engineer.txt").read_text(encoding="utf-8")
    return pipeline.run(cv, jd_text, NO_KB, approved, STAGES, analysis=flask_analysis(cv))


# ---- shared assertions (also used by tests/test_e2e_live.py) ------------------------------------


def skills_of(result: pipeline.RunResult) -> list[str]:
    assert result.writer is not None
    return result.writer.skills


def flat_terms(entries: list[str]) -> list[str]:
    return [t for entry in entries for t in skills_line_terms(entry)]


def assert_summary_word_budget(result: pipeline.RunResult, orig: str) -> None:
    new = result.writer.summary
    assert abs(words(new) - words(orig)) <= 0.10 * words(orig), (words(new), words(orig))


def assert_summary_terms_in_vocabulary(result: pipeline.RunResult, cv_text: str, kb_path: Path) -> None:
    vocab = set(allowed_vocabulary(cv_text, load_projects(kb_path)))
    # Matched JD skills are things the CV/KB has under another name; their tokens are allowed too.
    for skill in result.match.matched:
        vocab |= {t.lower() for t in raw_tokens(skill)}
    offending = suspicious_terms(result.writer.summary, vocab)
    assert offending == [], offending


def assert_summary_keeps_numbers(result: pipeline.RunResult, orig: str) -> None:
    new_numbers = set(extract_numbers(result.writer.summary))
    for number in FIXTURE_NUMBERS:
        assert number in orig
        assert number in new_numbers, (number, result.writer.summary)
    assert set(extract_numbers(orig)) <= new_numbers


def assert_skills_subset_of_inventory(result: pipeline.RunResult, inventory: set[str], approved: set[str]) -> None:
    allowed = {t.lower() for t in inventory} | {t.lower() for t in approved}
    skills = skills_of(result)
    assert skills, "skills line must not be empty"
    assert {t.lower() for t in flat_terms(skills)} <= allowed, set(flat_terms(skills)) - allowed
    assert "react" not in {t.lower() for t in flat_terms(skills)}


def assert_skills_order_mandatory_first(result: pipeline.RunResult) -> None:
    """Every entry holding a matched-mandatory skill comes before every entry holding only matched-nice skills."""
    skills = skills_of(result)
    mandatory = {s.lower() for s in result.keywords.mandatory}
    matched = {s.lower() for s in result.match.matched}
    to_jd: dict[str, set[str]] = {}  # inventory term -> every JD skill that resolved to it
    for jd_skill, term in result.match.aliases.items():
        to_jd.setdefault(term.lower(), set()).add(jd_skill.lower())

    def level(entry: str) -> int:
        terms = {t.lower() for t in [entry, *skills_line_terms(entry)]}
        jd_terms = terms | {s for t in terms for s in to_jd.get(t, ())}
        if jd_terms & matched & mandatory:
            return 0
        if jd_terms & matched:
            return 1
        return 2

    levels = [level(e) for e in skills]
    mandatory_idx = [i for i, lv in enumerate(levels) if lv == 0]
    nice_idx = [i for i, lv in enumerate(levels) if lv == 1]
    assert mandatory_idx, "expected at least one matched mandatory skill on the line"
    if nice_idx:
        assert max(mandatory_idx) < min(nice_idx), list(zip(skills, levels))
    # And nothing matched comes after the unmatched tail.
    matched_idx = mandatory_idx + nice_idx
    other_idx = [i for i, lv in enumerate(levels) if lv == 2]
    if other_idx:
        assert max(matched_idx) < min(other_idx), list(zip(skills, levels))


def assert_skills_count_within_budget(result: pipeline.RunResult, orig_line: str) -> None:
    orig_skills = skills_entries(orig_line)
    assert abs(len(skills_of(result)) - len(orig_skills)) <= 2, (len(skills_of(result)), len(orig_skills))


def assert_rewrites_applied(result: pipeline.RunResult) -> None:
    """Both paragraphs really changed and neither fell back to the original.

    Since SCRUM-16 a paragraph can also leave the writer clean and still not reach the
    document, because Agent 6 refused it: that is a pass here as long as the refusal is
    recorded with a reason. What must never happen is the writer itself giving up.
    """
    assert result.writer is not None
    assert not result.writer.summary_reverted, result.notes
    assert not result.writer.skills_reverted, result.notes
    blocked = {revert.para_id: revert.reason for revert in result.reverts}
    by_section = {section: pid for pid, section in result.sections.items()}
    for section in ("summary", "skills"):
        para_id = by_section[section]
        if para_id in result.rewrites:
            continue
        assert blocked.get(para_id), (section, result.notes, result.reverts)


# ---- tests -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def originals() -> dict[str, Para]:
    return editable_paras(cv_bytes())


@pytest.fixture(scope="module")
def results(llm_replay_module) -> dict[str, pipeline.RunResult]:
    return {case: run_case(case) for case in CASES}


@pytest.fixture(scope="module")
def llm_replay_module():
    """Module-scoped replay mode (the function-scoped `llm_replay` cannot back a module fixture)."""
    from resume_tailor import llm

    mp = pytest.MonkeyPatch()
    mp.setenv("LLM_MODE", "replay")
    mp.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    yield
    mp.undo()
    llm.set_provider(None)


@pytest.mark.parametrize("case", CASES)
def test_summary_word_budget(results, originals, case: str) -> None:
    assert_summary_word_budget(results[case], originals["summary"].text)


@pytest.mark.parametrize("case", CASES)
def test_summary_terms_in_vocabulary(results, case: str) -> None:
    cv_text = "\n".join(p.full_text for p in iter_paragraphs(load(cv_bytes())))
    assert_summary_terms_in_vocabulary(results[case], cv_text, KB_SAMPLE)


@pytest.mark.parametrize("case", CASES)
def test_summary_keeps_numbers(results, originals, case: str) -> None:
    assert_summary_keeps_numbers(results[case], originals["summary"].text)


@pytest.mark.parametrize("case", CASES)
def test_skills_subset_of_inventory(results, case: str) -> None:
    inventory = matching.build_inventory(iter_paragraphs(load(cv_bytes())), load_projects(KB_SAMPLE))
    assert_skills_subset_of_inventory(results[case], inventory, set())


@pytest.mark.parametrize("case", CASES)
def test_skills_order_mandatory_first(results, case: str) -> None:
    assert_skills_order_mandatory_first(results[case])


@pytest.mark.parametrize("case", CASES)
def test_skills_count_within_budget(results, originals, case: str) -> None:
    assert_skills_count_within_budget(results[case], originals["skills"].text)


@pytest.mark.parametrize("case", CASES)
def test_rewrites_are_applied(results, case: str) -> None:
    assert_rewrites_applied(results[case])


def test_adjacent_only_when_approved(llm_replay) -> None:
    without = run_flask_case(set())
    assert "flask" not in {t.lower() for t in flat_terms(skills_of(without))}, skills_of(without)
    assert "flask" not in without.writer.summary.lower()
    assert "Flask" in without.match.adjacent and "Flask" not in without.match.matched

    with_flask = run_flask_case({"Flask"})
    assert "Flask" in with_flask.match.matched and "Flask" not in with_flask.match.adjacent
    inventory = matching.build_inventory(iter_paragraphs(load(cv_bytes())), [])
    assert_skills_subset_of_inventory(with_flask, inventory, {"Flask"})
    assert_skills_subset_of_inventory(without, inventory, set())
    # Flask may appear now (and with a real recording it does: the candidate approved it).
    mentioned = "flask" in {t.lower() for t in flat_terms(skills_of(with_flask))} or "flask" in with_flask.writer.summary.lower()
    assert mentioned, (skills_of(with_flask), with_flask.writer.summary)
    assert_skills_order_mandatory_first(with_flask)


def test_python_checks_catch_violations(originals) -> None:
    """The deterministic guards that make the recording tests hold regardless of the model."""
    orig = originals["summary"].text
    vocab = {"judgeservice", "llm", "aws", "fastapi", "f1"}
    assert suspicious_terms("Shipped LLM systems at JudgeService on AWS.", vocab) == []
    assert suspicious_terms("Shipped LLM systems on SageMaker and Kubernetes.", vocab) == ["SageMaker", "Kubernetes"]
    # Sentence-initial ordinary words are exempt; acronyms never are.
    assert suspicious_terms("Delivered results. AWS work.", {"aws"}) == []
    assert suspicious_terms("Delivered results. GCP work.", {"aws"}) == ["GCP"]
    assert summary_skills_writer.missing_numbers("no numbers here", orig) == extract_numbers(orig)
    assert summary_skills_writer.invented_numbers("raised 45% and 0.98", {"0.98"}) == ["45%"]
    assert summary_skills_writer.forbidden_mentions("Ran Kubernetes clusters", ["Kubernetes", "React"]) == ["Kubernetes"]
    assert summary_skills_writer.word_bounds(86) == (78, 94)
    assert summary_skills_writer.case_for("x", {"Flask"}) == "x__flask" and summary_skills_writer.case_for("x", set()) == "x"
