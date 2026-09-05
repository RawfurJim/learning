"""SCRUM-14: Agent 3 (Experience Writer) contract tests. Replays recorded Gemini output.

Recordings (`LLM_MODE=record uv run pytest -m live -k experience_live` rewrites them):
  tests/recordings/experience/<case>__g<n>_<heading slug>.json
      one per project group (exp_meta sub-heading + its bullets) of the fixture CV, for
      each of the three JDs, with tests/fixtures/kb_sample.md and no approvals.
A `<...>_retry.json` exists only when the first answer broke a rule during recording.

The "bad recording" of `test_added_metrics_come_from_kb` is synthetic: it is written
into a temporary recordings dir (`LLM_RECORDINGS_DIR`) by the test itself and never
sits among the real recordings.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from lxml import etree

from resume_tailor import pipeline
from resume_tailor.agents import experience_writer
from resume_tailor.agents.summary_skills_writer import suspicious_terms, words
from resume_tailor.ats_score import contains
from resume_tailor.docx_io import Para, iter_paragraphs, load
from resume_tailor.knowledge import allowed_vocabulary, extract_numbers, load_projects, raw_tokens
from resume_tailor.schemas import BulletRewrite, ExperienceResult, ProjectFact
from resume_tailor.sections import classify
from resume_tailor.settings import DEFAULT_RECORDINGS_DIR

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_SAMPLE = FIXTURES_DIR / "kb_sample.md"
SAMPLE_CV = FIXTURES_DIR / "sample_cv.docx"

CASES = ["senior_ai_engineer", "ds_nlp", "ml_platform"]
STAGES = ["summary", "skills", "experience"]
PROFESSIONAL_EMPLOYER = "JudgeService"
BULLET_COUNT = 12
PERSONAL_COUNT = 2


def cv_bytes() -> bytes:
    return SAMPLE_CV.read_bytes()


def experience_paras(cv: bytes) -> tuple[list[Para], list[Para]]:
    """`(exp_meta paragraphs, exp_bullet paragraphs)` of the CV, in document order."""
    paras = iter_paragraphs(load(cv))
    sections = classify(paras)
    meta = [p for p in paras if sections[p.id] == "exp_meta"]
    bullets = [p for p in paras if sections[p.id] == "exp_bullet"]
    return meta, bullets


def jd_text(case: str) -> str:
    return (FIXTURES_DIR / "jds" / f"{case}.txt").read_text(encoding="utf-8")


def run_case(case: str) -> pipeline.RunResult:
    """Full pipeline (summary, skills and experience) on the fixture CV + kb_sample for one JD."""
    return pipeline.run(cv_bytes(), jd_text(case), KB_SAMPLE, set(), STAGES, case=case)


def run_experience(case: str, analysis: pipeline.Analysis) -> ExperienceResult:
    """Only the experience writer (record/live mode records nothing else)."""
    return pipeline.run_experience(cv_bytes(), analysis, KB_SAMPLE, approved_adjacent=())


def facts_by_name() -> dict[str, ProjectFact]:
    return {fact.name: fact for fact in load_projects(KB_SAMPLE)}


def fact_text(fact: ProjectFact) -> str:
    return " ".join([*fact.stack, *fact.keywords, fact.built, fact.problem])


def is_professional(exp: ExperienceResult, para_id: str) -> bool:
    fact = facts_by_name().get(exp.projects.get(para_id, ""))
    return fact is not None and fact.professional


# ---- shared assertions (also used by tests/test_experience_live.py) ---------------------------------


def assert_bullet_word_budget(exp: ExperienceResult) -> None:
    for bullet in exp.bullets:
        orig = exp.originals[bullet.para_id]
        assert abs(words(bullet.text) - words(orig)) <= 0.10 * words(orig), (bullet.para_id, words(bullet.text), words(orig))


def assert_bullet_numbers_preserved(exp: ExperienceResult) -> None:
    for bullet in exp.bullets:
        orig = exp.originals[bullet.para_id]
        assert set(extract_numbers(orig)) <= set(extract_numbers(bullet.text)), (bullet.para_id, bullet.text)


def assert_bullet_terms_in_vocabulary(exp: ExperienceResult, matched: list[str], cv_text: str) -> None:
    vocab = set(allowed_vocabulary(cv_text, load_projects(KB_SAMPLE)))
    for skill in matched:  # matched JD skills are things the CV/KB has under another name
        vocab |= {t.lower() for t in raw_tokens(skill)}
    for bullet in exp.bullets:
        offending = suspicious_terms(bullet.text, vocab)
        assert offending == [], (bullet.para_id, offending, bullet.text)


def assert_added_metrics_come_from_kb(exp: ExperienceResult) -> None:
    facts = facts_by_name()
    for bullet in exp.bullets:
        fact = facts.get(exp.projects.get(bullet.para_id, ""))
        orig_numbers = set(extract_numbers(exp.originals[bullet.para_id]))
        for metric in bullet.used_kb_metrics:
            assert fact is not None, (bullet.para_id, metric)
            assert metric in fact.metrics, (bullet.para_id, metric, fact.metrics)
        # Every new number in the text is explained by a reported KB metric.
        allowed = orig_numbers | {n for m in bullet.used_kb_metrics for n in extract_numbers(m)}
        assert set(extract_numbers(bullet.text)) <= allowed, (bullet.para_id, bullet.text, bullet.used_kb_metrics)


def assert_jd_keywords_woven(exp: ExperienceResult, matched: list[str], minimum: int) -> None:
    matched_lower = {m.lower() for m in matched}
    woven = 0
    for bullet in exp.bullets:
        for keyword in bullet.jd_keywords_used:
            assert contains(bullet.text, keyword), (bullet.para_id, keyword, bullet.text)
        if any(k.lower() in matched_lower for k in bullet.jd_keywords_used):
            woven += 1
    assert woven >= minimum, (woven, [(b.para_id, b.jd_keywords_used) for b in exp.bullets])


def assert_professional_first(exp: ExperienceResult, matched: list[str]) -> None:
    facts = facts_by_name()
    professional = [b for b in exp.bullets if is_professional(exp, b.para_id)]
    personal = [b for b in exp.bullets if not is_professional(exp, b.para_id)]
    assert len(personal) == PERSONAL_COUNT and len(professional) == BULLET_COUNT - PERSONAL_COUNT
    # Every professional bullet has JD keywords before any personal bullet does.
    if any(b.jd_keywords_used for b in personal):
        assert all(b.jd_keywords_used for b in professional), [(b.para_id, b.jd_keywords_used) for b in professional]
    professional_text = " ".join(fact_text(f) for f in facts.values() if f.professional)
    matched_lower = {m.lower() for m in matched}
    for bullet in personal:
        orig = exp.originals[bullet.para_id]
        if bullet.text == orig:
            continue
        fact = facts[exp.projects[bullet.para_id]]
        only_here = [
            k
            for k in bullet.jd_keywords_used
            if k.lower() in matched_lower
            and not contains(orig, k)
            and contains(fact_text(fact), k)
            and not contains(professional_text, k)
        ]
        assert only_here, (bullet.para_id, bullet.jd_keywords_used, bullet.text)


def assert_exp_meta_untouched(result: pipeline.RunResult, original: bytes) -> None:
    paras_a = iter_paragraphs(load(original))
    paras_b = iter_paragraphs(load(result.output))
    sections = classify(paras_a)
    by_id_b = {p.id: p for p in paras_b}
    checked = 0
    for pa in paras_a:
        if sections[pa.id] != "exp_meta":
            continue
        pb = by_id_b[pa.id]
        assert pa.full_text == pb.full_text, pa.id
        assert etree.tostring(pa.paragraph._p) == etree.tostring(pb.paragraph._p), pa.id
        checked += 1
    assert checked == 5, checked  # employer line + 3 project sub-headings + Independent Projects line


# ---- tests -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm_replay_module():
    from resume_tailor import llm

    mp = pytest.MonkeyPatch()
    mp.setenv("LLM_MODE", "replay")
    mp.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    yield
    mp.undo()
    llm.set_provider(None)


@pytest.fixture(scope="module")
def results(llm_replay_module) -> dict[str, pipeline.RunResult]:
    return {case: run_case(case) for case in CASES}


def experience_of(result: pipeline.RunResult) -> ExperienceResult:
    assert result.experience is not None, result.notes
    assert len(result.experience.bullets) == BULLET_COUNT
    return result.experience


@pytest.mark.parametrize("case", CASES)
def test_bullet_word_budget_all(results, case: str) -> None:
    assert_bullet_word_budget(experience_of(results[case]))


@pytest.mark.parametrize("case", CASES)
def test_bullet_numbers_preserved(results, case: str) -> None:
    assert_bullet_numbers_preserved(experience_of(results[case]))


@pytest.mark.parametrize("case", CASES)
def test_bullet_terms_in_vocabulary(results, case: str) -> None:
    cv_text = "\n".join(p.full_text for p in iter_paragraphs(load(cv_bytes())))
    assert_bullet_terms_in_vocabulary(experience_of(results[case]), results[case].match.matched, cv_text)


@pytest.mark.parametrize("case", CASES)
def test_added_metrics_come_from_kb(results, case: str, llm_replay_module, monkeypatch, tmp_path: Path) -> None:
    assert_added_metrics_come_from_kb(experience_of(results[case]))

    # A deliberately bad answer: the first bullet of group 1 reports a metric "3x" that is
    # not in the knowledge base. Synthetic, written to a scratch recordings dir.
    scratch = tmp_path / "recordings"
    shutil.copytree(DEFAULT_RECORDINGS_DIR, scratch)
    result = results[case]
    exp = experience_of(result)
    analysis = pipeline.analyse(cv_bytes(), jd_text(case), KB_SAMPLE, case=case)
    inputs = pipeline.experience_inputs(cv_bytes(), analysis, KB_SAMPLE, approved_adjacent=())
    group = inputs.groups[0]
    first = exp.bullets[0]
    assert group.bullets[0].id == first.para_id
    bad = {
        "bullets": [
            {
                "para_id": first.para_id,
                "text": exp.originals[first.para_id] + " Cut latency 3x.",
                "used_kb_metrics": ["3x"],
                "jd_keywords_used": [],
            }
        ]
    }
    for name in (experience_writer.group_case(case, group), experience_writer.group_case(case, group) + "_retry"):
        (scratch / experience_writer.AGENT / f"{name}.json").write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setenv("LLM_RECORDINGS_DIR", str(scratch))
    try:
        rerun = run_experience(case, analysis)
    finally:
        monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)

    rejected = next(b for b in rerun.bullets if b.para_id == first.para_id)
    assert rejected.text == exp.originals[first.para_id]
    assert rejected.used_kb_metrics == [] and "3x" not in rejected.text
    assert first.para_id in rerun.reverted
    assert any("3x" in note for note in rerun.notes), rerun.notes
    assert_added_metrics_come_from_kb(rerun)
    # The check itself names the offending metric.
    problems = experience_writer.check_bullet(BulletRewrite(**bad["bullets"][0]), group, inputs)
    assert any("3x" in p for p in problems), problems


def test_jd_keywords_woven_when_truthful(results) -> None:
    result = results["senior_ai_engineer"]
    assert_jd_keywords_woven(experience_of(result), result.match.matched, minimum=6)


@pytest.mark.parametrize("case", CASES)
def test_professional_first(results, case: str) -> None:
    assert_professional_first(experience_of(results[case]), results[case].match.matched)


@pytest.mark.parametrize("case", CASES)
def test_exp_meta_untouched(results, case: str) -> None:
    assert_exp_meta_untouched(results[case], cv_bytes())


def test_bullets_grouped_by_project() -> None:
    """Groups follow the sub-headings; bullets map to their knowledge-base project."""
    cv = cv_bytes()
    paras = iter_paragraphs(load(cv))
    groups = experience_writer.group_bullets(paras, classify(paras), load_projects(KB_SAMPLE))
    assert [len(g.bullets) for g in groups] == [3, 4, 3, 2]
    assert [g.heading.id for g in groups] == ["p6", "p10", "p15", "p19"]
    names = [{g.facts[b.id].name if g.facts[b.id] else None for b in g.bullets} for g in groups]
    assert names[0] == {"AI-Powered Review Response System"}
    assert names[1] == {"Customer Insight AI Agent & Sentiment Platform"}
    assert names[2] == {"RAG Documentation Engine, QA Automation & Mentoring"}
    assert names[3] == {"ResumeBoost (ML-Based Resume Optimiser)", "Data Warehouse for Government Spending Insights"}
    assert all(g.facts[b.id].professional for g in groups[:3] for b in g.bullets)
    assert not any(groups[3].facts[b.id].professional for b in groups[3].bullets)


def test_experience_python_checks(results, llm_replay_module) -> None:
    """The deterministic guards that make the recording tests hold regardless of the model."""
    result = results["senior_ai_engineer"]
    exp = experience_of(result)
    analysis = pipeline.analyse(cv_bytes(), jd_text("senior_ai_engineer"), KB_SAMPLE, case="senior_ai_engineer")
    inputs = pipeline.experience_inputs(cv_bytes(), analysis, KB_SAMPLE, approved_adjacent=())
    group = inputs.groups[0]
    first = group.bullets[0]
    orig = first.text
    assert exp.originals[first.id] == orig

    def check(text: str, metrics: list[str] = [], keywords: list[str] = []) -> list[str]:
        rewrite = BulletRewrite(para_id=first.id, text=text, used_kb_metrics=metrics, jd_keywords_used=keywords)
        return experience_writer.check_bullet(rewrite, group, inputs)

    assert check(orig) == []
    assert any("words" in p for p in check(orig + " " + " ".join(["extra"] * 10)))
    assert any("93%" in p for p in check(orig.replace("93%", "most")))
    assert any("Kubernetes" in p for p in check(orig + " Ran it on Kubernetes."))
    assert any("2x" in p for p in check(orig + " Doubled throughput 2x."))  # metric of another project
    assert check(orig + " Distilled 8B from 70B.", metrics=["8B model distilled from 70B"]) == []
    assert any("3x" in p for p in check(orig + " Cut latency 3x.", metrics=["3x"]))

    # Personal-project bullets: only a matched keyword that applies to that project alone justifies a change.
    personal = inputs.groups[3]
    resume_boost = personal.bullets[0]
    rewrite = BulletRewrite(para_id=resume_boost.id, text=resume_boost.text.replace("tool", "Python tool"), jd_keywords_used=["Python"])
    assert any("unchanged" in p for p in experience_writer.check_bullet(rewrite, personal, inputs))  # Python is in the work projects too
    rewrite = BulletRewrite(para_id=resume_boost.id, text=resume_boost.text.replace("CI/CD pipeline", "GitHub Actions CI/CD pipeline"), jd_keywords_used=["GitHub Actions"])
    assert experience_writer.check_bullet(rewrite, personal, inputs) == []


def test_app_accept_reject(llm_replay, jd, sample_cv_path) -> None:
    """AppTest: reject bullet 2 -> the assembled docx keeps bullet 2's original text and bullet 1's new text."""
    from streamlit.testing.v1 import AppTest

    from resume_tailor.settings import REPO_ROOT

    cv = sample_cv_path.read_bytes()
    _, bullets = experience_paras(cv)
    first, second = bullets[0], bullets[1]

    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=60)
    at.session_state["llm_case"] = "senior_ai_engineer"
    at.session_state["cv_bytes"] = cv
    at.session_state["kb_path"] = str(KB_SAMPLE)
    at.run()
    at.text_area(key="jd_text").input(jd("senior_ai_engineer")).run()
    at.button(key="analyse").click().run()
    at.button(key="rewrite").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]

    result: pipeline.RunResult = at.session_state["run_result"]
    assert first.id in result.rewrites and second.id in result.rewrites, result.notes
    assert result.rewrites[first.id] != first.text and result.rewrites[second.id] != second.text

    # Every rewritten bullet starts accepted; the download equals the pipeline output.
    accept_keys = {f"accept::{pid}" for pid, section in result.sections.items() if section == "exp_bullet" and pid in result.rewrites}
    assert {cb.key for cb in at.checkbox if str(cb.key).startswith("accept::")} == accept_keys
    assert all(at.session_state[key] is True for key in accept_keys)
    texts = lambda b: {p.id: p.text for p in iter_paragraphs(load(b))}  # noqa: E731
    assert texts(at.session_state["tailored_docx"]) == texts(result.output)

    at.checkbox(key=f"accept::{second.id}").uncheck().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state[f"accept::{second.id}"] is False
    out = texts(at.session_state["tailored_docx"])
    assert out[second.id] == second.text
    assert out[first.id] == result.rewrites[first.id]
    assert len(at.download_button) == 1
    for para in iter_paragraphs(load(at.session_state["tailored_docx"])):
        if para.id not in result.rewrites:
            assert para.text == result.originals.get(para.id, para.text)

    # Accepting it again restores the new text.
    at.checkbox(key=f"accept::{second.id}").check().run()
    assert texts(at.session_state["tailored_docx"])[second.id] == result.rewrites[second.id]
