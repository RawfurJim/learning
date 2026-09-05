"""Streamlit UI: paste a JD, click Analyse, see intent and keyword buckets (SCRUM-11);
upload a CV too and see Matched / Adjacent / Missing, ranked projects and ATS coverage (SCRUM-12).
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from resume_tailor.settings import REPO_ROOT

APP = REPO_ROOT / "app.py"
KB_SAMPLE = Path(__file__).resolve().parent / "fixtures" / "kb_sample.md"


def _rendered_text(at: AppTest) -> str:
    parts = [el.value for el in at.markdown] + [el.value for el in at.text]
    parts += [el.value for el in at.caption] + [el.value for el in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_app_shows_analysis(llm_replay, jd) -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["llm_case"] = "senior_ai_engineer"  # replay the committed recording
    at.run()
    assert not at.exception

    at.text_area(key="jd_text").input(jd("senior_ai_engineer")).run()
    at.button(key="analyse").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]

    text = _rendered_text(at)
    assert "senior" in text.lower()
    for skill in ("Python", "PyTorch", "RAG", "AWS"):
        assert skill in text, skill
    assert "Kubernetes" in text


def test_app_starts_without_analysis(llm_replay) -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.button(key="analyse")
    assert at.text_area(key="jd_text")


def _column_text(column) -> str:
    parts = [el.value for el in column.markdown] + [el.value for el in column.subheader]
    return "\n".join(str(p) for p in parts)


def test_app_buckets_render(llm_replay, jd, sample_cv_path) -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["llm_case"] = "senior_ai_engineer"
    # AppTest cannot drive st.file_uploader, so the app also accepts the CV bytes and the
    # knowledge-base path through session_state (see `cv_bytes` / `kb_path` in app.py).
    at.session_state["cv_bytes"] = sample_cv_path.read_bytes()
    at.session_state["kb_path"] = str(KB_SAMPLE)
    at.run()
    assert not at.exception

    at.text_area(key="jd_text").input(jd("senior_ai_engineer")).run()
    at.button(key="analyse").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]

    buckets = {el.value: el for el in at.subheader}
    assert {"Matched", "Adjacent", "Missing"} <= set(buckets), list(buckets)

    columns = {}
    for column in at.columns:
        for header in column.subheader:
            columns[header.value] = column
    assert {"Matched", "Adjacent", "Missing"} <= set(columns), list(columns)
    missing_text = _column_text(columns["Missing"])
    assert "React" in missing_text, missing_text
    assert "React" not in _column_text(columns["Matched"])
    for skill in ("Python", "PyTorch", "RAG", "AWS"):
        assert skill in _column_text(columns["Matched"]), skill

    # Adjacent suggestions are opt-in: every checkbox starts unticked.
    assert all(cb.value is False for cb in at.checkbox)

    text = _rendered_text(at)
    assert "Ranked projects" in text or "ranked projects" in text.lower()
    assert "JudgeService" in text or "Review Response" in text
    assert any("coverage" in str(m.label).lower() for m in at.metric), [m.label for m in at.metric]
