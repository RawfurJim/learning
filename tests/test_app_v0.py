"""Streamlit v0: paste a JD, click Analyse, see intent and keyword buckets (SCRUM-11)."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from resume_tailor.settings import REPO_ROOT

APP = REPO_ROOT / "app.py"


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
