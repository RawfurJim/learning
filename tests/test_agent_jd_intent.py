"""Agent 1 (JD Intent) contract tests (names from SCRUM-11). Replays recorded Gemini output."""

from __future__ import annotations

from resume_tailor.agents import jd_intent
from resume_tailor.schemas import JDIntent


def assert_senior_ai_engineer_intent(intent: JDIntent) -> None:
    """Shared with the live test: what a correct reading of senior_ai_engineer.txt looks like."""
    assert intent.seniority == "senior"
    assert len(intent.top_priorities) == 5
    lowered = [p.lower() for p in intent.top_priorities]
    assert any(
        "llm" in p or "serving" in p or "fine-tun" in p or "fine tun" in p for p in lowered
    ), intent.top_priorities
    assert any("evaluat" in p for p in lowered), intent.top_priorities
    assert intent.tone.strip()
    assert intent.role_summary.strip()


def test_intent_senior_ai_engineer(llm_replay, jd) -> None:
    intent = jd_intent.run(jd("senior_ai_engineer"), case="senior_ai_engineer")
    assert_senior_ai_engineer_intent(intent)


def test_intent_ds_nlp_is_mid_or_junior(llm_replay, jd) -> None:
    intent = jd_intent.run(jd("ds_nlp"), case="ds_nlp")
    assert intent.seniority in {"junior", "mid"}
    domain = intent.domain.lower()
    assert "nlp" in domain or "natural language" in domain, intent.domain
    assert len(intent.top_priorities) == 5
