"""Real Gemini run of both agents on senior_ai_engineer.txt (`uv run pytest -m live`).

`LLM_MODE=record uv run pytest -m live` (re)writes the two senior_ai_engineer recordings.
"""

from __future__ import annotations

import os

import pytest
from test_agent_jd_intent import assert_senior_ai_engineer_intent
from test_agent_keywords import assert_senior_ai_engineer_keywords

from resume_tailor import llm
from resume_tailor.agents import jd_intent, keywords


@pytest.mark.live
def test_agents_live(monkeypatch: pytest.MonkeyPatch, jd) -> None:
    mode = "record" if os.environ.get("LLM_MODE") == "record" else "live"
    monkeypatch.setenv("LLM_MODE", mode)
    monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    try:
        text = jd("senior_ai_engineer")
        assert_senior_ai_engineer_intent(jd_intent.run(text, case="senior_ai_engineer"))
        assert_senior_ai_engineer_keywords(keywords.run(text, case="senior_ai_engineer"))
        usage = llm.last_usage()
        assert usage is not None and usage.input_tokens > 0
    finally:
        llm.set_provider(None)
