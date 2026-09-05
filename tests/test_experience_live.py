"""Real Gemini run of the experience writer on the fixture CV x 3 JDs (`uv run pytest -m live`).

`LLM_MODE=record uv run pytest -m live -k experience_live` (re)writes every recording
that tests/test_agent_experience.py replays:
  tests/recordings/experience/<case>__g<n>_<heading slug>.json   (4 groups x 3 JDs)
  (+ `<...>_retry.json` whenever a group's first answer broke a rule and was retried)
Everything upstream (JD intent, keywords, alias, project ranking) is always replayed from
its committed recordings, so the writer's inputs are identical in record and replay mode
and no other recording is touched.
"""

from __future__ import annotations

import os

import pytest
from test_agent_experience import (
    CASES,
    KB_SAMPLE,
    assert_added_metrics_come_from_kb,
    assert_bullet_numbers_preserved,
    assert_bullet_terms_in_vocabulary,
    assert_bullet_word_budget,
    assert_jd_keywords_woven,
    assert_professional_first,
    cv_bytes,
    jd_text,
    run_experience,
)

from resume_tailor import llm, pipeline
from resume_tailor.docx_io import iter_paragraphs, load


@pytest.mark.live
def test_experience_live(monkeypatch: pytest.MonkeyPatch) -> None:
    mode = "record" if os.environ.get("LLM_MODE") == "record" else "live"
    monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    cv = cv_bytes()
    cv_text = "\n".join(p.full_text for p in iter_paragraphs(load(cv)))
    try:
        for case in CASES:
            monkeypatch.setenv("LLM_MODE", "replay")
            analysis = pipeline.analyse(cv, jd_text(case), KB_SAMPLE, case=case)

            monkeypatch.setenv("LLM_MODE", mode)
            exp = run_experience(case, analysis)
            assert len(exp.bullets) == 12
            assert_bullet_word_budget(exp)
            assert_bullet_numbers_preserved(exp)
            assert_bullet_terms_in_vocabulary(exp, analysis.match.matched, cv_text)
            assert_added_metrics_come_from_kb(exp)
            assert_professional_first(exp, analysis.match.matched)
            if case == "senior_ai_engineer":
                assert_jd_keywords_woven(exp, analysis.match.matched, minimum=6)
            # A fallback is allowed (the rules win), but never silent.
            for para_id in exp.reverted:
                assert any(para_id in note for note in exp.notes), (para_id, exp.notes)
            assert exp.input_tokens > 0 and exp.calls >= 4
    finally:
        llm.set_provider(None)
