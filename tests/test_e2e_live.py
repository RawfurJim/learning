"""Real Gemini run of the summary & skills writer end to end (`uv run pytest -m live`).

`LLM_MODE=record uv run pytest -m live -k e2e_live` (re)writes every recording that
tests/test_agent_summary_skills.py, tests/test_e2e.py and test_app_download_available replay:
  tests/recordings/summary_skills/{senior_ai_engineer,ds_nlp,ml_platform}.json
  tests/recordings/summary_skills/senior_ai_engineer_cv_only.json
  tests/recordings/summary_skills/senior_ai_engineer_cv_only__flask.json
  tests/recordings/alias/{senior_ai_engineer,ds_nlp,ml_platform}.json
  tests/recordings/project_rank/{senior_ai_engineer,ds_nlp,ml_platform}.json
  (+ `<case>_retry.json` whenever the first answer broke a rule and was retried)
The JD intent and keywords are always replayed from their committed recordings so the
writer's inputs are identical in record and replay mode.
"""

from __future__ import annotations

import os

import pytest
from test_agent_summary_skills import (
    CASES,
    FIXTURES_DIR,
    KB_SAMPLE,
    NO_KB,
    STAGES,
    assert_rewrites_applied,
    assert_skills_count_within_budget,
    assert_skills_order_mandatory_first,
    assert_skills_subset_of_inventory,
    assert_summary_keeps_numbers,
    assert_summary_terms_in_vocabulary,
    assert_summary_word_budget,
    cv_bytes,
    editable_paras,
    flask_analysis,
    flat_terms,
    skills_of,
)
from test_e2e import assert_valid_tailored_docx

from resume_tailor import llm, matching, pipeline
from resume_tailor.agents import jd_intent, keywords
from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.knowledge import load_projects


@pytest.mark.live
def test_e2e_live(monkeypatch: pytest.MonkeyPatch, jd) -> None:
    mode = "record" if os.environ.get("LLM_MODE") == "record" else "live"
    monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    cv = cv_bytes()
    originals = editable_paras(cv)
    cv_text = "\n".join(p.full_text for p in iter_paragraphs(load(cv)))
    inventory = matching.build_inventory(iter_paragraphs(load(cv)), load_projects(KB_SAMPLE))
    try:
        for case in CASES:
            monkeypatch.setenv("LLM_MODE", "replay")
            text = jd(case)
            analysis_inputs = (jd_intent.run(text, case=case), keywords.run(text, case=case))

            monkeypatch.setenv("LLM_MODE", mode)
            analysis = pipeline.analyse(cv, text, KB_SAMPLE, case=case, jd_analysis=analysis_inputs)
            result = pipeline.run(cv, text, KB_SAMPLE, set(), STAGES, analysis=analysis)
            assert_valid_tailored_docx(result, cv)
            assert_summary_word_budget(result, originals["summary"].text)
            assert_summary_terms_in_vocabulary(result, cv_text, KB_SAMPLE)
            assert_summary_keeps_numbers(result, originals["summary"].text)
            assert_skills_subset_of_inventory(result, inventory, set())
            assert_skills_order_mandatory_first(result)
            assert_skills_count_within_budget(result, originals["skills"].text)
            assert_rewrites_applied(result)
            assert result.usage.input_tokens > 0

        # Flask: adjacent to FastAPI, CV-only inventory; forbidden unless approved.
        monkeypatch.setenv("LLM_MODE", "replay")
        analysis = flask_analysis(cv)
        text = jd("senior_ai_engineer")
        monkeypatch.setenv("LLM_MODE", mode)
        without = pipeline.run(cv, text, NO_KB, set(), STAGES, analysis=analysis)
        assert "flask" not in {t.lower() for t in flat_terms(skills_of(without))}
        assert "flask" not in without.writer.summary.lower()
        with_flask = pipeline.run(cv, text, NO_KB, {"Flask"}, STAGES, analysis=analysis)
        cv_only = matching.build_inventory(iter_paragraphs(load(cv)), [])
        assert_skills_subset_of_inventory(with_flask, cv_only, {"Flask"})
        assert "flask" in {t.lower() for t in flat_terms(skills_of(with_flask))} or "flask" in with_flask.writer.summary.lower()
    finally:
        llm.set_provider(None)
