"""Real Gemini run of the alias and project-ranking calls (`uv run pytest -m live`).

`LLM_MODE=record uv run pytest -m live` (re)writes every recording that
tests/test_matching.py and test_app_buckets_render replay:
  tests/recordings/alias/{pytorch_aliases,flask_vs_fastapi,react_kubernetes,senior_ai_engineer}.json
  tests/recordings/project_rank/senior_ai_engineer.json
The JD intent and keywords are always replayed from their committed recordings so the
matching inputs are identical in record and replay mode.
"""

from __future__ import annotations

import os

import pytest
from test_matching import (
    CASE_FLASK,
    CASE_MISSING,
    CASE_RANK,
    CASE_TORCH,
    FIXTURES_DIR,
    KB_SAMPLE,
    SMALL_INVENTORY,
    _keywords,
    assert_flask_adjacent,
    assert_react_kubernetes_missing,
    assert_top_ranked_are_professional,
    assert_torch_all_matched,
)

from resume_tailor import llm, matching
from resume_tailor.agents import jd_intent, keywords
from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.knowledge import load_projects


@pytest.mark.live
def test_matching_live(monkeypatch: pytest.MonkeyPatch, jd) -> None:
    mode = "record" if os.environ.get("LLM_MODE") == "record" else "live"
    monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    try:
        monkeypatch.setenv("LLM_MODE", "replay")
        text = jd("senior_ai_engineer")
        intent = jd_intent.run(text, case="senior_ai_engineer")
        jd_keywords = keywords.run(text, case="senior_ai_engineer")

        monkeypatch.setenv("LLM_MODE", mode)
        facts = load_projects(KB_SAMPLE)
        inventory = matching.build_inventory(iter_paragraphs(load(FIXTURES_DIR / "sample_cv.docx")), facts)

        assert_torch_all_matched(matching.match(_keywords(["PyTorch", "Pytorch", "Torch"]), inventory, facts, case=CASE_TORCH))
        assert_flask_adjacent(matching.match(_keywords(["Python", "Flask"], ["Docker"]), SMALL_INVENTORY, [], case=CASE_FLASK))
        assert_react_kubernetes_missing(
            matching.match(_keywords(["Python", "React"], ["Kubernetes"]), inventory, facts, case=CASE_MISSING)
        )

        # The app's own run on the senior_ai_engineer JD (replayed by test_app_buckets_render).
        result = matching.match(jd_keywords, inventory, facts, case=CASE_RANK)
        assert "React" in result.missing and "Kubernetes" in result.missing, result
        assert {"Python", "PyTorch", "RAG", "AWS", "Docker"} <= set(result.matched), result.matched
        assert_top_ranked_are_professional(matching.rank_projects(intent, facts, case=CASE_RANK), facts)

        usage = llm.last_usage()
        assert usage is not None and usage.input_tokens > 0
    finally:
        llm.set_provider(None)
