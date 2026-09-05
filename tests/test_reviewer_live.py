"""SCRUM-16: real Gemini run of Agent 6's semantic-drift check (`uv run pytest -m live`).

`LLM_MODE=record uv run pytest -m live -k reviewer_live` (re)writes
`tests/recordings/reviewer/{semantic_drift_team,clean_rewrite}.json`, the two recordings
`tests/test_reviewer.py` replays. The drift case adds a claim ("led a team of six
engineers") that neither the original bullet nor the project notes support; the clean case
is the same bullet reworded, so a correct model accepts it.
"""

from __future__ import annotations

import os

import pytest
from test_reviewer import CLEAN_CASE, DRIFT_CASE, ORIGINAL, KB_SAMPLE, cv_text

from resume_tailor import llm
from resume_tailor.agents import reviewer
from resume_tailor.knowledge import allowed_vocabulary, load_projects

DRIFTED = (
    "Led a team of six engineers building an LLM service that drafts three reply options for "
    "every customer review, with sentiment-conditioned prompts and per-dealership rules; the "
    "dealer publishes one. Roughly 93% of drafts go out unedited, across over 100k reviews a month."
)
CLEAN = (
    "Built and shipped an LLM service that drafts three reply options for every customer "
    "review (detailed, professional, short), applying sentiment-conditioned prompts and "
    "per-dealership rules; the dealer picks one and publishes it. Roughly 93% of drafts go "
    "out unedited, across over 100k reviews a month."
)


@pytest.mark.live
def test_reviewer_live(monkeypatch: pytest.MonkeyPatch) -> None:
    mode = "record" if os.environ.get("LLM_MODE") == "record" else "live"
    monkeypatch.setenv("LLM_MODE", mode)
    monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    try:
        facts = load_projects(KB_SAMPLE)
        vocab = allowed_vocabulary(cv_text(), facts)
        fact = {f.name: f for f in facts}["AI-Powered Review Response System"]

        drifted = reviewer.review(ORIGINAL, DRIFTED, vocab, fact, case=DRIFT_CASE)
        assert drifted.accept is False and drifted.reason.startswith("unsupported claim:")

        clean = reviewer.review(ORIGINAL, CLEAN, vocab, fact, case=CLEAN_CASE)
        assert clean.accept is True, clean.reason
    finally:
        llm.set_provider(None)
