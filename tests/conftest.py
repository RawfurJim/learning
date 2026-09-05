"""Shared fixtures: `.env` loading, the `live` marker, and the `llm_replay` fixture."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from resume_tailor import llm
from resume_tailor.settings import REPO_ROOT

load_dotenv(REPO_ROOT / ".env", override=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    skip = pytest.mark.skip(reason="GEMINI_API_KEY not set; live Gemini tests skipped")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def llm_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force replay mode against the committed recordings."""
    monkeypatch.setenv("LLM_MODE", "replay")
    monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    llm.reset_usage()
    yield
    llm.set_provider(None)


@pytest.fixture
def scratch_recordings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point recordings at a temp dir so record-mode tests never touch tests/recordings."""
    recordings = tmp_path / "recordings"
    monkeypatch.setenv("LLM_RECORDINGS_DIR", str(recordings))
    return recordings
