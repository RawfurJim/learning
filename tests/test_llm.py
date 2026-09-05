"""Contract tests for `resume_tailor.llm` (names are the ticket's contract, SCRUM-8)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from resume_tailor import llm
from resume_tailor.llm import LLMOutputError, ProviderResponse, RecordingMissing, Usage
from resume_tailor.schemas import Ping
from resume_tailor.settings import REPO_ROOT, Settings


class StubProvider:
    """Returns scripted texts in order and counts calls."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls = 0

    def generate(self, prompt: str, schema: type[BaseModel], *, model: str) -> ProviderResponse:
        self.calls += 1
        text = self.texts.pop(0)
        return ProviderResponse(text=text, usage=Usage(input_tokens=11, output_tokens=3, model=model))


@pytest.fixture
def llm_stub(monkeypatch: pytest.MonkeyPatch, scratch_recordings: Path):
    """Live mode with a stub provider (no network)."""
    monkeypatch.setenv("LLM_MODE", "live")
    llm.reset_usage()

    def install(*texts: str) -> StubProvider:
        stub = StubProvider(*texts)
        llm.set_provider(stub)
        return stub

    yield install
    llm.set_provider(None)


def test_replay_returns_recorded_model(llm_replay) -> None:
    result = llm.generate_json("any", Ping, case="ping")
    assert result == Ping(ok=True)
    assert llm.recording_path("ping") == REPO_ROOT / "tests" / "recordings" / "llm" / "ping.json"


def test_replay_missing_recording_raises(llm_replay) -> None:
    with pytest.raises(RecordingMissing) as excinfo:
        llm.generate_json("any", Ping, case="no_such_case")
    expected = REPO_ROOT / "tests" / "recordings" / "llm" / "no_such_case.json"
    assert str(expected) in str(excinfo.value)


def test_invalid_json_retries_once_then_raises(llm_stub) -> None:
    stub = llm_stub("garbage", "still garbage")
    with pytest.raises(LLMOutputError):
        llm.generate_json("any", Ping, case="retry")
    assert stub.calls == 2

    stub = llm_stub("garbage", '{"ok": true}')
    assert llm.generate_json("any", Ping, case="retry") == Ping(ok=True)
    assert stub.calls == 2


def test_usage_is_tracked(llm_stub) -> None:
    llm.reset_usage()
    assert llm.last_usage() is None
    llm_stub('{"ok": true}')
    llm.generate_json("any", Ping, case="usage")
    usage = llm.last_usage()
    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 3
    assert usage.model == Settings.from_env().llm_model


def test_record_mode_writes_then_replays(llm_stub, monkeypatch: pytest.MonkeyPatch, scratch_recordings: Path) -> None:
    monkeypatch.setenv("LLM_MODE", "record")
    llm_stub('{"ok": false}')
    assert llm.generate_json("any", Ping, case="agent_x/some_case") == Ping(ok=False)
    path = scratch_recordings / "agent_x" / "some_case.json"
    assert path.exists()

    monkeypatch.setenv("LLM_MODE", "replay")
    assert llm.generate_json("any", Ping, case="agent_x/some_case") == Ping(ok=False)
    assert llm.last_usage() == Usage(input_tokens=0, output_tokens=0, model="replay")


def test_settings_reads_env() -> None:
    settings = Settings.from_env(env_file=REPO_ROOT / ".env.example", environ={})
    assert settings.gemini_api_key is None
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.6-flash"
    assert settings.llm_mode == "replay"

    overridden = Settings.from_env(env_file=REPO_ROOT / ".env.example", environ={"LLM_MODE": "live"})
    assert overridden.llm_mode == "live"


@pytest.mark.live
def test_gemini_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Gemini call. `LLM_MODE=record uv run pytest -m live` (re)writes tests/recordings/llm/ping.json."""
    mode = "record" if os.environ.get("LLM_MODE") == "record" else "live"
    monkeypatch.setenv("LLM_MODE", mode)
    monkeypatch.delenv("LLM_RECORDINGS_DIR", raising=False)
    llm.set_provider(None)
    llm.reset_usage()

    result = llm.generate_json(
        'Return exactly this JSON object and nothing else: {"ok": true}', Ping, case="ping"
    )
    assert result == Ping(ok=True)

    usage = llm.last_usage()
    assert usage is not None and usage.input_tokens > 0 and usage.output_tokens > 0
    assert usage.model == Settings.from_env().llm_model
    if mode == "record":
        assert llm.recording_path("ping").exists()
    llm.set_provider(None)
