"""SCRUM-15: identical runs are served from `.cache/` (tests use an isolated temp cache dir)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_agent_summary_skills import KB_SAMPLE, STAGES, cv_bytes

from resume_tailor import cache, llm, pipeline
from resume_tailor.docx_io import iter_paragraphs, load
from resume_tailor.settings import Settings

JDS = KB_SAMPLE.parent / "jds"


def _jd(name: str) -> str:
    return (JDS / f"{name}.txt").read_text(encoding="utf-8")


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Route every `llm.generate_json` call through a counter (agents call it as `llm.generate_json`)."""
    calls: list[str] = []
    original = llm.generate_json

    def spy(prompt, schema, *, case):
        calls.append(case)
        return original(prompt, schema, case=case)

    monkeypatch.setattr(llm, "generate_json", spy)
    return calls


def test_second_run_hits_cache(llm_replay, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    original = cv_bytes()
    jd_text = _jd("senior_ai_engineer")

    first = pipeline.run(original, jd_text, KB_SAMPLE, set(), STAGES, case="senior_ai_engineer")
    n = len(calls)
    assert n >= 4, calls  # intent, keywords, alias, ranking, writer...
    assert first.rewrites and not first.from_cache

    second = pipeline.run(original, jd_text, KB_SAMPLE, set(), STAGES, case="senior_ai_engineer")
    assert len(calls) == n, calls[n:]  # 0 new LLM calls
    assert second.from_cache
    assert second.output == first.output  # identical bytes, not just identical text
    assert second.source == first.source
    assert second.rewrites == first.rewrites and second.sections == first.sections
    assert second.usage == first.usage

    # Accept/reject still works from the cached result.
    para_id = next(iter(second.rewrites))
    rebuilt = pipeline.assemble(second, rejected={para_id})
    texts = {p.id: p.text for p in iter_paragraphs(load(rebuilt))}
    assert texts[para_id] == second.originals[para_id]


def test_cache_key_changes_with_inputs() -> None:
    cv, kb = b"cv-bytes", "kb text"
    settings = {"provider": "gemini", "model": "gemini-3.6-flash"}
    base = cache.cache_key(cv, "senior AI engineer", kb, settings)
    assert len(base) == 64 and base == cache.cache_key(cv, "senior AI engineer", kb, settings)
    assert cache.cache_key(cv, "data scientist NLP", kb, settings) != base
    assert cache.cache_key(b"other cv", "senior AI engineer", kb, settings) != base
    assert cache.cache_key(cv, "senior AI engineer", "other kb", settings) != base
    assert cache.cache_key(cv, "senior AI engineer", kb, {**settings, "model": "gemini-2.5-flash"}) != base


def test_cache_get_put_clear(llm_replay, tmp_path: Path) -> None:
    result = pipeline.run(cv_bytes(), _jd("senior_ai_engineer"), KB_SAMPLE, set(), STAGES, case="senior_ai_engineer")
    key = cache.cache_key(b"x", "y", "z", {})
    assert cache.get(key, pipeline.RunResult, cache_dir=tmp_path) is None
    path = cache.put(key, result, cache_dir=tmp_path)
    assert path.parent == tmp_path and path.suffix == ".json"
    json.loads(path.read_text(encoding="utf-8"))  # stored as JSON
    loaded = cache.get(key, pipeline.RunResult, cache_dir=tmp_path)
    assert loaded is not None and loaded.output == result.output and loaded.ranked == result.ranked
    assert cache.clear(cache_dir=tmp_path) == 1
    assert cache.get(key, pipeline.RunResult, cache_dir=tmp_path) is None
    assert cache.clear(cache_dir=tmp_path) == 0


def test_cache_dir_is_gitignored() -> None:
    from resume_tailor.settings import REPO_ROOT

    assert ".cache/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert Settings.from_env(env_file=None, environ={}).cache_dir == REPO_ROOT / ".cache"


def test_cache_bypassed_in_record_mode(llm_replay, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    original = cv_bytes()
    jd_text = _jd("senior_ai_engineer")
    pipeline.run(original, jd_text, KB_SAMPLE, set(), STAGES, case="senior_ai_engineer")
    n = len(calls)

    # `use_cache=False` re-runs everything even though a cached result exists.
    again = pipeline.run(original, jd_text, KB_SAMPLE, set(), STAGES, case="senior_ai_engineer", use_cache=False)
    assert len(calls) == 2 * n and not again.from_cache

    # Record mode never reads the cache, so the cached run above must not short-circuit the
    # run below: it has to reach the provider, which is unbuildable here (unknown provider, blank
    # keys; blank env values override `.env`), so it fails fast with ConfigError and no network.
    scratch_recordings = tmp_path / "recordings"  # record mode must never write into tests/recordings
    monkeypatch.setenv("LLM_RECORDINGS_DIR", str(scratch_recordings))
    monkeypatch.setenv("LLM_MODE", "record")
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    llm.set_provider(None)
    with pytest.raises(llm.ConfigError):
        pipeline.run(original, jd_text, KB_SAMPLE, set(), STAGES, case="senior_ai_engineer")
    assert not list(scratch_recordings.rglob("*.json"))
