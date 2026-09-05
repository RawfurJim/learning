"""SCRUM-15: provider switch (Gemini / Groq / Ollama), price table and cost estimate. No network."""

from __future__ import annotations

import json

import pytest
from test_agent_summary_skills import KB_SAMPLE, STAGES, cv_bytes

from resume_tailor import llm, pipeline
from resume_tailor.llm import ConfigError, GeminiProvider, GroqProvider, OllamaProvider, ProviderConfigError, Usage
from resume_tailor.schemas import Ping
from resume_tailor.settings import Settings

ENV_OFF = {"GEMINI_API_KEY": "", "GROQ_API_KEY": ""}


def _settings(**env: str) -> Settings:
    return Settings.from_env(env_file=None, environ={**ENV_OFF, **env})


def test_provider_selection() -> None:
    assert isinstance(llm.build_provider(_settings(LLM_PROVIDER="groq", GROQ_API_KEY="gsk_test")), GroqProvider)
    assert isinstance(llm.build_provider(_settings(LLM_PROVIDER="ollama")), OllamaProvider)
    with pytest.raises(ConfigError):
        llm.build_provider(_settings(LLM_PROVIDER="openai"))
    with pytest.raises(ConfigError):  # groq needs its key
        llm.build_provider(_settings(LLM_PROVIDER="groq"))
    with pytest.raises(ConfigError):  # gemini needs its key
        llm.build_provider(_settings(LLM_PROVIDER="gemini"))
    assert ProviderConfigError is ConfigError  # SCRUM-8 name still works


def test_provider_selection_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`generate_json` in live mode builds the provider named by LLM_PROVIDER; unknown -> ConfigError."""
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    llm.set_provider(None)
    with pytest.raises(ConfigError):
        llm.generate_json("any", Ping, case="x")
    llm.set_provider(None)


class _FakePost:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, payload: dict, headers: dict) -> dict:
        self.requests.append((url, payload, headers))
        return self.response


def test_groq_provider_request_and_usage() -> None:
    post = _FakePost(
        {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }
    )
    provider = GroqProvider("gsk_test", post=post)
    response = provider.generate("Return {\"ok\": true}", Ping, model="llama-3.3-70b-versatile")
    assert json.loads(response.text) == {"ok": True}
    assert response.usage == Usage(input_tokens=42, output_tokens=7, model="llama-3.3-70b-versatile")
    url, payload, headers = post.requests[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer gsk_test"
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["response_format"] == {"type": "json_object"}
    assert "ok" in payload["messages"][-1]["content"]  # schema is described in the prompt


def test_ollama_provider_request_and_usage() -> None:
    post = _FakePost({"message": {"content": '{"ok": false}'}, "prompt_eval_count": 10, "eval_count": 3})
    provider = OllamaProvider(base_url="http://ollama.local:11434/", post=post)
    response = provider.generate("Return {\"ok\": false}", Ping, model="llama3.1")
    assert json.loads(response.text) == {"ok": False}
    assert response.usage == Usage(input_tokens=10, output_tokens=3, model="llama3.1")
    url, payload, _ = post.requests[0]
    assert url == "http://ollama.local:11434/api/chat"
    assert payload["model"] == "llama3.1" and payload["stream"] is False
    assert payload["format"] == "json"


def test_gemini_provider_needs_no_network_to_construct() -> None:
    assert isinstance(GeminiProvider("fake-key"), GeminiProvider)


def test_price_table_and_estimate_cost() -> None:
    price = llm.price_for("gemini-3.6-flash")
    assert price is not None and 0 < price.input_gbp_per_1m < price.output_gbp_per_1m
    assert llm.price_for("gemini-3.6-flash-preview") == price  # version suffixes share the base price
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, model="gemini-3.6-flash")
    assert llm.estimate_cost(usage) == pytest.approx(price.input_gbp_per_1m + price.output_gbp_per_1m)
    assert llm.estimate_cost(Usage(0, 0, "gemini-3.6-flash")) == 0.0
    assert llm.estimate_cost(Usage(10_000, 10_000, "llama3.1")) == 0.0  # local Ollama models are free
    assert llm.estimate_cost(Usage(10_000, 10_000, "no-such-model")) == 0.0 and llm.price_for("no-such-model") is None
    # A typical run (~30k in / 5k out on gemini-3.6-flash) stays under the PRD's 1p budget... or says so.
    assert llm.estimate_cost(Usage(30_000, 5_000, "gemini-3.6-flash")) < 0.05


def test_usage_totals(llm_replay, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replayed recordings carry no tokens, so the spy stamps each call with a Gemini-sized usage."""
    monkeypatch.setenv("LLM_MODEL", "gemini-3.6-flash")
    original = llm.generate_json
    per_call = Usage(input_tokens=2_000, output_tokens=400, model="gemini-3.6-flash")
    n_calls = 0

    def stamped(prompt, schema, *, case):
        nonlocal n_calls
        n_calls += 1
        result = original(prompt, schema, case=case)
        llm.record_usage(per_call)
        return result

    monkeypatch.setattr(llm, "generate_json", stamped)
    jd_text = (KB_SAMPLE.parent / "jds" / "senior_ai_engineer.txt").read_text(encoding="utf-8")
    result = pipeline.run(cv_bytes(), jd_text, KB_SAMPLE, set(), STAGES, case="senior_ai_engineer", use_cache=False)

    usage = result.usage
    assert usage.calls == n_calls >= 4
    assert usage.total_input_tokens == usage.input_tokens == n_calls * 2_000
    assert usage.total_output_tokens == usage.output_tokens == n_calls * 400
    assert usage.model == "gemini-3.6-flash"
    assert usage.estimated_cost_gbp > 0
    assert usage.estimated_cost_gbp == pytest.approx(
        llm.estimate_cost(Usage(usage.total_input_tokens, usage.total_output_tokens, "gemini-3.6-flash"))
    )
    dumped = usage.model_dump()
    assert {"total_input_tokens", "total_output_tokens", "estimated_cost_gbp"} <= set(dumped)
