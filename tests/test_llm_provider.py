from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.error import URLError

import pytest

from src.llm.provider import (
    LLMProviderError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    create_llm_provider,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_provider_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("EQUIMASTER_LLM_PROVIDER", raising=False)
    assert create_llm_provider() is None


def test_provider_requires_base_url_and_model(monkeypatch):
    monkeypatch.setenv("EQUIMASTER_LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("EQUIMASTER_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EQUIMASTER_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="BASE_URL and EQUIMASTER_LLM_MODEL"):
        create_llm_provider()


def test_provider_rejects_insecure_remote_endpoint():
    with pytest.raises(ValueError, match="must use HTTPS"):
        OpenAICompatibleConfig("http://api.example.com/v1", "test-model")
    OpenAICompatibleConfig("http://127.0.0.1:11434/v1", "local-model")


def test_provider_sends_messages_and_parses_content(monkeypatch):
    import src.llm.provider as provider_module
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        return _Response({"choices": [{"message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://llm.example.com/v1/", "test-model", "secret", 12)
    )
    content = provider.generate(
        [{"role": "system", "content": "Return JSON."}, {"role": "user", "content": "Analyze."}],
        {"type": "json_object"},
    )

    assert content == '{"ok":true}'
    assert captured["url"] == "https://llm.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 12
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_provider_surfaces_connection_errors(monkeypatch):
    import src.llm.provider as provider_module

    def fail_urlopen(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(provider_module, "urlopen", fail_urlopen)
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig("http://localhost:11434/v1", "local"))
    with pytest.raises(LLMProviderError, match="connection failed"):
        provider.generate([{"role": "user", "content": "hello"}])


def test_provider_rejects_invalid_messages():
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig("http://localhost:11434/v1", "local"))
    with pytest.raises(ValueError, match="supported role"):
        provider.generate([{"role": "tool", "content": "unexpected"}])


def test_grounded_synthesis_validates_and_returns_cited_claims():
    from src.llm.evidence import synthesize_retrieved_evidence

    class FakeProvider:
        def __init__(self):
            self.messages = None
            self.response_format = None

        def generate(self, messages, response_format=None):
            self.messages = messages
            self.response_format = response_format
            return json.dumps({
                "stance": "bullish",
                "summary": "The article reports higher revenue.",
                "summary_evidence_ids": ["chunk-1"],
                "claims": [{"text": "Revenue increased.", "evidence_ids": ["chunk-1"]}],
            })

    fake = FakeProvider()
    article = {
        "chunk_id": "chunk-1", "ticker": "TEST", "title": "TEST reports growth",
        "source": "Reuters", "published_at": "2026-01-01T00:00:00+00:00",
        "observed_at": "2026-01-01T01:00:00+00:00",
        "url": "https://reuters.com/story", "chunk_text": "Revenue increased.",
    }

    result = synthesize_retrieved_evidence(
        "test", [article], datetime(2026, 1, 2, tzinfo=timezone.utc), provider=fake
    )

    assert result["ticker"] == "TEST"
    assert result["stance"] == "bullish"
    assert result["llm_model_id"] == "FakeProvider"
    assert result["claims"][0]["evidence_ids"] == ["chunk-1"]
    assert fake.response_format == {"type": "json_object"}
    user_payload = json.loads(fake.messages[1]["content"])
    assert user_payload["evidence"][0]["evidence_id"] == "chunk-1"


def test_grounded_synthesis_rejects_unknown_citations_and_future_evidence():
    from src.llm.evidence import synthesize_retrieved_evidence

    class FakeProvider:
        calls = 0
        content = ""

        def generate(self, messages, response_format=None):
            self.calls += 1
            return self.content

    fake = FakeProvider()
    article = {
        "chunk_id": "chunk-1", "ticker": "TEST", "published_at": "2026-01-01",
        "observed_at": "2026-01-01", "chunk_text": "Revenue increased.",
    }
    fake.content = json.dumps({
        "stance": "neutral", "summary": "Unsupported citation.",
        "summary_evidence_ids": ["unknown"],
        "claims": [],
    })
    with pytest.raises(LLMProviderError, match="unknown evidence ID"):
        synthesize_retrieved_evidence(
            "TEST", [article], datetime(2026, 1, 2, tzinfo=timezone.utc), provider=fake
        )

    article["observed_at"] = "2026-01-03T00:00:00+00:00"
    calls_before = fake.calls
    with pytest.raises(ValueError, match="cutoff"):
        synthesize_retrieved_evidence(
            "TEST", [article], datetime(2026, 1, 2, tzinfo=timezone.utc), provider=fake
        )
    assert fake.calls == calls_before


def test_grounded_synthesis_fails_explicitly_when_llm_is_disabled(monkeypatch):
    import src.llm.evidence as evidence_module
    from src.llm.provider import LLMDisabledError

    monkeypatch.setattr(evidence_module, "create_llm_provider", lambda: None)
    with pytest.raises(LLMDisabledError, match="Configure EQUIMASTER_LLM_PROVIDER"):
        evidence_module.synthesize_retrieved_evidence(
            "TEST", [], datetime(2026, 1, 2, tzinfo=timezone.utc)
        )
