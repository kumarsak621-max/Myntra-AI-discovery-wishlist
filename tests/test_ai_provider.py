from __future__ import annotations

from app.ai.provider import (
    AIError,
    GeminiAIService,
    QUOTA_MESSAGE,
)
from app.ai.provider import test_gemini_connection as probe_gemini_connection
from app.config import Settings


class FakeResponse:
    def __init__(self, text: str = "") -> None:
        self.text = text


class FakeAPIError(Exception):
    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class FakeModels:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self.responses:
            raise AssertionError("unexpected extra Gemini call")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.models = FakeModels(responses)


def _provider(**kwargs) -> GeminiAIService:
    values = {
        "gemini_api_key": "unit-test-key",
        "gemini_model": "gemini-2.5-flash",
        "ai_retry_attempts": 4,
        "ai_rate_limit_seconds": 0,
    }
    values.update(kwargs)
    return GeminiAIService(Settings(**values))


def _attach_client(monkeypatch, client: FakeClient) -> None:
    monkeypatch.setattr(GeminiAIService, "_client", lambda self: client)


def test_complete_json_success(monkeypatch):
    client = FakeClient([FakeResponse('{"relevance":"none"}')])
    _attach_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    text = _provider().complete_json(system="s", user="u")
    assert text == '{"relevance":"none"}'
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"
    assert client.models.calls[0]["contents"] == "u"


def test_strips_google_prefix(monkeypatch):
    client = FakeClient([FakeResponse('{"ok":true}')])
    _attach_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = _provider(gemini_model="google/gemini-2.5-flash")
    assert provider.model == "gemini-2.5-flash"
    provider.complete_json(system="s", user="u")
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"


def test_empty_response_is_failure(monkeypatch):
    client = FakeClient([FakeResponse("")])
    _attach_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    try:
        _provider(ai_retry_attempts=1).complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert "empty" in str(exc).lower()


def test_transient_5xx_then_success(monkeypatch):
    client = FakeClient(
        [
            FakeAPIError("internal", code=503),
            FakeResponse('{"relevance":"low"}'),
        ]
    )
    _attach_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    text = _provider().complete_json(system="s", user="u")
    assert "relevance" in text
    assert len(client.models.calls) == 2


def test_timeout_then_success(monkeypatch):
    client = FakeClient(
        [
            FakeAPIError("The Gemini request timed out."),
            FakeResponse('{"relevance":"none"}'),
        ]
    )
    _attach_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    text = _provider().complete_json(system="s", user="u")
    assert "relevance" in text
    assert len(client.models.calls) == 2


def test_401_is_not_retried(monkeypatch):
    client = FakeClient([FakeAPIError("invalid api key AIzaSySECRET", code=401)])
    _attach_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    try:
        _provider().complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert "API key" in str(exc)
        assert "AIzaSySECRET" not in str(exc)
        assert "unit-test-key" not in str(exc)
    assert len(client.models.calls) == 1


def test_quota_maps_to_clear_message(monkeypatch):
    client = FakeClient(
        [
            FakeAPIError("RESOURCE_EXHAUSTED", code=429),
            FakeAPIError("RESOURCE_EXHAUSTED", code=429),
        ]
    )
    _attach_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    try:
        _provider(ai_retry_attempts=2).complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert str(exc) == QUOTA_MESSAGE
        assert exc.retryable is True
    assert len(client.models.calls) == 2


def test_connection_test_reports_missing_key():
    result = probe_gemini_connection(Settings(gemini_api_key="", gemini_model="gemini-2.5-flash"))
    assert result["ok"] is False
    assert result["status"] == "FAILED"
    assert result["credentials"] == "Missing"
    assert result["provider"] == "Google Gemini"
    assert result["model"] == "gemini-2.5-flash"
    assert result["error"] == "Gemini API key is not configured."
    assert "AIza" not in str(result)


def test_connection_test_success(monkeypatch):
    client = FakeClient([FakeResponse("ok")])

    class DummyTypes:
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    class DummyGenai:
        Client = lambda api_key=None: client  # noqa: E731

    import sys
    import types as pytypes

    dummy = pytypes.ModuleType("google")
    dummy_genai = pytypes.ModuleType("google.genai")
    dummy_genai.Client = DummyGenai.Client
    dummy_types = pytypes.ModuleType("google.genai.types")
    dummy_types.GenerateContentConfig = DummyTypes.GenerateContentConfig
    dummy.genai = dummy_genai
    monkeypatch.setitem(sys.modules, "google", dummy)
    monkeypatch.setitem(sys.modules, "google.genai", dummy_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", dummy_types)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    result = probe_gemini_connection(
        Settings(gemini_api_key="unit-test-key", gemini_model="gemini-2.5-flash")
    )
    assert result["ok"] is True
    assert result["status"] == "SUCCESS"
    assert result["error"] is None
    assert result["credentials"] == "Configured"
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"
    assert "unit-test-key" not in str(result)


def test_missing_key_does_not_call_gemini(monkeypatch):
    called = {"n": 0}

    def boom(self):
        called["n"] += 1
        raise AssertionError("must not contact Gemini without a key")

    monkeypatch.setattr(GeminiAIService, "_client", boom)
    try:
        GeminiAIService(Settings(gemini_api_key="")).complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert str(exc) == "Gemini API key is not configured."
    assert called["n"] == 0
