from __future__ import annotations

import httpx

from app.ai.provider import (
    AIError,
    AUTH_401_MESSAGE,
    CREDIT_402_MESSAGE,
    OpenRouterAIService,
    QUOTA_MESSAGE,
)
from app.ai.provider import test_openrouter_connection as probe_openrouter_connection
from app.config import Settings


class FakeResponse:
    def __init__(self, status_code: int, text: str = "", payload=None, headers=None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if not self.responses:
            raise AssertionError("unexpected extra HTTP call")
        return self.responses.pop(0)


def _provider(**kwargs) -> OpenRouterAIService:
    values = {
        "openrouter_api_key": "unit-test-key",
        "openrouter_model": "google/gemini-2.5-flash",
        "ai_retry_attempts": 4,
        "ai_rate_limit_seconds": 0,
        "ai_http_timeout_seconds": 5,
    }
    values.update(kwargs)
    return OpenRouterAIService(Settings(**values))


def test_gemini_openrouter_skips_json_object_mode(monkeypatch):
    client = FakeClient(
        [
            FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"relevance":"none"}'}}]},
            )
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    text = _provider().complete_json(system="s", user="u")
    assert text == '{"relevance":"none"}'
    assert "response_format" not in client.calls[0]["json"]
    assert client.calls[0]["json"]["model"] == "google/gemini-2.5-flash"
    assert "unit-test-key" not in str(client.calls[0]["headers"]["Authorization"]) or True
    assert client.calls[0]["headers"]["Authorization"] == "Bearer unit-test-key"
    assert client.calls[0]["json"]["max_tokens"] == 2000
    assert client.calls[0]["json"]["max_tokens"] != 65535


def test_json_mode_400_retries_without_response_format(monkeypatch):
    client = FakeClient(
        [
            FakeResponse(400, text='{"error":{"message":"JSON mode is not supported"}}'),
            FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"ok":true}'}}]},
            ),
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = _provider(openrouter_model="openai/gpt-4o-mini")
    text = provider.complete_json(system="s", user="u")
    assert text == '{"ok":true}'
    assert client.calls[0]["json"].get("response_format") == {"type": "json_object"}
    assert "response_format" not in client.calls[1]["json"]


def test_transient_5xx_then_success(monkeypatch):
    client = FakeClient(
        [
            FakeResponse(503, text="upstream unavailable"),
            FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"relevance":"low"}'}}]},
            ),
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    text = _provider().complete_json(system="s", user="u")
    assert "relevance" in text
    assert len(client.calls) == 2


def test_timeout_then_success(monkeypatch):
    class TimeoutThenOk(FakeClient):
        def post(self, url, headers=None, json=None):
            self.calls.append({"url": url, "json": json, "headers": headers})
            if len(self.calls) == 1:
                raise httpx.TimeoutException("timed out")
            return self.responses.pop(0)

    client = TimeoutThenOk(
        [
            FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"relevance":"none"}'}}]},
            )
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    text = _provider().complete_json(system="s", user="u")
    assert "relevance" in text
    assert len(client.calls) == 2


def test_401_is_not_retried(monkeypatch):
    client = FakeClient([FakeResponse(401, text='{"error":{"message":"invalid key"}}')])
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    try:
        _provider().complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert str(exc) == AUTH_401_MESSAGE
        assert exc.http_status == 401
        assert "unit-test-key" not in str(exc)
    assert len(client.calls) == 1


def test_402_is_not_retried_and_uses_credit_message(monkeypatch):
    client = FakeClient(
        [
            FakeResponse(
                402,
                text='{"error":{"message":"You requested up to 65535 tokens, but can only afford 16000."}}',
            )
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    try:
        _provider().complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert str(exc) == CREDIT_402_MESSAGE
        assert exc.http_status == 402
        assert exc.retryable is False
        assert "65535" not in str(exc)
    assert len(client.calls) == 1
    assert client.calls[0]["json"]["max_tokens"] == 2000
    assert client.calls[0]["json"]["max_tokens"] != 65535


def test_quota_maps_to_clear_message(monkeypatch):
    client = FakeClient(
        [
            FakeResponse(429, text="rate limited"),
            FakeResponse(429, text="rate limited"),
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    try:
        _provider(ai_retry_attempts=2).complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert str(exc) == QUOTA_MESSAGE
        assert exc.retryable is True
    assert len(client.calls) == 2


def test_bare_gemini_model_gets_openrouter_prefix():
    provider = _provider(openrouter_model="gemini-2.5-flash")
    assert provider.model == "google/gemini-2.5-flash"


def test_connection_test_reports_missing_key():
    result = probe_openrouter_connection(
        Settings(openrouter_api_key="", openrouter_model="google/gemini-2.5-flash")
    )
    assert result["ok"] is False
    assert result["status"] == "FAILED"
    assert result["credentials"] == "Missing"
    assert result["provider"] == "OpenRouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["error"]
    assert "not configured" in result["error"].lower()
    assert "OPENROUTER_API_KEY" in result["error"]
    assert "Gemini" not in result["error"]
    assert "sk-or" not in str(result)


def test_connection_test_success(monkeypatch):
    client = FakeClient(
        [
            FakeResponse(
                200,
                payload={"choices": [{"message": {"content": "ok"}}]},
            )
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    result = probe_openrouter_connection(
        Settings(openrouter_api_key="unit-test-key", openrouter_model="google/gemini-2.5-flash")
    )
    assert result["ok"] is True
    assert result["status"] == "SUCCESS"
    assert result["error"] is None
    assert result["credentials"] == "Configured"
    assert client.calls[0]["json"]["model"] == "google/gemini-2.5-flash"
    assert "response_format" not in client.calls[0]["json"]
    messages = client.calls[0]["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "unit-test-key" not in str(result)


def test_missing_key_does_not_call_openrouter(monkeypatch):
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("must not contact OpenRouter without a key")

    monkeypatch.setattr("httpx.Client", boom)
    try:
        OpenRouterAIService(Settings(openrouter_api_key="")).complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert "not configured" in str(exc).lower()
        assert "OPENROUTER_API_KEY" in str(exc)
        assert "Gemini" not in str(exc)
    assert called["n"] == 0


def test_list_content_parts_are_extracted():
    from app.ai.provider import _message_content

    text = _message_content(
        {"content": [{"type": "text", "text": '{"relevance":"none"}'}]}
    )
    assert text == '{"relevance":"none"}'


def test_400_includes_openrouter_error_body(monkeypatch):
    client = FakeClient(
        [FakeResponse(400, text='{"error":{"message":"Provider returned error"}}')]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    try:
        _provider().complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert exc.http_status == 400
        assert "400" in str(exc)
        assert "Provider returned error" in str(exc)


def test_quoted_secret_is_stripped_from_authorization(monkeypatch):
    client = FakeClient(
        [
            FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"relevance":"none"}'}}]},
            )
        ]
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **k: client)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _provider(openrouter_api_key='  "sk-or-v1-quotedtest"  ').complete_json(system="s", user="u")
    auth = client.calls[0]["headers"]["Authorization"]
    assert auth == "Bearer sk-or-v1-quotedtest"
    assert auth.count("Bearer") == 1
    assert '"' not in auth
    assert client.calls[0]["url"].endswith("/chat/completions")
    assert "generativelanguage.googleapis.com" not in client.calls[0]["url"]


def test_gemini_style_key_is_rejected_before_http(monkeypatch):
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("must not call OpenRouter with a Gemini key")

    monkeypatch.setattr("httpx.Client", boom)
    try:
        _provider(openrouter_api_key="AIzaSyDummyGeminiKey").complete_json(system="s", user="u")
        raise AssertionError("should have failed")
    except AIError as exc:
        assert "sk-or-" in str(exc)
        assert "AIza" not in str(exc)
    assert called["n"] == 0
