from __future__ import annotations

import httpx

from app.ai.provider import AIError, AIProvider
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

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "https://x"), response=httpx.Response(self.status_code))


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


def _provider(**kwargs) -> AIProvider:
    values = {
        "openrouter_api_key": "unit-test-key",
        "openrouter_model": "google/gemini-2.5-flash",
        "ai_retry_attempts": 4,
        "ai_rate_limit_seconds": 0,
    }
    values.update(kwargs)
    return AIProvider(Settings(**values))


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
        assert "401" in str(exc)
        assert "unit-test-key" not in str(exc)
    assert len(client.calls) == 1


def test_connection_test_reports_missing_key():
    from app.ai.provider import test_openrouter_connection

    result = test_openrouter_connection(Settings(openrouter_api_key="", openrouter_model="google/gemini-2.5-flash"))
    assert result["ok"] is False
    assert result["status"] == "FAILED"
    assert result["credentials"] == "Missing"
    assert result["model"] == "google/gemini-2.5-flash"
    assert "not configured" in (result["error"] or "").lower()
    assert "sk-or" not in str(result)


def test_connection_test_success(monkeypatch):
    from app.ai.provider import test_openrouter_connection

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
    result = test_openrouter_connection(
        Settings(openrouter_api_key="unit-test-key", openrouter_model="google/gemini-2.5-flash")
    )
    assert result["ok"] is True
    assert result["status"] == "SUCCESS"
    assert result["http_status"] == 200
    assert result["error"] is None
    assert client.calls[0]["json"]["model"] == "google/gemini-2.5-flash"
    assert "response_format" not in client.calls[0]["json"]
    assert "unit-test-key" not in str(result)
