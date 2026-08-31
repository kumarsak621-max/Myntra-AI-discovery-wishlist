"""AI provider gateway. Keys and model names come only from Settings."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _redact(text: str) -> str:
    import re

    cleaned = text or ""
    cleaned = re.sub(r"sk-or-[A-Za-z0-9_-]+", "[redacted]", cleaned)
    cleaned = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", cleaned)
    cleaned = re.sub(r"AIza[0-9A-Za-z_-]+", "[redacted]", cleaned)
    return cleaned


def redact_secrets(text: str) -> str:
    return _redact(text)


def _backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    wait = float(retry_after) if retry_after and retry_after > 0 else float(2 ** attempt)
    return min(8.0, max(0.5, wait))


def _error_snippet(body: str) -> str:
    text = _redact((body or "").strip())
    try:
        payload = json.loads(text)
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            message = err.get("message") or err.get("metadata") or err
            return _redact(str(message))[:400]
        if isinstance(err, str):
            return _redact(err)[:400]
    except json.JSONDecodeError:
        pass
    return text[:400]


def _client_http_message(status_code: int, body: str) -> str:
    snippet = _error_snippet(body)
    if status_code in {401, 403}:
        return (
            "OpenRouter rejected the API key (HTTP "
            f"{status_code}). OpenRouter API key is not configured correctly."
        )
    if status_code == 400:
        return f"OpenRouter rejected the request (HTTP 400). {snippet}"
    return f"AI HTTP {status_code}: {snippet}"


def _is_json_mode_error(body: str) -> bool:
    lowered = (body or "").lower()
    return any(
        token in lowered
        for token in (
            "json mode",
            "json_object",
            "response_format",
            "not supported",
            "unsupported",
        )
    )


class AIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
    ) -> None:
        super().__init__(_redact(message))
        self.retryable = retryable
        self.http_status = http_status


class AIProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def provider_name(self) -> str:
        return self.settings.ai_provider.lower()

    @property
    def model(self) -> str:
        return self.settings.resolved_model

    def available(self) -> bool:
        return self.settings.has_ai_credentials

    def complete_json(self, *, system: str, user: str) -> str:
        if not self.available():
            raise AIError(
                "OpenRouter API key is not configured.",
                http_status=None,
            )
        if self.provider_name == "gemini":
            return self._gemini(system, user)
        return self._openrouter(system, user)

    def _openrouter(self, system: str, user: str) -> str:
        url = self.settings.openrouter_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kumarsak621-max/Myntra-AI-discovery-wishlist",
            "X-Title": "Myntra Discovery Engine",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        # Gemini via OpenRouter often rejects OpenAI json_object mode (HTTP 400).
        model_l = str(self.model).lower()
        prefer_json_object = not (model_l.startswith("google/") or "gemini" in model_l)
        return self._post_chat(url, headers, body, prefer_json_object=prefer_json_object)

    def _gemini(self, system: str, user: str) -> str:
        model = self.model
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        headers = {"x-goog-api-key": self.settings.gemini_api_key, "Content-Type": "application/json"}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        last_error: AIError | None = None
        attempts = max(1, int(self.settings.ai_retry_attempts))
        for attempt in range(1, attempts + 1):
            try:
                with httpx.Client(timeout=90.0) as client:
                    response = client.post(url, headers=headers, json=body)
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = AIError(
                            _client_http_message(response.status_code, response.text),
                            retryable=True,
                            http_status=response.status_code,
                        )
                        time.sleep(_backoff_seconds(attempt))
                        continue
                    if 400 <= response.status_code < 500:
                        raise AIError(
                            _client_http_message(response.status_code, response.text),
                            http_status=response.status_code,
                        )
                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates") or []
                    if not candidates:
                        raise AIError("Gemini returned no candidates.")
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    text = "".join(p.get("text") or "" for p in parts)
                    if not text:
                        raise AIError("Gemini returned empty text")
                    return text
            except AIError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                time.sleep(_backoff_seconds(attempt))
            except httpx.TimeoutException:
                last_error = AIError("The AI request timed out. Try fewer reviews, then retry.", retryable=True)
                time.sleep(_backoff_seconds(attempt))
            except httpx.HTTPError as exc:
                last_error = AIError("Network error contacting the AI provider.", retryable=True)
                logger.warning("AI call failed attempt %s: %s", attempt, _redact(str(exc)))
                time.sleep(_backoff_seconds(attempt))
        raise last_error or AIError("Gemini request failed")

    def _post_chat(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        prefer_json_object: bool,
    ) -> str:
        last_error: AIError | None = None
        json_object_enabled = prefer_json_object
        attempts = max(1, int(self.settings.ai_retry_attempts))
        for attempt in range(1, attempts + 1):
            payload = dict(body)
            if json_object_enabled:
                payload["response_format"] = {"type": "json_object"}
            try:
                with httpx.Client(timeout=90.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    if response.status_code == 429:
                        retry_after = None
                        raw_retry = response.headers.get("Retry-After")
                        try:
                            retry_after = float(raw_retry) if raw_retry else None
                        except (TypeError, ValueError):
                            retry_after = None
                        last_error = AIError(
                            "The AI provider rate-limited this request (HTTP 429). Retrying.",
                            retryable=True,
                            http_status=429,
                        )
                        time.sleep(_backoff_seconds(attempt, retry_after))
                        continue
                    if response.status_code == 400 and json_object_enabled and _is_json_mode_error(response.text):
                        json_object_enabled = False
                        last_error = AIError(
                            _client_http_message(400, response.text),
                            retryable=True,
                            http_status=400,
                        )
                        logger.warning("JSON response_format rejected; retrying without it.")
                        continue
                    if 400 <= response.status_code < 500:
                        raise AIError(
                            _client_http_message(response.status_code, response.text),
                            http_status=response.status_code,
                        )
                    if response.status_code >= 500:
                        last_error = AIError(
                            f"AI provider error HTTP {response.status_code}. Retrying.",
                            retryable=True,
                            http_status=response.status_code,
                        )
                        time.sleep(_backoff_seconds(attempt))
                        continue
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise AIError("AI returned a non-JSON HTTP body.") from exc
                    choices = data.get("choices") or []
                    if not choices:
                        raise AIError("AI returned no choices in the response.")
                    content = (choices[0].get("message") or {}).get("content") or ""
                    if not content:
                        raise AIError("AI returned empty content")
                    return content
            except AIError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                time.sleep(_backoff_seconds(attempt))
            except httpx.TimeoutException:
                last_error = AIError(
                    "The AI request timed out. Try fewer reviews, then retry.",
                    retryable=True,
                )
                logger.warning("AI timeout attempt %s", attempt)
                time.sleep(_backoff_seconds(attempt))
            except httpx.HTTPError as exc:
                last_error = AIError("Network error contacting the AI provider.", retryable=True)
                logger.warning("AI call failed attempt %s: %s", attempt, _redact(str(exc)))
                time.sleep(_backoff_seconds(attempt))
        raise last_error or AIError("AI request failed")


def test_openrouter_connection(settings: Settings | None = None) -> dict[str, Any]:
    """Minimal live OpenRouter request. Never logs or returns the API key."""
    from config.settings import reload_settings

    settings = settings or reload_settings()
    model = (settings.openrouter_model or settings.ai_model or "google/gemini-2.5-flash").strip()
    configured = bool((settings.openrouter_api_key or "").strip())
    result: dict[str, Any] = {
        "provider": "OpenRouter",
        "model": model,
        "credentials": "Configured" if configured else "Missing",
        "ok": False,
        "status": "FAILED",
        "http_status": None,
        "error": None,
    }
    if not configured:
        result["error"] = "OpenRouter API key is not configured."
        return result

    url = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/kumarsak621-max/Myntra-AI-discovery-wishlist",
        "X-Title": "Myntra Discovery Engine",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word ok."}],
        "temperature": 0,
        "max_tokens": 8,
    }
    last_error = "OpenRouter connection test failed."
    http_status: int | None = None
    attempts = min(3, max(1, int(settings.ai_retry_attempts or 3)))
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=body)
            http_status = response.status_code
            if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                time.sleep(_backoff_seconds(attempt))
                last_error = _client_http_message(response.status_code, response.text)
                continue
            if response.status_code in {401, 403}:
                result["http_status"] = response.status_code
                result["error"] = _client_http_message(response.status_code, response.text)
                return result
            if response.status_code >= 400:
                result["http_status"] = response.status_code
                result["error"] = _client_http_message(response.status_code, response.text)
                return result
            try:
                data = response.json()
            except ValueError:
                result["http_status"] = response.status_code
                result["error"] = "OpenRouter returned a non-JSON HTTP body."
                return result
            err = data.get("error") if isinstance(data, dict) else None
            if err:
                result["http_status"] = response.status_code
                result["error"] = _redact(str(err.get("message") if isinstance(err, dict) else err))[:400]
                return result
            choices = data.get("choices") or []
            content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
            if not str(content).strip():
                result["http_status"] = response.status_code
                result["error"] = "OpenRouter returned empty content."
                return result
            result["ok"] = True
            result["status"] = "SUCCESS"
            result["http_status"] = response.status_code
            result["error"] = None
            return result
        except httpx.TimeoutException:
            last_error = "The OpenRouter request timed out."
            http_status = None
            if attempt < attempts:
                time.sleep(_backoff_seconds(attempt))
                continue
        except httpx.HTTPError:
            last_error = "Network error contacting OpenRouter."
            if attempt < attempts:
                time.sleep(_backoff_seconds(attempt))
                continue
    result["http_status"] = http_status
    result["error"] = last_error
    return result
