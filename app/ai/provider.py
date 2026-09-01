"""OpenRouter AI gateway. Keys and model names come only from Settings."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from app.config import Settings, get_settings
from config.settings import (
    clamp_max_tokens,
    normalize_openrouter_api_key,
    normalize_openrouter_model,
    resolve_openrouter_credentials,
)

logger = logging.getLogger(__name__)

QUOTA_MESSAGE = "OpenRouter rate limit reached. Please try again later."
CREDIT_402_MESSAGE = (
    "OpenRouter token/credit limit reached. Reduce max_tokens or add credits."
)
MISSING_KEY_MESSAGE = (
    "OpenRouter API key is not configured. "
    "Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
)
AUTH_401_MESSAGE = (
    "OpenRouter authentication failed (HTTP 401). "
    "The key was found, but OpenRouter rejected the credential. "
    "Create/check the OpenRouter API key in Streamlit Secrets."
)
GEMINI_KEY_MESSAGE = (
    "OPENROUTER_API_KEY looks like a Google Gemini key, not an OpenRouter key. "
    "Use an OpenRouter key that starts with sk-or- in Streamlit Secrets."
)
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


def _redact(text: str) -> str:
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


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    raw = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _supports_json_object(model: str) -> bool:
    lowered = (model or "").lower()
    return "gemini" not in lowered


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


class OpenRouterAIService:
    """Production AI path: OpenRouter chat completions only."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.last_usage: dict[str, int] = {}

    def _max_output_tokens(self) -> int:
        return clamp_max_tokens(getattr(self.settings, "ai_max_tokens", 2000))

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model(self) -> str:
        return normalize_openrouter_model(self.settings.resolved_model)

    @property
    def _endpoint(self) -> str:
        base = (self.settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        if "generativelanguage.googleapis.com" in base:
            return OPENROUTER_CHAT_URL
        return f"{base}/chat/completions"

    def _api_key(self) -> str:
        live = resolve_openrouter_credentials()
        if live.get("source") == "Streamlit Secrets" and live.get("key"):
            key = live["key"]
        else:
            key = normalize_openrouter_api_key(self.settings.openrouter_api_key) or live.get("key") or ""
        if key.startswith("AIza"):
            raise AIError(GEMINI_KEY_MESSAGE)
        return key

    def available(self) -> bool:
        try:
            return bool(self._api_key())
        except AIError:
            return False

    def close(self) -> None:
        """No persistent HTTP client; present for pipeline finally-blocks."""
        return None

    def _headers(self) -> dict[str, str]:
        key = self._api_key()
        if not key:
            raise AIError(MISSING_KEY_MESSAGE)
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kumarsak621-max/Myntra-AI-discovery-wishlist",
            "X-Title": "Myntra AI Discovery Engine",
        }

    def _timeout(self) -> httpx.Timeout:
        seconds = float(getattr(self.settings, "ai_http_timeout_seconds", 60) or 60)
        return httpx.Timeout(seconds, connect=min(15.0, seconds))

    def complete_json(self, *, system: str, user: str) -> str:
        if not self._api_key():
            raise AIError(MISSING_KEY_MESSAGE)
        last_error: AIError | None = None
        attempts = max(1, int(self.settings.ai_retry_attempts or 5))
        use_json_object = _supports_json_object(self.model)
        for attempt in range(1, attempts + 1):
            try:
                text = self._post_chat(
                    system=system,
                    user=user,
                    json_object=use_json_object,
                )
                if not text:
                    raise AIError("OpenRouter returned empty content.")
                return text
            except AIError as exc:
                if (
                    exc.http_status == 400
                    and use_json_object
                    and "json" in str(exc).lower()
                ):
                    use_json_object = False
                    last_error = exc
                    time.sleep(_backoff_seconds(attempt))
                    continue
                if not exc.retryable:
                    raise
                last_error = exc
                time.sleep(_backoff_seconds(attempt))
            except Exception as exc:
                mapped = _map_openrouter_exception(exc)
                if not mapped.retryable:
                    raise mapped
                last_error = mapped
                logger.warning("OpenRouter call failed attempt %s: %s", attempt, mapped)
                time.sleep(_backoff_seconds(attempt))
        raise last_error or AIError("OpenRouter request failed")

    def _post_chat(self, *, system: str, user: str, json_object: bool) -> str:
        max_tokens = self._max_output_tokens()
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(timeout=self._timeout()) as client:
                response = client.post(self._endpoint, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise AIError("The OpenRouter request timed out. Try fewer reviews, then retry.", retryable=True) from exc
        except httpx.ConnectError as exc:
            raise AIError("Network error contacting OpenRouter.", retryable=True) from exc
        text, usage = _content_from_response(response)
        self.last_usage = usage
        return text


AIProvider = OpenRouterAIService


def _message_content(message: Any) -> str:
    """Read OpenRouter chat content from string or multimodal list parts."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        joined = "".join(parts).strip()
        if joined:
            return joined
    return ""


def _openrouter_error_text(response: httpx.Response) -> str:
    body = _redact(response.text or "")
    try:
        payload = response.json()
    except ValueError:
        return body[:300]
    if not isinstance(payload, dict):
        return body[:300]
    err = payload.get("error")
    if isinstance(err, dict):
        return _redact(str(err.get("message") or err.get("code") or err))[:300]
    if isinstance(err, str):
        return _redact(err)[:300]
    return body[:300]


def _content_from_response(response: httpx.Response) -> tuple[str, dict[str, int]]:
    status = response.status_code
    snippet = _openrouter_error_text(response)
    if status == 401:
        raise AIError(AUTH_401_MESSAGE, http_status=401)
    if status == 403:
        raise AIError(
            "OpenRouter authentication failed (HTTP 403). "
            "The key was found, but OpenRouter rejected access to this resource.",
            http_status=403,
        )
    if status == 404:
        raise AIError(
            f"Configured model/endpoint unavailable (HTTP 404). "
            f"{snippet or 'Set OPENROUTER_MODEL to a model your OpenRouter account can use.'}",
            http_status=404,
        )
    if status == 429:
        raise AIError(QUOTA_MESSAGE, retryable=True, http_status=429)
    if status == 402:
        raise AIError(CREDIT_402_MESSAGE, http_status=402)
    if status in {500, 502, 503, 504}:
        raise AIError(
            f"OpenRouter/provider server error (HTTP {status}). Retrying.",
            retryable=True,
            http_status=status,
        )
    if status == 400:
        raise AIError(
            f"Invalid OpenRouter request (HTTP 400). {snippet or 'request failed'}",
            http_status=400,
        )
    if status >= 400:
        retryable = status >= 500
        raise AIError(
            f"OpenRouter analysis failed: HTTP {status}. {snippet or 'request failed'}",
            retryable=retryable,
            http_status=status,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise AIError("OpenRouter returned invalid JSON.") from exc
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise AIError("OpenRouter returned no choices.")
    message = (choices[0] or {}).get("message") or {}
    refusal = message.get("refusal") if isinstance(message, dict) else None
    if refusal:
        raise AIError(f"OpenRouter analysis failed: model refused the request. {_redact(str(refusal))[:200]}")
    text = _message_content(message)
    if not text:
        raise AIError("OpenRouter returned empty content.")
    usage_raw = payload.get("usage") if isinstance(payload, dict) else None
    usage = {}
    if isinstance(usage_raw, dict):
        try:
            usage = {
                "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
                "total_tokens": int(usage_raw.get("total_tokens") or 0),
            }
        except (TypeError, ValueError):
            usage = {}
    return text, usage


def _map_openrouter_exception(exc: BaseException) -> AIError:
    if isinstance(exc, AIError):
        return exc
    message = _redact(str(exc) or exc.__class__.__name__)
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return AIError("The OpenRouter request timed out. Try fewer reviews, then retry.", retryable=True)
    if "connect" in lowered or "network" in lowered:
        return AIError("Network error contacting OpenRouter.", retryable=True)
    return AIError(message[:400])


def test_openrouter_connection(settings: Settings | None = None) -> dict[str, Any]:
    """Minimal live OpenRouter request. Never logs or returns the API key."""
    from config.settings import reload_settings

    settings = settings or reload_settings()
    creds = resolve_openrouter_credentials()
    key = creds["key"] or normalize_openrouter_api_key(settings.openrouter_api_key)
    model = normalize_openrouter_model(settings.openrouter_model or settings.resolved_model)
    configured = bool(key)
    result: dict[str, Any] = {
        "provider": "OpenRouter",
        "model": model,
        "credentials": "Configured" if configured else "Missing",
        "secret_source": creds["source"] if creds["key"] else ("Environment" if configured else "Missing"),
        "key_format": creds["key_format"] if creds["key"] else (
            "VALID PREFIX" if key.startswith("sk-or-") else ("INVALID PREFIX" if key else "MISSING")
        ),
        "key_prefix": creds["key_prefix"] if creds["key"] else ("sk-or-v1-..." if key.startswith("sk-or-v1-") else "none"),
        "ok": False,
        "status": "FAILED",
        "http_status": None,
        "error": None,
        "endpoint": OPENROUTER_CHAT_URL,
    }
    if not configured:
        result["error"] = MISSING_KEY_MESSAGE
        return result
    if key.startswith("AIza"):
        result["error"] = GEMINI_KEY_MESSAGE
        result["key_format"] = "INVALID PREFIX"
        result["key_prefix"] = "gemini-style (invalid for OpenRouter)"
        return result

    service = OpenRouterAIService(settings)
    attempts = min(3, max(1, int(settings.ai_retry_attempts or 3)))
    last_error = "OpenRouter connection test failed."
    for attempt in range(1, attempts + 1):
        try:
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": 16,
                "messages": [
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Reply with the word OK."},
                ],
            }
            with httpx.Client(timeout=service._timeout()) as client:
                response = client.post(service._endpoint, headers=service._headers(), json=payload)
            result["http_status"] = response.status_code
            text, _usage = _content_from_response(response)
            if not text:
                result["error"] = "OpenRouter returned empty content."
                return result
            result["ok"] = True
            result["status"] = "SUCCESS"
            result["error"] = None
            return result
        except Exception as exc:
            mapped = _map_openrouter_exception(exc)
            last_error = str(mapped)
            result["http_status"] = mapped.http_status
            if mapped.retryable and attempt < attempts:
                time.sleep(_backoff_seconds(attempt, None))
                continue
            result["error"] = last_error
            return result
    result["error"] = last_error
    return result
