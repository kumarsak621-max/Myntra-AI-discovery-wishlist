"""Google Gemini AI gateway. Keys and model names come only from Settings."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.config import Settings, get_settings
from config.settings import normalize_gemini_model

logger = logging.getLogger(__name__)

QUOTA_MESSAGE = "Gemini API quota/rate limit reached. Please try again later."
MISSING_KEY_MESSAGE = "Gemini API key is not configured."


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


def _error_status(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    return None


def _is_quota_error(message: str, status: int | None) -> bool:
    if status == 429:
        return True
    lowered = (message or "").lower()
    return any(
        token in lowered
        for token in (
            "resource_exhausted",
            "resource exhausted",
            "quota",
            "rate limit",
            "too many requests",
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


class GeminiAIService:
    """Production AI path: official Google Gemini API only."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return normalize_gemini_model(self.settings.resolved_model)

    def available(self) -> bool:
        return bool((self.settings.gemini_api_key or "").strip())

    def _client(self):
        from google import genai

        return genai.Client(api_key=self.settings.gemini_api_key)

    def complete_json(self, *, system: str, user: str) -> str:
        if not self.available():
            raise AIError(MISSING_KEY_MESSAGE)
        from google.genai import types

        last_error: AIError | None = None
        attempts = max(1, int(self.settings.ai_retry_attempts or 5))
        for attempt in range(1, attempts + 1):
            try:
                response = self._client().models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                text = (getattr(response, "text", None) or "").strip()
                if not text:
                    raise AIError("Gemini returned empty content.")
                return text
            except AIError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                time.sleep(_backoff_seconds(attempt))
            except Exception as exc:
                mapped = _map_gemini_exception(exc)
                if not mapped.retryable:
                    raise mapped
                last_error = mapped
                logger.warning("Gemini call failed attempt %s: %s", attempt, mapped)
                time.sleep(_backoff_seconds(attempt))
        raise last_error or AIError("Gemini request failed")


AIProvider = GeminiAIService


def _map_gemini_exception(exc: BaseException) -> AIError:
    status = _error_status(exc)
    message = _redact(str(exc) or exc.__class__.__name__)
    if _is_quota_error(message, status):
        return AIError(QUOTA_MESSAGE, retryable=True, http_status=status or 429)
    lowered = message.lower()
    if status in {401, 403} or "api key" in lowered or "unauthenticated" in lowered or "permission" in lowered:
        return AIError(
            "Gemini rejected the API key. Check GEMINI_API_KEY in Streamlit Secrets or .env.",
            http_status=status or 403,
        )
    if status in {500, 502, 503, 504} or "unavailable" in lowered or "internal" in lowered:
        return AIError(
            f"Gemini provider error HTTP {status or '5xx'}. Retrying.",
            retryable=True,
            http_status=status,
        )
    if "timeout" in lowered or "timed out" in lowered or "deadline" in lowered:
        return AIError("The Gemini request timed out. Try fewer reviews, then retry.", retryable=True)
    if "connect" in lowered or "network" in lowered:
        return AIError("Network error contacting Gemini.", retryable=True)
    return AIError(message[:400], http_status=status)


def test_gemini_connection(settings: Settings | None = None) -> dict[str, Any]:
    """Minimal live Gemini request. Never logs or returns the API key."""
    from config.settings import reload_settings

    settings = settings or reload_settings()
    model = normalize_gemini_model(settings.gemini_model or settings.resolved_model)
    configured = bool((settings.gemini_api_key or "").strip())
    result: dict[str, Any] = {
        "provider": "Google Gemini",
        "model": model,
        "credentials": "Configured" if configured else "Missing",
        "ok": False,
        "status": "FAILED",
        "http_status": None,
        "error": None,
    }
    if not configured:
        result["error"] = MISSING_KEY_MESSAGE
        return result

    from google.genai import types

    attempts = min(3, max(1, int(settings.ai_retry_attempts or 3)))
    last_error = "Gemini connection test failed."
    for attempt in range(1, attempts + 1):
        try:
            from google import genai

            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=model,
                contents="Reply with the single word ok.",
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=16),
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                result["error"] = "Gemini returned empty content."
                return result
            result["ok"] = True
            result["status"] = "SUCCESS"
            result["http_status"] = 200
            result["error"] = None
            return result
        except Exception as exc:
            mapped = _map_gemini_exception(exc)
            last_error = str(mapped)
            result["http_status"] = mapped.http_status
            if mapped.retryable and attempt < attempts:
                time.sleep(_backoff_seconds(attempt))
                continue
            result["error"] = last_error
            return result
    result["error"] = last_error
    return result
