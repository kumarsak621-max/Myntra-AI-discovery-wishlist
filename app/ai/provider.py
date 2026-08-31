"""AI provider gateway. Keys and model names come only from Settings."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _client_http_message(status_code: int, body: str) -> str:
    snippet = _redact((body or "")[:300])
    if status_code in {401, 403}:
        return (
            "OpenRouter rejected the API key (HTTP "
            f"{status_code}). Check OPENROUTER_API_KEY in Streamlit Secrets or .env."
        )
    if status_code == 400:
        return f"OpenRouter rejected the request (HTTP 400). {snippet}"
    return f"AI HTTP {status_code}: {snippet}"


def _redact(text: str) -> str:
    import re

    cleaned = text or ""
    cleaned = re.sub(r"sk-or-[A-Za-z0-9_-]+", "[redacted]", cleaned)
    cleaned = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", cleaned)
    cleaned = re.sub(r"AIza[0-9A-Za-z_-]+", "[redacted]", cleaned)
    return cleaned


class AIError(RuntimeError):
    pass


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
                f"No credentials configured for AI_PROVIDER={self.provider_name}. "
                "Set OPENROUTER_API_KEY (or GEMINI_API_KEY)."
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
            "response_format": {"type": "json_object"},
        }
        return self._post_chat(url, headers, body)

    def _gemini(self, system: str, user: str) -> str:
        # OpenAI-compatible Gemini API via Google's OpenAI endpoint, or generateContent.
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
        last_error = None
        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=90.0) as client:
                    response = client.post(url, headers=headers, json=body)
                    if response.status_code == 429:
                        time.sleep(2 ** attempt)
                        last_error = AIError(f"Gemini rate limited: {response.text[:300]}")
                        continue
                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates") or []
                    if not candidates:
                        raise AIError(f"Gemini returned no candidates: {data}")
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    text = "".join(p.get("text") or "" for p in parts)
                    if not text:
                        raise AIError("Gemini returned empty text")
                    return text
            except httpx.HTTPError as exc:
                last_error = AIError(str(exc))
                time.sleep(2 ** (attempt - 1))
        raise last_error or AIError("Gemini request failed")

    def _post_chat(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=90.0) as client:
                    response = client.post(url, headers=headers, json=body)
                    if response.status_code == 429:
                        wait = float(response.headers.get("Retry-After") or 2 ** attempt)
                        time.sleep(min(wait, 20))
                        last_error = AIError(
                            "The AI provider rate-limited this request. Wait and retry."
                        )
                        continue
                    if 400 <= response.status_code < 500:
                        raise AIError(_client_http_message(response.status_code, response.text))
                    if response.status_code >= 500:
                        last_error = AIError(
                            f"AI provider error HTTP {response.status_code}. Retrying."
                        )
                        time.sleep(2 ** (attempt - 1))
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
            except AIError:
                raise
            except httpx.TimeoutException:
                last_error = AIError("The AI request timed out. Try fewer reviews, then retry.")
                logger.warning("AI timeout attempt %s", attempt)
                time.sleep(2 ** (attempt - 1))
            except httpx.HTTPError as exc:
                last_error = AIError("Network error contacting the AI provider.")
                logger.warning("AI call failed attempt %s: %s", attempt, _redact(str(exc)))
                time.sleep(2 ** (attempt - 1))
        raise AIError(str(last_error) if last_error else "AI request failed")
