"""AI provider gateway. Keys and model names come only from Settings."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


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
            "HTTP-Referer": "http://localhost:8000",
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
            f"{model}:generateContent?key={self.settings.gemini_api_key}"
        )
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
                    response = client.post(url, json=body)
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
                        last_error = AIError(f"Rate limited: {response.text[:300]}")
                        continue
                    if response.status_code >= 400:
                        raise AIError(
                            f"AI HTTP {response.status_code}: {response.text[:500]}"
                        )
                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise AIError(f"AI returned no choices: {data}")
                    content = (choices[0].get("message") or {}).get("content") or ""
                    if not content:
                        raise AIError("AI returned empty content")
                    return content
            except (httpx.HTTPError, AIError) as exc:
                last_error = exc
                logger.warning("AI call failed attempt %s: %s", attempt, exc)
                time.sleep(2 ** (attempt - 1))
        raise AIError(str(last_error) if last_error else "AI request failed")
