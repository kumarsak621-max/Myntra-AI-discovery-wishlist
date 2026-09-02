"""Central configuration. Official Myntra IDs and URLs live here only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import logging
import os
import re

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

# Official Myntra identities — single source of truth.
OFFICIAL_GOOGLE_PLAY_APP_ID = "com.myntra.android"
OFFICIAL_GOOGLE_PLAY_URL = "https://play.google.com/store/apps/details?id=com.myntra.android"
OFFICIAL_GOOGLE_PLAY_APP_NAME = "Myntra"

OFFICIAL_APPLE_APP_ID = "907394059"
OFFICIAL_APPLE_APP_URL = (
    "https://apps.apple.com/in/app/myntra-fashion-shopping-app/id907394059"
)
OFFICIAL_APPLE_APP_NAME = "Myntra Fashion Shopping App"

# Must never be used for Myntra collection.
BANNED_APP_IDS = frozenset({"com.grofers.customerapp", "960335206"})

DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"


def normalize_openrouter_model(value: str | None) -> str:
    """OpenRouter model id. Bare gemini-* ids get a google/ prefix."""
    text = (value or "").strip() or DEFAULT_OPENROUTER_MODEL
    if text.lower().startswith("models/"):
        text = text.split("/", 1)[1]
    if "/" not in text and text.lower().startswith("gemini"):
        text = f"google/{text}"
    return text.strip() or DEFAULT_OPENROUTER_MODEL


def normalize_openrouter_api_key(value: str | None) -> str:
    """Strip whitespace, wrapping quotes, and accidental Bearer prefixes. Never log the result."""
    text = str(value or "")
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\r"):
        text = text.replace(ch, "")
    text = text.strip()
    if text.lower().startswith("openrouter_api_key="):
        text = text.split("=", 1)[1].strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        text = text[1:-1].strip()
    if text.lower().startswith("bearer "):
        text = text[7:].strip()
        while len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
            text = text[1:-1].strip()
    text = re.sub(r"\s+", "", text)
    return text


def clamp_max_tokens(value) -> int:
    """Output-token cap for OpenRouter. Never allow the 65535 model default."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 2000
    return max(64, min(2000, number))


def clamp_batch_size(value) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 10
    return max(1, min(10, number))


DEFAULT_MAX_DATASET_REVIEWS = 300


def clamp_max_dataset_reviews(value) -> int:
    """Active real-review cap. 300 is a maximum, not a target to fabricate toward."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_MAX_DATASET_REVIEWS
    return max(1, min(10_000, number))


def openrouter_key_prefix_status(key: str) -> str:
    if not key:
        return "MISSING"
    if key.startswith("sk-or-"):
        return "VALID PREFIX"
    return "INVALID PREFIX"


def openrouter_key_prefix_mask(key: str) -> str:
    if not key:
        return "none"
    if key.startswith("sk-or-v1-"):
        return "sk-or-v1-..."
    if key.startswith("sk-or-"):
        return "sk-or-..."
    if key.startswith("AIza"):
        return "gemini-style (invalid for OpenRouter)"
    return "unrecognized"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ai_provider: str = "openrouter"
    ai_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ai_http_timeout_seconds: float = 60.0

    google_play_app_id: str = OFFICIAL_GOOGLE_PLAY_APP_ID
    google_play_max_reviews: int = 5000
    google_play_batch_size: int = 100
    google_play_language: str = "en"
    google_play_country: str = "in"

    apple_app_id: str = OFFICIAL_APPLE_APP_ID
    apple_primary_region: str = "in"
    apple_fallback_region: str = "us"
    apple_max_reviews: int = 5000

    database_url: str = "sqlite:///./myntra_discovery.db"

    collection_rate_limit_seconds: float = 1.0
    collection_retry_attempts: int = 3
    collection_window_days: int = 30
    google_play_window_safety_limit: int = 800
    apple_window_safety_limit: int = 500
    refresh_safety_limit: int = 150
    max_dataset_reviews: int = DEFAULT_MAX_DATASET_REVIEWS
    max_analysis_reviews: int = DEFAULT_MAX_DATASET_REVIEWS
    prune_excess_reviews: bool = False
    expected_app_name: str = OFFICIAL_GOOGLE_PLAY_APP_NAME
    expected_apple_app_name: str = OFFICIAL_APPLE_APP_NAME

    ai_max_review_chars: int = 4000
    ai_rate_limit_seconds: float = 0.4
    ai_analysis_batch_size: int = 300
    ai_request_batch_size: int = 10
    ai_batch_size: int | None = None
    analysis_batch_size: int | None = None
    ai_max_tokens: int = 2000
    max_output_tokens: int | None = None
    ai_retry_attempts: int = 5

    host: str = "127.0.0.1"
    port: int = 8000

    @model_validator(mode="after")
    def _force_openrouter_provider(self):
        """Leftover Gemini env vars must not switch the production path."""
        self.ai_provider = "openrouter"
        if not (self.openrouter_model or "").strip():
            self.openrouter_model = DEFAULT_OPENROUTER_MODEL
        if not (self.ai_model or "").strip():
            self.ai_model = DEFAULT_OPENROUTER_MODEL
        token_cap = self.max_output_tokens if self.max_output_tokens is not None else self.ai_max_tokens
        self.ai_max_tokens = clamp_max_tokens(token_cap)
        batch = (
            self.analysis_batch_size
            if self.analysis_batch_size is not None
            else (self.ai_batch_size if self.ai_batch_size is not None else self.ai_request_batch_size)
        )
        self.ai_request_batch_size = clamp_batch_size(batch)
        self.ai_batch_size = self.ai_request_batch_size
        self.analysis_batch_size = self.ai_request_batch_size
        self.max_dataset_reviews = clamp_max_dataset_reviews(self.max_dataset_reviews)
        self.max_analysis_reviews = clamp_max_dataset_reviews(
            self.max_analysis_reviews or self.max_dataset_reviews
        )
        return self

    @property
    def resolved_model(self) -> str:
        return normalize_openrouter_model(
            self.openrouter_model or self.ai_model or DEFAULT_OPENROUTER_MODEL
        )

    @property
    def has_ai_credentials(self) -> bool:
        return bool(normalize_openrouter_api_key(self.openrouter_api_key))

    @property
    def google_play_url(self) -> str:
        return f"https://play.google.com/store/apps/details?id={self.google_play_app_id}"

    @property
    def apple_app_store_url(self) -> str:
        if self.apple_app_id == OFFICIAL_APPLE_APP_ID:
            return OFFICIAL_APPLE_APP_URL
        return (
            f"https://apps.apple.com/{self.apple_primary_region}/app/id{self.apple_app_id}"
        )


def _streamlit_secret(name: str) -> str | None:
    """Read a secret from Streamlit Cloud when running inside Streamlit."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            return None
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
        if value is None:
            return None
        text = normalize_openrouter_api_key(str(value)) if name.endswith("API_KEY") else str(value).strip()
        return text or None
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Streamlit secrets are not available in this process")
        return None


def resolve_openrouter_credentials() -> dict:
    """Fresh read: Streamlit Secrets, then environment/.env. Includes the key for HTTP only."""
    source = "Missing"
    raw = ""
    secret = _streamlit_secret("OPENROUTER_API_KEY")
    if secret:
        source = "Streamlit Secrets"
        raw = secret
    else:
        env_raw = os.getenv("OPENROUTER_API_KEY")
        if normalize_openrouter_api_key(env_raw):
            source = "Environment"
            raw = env_raw or ""
        else:
            file_key = _dotenv_file_values().get("OPENROUTER_API_KEY") or ""
            if normalize_openrouter_api_key(file_key):
                source = "Environment"
                raw = file_key
    key = normalize_openrouter_api_key(raw)
    return {
        "key": key,
        "source": source if key else "Missing",
        "configured": bool(key),
        "key_format": openrouter_key_prefix_status(key),
        "key_prefix": openrouter_key_prefix_mask(key),
        "looks_like_gemini": key.startswith("AIza"),
    }


def _dotenv_file_values() -> dict[str, str]:
    """Read .env from disk. Used when process env still has a stale empty key."""
    from dotenv import dotenv_values

    path = ROOT_DIR / ".env"
    if not path.exists():
        return {}
    raw = dotenv_values(path)
    return {key: (value or "").strip() for key, value in raw.items() if key}


@lru_cache
def _env_settings() -> Settings:
    settings = Settings()
    file_vals = _dotenv_file_values()
    if not normalize_openrouter_api_key(settings.openrouter_api_key):
        file_key = normalize_openrouter_api_key(file_vals.get("OPENROUTER_API_KEY"))
        if file_key:
            settings.openrouter_api_key = file_key
    if not (settings.openrouter_model or "").strip():
        file_model = file_vals.get("OPENROUTER_MODEL") or ""
        if file_model:
            settings.openrouter_model = file_model
            settings.ai_model = file_model
    settings.openrouter_api_key = normalize_openrouter_api_key(settings.openrouter_api_key)
    settings.ai_provider = "openrouter"
    return settings


def get_settings() -> Settings:
    """Env/file base settings, then a fresh Streamlit Secrets overlay for the API key."""
    settings = _env_settings()
    creds = resolve_openrouter_credentials()
    if creds["key"]:
        settings.openrouter_api_key = creds["key"]
    else:
        settings.openrouter_api_key = normalize_openrouter_api_key(settings.openrouter_api_key)
    secret_model = _streamlit_secret("OPENROUTER_MODEL")
    if secret_model:
        settings.openrouter_model = secret_model
        settings.ai_model = secret_model
    secret_tokens = _streamlit_secret("AI_MAX_TOKENS") or _streamlit_secret("MAX_OUTPUT_TOKENS")
    if secret_tokens:
        settings.ai_max_tokens = clamp_max_tokens(secret_tokens)
    else:
        settings.ai_max_tokens = clamp_max_tokens(settings.ai_max_tokens)
    secret_batch = _streamlit_secret("ANALYSIS_BATCH_SIZE") or _streamlit_secret("AI_BATCH_SIZE")
    if secret_batch:
        settings.ai_batch_size = clamp_batch_size(secret_batch)
        settings.ai_request_batch_size = settings.ai_batch_size
        settings.analysis_batch_size = settings.ai_batch_size
    secret_limit = _streamlit_secret("MAX_DATASET_REVIEWS")
    if secret_limit:
        settings.max_dataset_reviews = clamp_max_dataset_reviews(secret_limit)
    else:
        settings.max_dataset_reviews = clamp_max_dataset_reviews(settings.max_dataset_reviews)
    secret_analysis = _streamlit_secret("MAX_ANALYSIS_REVIEWS")
    if secret_analysis:
        settings.max_analysis_reviews = clamp_max_dataset_reviews(secret_analysis)
    else:
        settings.max_analysis_reviews = clamp_max_dataset_reviews(
            settings.max_analysis_reviews or settings.max_dataset_reviews
        )
    settings.ai_provider = "openrouter"
    return settings


get_settings.cache_clear = _env_settings.cache_clear  # type: ignore[method-assign]


def get_ai_config() -> dict:
    """Safe diagnostics for the UI. Never includes the API key."""
    reload_settings()
    settings = get_settings()
    creds = resolve_openrouter_credentials()
    if not creds["configured"]:
        creds = {
            **creds,
            "configured": bool(settings.openrouter_api_key),
            "source": creds["source"] if not settings.openrouter_api_key else "Environment",
            "key_format": openrouter_key_prefix_status(settings.openrouter_api_key),
            "key_prefix": openrouter_key_prefix_mask(settings.openrouter_api_key),
        }
    return {
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "model": settings.resolved_model,
        "configured": bool(creds.get("configured") or settings.openrouter_api_key),
        "secret_source": creds.get("source") or "Missing",
        "key_format": creds.get("key_format") or "MISSING",
        "key_prefix": creds.get("key_prefix") or "none",
        "max_tokens": clamp_max_tokens(settings.ai_max_tokens),
        "batch_size": clamp_batch_size(
            settings.analysis_batch_size
            or settings.ai_batch_size
            or settings.ai_request_batch_size
            or 10
        ),
        "max_dataset_reviews": clamp_max_dataset_reviews(settings.max_dataset_reviews),
        "max_analysis_reviews": clamp_max_dataset_reviews(
            settings.max_analysis_reviews or settings.max_dataset_reviews
        ),
        "prune_excess_reviews": bool(settings.prune_excess_reviews),
        "missing_key_message": (
            "OpenRouter API key is not configured. "
            "Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
        ),
    }


def reload_settings() -> Settings:
    _env_settings.cache_clear()
    return get_settings()


def is_banned_app_id(app_id: str) -> bool:
    return (app_id or "").strip() in BANNED_APP_IDS


def is_official_myntra_app_id(app_id: str, platform: str = "") -> bool:
    value = (app_id or "").strip()
    if platform == "google_play":
        return value == OFFICIAL_GOOGLE_PLAY_APP_ID
    if platform == "apple_app_store":
        return value == OFFICIAL_APPLE_APP_ID
    return value in {OFFICIAL_GOOGLE_PLAY_APP_ID, OFFICIAL_APPLE_APP_ID}


def official_ids() -> frozenset[str]:
    return frozenset({OFFICIAL_GOOGLE_PLAY_APP_ID, OFFICIAL_APPLE_APP_ID})
