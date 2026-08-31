"""Central configuration. Official Myntra IDs and URLs live here only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import logging

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


def normalize_gemini_model(value: str | None) -> str:
    """Gemini API model id. Strips leftover OpenRouter-style google/ prefixes."""
    text = (value or "").strip() or "gemini-2.5-flash"
    if text.lower().startswith("google/"):
        text = text.split("/", 1)[1]
    if text.lower().startswith("models/"):
        text = text.split("/", 1)[1]
    return text.strip() or "gemini-2.5-flash"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ai_provider: str = "gemini"
    ai_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

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
    expected_app_name: str = OFFICIAL_GOOGLE_PLAY_APP_NAME
    expected_apple_app_name: str = OFFICIAL_APPLE_APP_NAME

    ai_max_review_chars: int = 4000
    ai_rate_limit_seconds: float = 0.4
    ai_analysis_batch_size: int = 60
    ai_request_batch_size: int = 10
    ai_retry_attempts: int = 5

    host: str = "127.0.0.1"
    port: int = 8000

    @model_validator(mode="after")
    def _force_gemini_provider(self):
        """Leftover AI_PROVIDER/OpenRouter env vars must not switch the production path."""
        self.ai_provider = "gemini"
        if not (self.gemini_model or "").strip():
            self.gemini_model = "gemini-2.5-flash"
        if not (self.ai_model or "").strip():
            self.ai_model = "gemini-2.5-flash"
        return self

    @property
    def resolved_model(self) -> str:
        return normalize_gemini_model(self.gemini_model or self.ai_model or "gemini-2.5-flash")

    @property
    def has_ai_credentials(self) -> bool:
        return bool((self.gemini_api_key or "").strip())

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
        if name not in st.secrets:
            return None
        value = st.secrets.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Streamlit secrets are not available in this process")
        return None


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    secret_key = _streamlit_secret("GEMINI_API_KEY")
    secret_model = _streamlit_secret("GEMINI_MODEL")
    if secret_key:
        settings.gemini_api_key = secret_key
    if secret_model:
        settings.gemini_model = secret_model
        settings.ai_model = secret_model
    settings.ai_provider = "gemini"
    return settings


def reload_settings() -> Settings:
    get_settings.cache_clear()
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
