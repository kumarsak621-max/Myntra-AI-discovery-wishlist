"""Configuration loading: local .env defaults and Streamlit Secrets overlay."""

from __future__ import annotations

from config.settings import (
    OFFICIAL_APPLE_APP_ID,
    OFFICIAL_GOOGLE_PLAY_APP_ID,
    Settings,
    _streamlit_secret,
    get_settings,
    reload_settings,
)


def test_official_myntra_ids_are_defaults():
    settings = Settings()
    assert settings.google_play_app_id == "com.myntra.android" == OFFICIAL_GOOGLE_PLAY_APP_ID
    assert settings.apple_app_id == "907394059" == OFFICIAL_APPLE_APP_ID
    assert settings.apple_primary_region == "in"
    assert settings.apple_fallback_region == "us"


def test_default_openrouter_model_without_key():
    settings = Settings(openrouter_api_key="", openrouter_model="")
    assert settings.resolved_model == "google/gemini-2.5-flash"
    assert settings.has_ai_credentials is False


def test_streamlit_secret_is_none_outside_streamlit():
    assert _streamlit_secret("OPENROUTER_API_KEY") is None
    assert _streamlit_secret("OPENROUTER_MODEL") is None


def test_streamlit_secrets_overlay(monkeypatch):
    import config.settings as settings_mod

    def fake_secret(name: str):
        mapping = {
            "OPENROUTER_API_KEY": "unit-test-openrouter-key",
            "OPENROUTER_MODEL": "google/gemini-2.5-flash",
        }
        return mapping.get(name)

    monkeypatch.setattr(settings_mod, "_streamlit_secret", fake_secret)
    settings_mod.get_settings.cache_clear()
    try:
        settings = settings_mod.get_settings()
        assert settings.openrouter_api_key == "unit-test-openrouter-key"
        assert settings.resolved_model == "google/gemini-2.5-flash"
        assert settings.has_ai_credentials is True
    finally:
        settings_mod.get_settings.cache_clear()


def test_reload_settings_returns_settings():
    settings = reload_settings()
    assert settings.google_play_app_id == OFFICIAL_GOOGLE_PLAY_APP_ID
    assert get_settings() is settings
