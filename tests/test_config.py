"""Configuration loading: local .env defaults and Streamlit Secrets overlay."""

from __future__ import annotations

from config.settings import (
    OFFICIAL_APPLE_APP_ID,
    OFFICIAL_GOOGLE_PLAY_APP_ID,
    Settings,
    _streamlit_secret,
    get_settings,
    normalize_openrouter_api_key,
    normalize_openrouter_model,
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
    assert settings.ai_provider == "openrouter"


def test_normalize_adds_google_prefix_for_bare_gemini():
    assert normalize_openrouter_model("gemini-2.5-flash") == "google/gemini-2.5-flash"
    assert normalize_openrouter_model("google/gemini-2.5-flash") == "google/gemini-2.5-flash"


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
        assert settings.ai_provider == "openrouter"
    finally:
        settings_mod.get_settings.cache_clear()


def test_reload_settings_returns_settings():
    settings = reload_settings()
    assert settings.google_play_app_id == OFFICIAL_GOOGLE_PLAY_APP_ID
    assert get_settings() is settings


def test_get_ai_config_never_returns_key():
    from config.settings import get_ai_config

    cfg = get_ai_config()
    assert cfg["provider"] == "openrouter"
    assert cfg["provider_label"] == "OpenRouter"
    assert cfg["model"]
    assert "openrouter_api_key" not in cfg
    assert "api_key" not in cfg
    dumped = str(cfg)
    assert "openrouter_api_key" not in cfg
    assert "api_key" not in cfg
    assert "sk-or-v1-" not in dumped or cfg.get("key_prefix") in {
        "sk-or-v1-...",
        "sk-or-...",
        "none",
        "unrecognized",
        "gemini-style (invalid for OpenRouter)",
        "MISSING",
    }
    import re as _re

    assert not _re.search(r"sk-or-v1-[A-Za-z0-9]{8,}", dumped)
    assert "AIzaSy" not in dumped
    if not cfg["configured"]:
        assert "OPENROUTER_API_KEY" in cfg["missing_key_message"]
        assert "Gemini" not in cfg["missing_key_message"]


def test_normalize_strips_quotes_and_bearer():
    assert normalize_openrouter_api_key('  "sk-or-v1-abc"  ') == "sk-or-v1-abc"
    assert normalize_openrouter_api_key("Bearer sk-or-v1-abc") == "sk-or-v1-abc"
    assert normalize_openrouter_api_key("'sk-or-v1-abc'") == "sk-or-v1-abc"
    assert normalize_openrouter_api_key('OPENROUTER_API_KEY="sk-or-v1-abc"') == "sk-or-v1-abc"


def test_file_key_used_when_process_env_empty(monkeypatch):
    import config.settings as settings_mod

    monkeypatch.setattr(
        settings_mod,
        "_dotenv_file_values",
        lambda: {"OPENROUTER_API_KEY": "sk-or-file-test-key"},
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    settings_mod.get_settings.cache_clear()
    try:
        settings = settings_mod.get_settings()
        assert settings.openrouter_api_key == "sk-or-file-test-key"
        assert settings.has_ai_credentials is True
    finally:
        settings_mod.get_settings.cache_clear()
