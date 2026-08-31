"""Compatibility wrapper. Canonical settings live in config.settings."""

from config.settings import (
    BANNED_APP_IDS,
    OFFICIAL_APPLE_APP_ID,
    OFFICIAL_APPLE_APP_NAME,
    OFFICIAL_APPLE_APP_URL,
    OFFICIAL_GOOGLE_PLAY_APP_ID,
    OFFICIAL_GOOGLE_PLAY_APP_NAME,
    OFFICIAL_GOOGLE_PLAY_URL,
    ROOT_DIR,
    Settings,
    get_ai_config,
    get_settings,
    is_banned_app_id,
    is_official_myntra_app_id,
    official_ids,
    reload_settings,
)

__all__ = [
    "BANNED_APP_IDS",
    "OFFICIAL_APPLE_APP_ID",
    "OFFICIAL_APPLE_APP_NAME",
    "OFFICIAL_APPLE_APP_URL",
    "OFFICIAL_GOOGLE_PLAY_APP_ID",
    "OFFICIAL_GOOGLE_PLAY_APP_NAME",
    "OFFICIAL_GOOGLE_PLAY_URL",
    "ROOT_DIR",
    "Settings",
    "get_ai_config",
    "get_settings",
    "is_banned_app_id",
    "is_official_myntra_app_id",
    "official_ids",
    "reload_settings",
]
