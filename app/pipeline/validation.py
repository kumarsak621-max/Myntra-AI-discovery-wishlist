"""Source identity validation for official Myntra store listings."""

from __future__ import annotations

from datetime import datetime, timezone

from config.settings import (
    OFFICIAL_APPLE_APP_ID,
    OFFICIAL_APPLE_APP_NAME,
    OFFICIAL_APPLE_APP_URL,
    OFFICIAL_GOOGLE_PLAY_APP_ID,
    OFFICIAL_GOOGLE_PLAY_APP_NAME,
    OFFICIAL_GOOGLE_PLAY_URL,
    is_banned_app_id,
    is_official_myntra_app_id,
)
from app.schemas import SourceValidation

MYNTRA_TOKENS = ("myntra",)
KNOWN_NON_MYNTRA = {
    "blinkit": "Blinkit",
    "grofers": "Grofers",
    "blink commerce": "Blinkit",
}

WARNING_NOT_MYNTRA = (
    "WARNING: Configured Google Play app does not appear to be Myntra. "
    "Data must not be presented as Myntra evidence."
)

GENERIC_NOT_MYNTRA = (
    "WARNING: Detected app does not appear to be Myntra. "
    "Data must not be presented as Myntra evidence. "
    "Treat as REFERENCE / NON-MYNTRA DATA."
)

BANNED_ID_MESSAGE = (
    "ERROR: This identifier is banned for Myntra collection "
    "(Blinkit/Grofers). Use com.myntra.android / 907394059."
)


def _haystack(*parts: str) -> str:
    return " ".join(p or "" for p in parts).lower()


def looks_like_myntra(app_name: str = "", developer: str = "", app_id: str = "") -> bool:
    blob = _haystack(app_name, developer, app_id)
    return any(token in blob for token in MYNTRA_TOKENS)


def detected_non_myntra_brand(app_name: str = "", developer: str = "", app_id: str = "") -> str | None:
    blob = _haystack(app_name, developer, app_id)
    for token, brand in KNOWN_NON_MYNTRA.items():
        if token in blob:
            return brand
    return None


def expected_for(platform: str) -> tuple[str, str, str]:
    if platform == "apple_app_store":
        return OFFICIAL_APPLE_APP_ID, OFFICIAL_APPLE_APP_NAME, OFFICIAL_APPLE_APP_URL
    return OFFICIAL_GOOGLE_PLAY_APP_ID, OFFICIAL_GOOGLE_PLAY_APP_NAME, OFFICIAL_GOOGLE_PLAY_URL


def validate_app_identity(
    *,
    platform: str,
    app_id: str,
    detected_app_name: str = "",
    detected_developer: str = "",
    region: str = "",
    expected_app: str = "",
    metadata: dict | None = None,
) -> SourceValidation:
    """PASS only for official Myntra IDs whose live listing looks like Myntra."""
    official_id, official_name, official_url = expected_for(platform)
    expected_label = expected_app or official_name
    banned = is_banned_app_id(app_id)
    official = is_official_myntra_app_id(app_id, platform)
    # App ID matching is not enough — the live listing name/developer must be Myntra.
    is_myntra = looks_like_myntra(detected_app_name, detected_developer)
    foreign = detected_non_myntra_brand(detected_app_name, detected_developer, app_id)

    valid = bool(official and is_myntra and not foreign and not banned)
    if valid:
        status = "VALID_MYNTRA"
        result = "PASS"
        warning = ""
    else:
        status = "INVALID_FOR_MYNTRA_ANALYSIS"
        result = "FAIL"
        if platform == "google_play":
            warning = WARNING_NOT_MYNTRA
        else:
            warning = GENERIC_NOT_MYNTRA
        if banned:
            warning = BANNED_ID_MESSAGE + " " + warning
        if not official:
            warning += (
                f" Configured ID {app_id} is not the official Myntra ID {official_id}."
            )
        if foreign:
            warning += f" Detected brand appears to be {foreign}."
        if detected_app_name:
            warning += (
                f" Configured ID: {app_id}. Detected app: {detected_app_name}. "
                f"Expected app: {expected_label}. Validation: FAIL."
            )

    meta = dict(metadata or {})
    meta.update(
        {
            "expected_id": official_id,
            "expected_url": official_url,
            "validation_result": result,
            "store_url": official_url if official else meta.get("url") or "",
        }
    )

    return SourceValidation(
        platform=platform,
        app_id=app_id,
        detected_app_name=detected_app_name or "",
        detected_developer=detected_developer or "",
        region=region or "",
        expected_app=expected_label,
        expected_id=official_id,
        expected_url=official_url,
        collection_date=datetime.now(timezone.utc),
        validation_status=status,
        validation_result=result,
        is_valid_for_myntra=valid,
        warning=warning,
        metadata=meta,
    )
