"""Blinkit/Grofers Play data must never be presented as Myntra evidence."""

from __future__ import annotations

from datetime import datetime, timezone

from app.api.routes import serialize_review
from app.collectors.google_play import GooglePlayCollector
from app.config import Settings
from app.models import Review
from app.pipeline.validation import WARNING_NOT_MYNTRA, validate_app_identity


def test_blinkit_identity_is_invalid_for_myntra():
    result = validate_app_identity(
        platform="google_play",
        app_id="com.grofers.customerapp",
        detected_app_name="Blinkit",
        detected_developer="Blink Commerce Private Limited",
        expected_app="Myntra",
    )
    assert result.is_valid_for_myntra is False
    assert result.validation_result == "FAIL"
    assert result.validation_status == "INVALID_FOR_MYNTRA_ANALYSIS"
    assert "Myntra" in result.warning
    assert "Blinkit" in result.warning
    assert result.data_classification == "REFERENCE / NON-MYNTRA DATA"
    assert WARNING_NOT_MYNTRA in result.warning


def test_myntra_identity_is_valid():
    result = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    assert result.is_valid_for_myntra is True
    assert result.validation_result == "PASS"
    assert result.validation_status == "VALID_MYNTRA"
    assert result.data_classification == "MYNTRA EVIDENCE"


def test_official_ids_are_the_defaults():
    from config.settings import (
        OFFICIAL_APPLE_APP_ID,
        OFFICIAL_GOOGLE_PLAY_APP_ID,
        Settings,
    )

    settings = Settings()
    assert settings.google_play_app_id == OFFICIAL_GOOGLE_PLAY_APP_ID == "com.myntra.android"
    assert settings.apple_app_id == OFFICIAL_APPLE_APP_ID == "907394059"
    assert "com.grofers.customerapp" not in {settings.google_play_app_id, settings.apple_app_id}
    assert "960335206" not in {settings.google_play_app_id, settings.apple_app_id}


def test_configured_package_id_is_not_rewritten():
    settings = Settings(google_play_app_id="com.grofers.customerapp")
    collector = GooglePlayCollector(settings=settings)
    assert collector.settings.google_play_app_id == "com.grofers.customerapp"


def test_serialized_blinkit_review_cannot_look_like_myntra(db):
    review = Review(
        source="google_play",
        source_review_id="abc",
        app_id="com.grofers.customerapp",
        app_name="Blinkit",
        developer="Blink Commerce Private Limited",
        text="Delivery was late.",
        is_valid_source=False,
        data_classification="REFERENCE / NON-MYNTRA DATA",
        collected_at=datetime.now(timezone.utc),
    )
    db.add(review)
    db.commit()
    payload = serialize_review(review)
    assert payload["is_valid_source"] is False
    assert payload["data_classification"] == "REFERENCE / NON-MYNTRA DATA"
    assert "must not be" in payload["warning"].lower()
    assert "myntra evidence" not in payload["data_classification"].lower()
    assert payload["app_name"] == "Blinkit"
