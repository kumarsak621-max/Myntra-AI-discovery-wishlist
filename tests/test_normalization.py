from __future__ import annotations

from datetime import datetime, timezone

from app.collectors.google_play import GooglePlayCollector
from app.config import Settings
from app.pipeline.validation import validate_app_identity


def test_normalize_maps_play_fields_and_flags_non_myntra():
    collector = GooglePlayCollector(
        settings=Settings(google_play_app_id="com.grofers.customerapp", google_play_country="in")
    )
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.grofers.customerapp",
        detected_app_name="Blinkit",
        detected_developer="Blink Commerce Private Limited",
        region="in",
    )
    raw = {
        "reviewId": "gp:1",
        "content": "Saved for later, not sure about quality.",
        "score": 3,
        "at": datetime(2023, 8, 1, tzinfo=timezone.utc),
        "reviewCreatedVersion": "8.1",
        "replyContent": "Thanks",
        "userName": "secret-user",
    }
    item = collector.normalize(raw, validation)
    assert item.source == "google_play"
    assert item.source_review_id == "gp:1"
    assert item.rating == 3
    assert item.app_version == "8.1"
    assert item.developer_reply == "Thanks"
    assert item.region == "in"
    assert item.is_valid_source is False
    assert "secret-user" not in str(item.raw_payload)
    assert item.text == "Saved for later, not sure about quality."
