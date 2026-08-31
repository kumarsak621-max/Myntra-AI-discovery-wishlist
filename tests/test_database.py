from __future__ import annotations

from datetime import datetime, timezone

from app.collectors.google_play import GooglePlayCollector
from app.config import Settings
from app.models import Review
from app.pipeline.validation import validate_app_identity
from app.schemas import NormalizedReview


def test_database_round_trip_never_overwrites_original_text(db):
    settings = Settings(collection_rate_limit_seconds=0)
    collector = GooglePlayCollector(settings=settings)
    original = "I love this dress but didn't buy because I don't know if the size will fit."
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    item = NormalizedReview(
        source="google_play",
        source_review_id="keep-me",
        app_id="com.myntra.android",
        app_name="Myntra",
        text=original,
        title="Fit worry",
        rating=3,
        review_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        source_url="https://example.test/r",
        is_valid_source=True,
        data_classification="MYNTRA EVIDENCE",
    )
    collector.save_raw(db, [item], validation)
    row = db.query(Review).one()
    assert row.text == original
    row.cleaned_text = "stripped"
    db.commit()
    again = db.query(Review).one()
    assert again.text == original
    assert again.is_valid_source is True


def test_empty_review_is_rejected(db):
    collector = GooglePlayCollector(settings=Settings(collection_rate_limit_seconds=0))
    validation = validate_app_identity(
        platform="google_play",
        app_id="x",
        detected_app_name="Myntra",
        detected_developer="Myntra",
    )
    stats = collector.save_raw(
        db,
        [NormalizedReview(source="google_play", source_review_id="e", app_id="x", text="   ")],
        validation,
    )
    assert stats.rejected == 1
    assert db.query(Review).count() == 0
