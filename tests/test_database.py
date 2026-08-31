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
    assert row.analysis is not None
    assert row.analysis.status == "pending"
    from app.pipeline.analysis import reviews_needing_analysis

    assert row.id in {r.id for r in reviews_needing_analysis(db)}
    row.cleaned_text = "stripped"
    db.commit()
    again = db.query(Review).one()
    assert again.text == original
    assert again.is_valid_source is True


def test_failed_analysis_is_retried(db):
    from app.pipeline.analysis import reviews_needing_analysis

    settings = Settings(collection_rate_limit_seconds=0)
    collector = GooglePlayCollector(settings=settings)
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    item = NormalizedReview(
        source="google_play",
        source_review_id="fail-once",
        app_id="com.myntra.android",
        app_name="Myntra",
        text="Size chart is missing so I did not buy.",
        is_valid_source=True,
        data_classification="MYNTRA EVIDENCE",
    )
    collector.save_raw(db, [item], validation)
    row = db.query(Review).one()
    row.analysis.status = "failed"
    row.analysis.content_hash = row.content_hash
    row.analysis.is_valid_json = False
    db.commit()
    needed = reviews_needing_analysis(db)
    assert row.id in {r.id for r in needed}


def test_analyzed_review_is_not_retried(db):
    from app.pipeline.analysis import reviews_needing_analysis

    settings = Settings(collection_rate_limit_seconds=0)
    collector = GooglePlayCollector(settings=settings)
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    item = NormalizedReview(
        source="google_play",
        source_review_id="already-ok",
        app_id="com.myntra.android",
        app_name="Myntra",
        text="Wishlisted a kurta but the size chart is confusing so I did not buy.",
        is_valid_source=True,
        data_classification="MYNTRA EVIDENCE",
    )
    collector.save_raw(db, [item], validation)
    row = db.query(Review).one()
    row.analysis.status = "analyzed"
    row.analysis.content_hash = row.content_hash
    row.analysis.is_valid_json = True
    db.commit()
    assert reviews_needing_analysis(db) == []


def test_duplicate_insert_is_skipped(db):
    settings = Settings(collection_rate_limit_seconds=0)
    collector = GooglePlayCollector(settings=settings)
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    item = NormalizedReview(
        source="google_play",
        source_review_id="dup-1",
        app_id="com.myntra.android",
        app_name="Myntra",
        text="Wishlisted a kurta but price is high.",
        is_valid_source=True,
        data_classification="MYNTRA EVIDENCE",
    )
    first = collector.save_raw(db, [item], validation)
    second = collector.save_raw(db, [item], validation)
    assert first.new == 1
    assert second.new == 0
    assert second.duplicates == 1
    assert db.query(Review).count() == 1


def test_database_diagnostics_counts_pending(db):
    from app.database import get_database_diagnostics
    from app.models import Analysis

    settings = Settings(collection_rate_limit_seconds=0)
    collector = GooglePlayCollector(settings=settings)
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    item = NormalizedReview(
        source="google_play",
        source_review_id="diag-1",
        app_id="com.myntra.android",
        app_name="Myntra",
        text="Saved to wishlist but size is unclear so I did not buy.",
        is_valid_source=True,
        data_classification="MYNTRA EVIDENCE",
    )
    collector.save_raw(db, [item], validation)
    diag = get_database_diagnostics(db)
    assert diag["google_play_reviews"] == 1
    assert diag["pending_reviews"] >= 1
    assert diag["analyzed_reviews"] == 0
    assert diag["ai_provider"] == "OpenRouter"
    assert diag["ai_model"]
    assert diag["last_successful_analysis_at"] is None
    assert diag["last_analysis_error"] is None
    assert db.query(Analysis).count() == 1


def test_ai_diagnostics_never_includes_key(db):
    from app.database import get_ai_diagnostics

    settings = Settings(collection_rate_limit_seconds=0)
    collector = GooglePlayCollector(settings=settings)
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    collector.save_raw(
        db,
        [
            NormalizedReview(
                source="google_play",
                source_review_id="ai-diag-1",
                app_id="com.myntra.android",
                app_name="Myntra",
                text="Saved to wishlist but size is unclear so I did not buy.",
                is_valid_source=True,
                data_classification="MYNTRA EVIDENCE",
            )
        ],
        validation,
    )
    diag = get_ai_diagnostics(db)
    assert diag["api_key_configured"] in {"YES", "NO"}
    assert diag["ai_provider"] == "OpenRouter"
    assert diag["ai_model"]
    assert "gemini_api_key" not in diag
    assert "openrouter_api_key" not in diag
    blob = str(diag)
    assert "sk-or-" not in blob
    assert diag["pending_reviews"] >= 1
    assert diag["analyzed_reviews"] == 0


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
