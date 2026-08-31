"""Last-30-day window filtering uses review timestamps, not collection time."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.pipeline.dates import filter_reviews_by_date, get_last_30_days_cutoff
from app.schemas import NormalizedReview


def test_cutoff_is_dynamic_thirty_days():
    now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    cutoff = get_last_30_days_cutoff(now)
    assert cutoff == datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    later = get_last_30_days_cutoff(now + timedelta(days=1))
    assert later > cutoff


def test_naive_datetime_is_treated_as_utc():
    cutoff = get_last_30_days_cutoff(datetime(2026, 8, 31, 0, 0))
    assert cutoff.tzinfo is not None
    assert cutoff.utcoffset() == timedelta(0)


def test_filter_excludes_old_review_even_if_collected_today():
    cutoff = get_last_30_days_cutoff(datetime(2026, 8, 31, tzinfo=timezone.utc))
    old = NormalizedReview(
        source="google_play",
        source_review_id="old",
        app_id="com.myntra.android",
        text="written 45 days ago",
        review_date=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    recent = NormalizedReview(
        source="google_play",
        source_review_id="new",
        app_id="com.myntra.android",
        text="written this week",
        review_date=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    undated = NormalizedReview(
        source="google_play",
        source_review_id="undated",
        app_id="com.myntra.android",
        text="no review date",
        review_date=None,
    )
    kept = filter_reviews_by_date([old, recent, undated], cutoff)
    assert [r.source_review_id for r in kept] == ["new"]


def test_label_window_momentum_is_descriptive_not_significant(db):
    from app.models import Analysis, Review
    from app.pipeline.quantification import label_window_momentum

    now = datetime(2026, 8, 31, tzinfo=timezone.utc)
    cutoff = get_last_30_days_cutoff(now)
    first_half = cutoff + timedelta(days=5)
    second_half = cutoff + timedelta(days=20)
    for i, stamp in enumerate([first_half] + [second_half] * 4):
        review = Review(
            source="google_play",
            source_review_id=f"m{i}",
            app_id="com.myntra.android",
            app_name="Myntra",
            text=f"review {i} about size uncertainty",
            review_date=stamp,
            is_valid_source=True,
            is_empty=False,
            is_duplicate=False,
        )
        db.add(review)
        db.flush()
        db.add(
            Analysis(
                review_id=review.id,
                content_hash=f"h{i}",
                is_valid_json=True,
                status="analyzed",
                barriers_json='["size uncertainty"]',
                relevance="high",
            )
        )
    db.commit()
    rows = label_window_momentum(db, "barriers", since=cutoff, now=now, myntra_only=True, min_count=3)
    assert rows
    assert rows[0]["label"] == "size uncertainty"
    assert rows[0]["momentum"] == "emerging"
    assert "not a statistical significance" in rows[0]["note"]
