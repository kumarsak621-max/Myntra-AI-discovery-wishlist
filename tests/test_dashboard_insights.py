from __future__ import annotations

from datetime import datetime, timezone

from app.models import Analysis, Review
from app.pipeline.quantification import WISHLIST_BEHAVIOR_TERMS, taxonomy_counts
from dashboard.insights import pm_insight


def _review(db, source_id: str, text: str) -> Review:
    row = Review(
        source="google_play",
        source_review_id=source_id,
        app_id="com.myntra.android",
        app_name="Myntra",
        text=text,
        title="",
        rating=3,
        review_date=datetime(2026, 8, 20, tzinfo=timezone.utc),
        is_valid_source=True,
        is_empty=False,
        is_synthetic=False,
        data_classification="MYNTRA EVIDENCE",
        content_hash=source_id,
    )
    db.add(row)
    db.flush()
    db.add(
        Analysis(
            review_id=row.id,
            content_hash=source_id,
            status="analyzed",
            is_valid_json=True,
            relevance="high",
            root_cause="size chart missing",
            barriers_json='["size"]',
            uncertainties_json='["Will it fit?"]',
            intent_json='["wishlist"]',
        )
    )
    db.commit()
    return row


def test_taxonomy_only_returns_matched_categories(db):
    _review(db, "t1", "Wishlisted this dress but the size chart is missing so I did not buy.")
    rows = taxonomy_counts(db, WISHLIST_BEHAVIOR_TERMS)
    labels = {r["label"] for r in rows}
    assert "Save for later" in labels or "Bookmarking" in labels
    assert "Occasion planning" not in labels
    assert all(r["count"] >= 1 for r in rows)


def test_pm_insight_does_not_invent_when_empty():
    text = pm_insight(topic="user problem", rows=[], analyzed=0)
    assert "insufficient" in text.lower()
    text2 = pm_insight(topic="user problem", rows=[], analyzed=12)
    assert "12" in text2
    assert "no user problem" in text2.lower()


def test_review_query_source_filter(db):
    from app.pipeline.quantification import review_query

    _review(db, "gp1", "Wishlisted on Google Play")
    row = Review(
        source="apple_app_store",
        source_review_id="ap1",
        app_id="907394059",
        app_name="Myntra",
        text="Apple store review",
        title="",
        rating=4,
        review_date=datetime(2026, 8, 21, tzinfo=timezone.utc),
        is_valid_source=True,
        is_empty=False,
        is_synthetic=False,
        data_classification="MYNTRA EVIDENCE",
        content_hash="ap1",
    )
    db.add(row)
    db.commit()
    play = review_query(db, myntra_only=True, source="google_play").all()
    apple = review_query(db, myntra_only=True, source="apple_app_store").all()
    assert all(r.source == "google_play" for r in play)
    assert all(r.source == "apple_app_store" for r in apple)
    assert len(play) == 1
    assert len(apple) == 1


def test_chat_insufficient_without_matches(db):
    from dashboard.chat import ask_product_assistant

    result = ask_product_assistant(db, "What do Martian shoppers think?", analyzed=0)
    assert "insufficient evidence" in result["answer"].lower()
    assert result["supporting_review_count"] == 0
