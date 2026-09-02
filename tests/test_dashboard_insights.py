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
            wishlist_signal="implicit",
            purchase_signal="none",
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


def test_wishlist_intent_split_requires_explicit_evidence(db):
    from app.pipeline.quantification import wishlist_intent_split

    _review(db, "t-intent", "I will buy this after the sale.")
    split = wishlist_intent_split(db)
    labels = {r["label"] for r in split["rows"]}
    assert "Occasion-driven shoppers" not in labels
    assert all(r["count"] >= 1 for r in split["rows"])


def test_root_cause_hierarchy_uses_review_ids(db):
    from app.pipeline.quantification import root_cause_hierarchy

    _review(db, "t-root", "Wishlisted this dress but the size chart is missing so I did not buy.")
    rows = root_cause_hierarchy(db)
    assert rows
    assert rows[0]["root_cause"] == "size chart missing"
    assert rows[0]["review_ids"]
    assert rows[0]["count"] >= 1


def test_pm_insight_card_and_root_cause_require_evidence():
    from dashboard.insights import derive_root_cause, pm_insight_card

    card = pm_insight_card(analyzed=0, problems=[], opportunities=[])
    assert "insufficient" in card["strongest_signal"].lower()
    root = derive_root_cause(
        analyzed=0,
        problems=[],
        barriers=[],
        uncertainties=[],
        wishlist=[],
    )
    assert root["statement"] == "Insufficient evidence to establish a reliable root cause."
    root2 = derive_root_cause(
        analyzed=20,
        problems=[{"problem": "Size chart missing", "frequency": 8}],
        barriers=[{"label": "Size/fit", "count": 6}],
        uncertainties=[{"label": "fit", "count": 5}],
        wishlist=[{"label": "saving for later", "count": 4}],
        hesitation_count=7,
    )
    assert root2["supported"] is True
    assert "Size chart missing" in root2["statement"]
    assert "saving for later" in root2["statement"]
    assert "30-day" in root2["statement"]


def test_pm_insight_card_uses_top_problem():
    from dashboard.insights import pm_insight_card

    card = pm_insight_card(
        analyzed=40,
        problems=[{"problem": "Delayed delivery", "frequency": 11, "confidence": 4}],
        opportunities=[],
        example="Delivery took 12 days",
    )
    assert card["strongest_signal"] == "Delayed delivery"
    assert "11" in card["evidence"]
    assert "Delivery took 12 days" in card["evidence"]
    assert "conversion" in card["caveat"].lower()


def test_discovery_questions_insufficient_without_analysis(db):
    from dashboard.questions import answer_discovery_questions
    from dashboard.chat import ask_product_assistant

    cards = answer_discovery_questions(db, {"themes": [], "problems": []}, analyzed=0)
    assert len(cards) == 11
    assert all("insufficient" in c["answer"].lower() for c in cards)
    result = ask_product_assistant(db, "What do Martian shoppers think?", analyzed=0)
    assert "enough evidence" in result["answer"].lower() or "insufficient" in result["answer"].lower()
    assert result["supporting_review_count"] == 0
    assert result.get("caveat")
