from __future__ import annotations

from datetime import datetime, timezone

from app.models import Analysis, Review
from app.pipeline.dataset import enforce_review_limit, select_keep_ids
from app.pipeline.dates import get_last_30_days_cutoff


def _add(
    db,
    *,
    source: str,
    source_id: str,
    day: int,
    text: str = "Saved to wishlist but the size chart is missing so I did not buy.",
    app_id: str | None = None,
    valid: bool = True,
    synthetic: bool = False,
) -> Review:
    if app_id is None:
        app_id = "com.myntra.android" if source == "google_play" else "907394059"
    row = Review(
        source=source,
        source_review_id=source_id,
        app_id=app_id,
        app_name="Myntra",
        text=text,
        title="Fit worry",
        rating=3,
        review_date=datetime(2026, 8, day, tzinfo=timezone.utc),
        source_url=f"https://example.test/{source}/{source_id}",
        region="in",
        is_valid_source=valid,
        is_synthetic=synthetic,
        is_empty=False,
        is_duplicate=False,
        data_classification="MYNTRA EVIDENCE" if valid else "REFERENCE / NON-MYNTRA DATA",
        content_hash=source_id,
    )
    db.add(row)
    db.flush()
    db.add(Analysis(review_id=row.id, content_hash=source_id, status="pending"))
    db.flush()
    return row


def test_under_limit_is_unchanged(db):
    kept_text = "Wishlisted a kurta. Waiting for a sale."
    row = _add(db, source="google_play", source_id="keep-1", day=10, text=kept_text)
    db.commit()
    result = enforce_review_limit(db, max_reviews=300)
    assert result["deleted"] == 0
    assert db.query(Review).count() == 1
    assert db.query(Review).one().text == kept_text
    assert row.analysis.status == "pending"


def test_source_balance_keeps_newest(db):
    originals = {}
    for day in range(1, 21):
        gp = _add(db, source="google_play", source_id=f"gp-{day}", day=day, text=f"google play review {day}")
        ap = _add(db, source="apple_app_store", source_id=f"ap-{day}", day=day, text=f"apple review {day}")
        originals[gp.id] = gp.text
        originals[ap.id] = ap.text
    db.commit()
    assert db.query(Review).count() == 40
    result = enforce_review_limit(db, max_reviews=10)
    assert result["deleted"] == 30
    assert db.query(Review).count() == 10
    assert db.query(Analysis).count() == 10
    google = db.query(Review).filter(Review.source == "google_play").all()
    apple = db.query(Review).filter(Review.source == "apple_app_store").all()
    assert len(google) == 5
    assert len(apple) == 5
    assert {r.review_date.day for r in google} == {16, 17, 18, 19, 20}
    assert {r.review_date.day for r in apple} == {16, 17, 18, 19, 20}
    for row in db.query(Review).all():
        assert row.text == originals[row.id]
        assert row.is_synthetic is False
        assert row.analysis is not None
        assert row.analysis.status == "pending"


def test_sparse_apple_fills_from_google_play(db):
    for day in range(1, 21):
        _add(db, source="google_play", source_id=f"gp-{day}", day=day)
    for day in range(1, 4):
        _add(db, source="apple_app_store", source_id=f"ap-{day}", day=day)
    db.commit()
    enforce_review_limit(db, max_reviews=10)
    google = db.query(Review).filter(Review.source == "google_play").count()
    apple = db.query(Review).filter(Review.source == "apple_app_store").count()
    assert google + apple == 10
    assert apple == 3
    assert google == 7


def test_does_not_fabricate_to_reach_limit(db):
    _add(db, source="google_play", source_id="only-one", day=12)
    db.commit()
    enforce_review_limit(db, max_reviews=300)
    assert db.query(Review).count() == 1
    assert db.query(Review).filter(Review.is_synthetic.is_(True)).count() == 0


def test_deletes_analysis_with_excess_reviews(db):
    rows = [_add(db, source="google_play", source_id=f"gp-{i}", day=i) for i in range(1, 6)]
    db.commit()
    keep_before = {rows[-1].id, rows[-2].id}
    enforce_review_limit(db, max_reviews=2)
    remaining = {r.id for r in db.query(Review).all()}
    assert remaining == keep_before
    assert db.query(Analysis).count() == 2
    assert db.query(Analysis).filter(Analysis.review_id.notin_(list(remaining))).count() == 0


def test_last_30_days_uses_review_date_not_collected_at(db):
    cutoff = get_last_30_days_cutoff()
    old = _add(db, source="google_play", source_id="old", day=1)
    old.review_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    recent = _add(db, source="google_play", source_id="new", day=20)
    recent.review_date = datetime.now(timezone.utc)
    db.commit()
    from app.pipeline.quantification import review_query

    window = review_query(db, since=cutoff).all()
    assert {r.source_review_id for r in window} == {"new"}


def test_select_keep_ids_prefers_half_split():
    class _R:
        def __init__(self, id, source, day):
            self.id = id
            self.source = source
            self.review_date = datetime(2026, 8, day, tzinfo=timezone.utc)
            self.collected_at = self.review_date

    reviews = [_R(i, "google_play", i) for i in range(1, 9)] + [
        _R(100 + i, "apple_app_store", i) for i in range(1, 9)
    ]
    keep = select_keep_ids(reviews, 8)
    google = {r.id for r in reviews if r.source == "google_play" and r.id in keep}
    apple = {r.id for r in reviews if r.source == "apple_app_store" and r.id in keep}
    assert len(google) == 4
    assert len(apple) == 4
    assert google == {5, 6, 7, 8}
    assert apple == {105, 106, 107, 108}
