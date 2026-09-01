"""Cap the active real-review dataset. Never invent reviews to fill the limit."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models import Analysis, Opportunity, Review, Segment, Source, Theme
from app.pipeline.dates import ensure_aware
from config.settings import clamp_max_dataset_reviews, get_settings, official_ids

logger = logging.getLogger(__name__)

GOOGLE_PLAY = "google_play"
APPLE = "apple_app_store"


def _epoch() -> datetime:
    return datetime.min.replace(tzinfo=timezone.utc)


def _newest_key(review: Review) -> tuple:
    review_date = ensure_aware(review.review_date)
    collected = ensure_aware(review.collected_at)
    return (
        1 if review_date is not None else 0,
        review_date or _epoch(),
        collected or _epoch(),
        int(review.id or 0),
    )


def _newest_first(reviews: list[Review]) -> list[Review]:
    return sorted(reviews, key=_newest_key, reverse=True)


def _usable_reviews(db: Session) -> list[Review]:
    """Real public reviews that may occupy the active dataset."""
    rows = (
        db.query(Review)
        .filter(
            Review.is_empty.is_(False),
            Review.is_synthetic.is_(False),
            Review.is_duplicate.is_(False),
        )
        .all()
    )
    allowed = official_ids()
    myntra = [r for r in rows if r.is_valid_source and (r.app_id or "") in allowed]
    return myntra if myntra else rows


def select_keep_ids(reviews: list[Review], max_reviews: int) -> set[int]:
    """Keep the newest real reviews, preferring a 50/50 source split when both exist."""
    if max_reviews <= 0:
        return set()
    if len(reviews) <= max_reviews:
        return {int(r.id) for r in reviews if r.id is not None}

    by_source: dict[str, list[Review]] = defaultdict(list)
    for review in _newest_first(reviews):
        by_source[review.source or "unknown"].append(review)

    google = list(by_source.get(GOOGLE_PLAY) or [])
    apple = list(by_source.get(APPLE) or [])
    others: list[Review] = []
    for source, items in by_source.items():
        if source not in {GOOGLE_PLAY, APPLE}:
            others.extend(items)
    others = _newest_first(others)

    half = max_reviews // 2
    google_n = min(len(google), half)
    apple_n = min(len(apple), half)
    leftover = _newest_first(google[google_n:] + apple[apple_n:] + others)
    extra_n = max_reviews - google_n - apple_n
    kept = google[:google_n] + apple[:apple_n] + leftover[:extra_n]
    return {int(r.id) for r in kept if r.id is not None}


def _loads_ids(raw: str) -> list[Any]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _prune_evidence_table(db: Session, model, keep_ids: set[int]) -> None:
    rows = db.query(model).all()
    for row in rows:
        evidence = _loads_ids(getattr(row, "evidence_ids_json", "") or "")
        kept = [item for item in evidence if item in keep_ids]
        if not kept:
            db.delete(row)
            continue
        row.evidence_ids_json = json.dumps(kept)
        if hasattr(row, "review_count"):
            row.review_count = len(kept)
        if hasattr(row, "quote_ids_json"):
            quotes = [item for item in _loads_ids(row.quote_ids_json) if item in keep_ids]
            row.quote_ids_json = json.dumps(quotes)
        if hasattr(row, "relevant_count"):
            row.relevant_count = len(kept)


def prune_discovery_evidence(db: Session, keep_ids: set[int]) -> None:
    """Drop dashboard evidence that points at deleted reviews."""
    _prune_evidence_table(db, Theme, keep_ids)
    _prune_evidence_table(db, Segment, keep_ids)
    _prune_evidence_table(db, Opportunity, keep_ids)


def _refresh_source_counts(db: Session) -> None:
    for row in db.query(Source).all():
        row.review_count = (
            db.query(Review)
            .filter(Review.source == row.platform, Review.app_id == row.app_id)
            .count()
        )


def enforce_review_limit(
    db: Session,
    max_reviews: int | None = None,
) -> dict[str, int]:
    """Keep at most `max_reviews` newest real reviews. Delete excess and their analysis.

    Does nothing when the usable corpus is already within the limit.
    Never inserts or fabricates reviews.
    """
    settings = get_settings()
    limit = clamp_max_dataset_reviews(
        max_reviews if max_reviews is not None else getattr(settings, "max_dataset_reviews", 300)
    )
    usable = _usable_reviews(db)
    keep_ids = select_keep_ids(usable, limit)
    all_ids = {int(row[0]) for row in db.query(Review.id).all()}
    drop_ids = all_ids - keep_ids

    result = {
        "max_reviews": limit,
        "before": len(all_ids),
        "kept": len(keep_ids),
        "deleted": 0,
        "analysis_deleted": 0,
    }
    if not drop_ids:
        return result

    drop_list = list(drop_ids)
    analysis_deleted = 0
    deleted = 0
    chunk_size = 400
    for index in range(0, len(drop_list), chunk_size):
        chunk = drop_list[index : index + chunk_size]
        analysis_deleted += (
            db.query(Analysis)
            .filter(Analysis.review_id.in_(chunk))
            .delete(synchronize_session=False)
        )
        deleted += db.query(Review).filter(Review.id.in_(chunk)).delete(synchronize_session=False)
    prune_discovery_evidence(db, keep_ids)
    _refresh_source_counts(db)
    db.commit()
    db.expire_all()
    result["deleted"] = int(deleted or 0)
    result["analysis_deleted"] = int(analysis_deleted or 0)
    result["kept"] = db.query(Review).count()
    logger.info(
        "Dataset limit %s: deleted %s reviews and %s analysis rows; kept %s",
        limit,
        result["deleted"],
        result["analysis_deleted"],
        result["kept"],
    )
    return result
