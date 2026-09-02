"""Cap the active real-review dataset. Never invent reviews to fill the limit."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models import Analysis, Opportunity, Review, Segment, Source, Theme
from app.pipeline.dates import ensure_aware, filter_reviews_by_date, get_last_30_days_cutoff
from config.settings import (
    analysis_review_limit,
    clamp_max_total_reviews,
    get_settings,
    official_ids,
    storage_review_limit,
)

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


def _storage_cap(max_reviews: int | None = None) -> int:
    if max_reviews is not None:
        return clamp_max_total_reviews(max_reviews)
    return storage_review_limit()


def _analysis_cap(max_reviews: int | None = None) -> int:
    if max_reviews is not None:
        return min(int(max_reviews), storage_review_limit())
    return analysis_review_limit()


def _is_analyzed(review: Review) -> bool:
    analysis = getattr(review, "analysis", None)
    if analysis is None:
        return False
    return analysis.status == "analyzed" and bool(analysis.is_valid_json)


def _usable_reviews(db: Session, *, since=None) -> list[Review]:
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
    pool = myntra if myntra else rows
    if since is not None:
        windowed = filter_reviews_by_date(pool, since)
        if windowed:
            return windowed
    return pool


def select_analysis_reviews(
    db: Session,
    max_reviews: int | None = None,
    *,
    last_30_days: bool = True,
    since=None,
) -> list[Review]:
    """Choose up to MAX_ANALYSIS_REVIEWS real reviews. Never deletes storage rows."""
    limit = _analysis_cap(max_reviews)
    cutoff = since
    if cutoff is None and last_30_days:
        cutoff = get_last_30_days_cutoff(days=int(getattr(get_settings(), "collection_window_days", 30) or 30))
    usable = _usable_reviews(db, since=cutoff)
    keep_ids = select_analysis_sample_ids(usable, limit)
    selected = [row for row in usable if int(row.id) in keep_ids]
    return _newest_first(selected)


def analysis_dataset_stats(
    db: Session,
    max_reviews: int | None = None,
    *,
    last_30_days: bool = True,
) -> dict[str, int]:
    """Counts for storage vs the AI analysis sample. Never invents reviews."""
    storage_cap = _storage_cap()
    sample_cap = _analysis_cap(max_reviews)
    available = _usable_reviews(db)
    selected = select_analysis_reviews(db, sample_cap, last_30_days=last_30_days)
    selected_ids = {int(row.id) for row in selected if row.id is not None}
    analyzed = 0
    pending = 0
    failed = 0
    sample_analyzed = 0
    sample_pending = 0
    sample_failed = 0
    google = 0
    apple = 0
    google_selected = 0
    apple_selected = 0
    for review in available:
        if review.source == GOOGLE_PLAY:
            google += 1
        elif review.source == APPLE:
            apple += 1
        analysis = review.analysis
        status = getattr(analysis, "status", "") if analysis is not None else ""
        if analysis is not None and status == "analyzed" and analysis.is_valid_json:
            analyzed += 1
            if review.id in selected_ids:
                sample_analyzed += 1
        elif analysis is not None and status == "failed":
            failed += 1
            if review.id in selected_ids:
                sample_failed += 1
        else:
            pending += 1
            if review.id in selected_ids:
                sample_pending += 1
    for review in selected:
        if review.source == GOOGLE_PLAY:
            google_selected += 1
        elif review.source == APPLE:
            apple_selected += 1
    batch_size = int(
        getattr(get_settings(), "analysis_batch_size", None)
        or getattr(get_settings(), "ai_request_batch_size", 10)
        or 10
    )
    batch_total = (len(selected) + batch_size - 1) // batch_size if selected else 0
    stored = db.query(Review).count()
    return {
        "available_reviews": len(available),
        "stored_reviews": stored,
        "selected_reviews": len(selected),
        "analyzed_reviews": analyzed,
        "pending_reviews": pending,
        "failed_reviews": failed,
        "sample_analyzed": sample_analyzed,
        "sample_pending": sample_pending,
        "sample_failed": sample_failed,
        "max_analysis_reviews": sample_cap,
        "max_discovery_reviews": sample_cap,
        "max_total_reviews": storage_cap,
        "max_dataset_reviews": storage_cap,
        "google_play_reviews": google,
        "apple_reviews": apple,
        "google_play_selected": google_selected,
        "apple_selected": apple_selected,
        "batch_size": batch_size,
        "batch_total": batch_total,
        "dataset_limit_reached": stored >= storage_cap,
    }


def select_keep_ids(reviews: list[Review], max_reviews: int) -> set[int]:
    """Keep the newest real reviews, preferring a 50/50 source split when both exist.

    If one source has fewer than half, unused slots are filled from the other source.
    Combined total never exceeds max_reviews. Never fabricates reviews.
    """
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


def select_storage_keep_ids(reviews: list[Review], max_reviews: int) -> set[int]:
    """Storage prune: keep analyzed reviews first, then newest with source diversity."""
    analyzed = [row for row in reviews if _is_analyzed(row)]
    keep = select_keep_ids(analyzed, max_reviews)
    remaining = max_reviews - len(keep)
    if remaining <= 0:
        return keep
    rest = [row for row in reviews if row.id is not None and int(row.id) not in keep]
    return keep | select_keep_ids(rest, remaining)


def select_analysis_sample_ids(reviews: list[Review], max_reviews: int) -> set[int]:
    """Deterministic representative AI sample: source, rating, and recency.

    Round-robin across (source, rating) buckets, newest first inside each bucket.
    Never fabricates reviews.
    """
    if max_reviews <= 0:
        return set()
    if len(reviews) <= max_reviews:
        return {int(r.id) for r in reviews if r.id is not None}

    buckets: dict[tuple[str, int], list[Review]] = defaultdict(list)
    for review in _newest_first(reviews):
        buckets[(review.source or "unknown", int(review.rating or 0))].append(review)

    kept: list[Review] = []
    seen: set[int] = set()
    keys = sorted(buckets.keys())
    while len(kept) < max_reviews:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            while bucket:
                candidate = bucket.pop(0)
                cid = int(candidate.id or 0)
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                kept.append(candidate)
                progressed = True
                break
            if len(kept) >= max_reviews:
                break
        if not progressed:
            break
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


def dataset_integrity(db: Session) -> dict[str, int]:
    """Duplicate and orphan checks after prune/collection."""
    from sqlalchemy import func

    stored = db.query(Review).count()
    dup_groups = (
        db.query(Review.source, Review.source_review_id, Review.app_id)
        .group_by(Review.source, Review.source_review_id, Review.app_id)
        .having(func.count(Review.id) > 1)
        .all()
    )
    hash_dupes = (
        db.query(Review.content_hash)
        .filter(Review.content_hash != "", Review.is_duplicate.is_(False))
        .group_by(Review.content_hash)
        .having(func.count(Review.id) > 1)
        .all()
    )
    review_ids = {int(row[0]) for row in db.query(Review.id).all()}
    if review_ids:
        orphan_analysis = db.query(Analysis).filter(~Analysis.review_id.in_(list(review_ids))).count()
    else:
        orphan_analysis = db.query(Analysis).count()
    orphan_evidence = 0
    for model in (Theme, Segment, Opportunity):
        for row in db.query(model).all():
            evidence = _loads_ids(getattr(row, "evidence_ids_json", "") or "")
            orphan_evidence += sum(1 for item in evidence if item not in review_ids)
    return {
        "stored_reviews": stored,
        "duplicate_source_ids": len(dup_groups),
        "duplicate_content_hashes": len(hash_dupes),
        "orphaned_analysis": int(orphan_analysis or 0),
        "orphaned_evidence_ids": int(orphan_evidence or 0),
    }


def enforce_review_limit(
    db: Session,
    max_reviews: int | None = None,
    *,
    prune: bool | None = None,
) -> dict[str, int]:
    """Delete excess stored reviews so production storage never exceeds MAX_TOTAL_REVIEWS.

    No-op when count <= cap. Explicit max_reviews (tests/admin) still prunes.
    Collection always passes prune=True after insert.
    """
    settings = get_settings()
    limit = _storage_cap(max_reviews)
    total = db.query(Review).count()
    enabled = True if prune is True else (bool(settings.prune_excess_reviews) if prune is None else bool(prune))
    if max_reviews is None and not enabled:
        usable = _usable_reviews(db)
        return {
            "max_reviews": limit,
            "before": total,
            "kept": min(len(usable), limit) if total <= limit else len(usable),
            "deleted": 0,
            "analysis_deleted": 0,
            "pruned": False,
        }
    if total <= limit and max_reviews is None:
        return {
            "max_reviews": limit,
            "before": total,
            "kept": total,
            "deleted": 0,
            "analysis_deleted": 0,
            "pruned": False,
        }
    usable = _usable_reviews(db)
    keep_ids = select_storage_keep_ids(usable, limit)
    all_ids = {int(row[0]) for row in db.query(Review.id).all()}
    drop_ids = all_ids - keep_ids

    result = {
        "max_reviews": limit,
        "before": len(all_ids),
        "kept": len(keep_ids),
        "deleted": 0,
        "analysis_deleted": 0,
        "pruned": True,
    }
    if not drop_ids:
        result["pruned"] = False
        result["kept"] = total
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
        "Storage limit %s: deleted %s reviews and %s analysis rows; kept %s. "
        "Rule: analyzed first, then newest by review_date, source-balanced.",
        limit,
        result["deleted"],
        result["analysis_deleted"],
        result["kept"],
    )
    return result
