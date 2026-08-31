"""Deterministic quantification. Percentages are never produced by the LLM."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models import Analysis, Review
from config.settings import official_ids

logger = logging.getLogger(__name__)


def pct(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((count / denominator) * 100, 2)


def _loads(raw: str) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def is_myntra_evidence(review: Review) -> bool:
    return bool(review.is_valid_source) and (review.app_id or "") in official_ids()


def review_query(db: Session, *, myntra_only: bool = False, since: datetime | None = None):
    q = db.query(Review).filter(Review.is_duplicate.is_(False), Review.is_empty.is_(False))
    if myntra_only:
        q = q.filter(Review.is_valid_source.is_(True), Review.app_id.in_(list(official_ids())))
    if since is not None:
        q = q.filter(Review.review_date.isnot(None), Review.review_date >= since)
    return q


def overview_metrics(db: Session, *, since: datetime | None = None, myntra_only: bool = False) -> dict[str, Any]:
    all_reviews = review_query(db, myntra_only=myntra_only, since=since).all()
    myntra = [r for r in all_reviews if is_myntra_evidence(r)]
    reference = [r for r in all_reviews if not is_myntra_evidence(r)]
    analyzed = [r for r in all_reviews if r.analysis and r.analysis.is_valid_json]
    relevant = [
        r
        for r in analyzed
        if r.analysis and r.analysis.relevance in {"high", "medium"}
    ]
    by_source = Counter(r.source for r in all_reviews)
    by_rating = Counter(str(r.rating) for r in all_reviews if r.rating is not None)
    by_classification = Counter(r.data_classification for r in all_reviews)
    ratings = [r.rating for r in all_reviews if r.rating is not None]
    dates = [r.review_date for r in all_reviews if r.review_date]
    barriers = label_distribution_safe(db, "barriers", myntra_only=True, since=since)
    wishlist = signal_counts(db, myntra_only=True, since=since)
    intents = label_distribution_safe(db, "intent", myntra_only=True, since=since)
    uncertainties = label_distribution_safe(db, "uncertainties", myntra_only=True, since=since)

    from app.models import Opportunity, Theme

    theme_count = db.query(Theme).count()
    opportunity_count = db.query(Opportunity).count()
    problems = root_cause_distribution(db, myntra_only=True, since=since)
    return {
        "total_reviews": len(all_reviews),
        "myntra_reviews": len(myntra),
        "reference_non_myntra_reviews": len(reference),
        "analyzed_reviews": len(analyzed),
        "relevant_reviews": len(relevant),
        "relevant_pct_of_analyzed": pct(len(relevant), len(analyzed)),
        "google_play_reviews": by_source.get("google_play", 0),
        "apple_reviews": by_source.get("apple_app_store", 0),
        "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "date_from": min(dates).isoformat() if dates else None,
        "date_to": max(dates).isoformat() if dates else None,
        "theme_count": theme_count,
        "opportunity_count": opportunity_count,
        "top_barriers": barriers[:5],
        "top_intents": intents[:5],
        "top_uncertainties": uncertainties[:5],
        "top_problems": problems[:5],
        "wishlist_signals": wishlist.get("wishlist_signal", 0),
        "purchase_hesitation": wishlist.get("purchase_hesitation", 0),
        "rating_1": sum(1 for r in all_reviews if r.rating == 1),
        "rating_2": sum(1 for r in all_reviews if r.rating == 2),
        "rating_3": sum(1 for r in all_reviews if r.rating == 3),
        "rating_4": sum(1 for r in all_reviews if r.rating == 4),
        "rating_5": sum(1 for r in all_reviews if r.rating == 5),
        "by_source": dict(by_source),
        "by_rating": dict(by_rating),
        "by_classification": dict(by_classification),
        "synthetic_count": sum(1 for r in all_reviews if r.is_synthetic),
    }


def label_distribution_safe(
    db: Session, field: str, *, myntra_only: bool = False, since: datetime | None = None
) -> list[dict[str, Any]]:
    try:
        return label_distribution(db, field, myntra_only=myntra_only, since=since)
    except Exception as exc:
        logger.warning("Could not compute %s distribution: %s", field, exc)
        return []


def label_distribution(
    db: Session,
    field: str,
    *,
    myntra_only: bool = False,
    relevant_only: bool = True,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only, since=since).all()
    items: list[tuple[Review, Analysis]] = []
    for review in rows:
        analysis = review.analysis
        if not analysis or not analysis.is_valid_json:
            continue
        if relevant_only and analysis.relevance not in {"high", "medium"}:
            continue
        items.append((review, analysis))

    denominator = len(items)
    counts: Counter[str] = Counter()
    by_source: dict[str, set[str]] = defaultdict(set)
    review_ids: dict[str, list[int]] = defaultdict(list)
    hesitant: Counter[str] = Counter()

    attr = {
        "intent": "intent_json",
        "barriers": "barriers_json",
        "uncertainties": "uncertainties_json",
        "product_category": "product_category_json",
        "decision_factors": "decision_factors_json",
    }[field]

    for review, analysis in items:
        labels = [str(x).strip() for x in _loads(getattr(analysis, attr)) if str(x).strip()]
        unique_labels = list(dict.fromkeys(labels))
        for label in unique_labels:
            key = label
            counts[key] += 1
            by_source[key].add(review.source)
            review_ids[key].append(review.id)
            if analysis.purchase_hesitation in {"explicit", "implicit"}:
                hesitant[key] += 1

    ranked = []
    for label, count in counts.most_common():
        ranked.append(
            {
                "label": label,
                "count": count,
                "percentage": pct(count, denominator),
                "denominator": denominator,
                "sources": sorted(by_source[label]),
                "source_count": len(by_source[label]),
                "hesitant_count": hesitant[label],
                "review_ids": review_ids[label][:50],
            }
        )
    return ranked


def signal_counts(db: Session, *, myntra_only: bool = False, since: datetime | None = None) -> dict[str, Any]:
    rows = review_query(db, myntra_only=myntra_only, since=since).all()
    analyzed = [r for r in rows if r.analysis and r.analysis.is_valid_json]
    denom = len(analyzed) or 1
    wishlist = sum(
        1 for r in analyzed if r.analysis.wishlist_signal in {"explicit", "implicit"}
    )
    hesitation = sum(
        1 for r in analyzed if r.analysis.purchase_hesitation in {"explicit", "implicit"}
    )
    purchase = sum(
        1
        for r in analyzed
        if r.analysis.purchase_signal in {"purchased", "intend_to_purchase", "hesitant", "abandoned"}
    )
    return {
        "analyzed": len(analyzed),
        "wishlist_signal": wishlist,
        "wishlist_pct": pct(wishlist, len(analyzed)),
        "purchase_hesitation": hesitation,
        "hesitation_pct": pct(hesitation, len(analyzed)),
        "purchase_related": purchase,
        "purchase_pct": pct(purchase, len(analyzed)),
        "denominator": len(analyzed),
        "note": "percent = count / analyzed_reviews × 100",
    }


def time_trends(db: Session, *, myntra_only: bool = False, since: datetime | None = None) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only, since=since).all()
    buckets: Counter[str] = Counter()
    for review in rows:
        dt: datetime | None = review.review_date or review.collected_at
        if not dt:
            continue
        buckets[dt.strftime("%Y-%m")] += 1
    return [{"month": k, "count": v} for k, v in sorted(buckets.items())]


def information_seeking(db: Session, *, myntra_only: bool = False) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only).all()
    counts: Counter[str] = Counter()
    details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    analyzed = 0
    for review in rows:
        analysis = review.analysis
        if not analysis or not analysis.is_valid_json:
            continue
        analyzed += 1
        for item in _loads(analysis.information_seeking_json):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "unspecified")
            counts[source] += 1
            details[source].append(
                {
                    "review_id": review.id,
                    "what": item.get("what"),
                    "why": item.get("why"),
                    "associated_with_hesitation": item.get("associated_with_hesitation"),
                    "myntra_appears_to_lack_info": item.get("myntra_appears_to_lack_info"),
                    "basis": item.get("basis"),
                    "is_valid_source": review.is_valid_source,
                }
            )
    return [
        {
            "source": src,
            "count": count,
            "percentage": pct(count, analyzed),
            "denominator": analyzed,
            "examples": details[src][:20],
        }
        for src, count in counts.most_common()
    ]


def daily_review_trends(db: Session, *, myntra_only: bool = False, since: datetime | None = None) -> list[dict[str, Any]]:
    """Counts by review_date day only. Collection time is never used."""
    rows = review_query(db, myntra_only=myntra_only, since=since).all()
    counts: Counter[str] = Counter()
    rating_sum: dict[str, list[int]] = defaultdict(list)
    rating_buckets: dict[str, Counter[int]] = defaultdict(Counter)
    for review in rows:
        if not review.review_date:
            continue
        day = review.review_date.strftime("%Y-%m-%d")
        counts[day] += 1
        if review.rating is not None:
            rating_sum[day].append(review.rating)
            rating_buckets[day][int(review.rating)] += 1
    out = []
    for day in sorted(counts):
        ratings = rating_sum.get(day) or []
        stars = rating_buckets.get(day) or Counter()
        out.append(
            {
                "day": day,
                "reviews": counts[day],
                "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
                "rating_1": stars.get(1, 0),
                "rating_2": stars.get(2, 0),
                "rating_3": stars.get(3, 0),
                "rating_4": stars.get(4, 0),
                "rating_5": stars.get(5, 0),
            }
        )
    return out


def _json_labels(analysis: Analysis, field: str) -> list[str]:
    attr = {
        "intent": "intent_json",
        "barriers": "barriers_json",
        "uncertainties": "uncertainties_json",
        "root_cause": None,
        "themes": None,
    }.get(field, "barriers_json")
    if field == "root_cause":
        statement = (analysis.root_cause or "").strip()
        return [statement] if statement else []
    if field == "themes":
        bag = []
        for key in ("barriers_json", "uncertainties_json", "intent_json"):
            bag.extend(str(x).strip() for x in _loads(getattr(analysis, key)) if str(x).strip())
        if (analysis.root_cause or "").strip():
            bag.append(analysis.root_cause.strip())
        return list(dict.fromkeys(bag))
    labels = [str(x).strip() for x in _loads(getattr(analysis, attr)) if str(x).strip()]
    return list(dict.fromkeys(labels))


def label_window_momentum(
    db: Session,
    field: str,
    *,
    since: datetime,
    myntra_only: bool = True,
    now: datetime | None = None,
    min_count: int = 3,
) -> list[dict[str, Any]]:
    """First-half vs second-half of the window. Descriptive only — not a significance test."""
    from app.pipeline.dates import ensure_aware, utcnow

    start = ensure_aware(since)
    end = ensure_aware(now) or utcnow()
    if start is None:
        return []
    midpoint = start + (end - start) / 2
    rows = review_query(db, myntra_only=myntra_only, since=start).all()
    first: Counter[str] = Counter()
    second: Counter[str] = Counter()
    evidence: dict[str, list[int]] = defaultdict(list)
    daily: dict[str, Counter[str]] = defaultdict(Counter)

    for review in rows:
        analysis = review.analysis
        if not analysis or not analysis.is_valid_json:
            continue
        stamp = ensure_aware(review.review_date)
        if stamp is None:
            continue
        labels = _json_labels(analysis, field)
        day = stamp.strftime("%Y-%m-%d")
        bucket = first if stamp < midpoint else second
        for label in labels:
            bucket[label] += 1
            evidence[label].append(review.id)
            daily[label][day] += 1

    names = set(first) | set(second)
    ranked = []
    for label in names:
        a = first[label]
        b = second[label]
        total = a + b
        if total < min_count:
            momentum = "insufficient data"
        elif b > a * 1.25 and b >= 2:
            momentum = "emerging"
        elif a > b * 1.25 and a >= 2:
            momentum = "declining"
        else:
            momentum = "stable"
        ranked.append(
            {
                "label": label,
                "count": total,
                "first_half": a,
                "second_half": b,
                "momentum": momentum,
                "review_ids": evidence[label][:50],
                "by_day": dict(daily[label]),
                "note": "Descriptive split of this window, not a statistical significance test.",
            }
        )
    ranked.sort(key=lambda x: (-int(x["count"]), str(x["label"])))
    return ranked


def root_cause_distribution(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None
) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only, since=since).all()
    counts: Counter[str] = Counter()
    review_ids: dict[str, list[int]] = defaultdict(list)
    analyzed = 0
    for review in rows:
        analysis = review.analysis
        if not analysis or not analysis.is_valid_json:
            continue
        analyzed += 1
        statement = (analysis.root_cause or "").strip()
        if not statement:
            continue
        counts[statement] += 1
        review_ids[statement].append(review.id)
    return [
        {
            "label": label,
            "count": count,
            "percentage": pct(count, analyzed),
            "denominator": analyzed,
            "review_ids": review_ids[label][:50],
        }
        for label, count in counts.most_common()
    ]


def problem_rows(db: Session, *, myntra_only: bool = True, since: datetime | None = None) -> list[dict[str, Any]]:
    """Aggregate root causes with frequency, severity, and purchase impact from real analyses."""
    from app.models import Analysis
    from app.pipeline.scoring import purchase_impact_from_hesitation, severity_from_strength

    dist = root_cause_distribution(db, myntra_only=myntra_only, since=since)
    out = []
    for item in dist:
        ids = item.get("review_ids") or []
        analyses = (
            db.query(Analysis)
            .filter(Analysis.review_id.in_(ids), Analysis.is_valid_json.is_(True))
            .all()
            if ids
            else []
        )
        strengths = [a.evidence_strength for a in analyses if a.evidence_strength]
        ratings = []
        hesitant = 0
        for analysis in analyses:
            review = analysis.review
            if review is not None and review.rating is not None:
                ratings.append(review.rating)
            if analysis.purchase_hesitation in {"explicit", "implicit"}:
                hesitant += 1
        out.append(
            {
                "problem": item["label"],
                "frequency": item["count"],
                "percentage": item["percentage"],
                "denominator": item["denominator"],
                "severity": severity_from_strength(strengths, ratings),
                "purchase_impact": purchase_impact_from_hesitation(hesitant, item["count"]),
                "supporting_reviews": item["count"],
                "review_ids": ids,
                "confidence": round(sum(a.confidence for a in analyses) / len(analyses), 2) if analyses else None,
                "analysis_timestamp": max((a.analyzed_at for a in analyses if a.analyzed_at), default=None),
                "model": next((a.model for a in analyses if a.model), ""),
                "analysis_version": next((a.analysis_version for a in analyses if a.analysis_version), ""),
            }
        )
    return out


def source_live_status(db: Session) -> dict[str, Any]:
    """Freshness for official Myntra sources. Never claims a live stream."""
    from app.models import CollectionRun, Source
    from app.pipeline.dates import ensure_aware
    from config.settings import OFFICIAL_APPLE_APP_ID, OFFICIAL_GOOGLE_PLAY_APP_ID

    def _latest_review(source: str, app_id: str) -> datetime | None:
        row = (
            db.query(Review)
            .filter(
                Review.source == source,
                Review.app_id == app_id,
                Review.is_empty.is_(False),
                Review.review_date.isnot(None),
            )
            .order_by(Review.review_date.desc())
            .first()
        )
        return ensure_aware(row.review_date) if row else None

    def _source_row(platform: str, app_id: str) -> Source | None:
        return (
            db.query(Source)
            .filter(Source.platform == platform, Source.app_id == app_id)
            .order_by(Source.last_collection_at.desc())
            .first()
        )

    last_run = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
    last_ok = (
        db.query(CollectionRun)
        .filter(CollectionRun.status.in_(["completed", "completed_with_errors"]))
        .order_by(CollectionRun.id.desc())
        .first()
    )
    return {
        "google_play": {
            "source": _source_row("google_play", OFFICIAL_GOOGLE_PLAY_APP_ID),
            "latest_review_at": _latest_review("google_play", OFFICIAL_GOOGLE_PLAY_APP_ID),
        },
        "apple_app_store": {
            "source": _source_row("apple_app_store", OFFICIAL_APPLE_APP_ID),
            "latest_review_at": _latest_review("apple_app_store", OFFICIAL_APPLE_APP_ID),
        },
        "last_run": last_run,
        "last_successful_run": last_ok,
    }
