"""Deterministic quantification. Percentages are never produced by the LLM."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models import Analysis, Review
from app.pipeline.labels import (
    merge_category_rows,
    normalize_label,
    normalize_label_list,
)
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


def review_query(
    db: Session,
    *,
    myntra_only: bool = False,
    since: datetime | None = None,
    source: str | None = None,
):
    q = db.query(Review).filter(Review.is_duplicate.is_(False), Review.is_empty.is_(False))
    if myntra_only:
        q = q.filter(Review.is_valid_source.is_(True), Review.app_id.in_(list(official_ids())))
    if since is not None:
        q = q.filter(Review.review_date.isnot(None), Review.review_date >= since)
    if source in {"google_play", "apple_app_store"}:
        q = q.filter(Review.source == source)
    return q


def overview_metrics(
    db: Session,
    *,
    since: datetime | None = None,
    myntra_only: bool = False,
    source: str | None = None,
) -> dict[str, Any]:
    all_reviews = review_query(db, myntra_only=myntra_only, since=since, source=source).all()
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
    barriers = label_distribution_safe(db, "barriers", myntra_only=True, since=since, source=source)
    wishlist = signal_counts(db, myntra_only=True, since=since, source=source)
    intents = label_distribution_safe(db, "intent", myntra_only=True, since=since, source=source)
    uncertainties = label_distribution_safe(db, "uncertainties", myntra_only=True, since=since, source=source)

    from app.models import Opportunity, Theme

    theme_count = db.query(Theme).count()
    opportunity_count = db.query(Opportunity).count()
    problems = root_cause_distribution(db, myntra_only=True, since=since, source=source)
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
    db: Session,
    field: str,
    *,
    myntra_only: bool = False,
    since: datetime | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    try:
        return label_distribution(db, field, myntra_only=myntra_only, since=since, source=source)
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
    source: str | None = None,
) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only, since=since, source=source).all()
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
        unique_labels = normalize_label_list(_loads(getattr(analysis, attr)))
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


def signal_counts(
    db: Session, *, myntra_only: bool = False, since: datetime | None = None, source: str | None = None
) -> dict[str, Any]:
    rows = review_query(db, myntra_only=myntra_only, since=since, source=source).all()
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


def hesitation_split(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> list[dict[str, Any]]:
    """Explicit vs implicit purchase hesitation. Placeholder 'none' is omitted."""
    rows = [
        r
        for r in review_query(db, myntra_only=myntra_only, since=since, source=source).all()
        if r.analysis and r.analysis.is_valid_json
    ]
    counts: Counter[str] = Counter()
    ids: dict[str, list[int]] = defaultdict(list)
    for review in rows:
        value = (review.analysis.purchase_hesitation or "").strip().lower()
        if value not in {"explicit", "implicit"}:
            continue
        label = "Explicit hesitation" if value == "explicit" else "Implicit hesitation"
        counts[label] += 1
        ids[label].append(review.id)
    return [
        {
            "label": label,
            "count": count,
            "percentage": pct(count, len(rows)),
            "denominator": len(rows),
            "review_ids": ids[label][:50],
        }
        for label, count in counts.most_common()
    ]


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
            source = normalize_label(item.get("source"))
            if not source:
                continue
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
        return [label for label in [normalize_label(analysis.root_cause)] if label]
    if field == "themes":
        bag = []
        for key in ("barriers_json", "uncertainties_json", "intent_json"):
            bag.extend(_loads(getattr(analysis, key)))
        if analysis.root_cause:
            bag.append(analysis.root_cause)
        return normalize_label_list(bag, keep_uncategorized_if_only_missing=True)
    return normalize_label_list(_loads(getattr(analysis, attr)))


def label_window_momentum(
    db: Session,
    field: str,
    *,
    since: datetime,
    myntra_only: bool = True,
    now: datetime | None = None,
    min_count: int = 3,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """First-half vs second-half of the window. Descriptive only — not a significance test."""
    from app.pipeline.dates import ensure_aware, utcnow

    start = ensure_aware(since)
    end = ensure_aware(now) or utcnow()
    if start is None:
        return []
    midpoint = start + (end - start) / 2
    rows = review_query(db, myntra_only=myntra_only, since=start, source=source).all()
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
            key = normalize_label(label)
            if not key:
                continue
            bucket[key] += 1
            evidence[key].append(review.id)
            daily[key][day] += 1

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
    return merge_category_rows(ranked)


def root_cause_distribution(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only, since=since, source=source).all()
    counts: Counter[str] = Counter()
    review_ids: dict[str, list[int]] = defaultdict(list)
    analyzed = 0
    for review in rows:
        analysis = review.analysis
        if not analysis or not analysis.is_valid_json:
            continue
        analyzed += 1
        statement = normalize_label(analysis.root_cause)
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


def problem_rows(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> list[dict[str, Any]]:
    """Aggregate root causes with frequency, severity, and purchase impact from real analyses."""
    from app.models import Analysis
    from app.pipeline.scoring import purchase_impact_from_hesitation, severity_from_strength

    dist = root_cause_distribution(db, myntra_only=myntra_only, since=since, source=source)
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


# Keyword taxonomies used only against real review text + stored analysis labels.
# A category appears only when at least one review actually matches.
WISHLIST_BEHAVIOR_TERMS = {
    "Save for later": ("wishlist", "wish list", "save for later", "saved", "bookmark", "bookmarked"),
    "Price monitoring": ("price", "expensive", "sale", "discount", "offer", "cheap", "costly", "value"),
    "Compare options": ("compar", "vs ", "versus", "alternative", "other brand", "options"),
    "Inspiration": ("inspir", "look", "style", "trend", "aesthetic"),
    "Like / desire": ("love", "like this", "want this", "must have", "crush"),
    "Occasion planning": ("occasion", "wedding", "party", "festival", "event", "outfit"),
    "Purchase intent": ("will buy", "going to buy", "intend", "planning to buy", "add to bag"),
    "Bookmarking": ("bookmark", "later", "maybe", "not now", "someday"),
}

BARRIER_TERMS = {
    "Price / Value": ("price", "expensive", "cheap", "value", "cost", "overpriced"),
    "Size / Fit": ("size", "fit", "sizing", "chart", "measurement"),
    "Quality": ("quality", "cheap fabric", "tear", "poor quality"),
    "Returns / Exchange": ("return", "exchange", "refund"),
    "Delivery": ("delivery", "shipping", "late", "courier"),
    "Product information": ("description", "details missing", "info", "information"),
    "Reviews / Ratings": ("review", "rating", "fake review"),
    "Color": ("color", "colour", "shade"),
    "Material": ("material", "fabric", "cotton", "silk"),
    "Styling": ("style", "look", "styling"),
    "Trust": ("trust", "fake", "scam", "authentic"),
    "Availability": ("out of stock", "unavailable", "sold out"),
}

UNCERTAINTY_TERMS = {
    "Fit": ("fit", "fitting"),
    "Size": ("size", "sizing", "chart"),
    "Fabric": ("fabric",),
    "Material": ("material",),
    "Quality": ("quality", "durable"),
    "Color": ("color", "colour"),
    "Look / Styling": ("look", "style", "styling"),
    "Returns": ("return", "exchange"),
    "Delivery": ("delivery", "shipping"),
    "Durability": ("durable", "lasting", "wear"),
    "Reviews": ("review", "rating"),
    "Social validation": ("friend", "family", "people say"),
}

COMPARISON_TERMS = {
    "Price": ("price", "cheaper", "expensive"),
    "Fit / Size": ("size", "fit"),
    "Quality": ("quality",),
    "Material": ("material", "fabric"),
    "Color": ("color", "colour"),
    "Reviews": ("review",),
    "Ratings": ("rating", "stars"),
    "Brand": ("brand",),
    "Style": ("style", "look"),
    "Features": ("feature",),
}

COMPARISON_METHOD_TERMS = {
    "Myntra/AJIO": ("myntra", "ajio"),
    "Other marketplaces": ("amazon", "flipkart", "meesho", "nykaa"),
    "Google Search": ("google", "searched"),
    "Social media": ("instagram", "youtube", "facebook", "reddit"),
    "Friends / Family": ("friend", "family", "sister", "mom"),
}

EXTERNAL_TERMS = {
    "Google": ("google", "googled"),
    "YouTube": ("youtube", "yt"),
    "Instagram": ("instagram", "insta"),
    "Reddit": ("reddit",),
    "Other marketplaces": ("amazon", "flipkart", "ajio", "nykaa"),
    "Friends / Family": ("friend", "family", "asked my"),
    "Price comparison": ("compare price", "cheaper on", "price comparison"),
}

SOCIAL_TERMS = {
    "Ratings": ("rating", "stars"),
    "Reviews": ("review", "reviews"),
    "Photos": ("photo", "picture", "image"),
    "Videos": ("video", "reel"),
    "Friends": ("friend",),
    "Family": ("family", "mom", "sister"),
    "Influencers": ("influencer", "blogger"),
    "Social media": ("instagram", "youtube", "social"),
}

PURCHASE_BEHAVIOR_TERMS = {
    "Immediate purchase intent": ("bought", "purchased", "ordered", "will buy"),
    "Delayed purchase": ("later", "wait", "postpone", "not now"),
    "Comparison before purchase": ("compar", "vs ", "alternative"),
    "Waiting for price changes": ("sale", "discount", "offer", "cheaper"),
    "Waiting for information": ("not sure", "confused", "need to know"),
    "Seeking validation": ("review", "rating", "ask"),
    "Bookmarking": ("wishlist", "save", "bookmark"),
    "Repeat purchase": ("again", "repeat", "reorder"),
    "Abandoned intent": ("cancelled", "abandoned", "did not buy", "didn't buy"),
}


def _review_evidence_blob(review: Review) -> str:
    analysis = review.analysis
    parts = [review.title or "", review.text or ""]
    if analysis and analysis.is_valid_json:
        parts.append(analysis.root_cause or "")
        for attr in (
            "intent_json",
            "barriers_json",
            "uncertainties_json",
            "decision_factors_json",
            "behavioral_signals_json",
            "information_seeking_json",
        ):
            raw = getattr(analysis, attr, "") or ""
            parts.append(raw)
        parts.append(analysis.purchase_signal or "")
        parts.append(analysis.wishlist_signal or "")
    return " ".join(parts).lower()


def taxonomy_counts(
    db: Session,
    taxonomy: dict[str, tuple[str, ...]],
    *,
    myntra_only: bool = True,
    since: datetime | None = None,
    analyzed_only: bool = True,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Count real reviews whose stored text/labels contain taxonomy terms. Never invents categories."""
    rows = review_query(db, myntra_only=myntra_only, since=since, source=source).all()
    if analyzed_only:
        rows = [r for r in rows if r.analysis and r.analysis.is_valid_json]
    denominator = len(rows)
    buckets: dict[str, list[int]] = {name: [] for name in taxonomy}
    for review in rows:
        blob = _review_evidence_blob(review)
        for name, terms in taxonomy.items():
            if any(term in blob for term in terms):
                buckets[name].append(review.id)
    ranked = []
    for name, ids in buckets.items():
        if not ids:
            continue
        ranked.append(
            {
                "label": name,
                "count": len(ids),
                "percentage": pct(len(ids), denominator),
                "denominator": denominator,
                "review_ids": ids[:50],
            }
        )
    ranked.sort(key=lambda item: (-int(item["count"]), str(item["label"])))
    return ranked


def purchase_signal_counts(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> list[dict[str, Any]]:
    rows = [
        r
        for r in review_query(db, myntra_only=myntra_only, since=since, source=source).all()
        if r.analysis and r.analysis.is_valid_json
    ]
    counts: Counter[str] = Counter()
    ids: dict[str, list[int]] = defaultdict(list)
    for review in rows:
        signal = review.analysis.purchase_signal or "none"
        if signal == "none":
            continue
        counts[signal] += 1
        ids[signal].append(review.id)
    return [
        {
            "label": label,
            "count": count,
            "percentage": pct(count, len(rows)),
            "denominator": len(rows),
            "review_ids": ids[label][:50],
        }
        for label, count in counts.most_common()
    ]


def explicit_age_mentions(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> list[dict[str, Any]]:
    """Only count reviews that explicitly mention an age number with an age word."""
    import re

    pattern = re.compile(r"\b(?:i(?:'m| am)?|age)\s*(\d{2})\b|\b(\d{2})\s*(?:years? old|yr)\b", re.I)
    rows = review_query(db, myntra_only=myntra_only, since=since, source=source).all()
    hits = []
    for review in rows:
        text = f"{review.title or ''} {review.text or ''}"
        match = pattern.search(text)
        if not match:
            continue
        age = match.group(1) or match.group(2)
        hits.append({"review_id": review.id, "age": int(age), "quote": text[:240]})
    return hits


SEGMENT_TERMS = {
    "Price-sensitive shoppers": ("price", "expensive", "cheap", "value", "cost", "overpriced"),
    "Fit-sensitive shoppers": ("size", "fit", "sizing", "chart"),
    "Quality-conscious shoppers": ("quality", "fabric", "material", "durable"),
    "Comparison shoppers": ("compar", "vs ", "versus", "alternative"),
    "Occasion-driven shoppers": ("occasion", "wedding", "party", "festival", "event"),
    "High-intent shoppers": ("will buy", "going to buy", "bought", "ordered", "purchased"),
    "Uncertainty-driven shoppers": ("not sure", "confused", "uncertain", "doubt"),
    "Bookmarking/save-for-later users": ("wishlist", "save for later", "bookmark", "later"),
}

PURCHASE_INTENT_TERMS = (
    "will buy",
    "going to buy",
    "intend to buy",
    "planning to buy",
    "add to bag",
    "added to bag",
    "bought",
    "purchased",
    "ordered",
)
BOOKMARK_TERMS = (
    "bookmark",
    "save for later",
    "saved for later",
    "maybe later",
    "not now",
    "someday",
)


def _blob_has(blob: str, terms: tuple[str, ...]) -> bool:
    return any(term in blob for term in terms)


def rating_distribution(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only, since=since, source=source).all()
    counts: Counter[str] = Counter()
    for review in rows:
        if review.rating is None:
            continue
        counts[str(int(review.rating))] += 1
    total = sum(counts.values())
    return [
        {
            "label": f"{star}★",
            "count": counts.get(str(star), 0),
            "percentage": pct(counts.get(str(star), 0), total),
            "star": star,
        }
        for star in (1, 2, 3, 4, 5)
        if counts.get(str(star), 0) > 0
    ]


def latest_review_cards(
    db: Session,
    *,
    since: datetime | None = None,
    source: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = (
        review_query(db, myntra_only=True, since=since, source=source)
        .order_by(Review.review_date.desc())
        .limit(limit)
        .all()
    )
    cards = []
    for review in rows:
        analysis = review.analysis
        cards.append(
            {
                "id": review.id,
                "text": (review.text or review.title or "")[:400],
                "rating": review.rating,
                "source": review.source,
                "region": review.region,
                "date": review.review_date.isoformat() if review.review_date else None,
                "source_review_id": review.source_review_id,
                "status": getattr(analysis, "status", "none") if analysis else "none",
            }
        )
    return cards


def wishlist_intent_split(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> dict[str, Any]:
    """Exclusive buckets from explicit text + stored signals. Never assumes liking = intent."""
    rows = [
        r
        for r in review_query(db, myntra_only=myntra_only, since=since, source=source).all()
        if r.analysis and r.analysis.is_valid_json
    ]
    buckets = {
        "Genuine Purchase Intent": [],
        "Bookmarking / Save-for-later": [],
        "Unclear Evidence": [],
    }
    for review in rows:
        blob = _review_evidence_blob(review)
        analysis = review.analysis
        has_wishlist = analysis.wishlist_signal in {"explicit", "implicit"} or _blob_has(blob, BOOKMARK_TERMS)
        if not has_wishlist:
            continue
        intent_hit = analysis.purchase_signal in {"purchased", "intend_to_purchase"} or _blob_has(
            blob, PURCHASE_INTENT_TERMS
        )
        bookmark_hit = _blob_has(blob, BOOKMARK_TERMS)
        if intent_hit and not bookmark_hit:
            buckets["Genuine Purchase Intent"].append(review.id)
        elif bookmark_hit and not intent_hit:
            buckets["Bookmarking / Save-for-later"].append(review.id)
        else:
            buckets["Unclear Evidence"].append(review.id)
    total = sum(len(v) for v in buckets.values())
    rows_out = [
        {
            "label": name,
            "count": len(ids),
            "percentage": pct(len(ids), total),
            "review_ids": ids[:50],
        }
        for name, ids in buckets.items()
        if ids
    ]
    return {
        "rows": rows_out,
        "total": total,
        "analyzed": len(rows),
        "limited": total == 0,
    }


def root_cause_hierarchy(
    db: Session, *, myntra_only: bool = True, since: datetime | None = None, source: str | None = None
) -> list[dict[str, Any]]:
    """Symptom → problem → root cause → behavior → impact → opportunity from stored analyses."""
    from collections import Counter as Ctr

    problems = problem_rows(db, myntra_only=myntra_only, since=since, source=source)
    out = []
    for item in problems:
        ids = item.get("review_ids") or []
        reviews = review_query(db, myntra_only=myntra_only, since=since, source=source).filter(Review.id.in_(ids)).all() if ids else []
        barriers: Ctr[str] = Ctr()
        uncertainties: Ctr[str] = Ctr()
        seeking: Ctr[str] = Ctr()
        hesitation = 0
        for review in reviews:
            analysis = review.analysis
            if not analysis or not analysis.is_valid_json:
                continue
            for label in normalize_label_list(_loads(analysis.barriers_json), keep_uncategorized_if_only_missing=False):
                barriers[label] += 1
            for label in normalize_label_list(
                _loads(analysis.uncertainties_json), keep_uncategorized_if_only_missing=False
            ):
                uncertainties[label] += 1
            for raw in _loads(analysis.information_seeking_json):
                if isinstance(raw, dict) and raw.get("source"):
                    key = normalize_label(raw.get("source"))
                    if key:
                        seeking[key] += 1
            if analysis.purchase_hesitation in {"explicit", "implicit"}:
                hesitation += 1
        top_barrier = barriers.most_common(1)[0][0] if barriers else ""
        top_unc = uncertainties.most_common(1)[0][0] if uncertainties else ""
        top_seek = seeking.most_common(1)[0][0] if seeking else ""
        if hesitation and item["frequency"]:
            symptom = "Purchase hesitation mentioned in supporting reviews"
        elif top_unc:
            symptom = f"Remaining uncertainty recorded: {top_unc}"
        else:
            symptom = "Named user problem in public reviews"
        if top_seek:
            behavior = f"Information seeking ({top_seek})"
        elif top_barrier:
            behavior = f"Barrier mentioned: {top_barrier}"
        elif top_unc:
            behavior = f"Users still asking about: {top_unc}"
        else:
            behavior = "No additional behavioral label extracted"
        impact = item["purchase_impact"]
        if impact >= 4:
            business = "High purchase-impact signal in this sample"
        elif impact >= 3:
            business = "Moderate purchase-impact signal in this sample"
        else:
            business = "Lower purchase-impact signal in this sample — inspect evidence before generalizing"
        opportunity = f"Investigate decision confidence around: {item['problem']}"
        out.append(
            {
                "root_cause": item["problem"],
                "symptom": symptom,
                "problem": item["problem"],
                "behavior": behavior,
                "business_impact": business,
                "opportunity": opportunity,
                "count": item["frequency"],
                "percentage": item["percentage"],
                "severity": item["severity"],
                "purchase_impact": item["purchase_impact"],
                "confidence": item.get("confidence"),
                "review_ids": ids,
                "top_barrier": top_barrier,
                "top_uncertainty": top_unc,
            }
        )
    return out
