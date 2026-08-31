"""Deterministic quantification. Percentages are never produced by the LLM."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models import Analysis, Review
from config.settings import official_ids


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


def review_query(db: Session, *, myntra_only: bool = False):
    q = db.query(Review).filter(Review.is_duplicate.is_(False), Review.is_empty.is_(False))
    if myntra_only:
        q = q.filter(Review.is_valid_source.is_(True), Review.app_id.in_(list(official_ids())))
    return q


def overview_metrics(db: Session) -> dict[str, Any]:
    all_reviews = review_query(db).all()
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
    return {
        "total_reviews": len(all_reviews),
        "myntra_reviews": len(myntra),
        "reference_non_myntra_reviews": len(reference),
        "analyzed_reviews": len(analyzed),
        "relevant_reviews": len(relevant),
        "relevant_pct_of_analyzed": pct(len(relevant), len(analyzed)),
        "by_source": dict(by_source),
        "by_rating": dict(by_rating),
        "by_classification": dict(by_classification),
        "synthetic_count": sum(1 for r in all_reviews if r.is_synthetic),
    }


def label_distribution(
    db: Session,
    field: str,
    *,
    myntra_only: bool = False,
    relevant_only: bool = True,
) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only).all()
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


def signal_counts(db: Session, *, myntra_only: bool = False) -> dict[str, Any]:
    rows = review_query(db, myntra_only=myntra_only).all()
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


def time_trends(db: Session, *, myntra_only: bool = False) -> list[dict[str, Any]]:
    rows = review_query(db, myntra_only=myntra_only).all()
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
