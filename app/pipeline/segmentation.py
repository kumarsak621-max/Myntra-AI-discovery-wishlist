"""Behavioral segments derived only from evidence in analyses."""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from sqlalchemy.orm import Session

from app.models import Review, Segment, utcnow

logger = logging.getLogger(__name__)


def _loads(raw: str) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# Map evidence → segment. Applied only when the review actually contains the signal.
RULES = [
    (
        "Comparison-heavy users",
        "inferred",
        "Users whose reviews mention comparing products or alternatives.",
        lambda a, signals, intents: "comparison" in " ".join(signals + intents).lower()
        or any("compar" in i.lower() for i in intents),
    ),
    (
        "Delayed purchasers",
        "inferred",
        "Users describing waiting, postponing, or later purchase.",
        lambda a, signals, intents: a.purchase_signal in {"hesitant", "abandoned"}
        or any(s in {"delayed_purchase", "waiting", "revisit"} for s in signals),
    ),
    (
        "Occasion-driven shoppers",
        "inferred",
        "Users mentioning occasions, events, or outfit planning.",
        lambda a, signals, intents: any(
            k in " ".join(intents + _loads(a.decision_factors_json)).lower()
            for k in ("occasion", "wedding", "party", "festival", "outfit")
        ),
    ),
    (
        "Budget-conscious shoppers",
        "inferred",
        "Users who explicitly discuss price, value, or cost — not assumed.",
        lambda a, signals, intents: any(
            k in " ".join(intents + _loads(a.barriers_json) + _loads(a.decision_factors_json)).lower()
            for k in ("price", "expensive", "cheap", "value", "budget", "cost")
        ),
    ),
    (
        "Return-concerned shoppers",
        "inferred",
        "Users mentioning returns, exchanges, or difficulty sending items back.",
        lambda a, signals, intents: "return_concern" in signals
        or any("return" in b.lower() or "exchange" in b.lower() for b in _loads(a.barriers_json)),
    ),
        (
            "Wishlist / save mentioners",
            "inferred",
            "Users who mention wishlist, save, or bag as a holding pattern.",
        lambda a, signals, intents: a.wishlist_signal in {"explicit", "implicit"}
        or any("wishlist" in i.lower() or "save" in i.lower() for i in intents),
    ),
    (
        "Fit-sensitive shoppers",
        "inferred",
        "Users whose reviews mention size, fit, or measurement uncertainty.",
        lambda a, signals, intents: "size_checking" in signals
        or any(
            k in " ".join(intents + _loads(a.barriers_json) + _loads(a.uncertainties_json)).lower()
            for k in ("size", "fit", "measurement", "chart")
        ),
    ),
]


def discover_segments(db: Session) -> list[Segment]:
    reviews = (
        db.query(Review)
        .filter(
            Review.is_duplicate.is_(False),
            Review.is_empty.is_(False),
            Review.is_valid_source.is_(True),
        )
        .all()
    )
    db.query(Segment).delete()
    buckets: dict[str, dict] = {}
    for name, basis, description, matcher in RULES:
        buckets[name] = {
            "basis": basis,
            "description": description,
            "ids": [],
            "myntra": 0,
            "sources": set(),
        }

    for review in reviews:
        analysis = review.analysis
        if not analysis or not analysis.is_valid_json:
            continue
        signals = [
            str(item.get("signal") if isinstance(item, dict) else item)
            for item in _loads(analysis.behavioral_signals_json)
        ]
        intents = [str(x) for x in _loads(analysis.intent_json)]
        for name, basis, description, matcher in RULES:
            try:
                matched = bool(matcher(analysis, signals, intents))
            except Exception as exc:
                logger.warning("Segment matcher failed for %s: %s", name, exc)
                matched = False
            if not matched:
                continue
            bucket = buckets[name]
            bucket["ids"].append(review.id)
            bucket["sources"].add(review.source)
            if review.is_valid_source:
                bucket["myntra"] += 1

    created: list[Segment] = []
    for name, bucket in buckets.items():
        if not bucket["ids"]:
            continue
        row = Segment(
            name=name,
            description=bucket["description"],
            basis=bucket["basis"],
            review_count=len(bucket["ids"]),
            myntra_review_count=bucket["myntra"],
            evidence_ids_json=json.dumps(bucket["ids"][:80]),
            sources_json=json.dumps(sorted(bucket["sources"])),
            updated_at=utcnow(),
        )
        db.add(row)
        created.append(row)
    db.commit()
    return created
