"""Build scored opportunities from barrier/uncertainty/root-cause groups."""

from __future__ import annotations

import json
from collections import defaultdict

from sqlalchemy.orm import Session

from app.models import Opportunity, Review, utcnow
from app.pipeline.labels import normalize_label
from app.pipeline.quantification import label_distribution, pct
from app.pipeline.scoring import (
    evidence_confidence_from_sources,
    frequency_from_share,
    opportunity_score,
    purchase_impact_from_hesitation,
    reach_from_volume,
    severity_from_strength,
)


def _loads(raw: str) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def rebuild_opportunities(db: Session) -> list[Opportunity]:
    db.query(Opportunity).delete()
    # Score primarily on Myntra-valid evidence; still compute a reference view.
    myntra_barriers = label_distribution(db, "barriers", myntra_only=True, relevant_only=True)
    all_barriers = label_distribution(db, "barriers", myntra_only=False, relevant_only=True)
    myntra_unc = label_distribution(db, "uncertainties", myntra_only=True, relevant_only=True)

    by_id = {r.id: r for r in db.query(Review).all()}

    candidates = []
    if myntra_barriers:
        for item in myntra_barriers:
            candidates.append(("barrier", item, True))
    else:
        # No Myntra-valid relevant barriers — keep reference candidates clearly flagged
        for item in all_barriers:
            candidates.append(("barrier", item, False))
    for item in myntra_unc:
        candidates.append(("uncertainty", item, True))

    built: list[Opportunity] = []
    seen_keys: set[str] = set()
    for kind, item, is_myntra_primary in candidates:
        key = f"{kind}:{(normalize_label(item.get('label')) or '').lower()}"
        if not normalize_label(item.get("label")):
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        reviews = [by_id[i] for i in item["review_ids"] if i in by_id]
        strengths = [
            r.analysis.evidence_strength
            for r in reviews
            if r.analysis and r.analysis.is_valid_json
        ]
        ratings = [r.rating for r in reviews]
        sources = item["sources"]
        has_myntra = any(r.is_valid_source for r in reviews)
        only_non = bool(reviews) and not has_myntra
        reach = reach_from_volume(item["count"], item["source_count"])
        frequency = frequency_from_share(item["count"], item["denominator"])
        impact = purchase_impact_from_hesitation(item["hesitant_count"], item["count"])
        severity = severity_from_strength(strengths, ratings)
        confidence = evidence_confidence_from_sources(
            item["source_count"], item["count"], has_myntra, only_non
        )
        score = opportunity_score(reach, frequency, impact, severity, confidence)

        if item["source_count"] >= 2 and has_myntra:
            cross = "cross-source"
        elif item["source_count"] >= 2 and only_non:
            cross = "cross-source-but-non-myntra"
        elif only_non:
            cross = "non-myntra-only"
        else:
            cross = "single-source"

        myntra_count = sum(1 for r in reviews if r.is_valid_source)
        know = (
            f"{item['count']} relevant reviews mention this {kind} "
            f"({item['percentage']}% of {item['denominator']} relevant analyzed reviews). "
            f"Sources: {', '.join(sources) or 'none'}."
        )
        dont = (
            "App-store reviews are not a complete picture of in-app wishlist behavior. "
            "We cannot observe actual 30-day conversion from public reviews. "
        )
        if only_non:
            dont += "This group is REFERENCE / NON-MYNTRA DATA and must not be treated as Myntra evidence. "
        if item["source_count"] < 2:
            dont += "Finding is not yet corroborated across independent sources."

        why = (
            "This pattern is linked to purchase hesitation in "
            f"{item['hesitant_count']} of {item['count']} grouped reviews "
            f"({pct(item['hesitant_count'], item['count'])}%). "
            "It deserves further investigation before any solution is proposed."
        )

        row = Opportunity(
            name=item["label"][:255],
            user_problem=item["label"],
            problem_key=key,
            reach=reach,
            frequency=frequency,
            purchase_impact=impact,
            severity=severity,
            evidence_confidence=confidence,
            score=score,
            relevant_count=item["count"],
            total_relevant=item["denominator"],
            percentage=item["percentage"],
            sources_json=json.dumps(sources),
            evidence_ids_json=json.dumps(item["review_ids"]),
            quote_ids_json=json.dumps(item["review_ids"][:8]),
            what_we_know=know,
            what_we_dont_know=dont,
            why_investigate=why,
            cross_source_status=cross,
            includes_non_myntra=any(not r.is_valid_source for r in reviews),
            myntra_only_count=myntra_count,
            updated_at=utcnow(),
        )
        built.append(row)

    built.sort(key=lambda o: (-o.score, -o.relevant_count, o.name.lower()))
    for index, row in enumerate(built, start=1):
        row.rank = index
        db.add(row)
    db.commit()
    return built
