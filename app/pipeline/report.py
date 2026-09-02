"""Programmatic discovery report. Quotes come only from stored reviews."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.models import Opportunity, Review, Segment, Source, Theme
from app.pipeline.labels import merge_category_rows
from app.pipeline.quantification import (
    information_seeking,
    label_distribution,
    overview_metrics,
    signal_counts,
    time_trends,
)


def _loads(raw: str) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def evidence_cards(db: Session, review_ids: list[int], limit: int = 5) -> list[dict[str, Any]]:
    if not review_ids:
        return []
    rows = db.query(Review).filter(Review.id.in_(review_ids[:40])).all()
    by_id = {r.id: r for r in rows}
    cards = []
    for rid in review_ids:
        review = by_id.get(rid)
        if review is None:
            continue
        text = (review.text or "").strip()
        if not text:
            continue
        quote = text if len(text) <= 280 else text[:277] + "..."
        cards.append(
            {
                "review_id": review.id,
                "quote": quote,
                "source": review.source,
                "source_url": review.source_url,
                "source_review_id": review.source_review_id,
                "date": review.review_date.isoformat() if review.review_date else None,
                "rating": review.rating,
                "app_name": review.app_name,
                "app_id": review.app_id,
                "region": review.region,
                "is_valid_source": review.is_valid_source,
                "data_classification": review.data_classification,
                "is_synthetic": review.is_synthetic,
            }
        )
        if len(cards) >= limit:
            break
    return cards


def build_report(db: Session) -> dict[str, Any]:
    overview = overview_metrics(db)
    sources = db.query(Source).all()
    opportunities = db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
    themes = db.query(Theme).order_by(Theme.review_count.desc()).all()
    segments = db.query(Segment).order_by(Segment.review_count.desc()).all()
    myntra_signals = signal_counts(db, myntra_only=True)
    all_signals = signal_counts(db, myntra_only=False)

    source_block = []
    for src in sources:
        source_block.append(
            {
                "platform": src.platform,
                "app_id": src.app_id,
                "detected_app_name": src.detected_app_name,
                "detected_developer": src.detected_developer,
                "region": src.region,
                "expected_app": src.expected_app,
                "validation_status": src.validation_status,
                "is_valid_for_myntra": src.is_valid_for_myntra,
                "warning": src.warning,
                "collection_date": src.collection_date.isoformat() if src.collection_date else None,
                "review_count": src.review_count,
            }
        )

    myntra_ok = overview["myntra_reviews"] > 0
    quality_notes = []
    if overview["reference_non_myntra_reviews"]:
        quality_notes.append(
            "Some collected reviews are REFERENCE / NON-MYNTRA DATA (for example a Google Play "
            "package that resolves to Blinkit/Grofers). They must not be presented as Myntra evidence."
        )
    if not myntra_ok:
        quality_notes.append(
            "No reviews currently pass Myntra source validation. Myntra-specific conclusions cannot be drawn."
        )
    if overview["synthetic_count"]:
        quality_notes.append(
            "SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA is present and must stay labelled."
        )

    top = []
    for opp in opportunities[:5]:
        top.append(
            {
                "rank": opp.rank,
                "opportunity": opp.name,
                "user_problem": opp.user_problem,
                "affected_users": {
                    "relevant_count": opp.relevant_count,
                    "myntra_only_count": opp.myntra_only_count,
                    "percentage_of_relevant": opp.percentage,
                    "denominator": opp.total_relevant,
                },
                "reach": opp.reach,
                "frequency": opp.frequency,
                "severity": opp.severity,
                "purchase_impact": opp.purchase_impact,
                "evidence_confidence": opp.evidence_confidence,
                "opportunity_score": opp.score,
                "score_formula": "reach × frequency × purchase_impact × severity × evidence_confidence",
                "supporting_sources": _loads(opp.sources_json),
                "cross_source_status": opp.cross_source_status,
                "includes_non_myntra": opp.includes_non_myntra,
                "evidence": evidence_cards(db, _loads(opp.evidence_ids_json), limit=5),
                "what_we_know": opp.what_we_know,
                "what_we_dont_know": opp.what_we_dont_know,
                "why_it_deserves_investigation": opp.why_investigate,
            }
        )

    primary = top[0] if top else None

    known = [
        "Public app-store reviews can be collected automatically from configured sources.",
        "Source identity is validated before any row is treated as Myntra evidence.",
    ]
    unknown = [
        "We cannot observe actual wishlist-add timestamps or 30-day purchase conversion from public reviews.",
        "App reviews over-represent extreme satisfaction and dissatisfaction.",
        "Users who quietly abandon a wishlist without reviewing the app are invisible here.",
        "Demographic attributes are not inferred without explicit evidence.",
    ]
    if not myntra_ok:
        unknown.append(
            "Until a verified Myntra Google Play package ID is configured, Play Store rows remain reference data."
        )

    return {
        "title": "Wishlist-to-Purchase Discovery Report",
        "business_goal": (
            "Increase the percentage of users who purchase at least one item from their wishlist "
            "within 30 days of adding it."
        ),
        "research_question": (
            "Why do users add products to their wishlist but fail to purchase them within 30 days?"
        ),
        "anti_solution_note": (
            "This report does not propose discounts, notifications, AI recommendations, or other features. "
            "It identifies problems that deserve further research."
        ),
        "sections": {
            "1_business_goal": {
                "goal": "Increase 30-day wishlist-to-purchase conversion.",
                "role_framing": "Growth / product discovery, not solution design.",
            },
            "2_data_collection_method": {
                "primary": "Automatic collection from public Google Play reviews (google-play-scraper) and Apple App Store iTunes RSS.",
                "secondary": "Optional CSV/JSON upload fallback.",
                "india_first": "Apple collector requests the Indian region first and falls back to US only when no written reviews are returned.",
            },
            "3_data_sources": source_block,
            "4_data_volume": overview,
            "5_data_quality": quality_notes,
            "6_wishlist_motivations": label_distribution(db, "intent", myntra_only=True),
            "7_purchase_barriers": label_distribution(db, "barriers", myntra_only=True),
            "8_uncertainties": label_distribution(db, "uncertainties", myntra_only=True),
            "9_decision_making_behavior": {
                "myntra_valid": myntra_signals,
                "all_sources_including_reference": all_signals,
            },
            "10_external_information_seeking": information_seeking(db, myntra_only=True),
            "11_user_segments": [
                {
                    "name": s.name,
                    "description": s.description,
                    "basis": s.basis,
                    "review_count": s.review_count,
                    "myntra_review_count": s.myntra_review_count,
                    "sources": _loads(s.sources_json),
                }
                for s in segments
            ],
            "12_category_differences": label_distribution(db, "product_category", myntra_only=False),
            "13_emergent_themes": merge_category_rows(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "review_count": t.review_count,
                        "myntra_review_count": t.myntra_review_count,
                        "reference_review_count": t.reference_review_count,
                        "sources": _loads(t.sources_json),
                        "is_emergent": t.is_emergent,
                        "evidence_ids": _loads(t.evidence_ids_json),
                    }
                    for t in themes
                ],
                label_keys=("name", "label"),
                count_keys=("review_count", "count"),
                id_keys=("evidence_ids", "review_ids"),
            ),
            "14_root_causes": [
                {
                    "name": o.user_problem,
                    "score": o.score,
                    "cross_source_status": o.cross_source_status,
                }
                for o in opportunities[:15]
            ],
            "15_opportunity_scoring": {
                "formula": "Opportunity Score = Reach × Frequency × Purchase Impact × Severity × Evidence Confidence (each 1–5)",
                "range": "1 to 3125",
                "ranked": top,
            },
            "16_cross_source_validation": {
                "note": (
                    "A problem that appears frequently in one source is not claimed as universal. "
                    "Blinkit/Grofers Play reviews cannot corroborate Myntra App Store findings."
                ),
                "items": [
                    {
                        "name": o.name,
                        "sources": _loads(o.sources_json),
                        "status": o.cross_source_status,
                        "includes_non_myntra": o.includes_non_myntra,
                    }
                    for o in opportunities[:20]
                ],
            },
            "17_strongest_evidence": top[:3],
            "18_contradictory_evidence": _contradictions(opportunities),
            "19_what_we_know": known,
            "20_what_we_dont_know": unknown,
            "21_recommended_next_research_steps": [
                "Confirm Myntra's verified Google Play package ID and re-collect Play reviews.",
                "Add public Reddit / YouTube collectors only after source identity rules are in place.",
                "Interview users who recently added to wishlist but did not purchase within 30 days.",
                "Instrument in-product wishlist events (add, revisit, size-chart open, purchase) before designing a solution.",
            ],
        },
        "top_opportunities": top,
        "single_most_promising_problem": {
            "item": primary,
            "why_first": (
                primary["why_it_deserves_investigation"]
                + " It ranks first because it has the highest deterministic opportunity score "
                f"({primary['opportunity_score']}) given current evidence."
                if primary
                else "No scored opportunities yet. Collect and analyze reviews first."
            ),
        },
        "time_trends": time_trends(db, myntra_only=False),
    }


def _contradictions(opportunities: list[Opportunity]) -> list[str]:
    notes = []
    sourcesets = [tuple(sorted(_loads(o.sources_json))) for o in opportunities[:10]]
    if opportunities:
        myntra_hits = [o for o in opportunities if o.myntra_only_count > 0]
        ref_only = [o for o in opportunities if o.myntra_only_count == 0 and o.includes_non_myntra]
        if ref_only and myntra_hits:
            notes.append(
                "Some high-frequency problems appear only in non-Myntra reference data and may not apply to Myntra."
            )
        if any(o.cross_source_status == "single-source" for o in opportunities[:5]):
            notes.append(
                "One or more top problems are currently single-source and should not be treated as universal."
            )
    if not notes:
        notes.append("No strong contradictions have been quantified yet; absence of conflict is not confirmation.")
    return notes
