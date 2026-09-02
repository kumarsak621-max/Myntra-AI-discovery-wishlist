"""Evidence-backed Product Manager assistant. Retrieves stored records; does not invent."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from app.models import Opportunity, Review, Segment, Theme
from app.pipeline.labels import merge_category_rows, normalize_category_label
from app.pipeline.quantification import _loads, problem_rows
from config.settings import official_ids

STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "do",
    "what", "why", "how", "when", "which", "who", "users", "user", "me", "show",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if t not in STOP}


def retrieve_evidence(db: Session, question: str, *, limit: int = 8) -> dict[str, Any]:
    tokens = _tokens(question)
    if not tokens:
        return {"reviews": [], "problems": [], "themes": [], "opportunities": [], "segments": []}

    rows = (
        db.query(Review)
        .filter(
            Review.is_empty.is_(False),
            Review.is_valid_source.is_(True),
            Review.app_id.in_(list(official_ids())),
        )
        .all()
    )
    scored: list[tuple[int, Review]] = []
    for review in rows:
        analysis = review.analysis
        blob = f"{review.title or ''} {review.text or ''}"
        if analysis and analysis.is_valid_json:
            blob += f" {analysis.root_cause or ''} {analysis.barriers_json} {analysis.uncertainties_json} {analysis.intent_json}"
        hay = blob.lower()
        score = sum(1 for token in tokens if token in hay)
        if score:
            scored.append((score, review))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    reviews = []
    for score, review in scored[:limit]:
        analysis = review.analysis
        reviews.append(
            {
                "id": review.id,
                "text": (review.text or "")[:400],
                "source": review.source,
                "rating": review.rating,
                "date": review.review_date.isoformat() if review.review_date else None,
                "region": review.region,
                "root_cause": getattr(analysis, "root_cause", "") if analysis else "",
                "barriers": _loads(getattr(analysis, "barriers_json", "[]")) if analysis else [],
                "score": score,
            }
        )

    q = question.lower()
    problems = [p for p in problem_rows(db, myntra_only=True) if any(t in (p.get("problem") or "").lower() for t in tokens)]
    themes = merge_category_rows(
        [
            {"name": t.name, "count": t.review_count, "evidence_ids": []}
            for t in db.query(Theme).order_by(Theme.review_count.desc()).all()
            if any(tok in normalize_category_label(t.name).lower() for tok in tokens) or "theme" in q
        ],
        label_keys=("name", "label"),
        count_keys=("count", "review_count"),
    )[:8]
    opportunities = merge_category_rows(
        [
            {"name": o.name, "score": o.score, "count": o.relevant_count}
            for o in db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
            if any(tok in normalize_category_label(o.name).lower() for tok in tokens) or "opportunit" in q
        ],
        label_keys=("name", "label"),
        count_keys=("count", "relevant_count"),
    )[:8]
    segments = merge_category_rows(
        [
            {"name": s.name, "count": s.review_count}
            for s in db.query(Segment).order_by(Segment.review_count.desc()).all()
            if any(tok in normalize_category_label(s.name).lower() for tok in tokens) or "segment" in q
        ],
        label_keys=("name", "label"),
        count_keys=("count", "review_count"),
    )[:8]
    if "opportunit" in q and not opportunities:
        opportunities = [
            {"name": o.name, "score": o.score, "count": o.relevant_count}
            for o in db.query(Opportunity).order_by(Opportunity.rank.asc()).limit(5).all()
        ]
    if ("problem" in q or "barrier" in q) and not problems:
        problems = problem_rows(db, myntra_only=True)[:5]
    return {
        "reviews": reviews,
        "problems": problems[:5],
        "themes": themes,
        "opportunities": opportunities,
        "segments": segments,
    }


def answer_from_evidence(question: str, evidence: dict[str, Any], *, analyzed: int) -> dict[str, Any]:
    reviews = evidence.get("reviews") or []
    problems = evidence.get("problems") or []
    opportunities = evidence.get("opportunities") or []
    themes = evidence.get("themes") or []
    if analyzed <= 0 and not reviews:
        return {
            "answer": "I don't have enough evidence in the collected review dataset to answer this reliably.",
            "evidence_summary": "No analyzed reviews are stored.",
            "supporting_review_count": 0,
            "themes": [],
            "review_ids": [],
            "confidence": None,
            "pm_implication": "Collect and analyze real reviews before drawing product conclusions.",
        }
    if not reviews and not problems and not opportunities:
        return {
            "answer": "I don't have enough evidence in the collected review dataset to answer this reliably.",
            "evidence_summary": f"{analyzed} reviews are analyzed, but none matched this question.",
            "supporting_review_count": 0,
            "themes": [t.get("name") for t in themes],
            "review_ids": [],
            "confidence": None,
            "pm_implication": "Try a more specific term that appears in stored barriers, problems, or review text.",
        }

    top_problem = problems[0]["problem"] if problems else None
    top_opp = opportunities[0]["name"] if opportunities else None
    pieces = []
    if top_problem:
        pieces.append(
            f"The strongest matching problem in stored analysis is “{top_problem}” "
            f"({problems[0].get('frequency')} supporting reviews)."
        )
    if top_opp:
        pieces.append(
            f"The highest-ranked matching opportunity is “{top_opp}” "
            f"(score {opportunities[0].get('score')}, {opportunities[0].get('count')} reviews)."
        )
    if reviews:
        pieces.append(
            f"{len(reviews)} stored reviews mention terms from the question; "
            "quotes below are copied from those records."
        )
    pieces.append(
        "Public app reviews do not contain Myntra’s actual wishlist-to-purchase conversion events."
    )
    return {
        "answer": " ".join(pieces),
        "evidence_summary": (
            f"Retrieved {len(reviews)} reviews, {len(problems)} problem groups, "
            f"{len(opportunities)} opportunities from the local database."
        ),
        "supporting_review_count": len(reviews),
        "themes": [t.get("name") for t in themes][:6],
        "review_ids": [r["id"] for r in reviews],
        "confidence": round(
            sum(1 for r in reviews if r.get("root_cause") or r.get("barriers")) / max(1, len(reviews)) * 5,
            2,
        ) if reviews else None,
        "pm_implication": (
            "Inspect the retrieved reviews before treating this as a company-wide pattern. "
            "This is discovery evidence, not a recommended feature."
        ),
        "quotes": [
            {
                "text": r["text"],
                "source": r["source"],
                "rating": r["rating"],
                "date": r["date"],
                "id": r["id"],
            }
            for r in reviews[:5]
            if r.get("text")
        ],
    }


def ask_product_assistant(db: Session, question: str, *, analyzed: int) -> dict[str, Any]:
    evidence = retrieve_evidence(db, question)
    return answer_from_evidence(question, evidence, analyzed=analyzed)
