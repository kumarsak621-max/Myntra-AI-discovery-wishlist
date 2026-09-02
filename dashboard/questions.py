"""Evidence-grounded answers for the ten discovery questions. Never invents quotes."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from dashboard.chat import ask_product_assistant, retrieve_evidence
from dashboard.insights import pm_insight

DISCOVERY_QUESTIONS = [
    "Why do users add fashion products to their wishlist?",
    "What prevents wishlisted products from eventually being purchased?",
    "What uncertainties remain after users have identified a product they like?",
    "What causes users to postpone a purchase?",
    "How do users compare multiple shortlisted products?",
    "What information do users seek outside Myntra/AJIO before purchasing?",
    "What role do fit, size, styling, price, reviews, occasion and social validation play?",
    "When do users use the wishlist as genuine purchase intent versus simply as a bookmarking mechanism?",
    "How do these behaviors differ across user segments?",
    "What unmet needs emerge consistently across user conversations?",
]

INSUFFICIENT = "Insufficient direct evidence in the collected public reviews."


def _rows_summary(rows: list[dict[str, Any]] | None, *, name="label", count="count") -> str:
    items = [r for r in (rows or []) if (r.get(count) or r.get("frequency") or 0)]
    if not items:
        return INSUFFICIENT
    parts = []
    for row in items[:5]:
        label = row.get(name) or row.get("problem") or row.get("label")
        n = row.get(count) or row.get("frequency") or 0
        pct = row.get("percentage")
        extra = f" ({pct}%)" if pct is not None else ""
        parts.append(f"{label}: {n}{extra}")
    return "; ".join(parts)


def _ids(rows: list[dict[str, Any]] | None) -> list[int]:
    out: list[int] = []
    for row in rows or []:
        for rid in row.get("review_ids") or row.get("evidence_ids") or []:
            try:
                out.append(int(rid))
            except (TypeError, ValueError):
                continue
    return list(dict.fromkeys(out))


def _pack(
    *,
    question: str,
    answer: str,
    rows: list[dict[str, Any]] | None,
    analyzed: int,
    themes: list[str],
    extra_ids: list[int] | None = None,
    confidence: Any = None,
) -> dict[str, Any]:
    ids = list(dict.fromkeys((extra_ids or []) + _ids(rows)))
    count = len(ids) or sum(int(r.get("count") or r.get("frequency") or 0) for r in (rows or []))
    if analyzed <= 0 or (not rows and not ids):
        return {
            "question": question,
            "answer": INSUFFICIENT,
            "evidence_summary": "No analyzed records matched this question.",
            "evidence_count": 0,
            "themes": [],
            "review_ids": [],
            "confidence": None,
        }
    return {
        "question": question,
        "answer": answer,
        "evidence_summary": _rows_summary(rows),
        "evidence_count": count,
        "themes": themes[:8],
        "review_ids": ids[:30],
        "confidence": confidence,
    }


def answer_discovery_questions(db: Session, data: dict[str, Any], *, analyzed: int) -> list[dict[str, Any]]:
    """One structured card per required question, using stored aggregates + retrieval."""
    theme_names = [t.get("name") for t in (data.get("themes") or []) if t.get("name")]
    wishlist = data.get("wishlist_beh") or data.get("intents") or []
    barriers = data.get("barriers") or data.get("barrier_tax") or []
    uncertainties = data.get("uncertainties") or data.get("unc_tax") or []
    compare = data.get("compare") or []
    compare_how = data.get("compare_how") or []
    external = data.get("external") or []
    social = data.get("social") or []
    segments = data.get("segments") or []
    problems = data.get("problems") or []
    intent_split = (data.get("wishlist_intent") or {}).get("rows") or []
    role_rows = []
    for bucket in (uncertainties, barriers, social, wishlist):
        role_rows.extend(bucket)

    specs = [
        (
            DISCOVERY_QUESTIONS[0],
            pm_insight(topic="wishlist behavior", rows=wishlist, analyzed=analyzed),
            wishlist,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[1],
            pm_insight(topic="purchase barrier", rows=barriers, analyzed=analyzed),
            barriers,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[2],
            pm_insight(topic="purchase uncertainty", rows=uncertainties, analyzed=analyzed),
            uncertainties,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[3],
            pm_insight(
                topic="purchase postponement",
                rows=barriers or uncertainties,
                analyzed=analyzed,
                extra="Postponement here is inferred from hesitation and barrier labels in public reviews, not checkout events.",
            ),
            barriers or uncertainties,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[4],
            pm_insight(topic="comparison factor", rows=compare or compare_how, analyzed=analyzed),
            compare or compare_how,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[5],
            pm_insight(topic="external information seeking", rows=external, analyzed=analyzed),
            external,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[6],
            pm_insight(
                topic="decision factor (fit, size, styling, price, reviews, occasion, social validation)",
                rows=role_rows,
                analyzed=analyzed,
            ),
            role_rows,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[7],
            pm_insight(
                topic="wishlist intent vs bookmarking",
                rows=intent_split,
                analyzed=analyzed,
                extra="Public reviews contain limited direct wishlist events and cannot calculate conversion.",
            ),
            intent_split,
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[8],
            pm_insight(
                topic="behavioral segment",
                rows=[{"label": s.get("name"), "count": s.get("review_count")} for s in segments],
                analyzed=analyzed,
            ),
            [{"label": s.get("name"), "count": s.get("review_count"), "review_ids": s.get("evidence_ids")} for s in segments],
            theme_names,
        ),
        (
            DISCOVERY_QUESTIONS[9],
            pm_insight(topic="unmet need / user problem", rows=problems, analyzed=analyzed),
            problems,
            theme_names,
        ),
    ]

    cards = []
    for question, answer, rows, themes in specs:
        retrieved = retrieve_evidence(db, question, limit=5)
        extra_ids = [r["id"] for r in retrieved.get("reviews") or []]
        confidences = [p.get("confidence") for p in (problems if rows is problems else []) if p.get("confidence")]
        confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
        if analyzed <= 0:
            packed = _pack(question=question, answer=INSUFFICIENT, rows=[], analyzed=0, themes=[])
        else:
            packed = _pack(
                question=question,
                answer=answer,
                rows=rows,
                analyzed=analyzed,
                themes=themes,
                extra_ids=extra_ids,
                confidence=confidence,
            )
            if packed["evidence_count"] == 0 and extra_ids:
                assistant = ask_product_assistant(db, question, analyzed=analyzed)
                packed["answer"] = assistant.get("answer") or INSUFFICIENT
                packed["evidence_count"] = assistant.get("supporting_review_count") or len(extra_ids)
                packed["review_ids"] = extra_ids[:30]
        cards.append(packed)
    return cards
