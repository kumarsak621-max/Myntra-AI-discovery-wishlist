"""Evidence-grounded answers for the ten discovery questions. Never invents quotes."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from dashboard.chat import ask_product_assistant, retrieve_evidence
from dashboard.insights import derive_root_cause, pm_insight
from app.pipeline.labels import normalize_label

DISCOVERY_QUESTIONS = [
    "Why do users add fashion products to wishlist?",
    "What prevents wishlisted products from being purchased?",
    "What uncertainties remain after users identify a product?",
    "What causes users to postpone a purchase?",
    "How do users compare shortlisted products?",
    "What information do users seek outside Myntra/AJIO before purchasing?",
    "What role do fit, size, styling, price, reviews, occasion and social validation play?",
    "Which wishlist behaviors indicate genuine purchase intent?",
    "How do behaviors differ across user segments?",
    "What unmet needs emerge consistently?",
    "What is the root cause of purchase hesitation?",
]

INSUFFICIENT = "Insufficient direct evidence in the collected public reviews."


def _rows_summary(rows: list[dict[str, Any]] | None, *, name="label", count="count") -> str:
    items = []
    for r in rows or []:
        label = normalize_label(r.get(name) or r.get("problem") or r.get("label"))
        n = r.get(count) or r.get("frequency") or 0
        if not label or not n:
            continue
        items.append((label, n, r.get("percentage")))
    if not items:
        return INSUFFICIENT
    parts = []
    for label, n, pct in items[:5]:
        extra = f" ({pct}%)" if pct is not None else ""
        parts.append(f"{label}: {n}{extra}")
    return "; ".join(parts)


def _ids(rows: list[dict[str, Any]] | None) -> list[int]:
    out: list[int] = []
    for row in rows or []:
        if not normalize_label(row.get("label") or row.get("problem") or row.get("name")):
            continue
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
    caveat: str = "",
) -> dict[str, Any]:
    ids = list(dict.fromkeys((extra_ids or []) + _ids(rows)))
    count = len(ids) or sum(int(r.get("count") or r.get("frequency") or 0) for r in (rows or []))
    themes = [t for t in themes if normalize_label(t)]
    if analyzed <= 0 or (not rows and not ids):
        return {
            "question": question,
            "answer": INSUFFICIENT,
            "evidence_summary": "No analyzed records matched this question.",
            "evidence_count": 0,
            "themes": [],
            "review_ids": [],
            "confidence": None,
            "caveat": "Public reviews are proxy evidence only.",
        }
    return {
        "question": question,
        "answer": answer,
        "evidence_summary": _rows_summary(rows),
        "evidence_count": count,
        "themes": themes[:8],
        "review_ids": ids[:30],
        "confidence": confidence,
        "caveat": caveat
        or "Public reviews do not measure actual wishlist-to-purchase conversion.",
    }


def answer_discovery_questions(db: Session, data: dict[str, Any], *, analyzed: int) -> list[dict[str, Any]]:
    """One structured card per required question, using stored aggregates + retrieval."""
    theme_names = [normalize_label(t.get("name")) for t in (data.get("themes") or [])]
    theme_names = [t for t in theme_names if t]
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
        (DISCOVERY_QUESTIONS[0], pm_insight(topic="wishlist behavior", rows=wishlist, analyzed=analyzed), wishlist),
        (DISCOVERY_QUESTIONS[1], pm_insight(topic="purchase barrier", rows=barriers, analyzed=analyzed), barriers),
        (DISCOVERY_QUESTIONS[2], pm_insight(topic="purchase uncertainty", rows=uncertainties, analyzed=analyzed), uncertainties),
        (
            DISCOVERY_QUESTIONS[3],
            pm_insight(
                topic="purchase postponement",
                rows=barriers or uncertainties,
                analyzed=analyzed,
                extra="Postponement here is inferred from hesitation and barrier labels in public reviews, not checkout events.",
            ),
            barriers or uncertainties,
        ),
        (DISCOVERY_QUESTIONS[4], pm_insight(topic="comparison factor", rows=compare or compare_how, analyzed=analyzed), compare or compare_how),
        (DISCOVERY_QUESTIONS[5], pm_insight(topic="external information seeking", rows=external, analyzed=analyzed), external),
        (
            DISCOVERY_QUESTIONS[6],
            pm_insight(
                topic="decision factor (fit, size, styling, price, reviews, occasion, social validation)",
                rows=role_rows,
                analyzed=analyzed,
            ),
            role_rows,
        ),
        (
            DISCOVERY_QUESTIONS[7],
            pm_insight(
                topic="wishlist intent vs bookmarking",
                rows=intent_split or wishlist,
                analyzed=analyzed,
                extra="Public reviews contain limited direct wishlist events and cannot calculate conversion.",
            ),
            intent_split or wishlist,
        ),
        (
            DISCOVERY_QUESTIONS[8],
            pm_insight(
                topic="behavioral segment",
                rows=[{"label": s.get("name"), "count": s.get("review_count")} for s in segments],
                analyzed=analyzed,
            ),
            [{"label": s.get("name"), "count": s.get("review_count"), "review_ids": s.get("evidence_ids")} for s in segments],
        ),
        (DISCOVERY_QUESTIONS[9], pm_insight(topic="unmet need / user problem", rows=problems, analyzed=analyzed), problems),
        (
            DISCOVERY_QUESTIONS[10],
            derive_root_cause(
                analyzed=analyzed,
                problems=problems,
                barriers=barriers,
                uncertainties=uncertainties,
                wishlist=wishlist,
                hesitation_count=int((data.get("signals") or {}).get("purchase_hesitation") or 0),
            )["statement"],
            problems or barriers or uncertainties or wishlist,
        ),
    ]

    cards = []
    for question, answer, rows in specs:
        retrieved = retrieve_evidence(db, question, limit=5)
        extra_ids = [r["id"] for r in retrieved.get("reviews") or []]
        if analyzed <= 0:
            packed = _pack(question=question, answer=INSUFFICIENT, rows=[], analyzed=0, themes=[])
        else:
            packed = _pack(
                question=question,
                answer=answer,
                rows=rows,
                analyzed=analyzed,
                themes=theme_names,
                extra_ids=extra_ids,
            )
            if packed["evidence_count"] == 0 and extra_ids:
                assistant = ask_product_assistant(db, question, analyzed=analyzed)
                packed["answer"] = assistant.get("answer") or INSUFFICIENT
                packed["evidence_count"] = assistant.get("supporting_review_count") or len(extra_ids)
                packed["review_ids"] = extra_ids[:30]
                packed["caveat"] = assistant.get("caveat") or packed["caveat"]
        cards.append(packed)
    return cards
