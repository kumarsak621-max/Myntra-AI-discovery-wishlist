"""Product-manager interpretations generated from stored aggregates only."""

from __future__ import annotations

from typing import Any

from app.pipeline.labels import normalize_label


def _n(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _valid_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for row in rows or []:
        if _n(row.get("count") or row.get("frequency") or row.get("supporting_reviews") or row.get("relevant_count") or row.get("review_count")) <= 0:
            continue
        name = normalize_label(
            row.get("label") or row.get("problem") or row.get("name") or row.get("root_cause")
        )
        if not name:
            continue
        item = dict(row)
        item["label"] = name
        out.append(item)
    return out


def pm_insight(
    *,
    topic: str,
    rows: list[dict[str, Any]] | None,
    analyzed: int,
    extra: str = "",
) -> str:
    """2–4 sentences from real counts. Returns an insufficiency note when empty."""
    rows = _valid_rows(rows)
    if analyzed <= 0:
        return (
            "There is insufficient analyzed evidence in the current review dataset "
            "to write a Product Manager interpretation for this section."
        )
    if not rows:
        return (
            f"Among {analyzed} analyzed reviews, no {topic} labels were extracted. "
            "That is a statement about this public-review sample, not about all Myntra users."
        )
    top = rows[0]
    name = str(top.get("label") or "this signal")
    count = _n(top.get("count") or top.get("frequency") or top.get("supporting_reviews") or top.get("relevant_count"))
    pct = top.get("percentage")
    pct_text = f" ({pct}% of the analyzed sample)" if pct is not None else ""
    second = rows[1] if len(rows) > 1 else None
    follow = ""
    if second:
        follow = (
            f" The next most common signal is {second.get('label')} "
            f"({_n(second.get('count') or second.get('frequency'))} reviews)."
        )
    extra = extra.strip() or (
        "This may affect decision confidence on the path from wishlist interest to purchase, "
        "but public reviews do not prove causality or the actual conversion rate."
    )
    return (
        f"{name} is the strongest {topic} in this sample, mentioned in {count} analyzed reviews{pct_text}. "
        f"{follow} {extra} "
        f"A Product Manager should inspect the supporting reviews before treating this as a company-wide pattern."
    ).strip()


def pm_insight_card(
    *,
    analyzed: int,
    problems: list[dict[str, Any]] | None,
    opportunities: list[dict[str, Any]] | None,
    evidence_count: int = 0,
    example: str = "",
    confidence: Any = None,
) -> dict[str, str]:
    """Structured PM Insight for the dedicated dashboard card."""
    problems = _valid_rows(problems)
    opportunities = _valid_rows(opportunities)
    if analyzed <= 0 or (not problems and not opportunities):
        return {
            "strongest_signal": "Insufficient evidence.",
            "why_it_matters": "There is not enough analyzed public-review evidence yet.",
            "evidence": "0 supporting reviews.",
            "confidence": "n/a",
            "caveat": "Public reviews are proxy evidence only and do not measure actual wishlist-to-purchase conversion.",
        }
    top = problems[0] if problems else opportunities[0]
    name = top.get("label") or top.get("name") or top.get("problem")
    count = _n(top.get("count") or top.get("frequency") or top.get("relevant_count") or evidence_count)
    score = top.get("score")
    why = (
        f"{name} is the strongest evidenced signal in this sample. "
        "It sits on the path from saving/shortlisting to purchase confidence. "
        + (f"Opportunity score {score} is computed in Python, not by the model. " if score is not None else "")
    )
    evidence = f"{count} supporting reviews in the analyzed sample of {analyzed}."
    if example:
        evidence += f" Example: “{example}”"
    conf = f"{confidence}/5" if confidence is not None else "Inspect supporting reviews; strength varies by quote."
    return {
        "strongest_signal": str(name),
        "why_it_matters": why.strip(),
        "evidence": evidence,
        "confidence": conf,
        "caveat": (
            "Public app reviews do not expose actual wishlist-add or 30-day purchase events. "
            "Do not treat this as the conversion rate."
        ),
    }


def derive_root_cause(
    *,
    analyzed: int,
    problems: list[dict[str, Any]] | None,
    barriers: list[dict[str, Any]] | None,
    uncertainties: list[dict[str, Any]] | None,
    wishlist: list[dict[str, Any]] | None,
    hesitation_count: int = 0,
) -> dict[str, Any]:
    """Connect behavior → problem → barrier/uncertainty → hesitation → business metric from evidence."""
    problems = _valid_rows(problems)
    barriers = _valid_rows(barriers)
    uncertainties = _valid_rows(uncertainties)
    wishlist = _valid_rows(wishlist)
    if analyzed <= 0:
        return {
            "supported": False,
            "statement": "Insufficient evidence to establish a reliable root cause.",
            "chain": {},
        }
    if not problems and not barriers and not uncertainties and not wishlist and hesitation_count <= 0:
        return {
            "supported": False,
            "statement": "Insufficient evidence to establish a reliable root cause.",
            "chain": {},
        }
    behavior = wishlist[0]["label"] if wishlist else "saving or shortlisting mentioned in public reviews"
    problem = problems[0]["label"] if problems else None
    friction = (barriers[0]["label"] if barriers else None) or (uncertainties[0]["label"] if uncertainties else None)
    parts = [
        f"Users show {behavior} in public reviews",
    ]
    if problem:
        parts.append(f"the strongest evidenced problem is {problem}")
    if friction:
        parts.append(f"unresolved {friction} appears as a barrier or uncertainty")
    if hesitation_count:
        parts.append(
            f"purchase hesitation is labelled in {hesitation_count} of {analyzed} analyzed reviews"
        )
    parts.append(
        "this is a proxy signal for delayed purchase confidence, not a measured 30-day wishlist conversion rate"
    )
    statement = "; ".join(parts) + "."
    statement = statement[0].upper() + statement[1:]
    return {
        "supported": True,
        "statement": statement,
        "chain": {
            "user_behavior": behavior,
            "problem": problem or "not established",
            "uncertainty_or_barrier": friction or "not established",
            "purchase_hesitation": hesitation_count,
            "business_metric": "30-day wishlist-to-purchase conversion (not directly observed)",
        },
    }


def why_this_matters(row: dict[str, Any], *, analyzed: int) -> str:
    """Grounded 'why this matters' copy for a single root-cause row."""
    if analyzed <= 0:
        return "Insufficient evidence to establish a reliable root cause."
    name = normalize_label(row.get("root_cause") or row.get("problem")) or "this cause"
    count = _n(row.get("count") or row.get("frequency"))
    impact = row.get("purchase_impact")
    behavior = row.get("behavior") or ""
    return (
        f"{name} is linked to {count} analyzed reviews in this sample"
        f"{f' with purchase-impact {impact}/5' if impact is not None else ''}. "
        f"{behavior + '. ' if behavior else ''}"
        "This is a discovery signal from public reviews, not proof of the actual conversion rate. "
        "A Product Manager should read the supporting reviews before prioritizing work."
    )


def wishlist_conversion_copy(signals: dict[str, Any]) -> str:
    analyzed = _n(signals.get("analyzed") or signals.get("denominator"))
    if analyzed <= 0:
        return (
            "Wishlist → purchase conversion cannot be computed from public reviews. "
            "No analyzed reviews are available yet."
        )
    return (
        f"This is an evidence-based opportunity indicator from {analyzed} analyzed public reviews, "
        "not Myntra's actual wishlist-to-purchase conversion rate. "
        f"Wishlist-related language appears in {signals.get('wishlist_signal', 0)} reviews "
        f"({signals.get('wishlist_pct', 0)}%). "
        f"Purchase hesitation appears in {signals.get('purchase_hesitation', 0)} reviews "
        f"({signals.get('hesitation_pct', 0)}%). "
        "Public store reviews do not include in-app wishlist or checkout events."
    )


def funnel_stages(
    *,
    analyzed: int,
    wishlist: int,
    intent: int,
    hesitation: int,
    barriers: int,
    uncertainties: int,
    abandoned: int,
    comparison: int = 0,
) -> list[dict[str, Any]]:
    """Behavioral decomposition counts. Missing stages stay at 0 instead of being invented."""
    return [
        {"stage": "Wishlist / save", "count": wishlist, "note": "Reviews mentioning save/wishlist/later"},
        {"stage": "Consideration", "count": barriers, "note": "Named product evaluation barriers"},
        {"stage": "Uncertainty", "count": uncertainties, "note": "Extracted uncertainties"},
        {"stage": "Comparison", "count": comparison, "note": "Comparison behavior in this sample"},
        {"stage": "Hesitation", "count": hesitation, "note": "purchase_hesitation explicit/implicit"},
        {"stage": "Purchase decision", "count": intent, "note": "intend_to_purchase / purchased"},
        {
            "stage": "Purchase (proxy only)",
            "count": abandoned,
            "note": "Abandoned intent mentions — not an actual conversion rate",
        },
    ]
