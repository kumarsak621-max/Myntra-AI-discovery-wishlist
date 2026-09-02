"""Product-manager interpretations generated from stored aggregates only."""

from __future__ import annotations

from typing import Any


def _n(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def pm_insight(
    *,
    topic: str,
    rows: list[dict[str, Any]] | None,
    analyzed: int,
    extra: str = "",
) -> str:
    """2–4 sentences from real counts. Returns an insufficiency note when empty."""
    rows = [r for r in (rows or []) if _n(r.get("count") or r.get("frequency") or r.get("supporting_reviews"))]
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
    name = str(top.get("label") or top.get("problem") or top.get("name") or "this signal")
    count = _n(top.get("count") or top.get("frequency") or top.get("supporting_reviews") or top.get("relevant_count"))
    pct = top.get("percentage")
    pct_text = f" ({pct}% of the analyzed sample)" if pct is not None else ""
    second = rows[1] if len(rows) > 1 else None
    follow = ""
    if second:
        follow = (
            f" The next most common signal is {second.get('label') or second.get('problem')} "
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


def why_this_matters(row: dict[str, Any], *, analyzed: int) -> str:
    """Grounded 'why this matters' copy for a single root-cause row."""
    if analyzed <= 0:
        return "Insufficient evidence for reliable root-cause analysis."
    name = row.get("root_cause") or row.get("problem") or "this cause"
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
) -> list[dict[str, Any]]:
    """Behavioral decomposition counts. Missing stages stay at 0 instead of being invented."""
    return [
        {"stage": "Wishlist language", "count": wishlist, "note": "Reviews mentioning save/wishlist/later"},
        {"stage": "Purchase intent signals", "count": intent, "note": "intend_to_purchase / purchased"},
        {"stage": "Information uncertainty", "count": uncertainties, "note": "Extracted uncertainties"},
        {"stage": "Named purchase barriers", "count": barriers, "note": "Reviews with at least one extracted barrier"},
        {"stage": "Purchase hesitation", "count": hesitation, "note": "purchase_hesitation explicit/implicit"},
        {"stage": "Abandoned intent", "count": abandoned, "note": "purchase_signal = abandoned"},
    ]
