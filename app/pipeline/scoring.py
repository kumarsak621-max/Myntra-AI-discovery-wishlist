"""Deterministic opportunity scoring. The LLM never computes the final score."""

from __future__ import annotations

from collections.abc import Sequence


def clamp_1_5(value: int | float) -> int:
    return max(1, min(5, int(round(value))))


def opportunity_score(
    reach: int,
    frequency: int,
    purchase_impact: int,
    severity: int,
    evidence_confidence: int,
) -> int:
    return (
        clamp_1_5(reach)
        * clamp_1_5(frequency)
        * clamp_1_5(purchase_impact)
        * clamp_1_5(severity)
        * clamp_1_5(evidence_confidence)
    )


def frequency_from_share(count: int, denominator: int) -> int:
    if denominator <= 0 or count <= 0:
        return 1
    pct = (count / denominator) * 100
    if pct < 2:
        return 1
    if pct < 5:
        return 2
    if pct < 10:
        return 3
    if pct < 20:
        return 4
    return 5


def reach_from_volume(count: int, source_count: int) -> int:
    volume = 1
    if count >= 5:
        volume = 2
    if count >= 15:
        volume = 3
    if count >= 40:
        volume = 4
    if count >= 80:
        return 5
    return clamp_1_5(volume + max(0, source_count - 1))


def purchase_impact_from_hesitation(hesitant_count: int, group_count: int) -> int:
    if group_count <= 0:
        return 1
    share = hesitant_count / group_count
    if share < 0.15:
        return 1
    if share < 0.35:
        return 2
    if share < 0.55:
        return 3
    if share < 0.75:
        return 4
    return 5


def severity_from_strength(strengths: Sequence[int], rating_proxy: Sequence[int | None]) -> int:
    if strengths:
        avg = sum(strengths) / len(strengths)
    else:
        avg = 1
    low_ratings = [r for r in rating_proxy if r is not None and r <= 2]
    bump = 1 if len(low_ratings) >= max(1, len(rating_proxy) // 3) else 0
    return clamp_1_5(avg + bump)


def evidence_confidence_from_sources(
    source_count: int,
    count: int,
    has_myntra: bool,
    has_non_myntra_only: bool,
) -> int:
    """Independent sources raise confidence. One non-Myntra source cannot look universal."""
    if count <= 0:
        return 1
    if has_non_myntra_only:
        base = 2 if count >= 10 else 1
        return base
    score = 1
    if count >= 3:
        score = 2
    if count >= 10:
        score = 3
    if source_count >= 2 and has_myntra:
        score = max(score, 4)
    if source_count >= 3 and has_myntra and count >= 15:
        score = 5
    if has_myntra and source_count == 1 and count >= 8:
        score = max(score, 3)
    return clamp_1_5(score)
