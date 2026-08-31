"""Dynamic date windows. Never hard-code calendar dates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_last_30_days_cutoff(now: datetime | None = None, *, days: int = 30) -> datetime:
    """Return timezone-aware UTC cutoff: now minus `days` (default 30)."""
    current = ensure_aware(now) or utcnow()
    return current - timedelta(days=days)


def window_start(hours: int | None = None, days: int | None = None, now: datetime | None = None) -> datetime:
    current = ensure_aware(now) or utcnow()
    if hours is not None:
        return current - timedelta(hours=hours)
    if days is not None:
        return current - timedelta(days=days)
    return get_last_30_days_cutoff(current)


def _review_timestamp(item: Any, date_attr: str = "review_date") -> datetime | None:
    if isinstance(item, dict):
        value = item.get(date_attr)
    else:
        value = getattr(item, date_attr, None)
    return ensure_aware(value)


def filter_reviews_by_date(
    reviews: Iterable[Any],
    cutoff: datetime,
    *,
    date_attr: str = "review_date",
) -> list[Any]:
    """Keep reviews whose *review* timestamp is on or after cutoff.

    Collection time is ignored. Items without a review timestamp are excluded
    from a dated window (they remain in all-time storage).
    """
    threshold = ensure_aware(cutoff)
    if threshold is None:
        return list(reviews)
    kept: list[Any] = []
    for item in reviews:
        stamp = _review_timestamp(item, date_attr)
        if stamp is None:
            continue
        if stamp >= threshold:
            kept.append(item)
    return kept


def humanize_ago(moment: datetime | None, now: datetime | None = None) -> str:
    stamp = ensure_aware(moment)
    if stamp is None:
        return "never"
    current = ensure_aware(now) or utcnow()
    delta = current - stamp
    seconds = int(delta.total_seconds())
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''} ago"
