"""Normalize missing/empty category labels before aggregation and charts.

AI outputs and clustering uniqueness used to produce:
none, none (2), none (3), none (4)
Those are the same missing-value bucket and must display as one Uncategorized row.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

UNCATEGORIZED = "Uncategorized"

_MISSING_EXACT = {
    "",
    "none",
    "null",
    "nil",
    "nan",
    "n/a",
    "na",
    "unknown",
    "undefined",
    "-",
    "--",
    ".",
}

_MISSING_SUFFIX = re.compile(
    r"^(none|null|n/?a|na|unknown|undefined|nil|nan)(\s*\(\d+\))?$",
    re.IGNORECASE,
)


def normalize_category_label(value: Any) -> str:
    """Map missing/placeholder labels to Uncategorized. Keep real category text."""
    if value is None:
        return UNCATEGORIZED
    if isinstance(value, float) and value != value:  # NaN
        return UNCATEGORIZED
    text = " ".join(str(value).split()).strip()
    if not text:
        return UNCATEGORIZED
    if text.lower() == UNCATEGORIZED.lower():
        return UNCATEGORIZED
    if text.lower() in _MISSING_EXACT:
        return UNCATEGORIZED
    if _MISSING_SUFFIX.fullmatch(text):
        return UNCATEGORIZED
    return text


def is_placeholder_label(value: Any) -> bool:
    """True for missing/null/none/none (N) values — not for a real 'Uncategorized' label."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    text = " ".join(str(value).split()).strip()
    if not text:
        return True
    if text.lower() == UNCATEGORIZED.lower():
        return False
    return normalize_category_label(text) == UNCATEGORIZED


def stored_category_text(value: Any) -> str:
    """Persist empty string for placeholders; keep meaningful labels."""
    if is_placeholder_label(value):
        return ""
    return " ".join(str(value).split()).strip()


def normalize_label_list(values: Any, *, keep_uncategorized_if_only_missing: bool = True) -> list[str]:
    """Normalize a list of labels. Dummy 'none' items are dropped when real labels exist."""
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    real: list[str] = []
    seen: set[str] = set()
    saw_placeholder = False
    for item in values:
        if is_placeholder_label(item):
            saw_placeholder = True
            continue
        label = normalize_category_label(item)
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        real.append(label)
    if real:
        return real
    if keep_uncategorized_if_only_missing and saw_placeholder:
        return [UNCATEGORIZED]
    return []


def validate_chart_categories(labels: list[Any] | None) -> list[str]:
    """Return leftover duplicate-missing labels that should not be rendered."""
    issues: list[str] = []
    for raw in labels or []:
        text = " ".join(str(raw or "").split()).strip()
        if not text:
            continue
        if text.lower() == UNCATEGORIZED.lower():
            continue
        if text.lower() in _MISSING_EXACT or _MISSING_SUFFIX.fullmatch(text):
            issues.append(text)
    return sorted(set(issues))


def merge_category_rows(
    rows: list[dict[str, Any]] | None,
    *,
    label_keys: tuple[str, ...] = ("label", "problem", "name", "root_cause", "theme", "segment", "barrier"),
    count_keys: tuple[str, ...] = ("count", "frequency", "review_count", "relevant_count"),
    id_keys: tuple[str, ...] = ("review_ids", "evidence_ids"),
) -> list[dict[str, Any]]:
    """Normalize labels then sum counts and union review ids. One Uncategorized row max."""
    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows or []:
        raw_label = ""
        for key in label_keys:
            if row.get(key) not in (None, ""):
                raw_label = row.get(key)
                break
        label = normalize_category_label(raw_label)
        count = 0
        for key in count_keys:
            if row.get(key) not in (None, ""):
                try:
                    count = int(row.get(key) or 0)
                    break
                except (TypeError, ValueError):
                    count = 0
        ids: list[int] = []
        for key in id_keys:
            for item in row.get(key) or []:
                try:
                    ids.append(int(item))
                except (TypeError, ValueError):
                    continue
        if label not in buckets:
            merged = dict(row)
            merged["label"] = label
            for key in label_keys:
                if key in row:
                    merged[key] = label
            merged["count"] = 0
            for key in count_keys:
                if key in merged:
                    try:
                        int(merged.get(key) or 0)
                    except (TypeError, ValueError):
                        continue
                    merged[key] = 0
            for key in ("first_half", "second_half", "hesitant_count"):
                if key in merged:
                    merged[key] = 0
            if "by_day" in merged:
                merged["by_day"] = {}
            merged["review_ids"] = []
            buckets[label] = merged
            order.append(label)
        bucket = buckets[label]
        bucket["count"] = int(bucket.get("count") or 0) + count
        for key in count_keys:
            if key == "count" or row.get(key) in (None, ""):
                continue
            try:
                extra = int(row.get(key) or 0)
            except (TypeError, ValueError):
                extra = 0
            if extra:
                bucket[key] = int(bucket.get(key) or 0) + extra
        existing_ids = list(bucket.get("review_ids") or [])
        bucket["review_ids"] = list(dict.fromkeys(existing_ids + ids))[:80]
        if "evidence_ids" in row or "evidence_ids" in bucket:
            bucket["evidence_ids"] = list(bucket["review_ids"])
        if row.get("hesitant_count"):
            bucket["hesitant_count"] = int(bucket.get("hesitant_count") or 0) + int(row.get("hesitant_count") or 0)
        if row.get("percentage") is not None and row.get("denominator") is not None:
            bucket["denominator"] = int(row.get("denominator") or bucket.get("denominator") or 0)
        if row.get("by_day"):
            days = bucket.setdefault("by_day", {})
            for day, value in (row.get("by_day") or {}).items():
                days[day] = int(days.get(day) or 0) + int(value or 0)
        if row.get("first_half") is not None:
            bucket["first_half"] = int(bucket.get("first_half") or 0) + int(row.get("first_half") or 0)
        if row.get("second_half") is not None:
            bucket["second_half"] = int(bucket.get("second_half") or 0) + int(row.get("second_half") or 0)
    ranked = [buckets[name] for name in order]
    for item in ranked:
        denom = int(item.get("denominator") or 0)
        if denom > 0 and item.get("count") is not None:
            item["percentage"] = round(100.0 * int(item["count"]) / denom, 1)
    ranked.sort(key=lambda item: (-int(item.get("count") or item.get("frequency") or 0), str(item.get("label"))))
    return ranked
