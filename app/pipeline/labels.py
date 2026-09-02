"""Normalize missing/empty category labels before aggregation and charts.

Placeholder values such as none / None / N/A must never become chart categories,
opportunity names, or discovery findings.
"""

from __future__ import annotations

import re
from typing import Any

# Kept for older callers; treated as missing if it appears as a stored label.
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
    "uncategorized",
    "no evidence",
    "not mentioned",
    "not specified",
    "none mentioned",
    "no mention",
    "-",
    "--",
    ".",
}

_MISSING_SUFFIX = re.compile(
    r"^(none|null|n/?a|na|unknown|undefined|nil|nan|uncategorized)(\s*\(\d+\))?$",
    re.IGNORECASE,
)


def normalize_label(value: Any) -> str | None:
    """Return a cleaned label, or None when the value is missing/placeholder."""
    if value is None or value is False:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _MISSING_EXACT:
        return None
    if _MISSING_SUFFIX.fullmatch(text):
        return None
    return text


def normalize_category_label(value: Any) -> str:
    """Compatibility wrapper. Missing labels become '' rather than a fake category."""
    return normalize_label(value) or ""


def is_placeholder_label(value: Any) -> bool:
    return normalize_label(value) is None


def stored_category_text(value: Any) -> str:
    """Persist empty string for placeholders; keep meaningful labels."""
    return normalize_label(value) or ""


def normalize_label_list(values: Any, *, keep_uncategorized_if_only_missing: bool = False) -> list[str]:
    """Normalize a list of labels. Placeholders are dropped and never displayed."""
    del keep_uncategorized_if_only_missing  # never keep a fake missing category
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    real: list[str] = []
    seen: set[str] = set()
    for item in values:
        label = normalize_label(item)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        real.append(label)
    return real


def validate_chart_categories(labels: list[Any] | None) -> list[str]:
    """Return leftover missing labels that should not be rendered."""
    issues: list[str] = []
    for raw in labels or []:
        if normalize_label(raw) is None and str(raw or "").strip():
            issues.append(str(raw).strip())
    return sorted(set(issues))


def merge_category_rows(
    rows: list[dict[str, Any]] | None,
    *,
    label_keys: tuple[str, ...] = ("label", "problem", "name", "root_cause", "theme", "segment", "barrier"),
    count_keys: tuple[str, ...] = ("count", "frequency", "review_count", "relevant_count"),
    id_keys: tuple[str, ...] = ("review_ids", "evidence_ids"),
) -> list[dict[str, Any]]:
    """Normalize labels then sum counts. Placeholder labels are omitted entirely."""
    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows or []:
        raw_label = ""
        for key in label_keys:
            if row.get(key) not in (None, ""):
                raw_label = row.get(key)
                break
        label = normalize_label(raw_label)
        if not label:
            continue
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
