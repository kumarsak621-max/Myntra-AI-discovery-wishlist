"""AI analysis JSON helpers — extract, validate, and quote-check against original text."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.schemas import (
    BehavioralSignalItem,
    InformationSeekingItem,
    ReviewAnalysisSchema,
    RootCauseItem,
)

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not str(text).strip():
        raise ValueError("Empty AI response")
    raw = str(text).strip()
    fenced = FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
        raise ValueError("AI JSON root is not an object")
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in AI response") from None
        payload = json.loads(raw[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("AI JSON root is not an object")
        return payload


def _quote_in_source(quote: str, source_text: str) -> bool:
    needle = " ".join((quote or "").split()).strip().lower()
    hay = " ".join((source_text or "").split()).strip().lower()
    if len(needle) < 8:
        return False
    return needle in hay


def sanitize_quotes(analysis: ReviewAnalysisSchema, original_text: str) -> ReviewAnalysisSchema:
    """Drop quotes that are not substrings of the original review. Never invent replacements."""
    title_and_body = original_text or ""
    kept_seeking: list[InformationSeekingItem] = []
    for item in analysis.information_seeking:
        if item.basis == "explicit" and item.quote and not _quote_in_source(item.quote, title_and_body):
            item.quote = ""
            item.basis = "inferred"
        kept_seeking.append(item)
    analysis.information_seeking = kept_seeking

    kept_signals: list[BehavioralSignalItem] = []
    for item in analysis.behavioral_signals:
        if item.basis == "explicit" and item.quote and not _quote_in_source(item.quote, title_and_body):
            item.quote = ""
            item.basis = "inferred"
        kept_signals.append(item)
    analysis.behavioral_signals = kept_signals
    return analysis


def normalize_root_cause(value: RootCauseItem | str | dict | None) -> RootCauseItem:
    if isinstance(value, RootCauseItem):
        return value
    if isinstance(value, dict):
        return RootCauseItem(
            observed=str(value.get("observed") or ""),
            inferred=str(value.get("inferred") or ""),
            hypothesized=str(value.get("hypothesized") or ""),
            statement=str(value.get("statement") or ""),
        )
    statement = str(value or "").strip()
    return RootCauseItem(statement=statement, hypothesized=statement)


def validate_analysis_payload(payload: dict[str, Any], original_text: str) -> ReviewAnalysisSchema:
    parsed = ReviewAnalysisSchema.model_validate(payload)
    parsed.root_cause = normalize_root_cause(parsed.root_cause)
    return sanitize_quotes(parsed, original_text)


def try_validate_analysis(raw_response: str, original_text: str) -> tuple[ReviewAnalysisSchema | None, str]:
    try:
        payload = extract_json_object(raw_response)
        return validate_analysis_payload(payload, original_text), ""
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        return None, str(exc)
