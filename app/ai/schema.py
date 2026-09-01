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


def extract_json_value(text: str) -> Any:
    if not text or not str(text).strip():
        raise ValueError("Empty AI response")
    raw = str(text).strip()
    fenced = FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, (dict, list)):
            return payload
        raise ValueError("AI JSON root is not an object or array")
    except json.JSONDecodeError:
        snippets: list[tuple[int, str]] = []
        start_obj, end_obj = raw.find("{"), raw.rfind("}")
        start_arr, end_arr = raw.find("["), raw.rfind("]")
        if start_obj != -1 and end_obj > start_obj:
            snippets.append((start_obj, raw[start_obj : end_obj + 1]))
        if start_arr != -1 and end_arr > start_arr:
            snippets.append((start_arr, raw[start_arr : end_arr + 1]))
        if not snippets:
            raise ValueError("No JSON object found in AI response") from None
        snippets.sort()
        last_err: Exception | None = None
        for _, snippet in snippets:
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError as exc:
                last_err = exc
                continue
            if isinstance(payload, (dict, list)):
                return payload
        raise ValueError("No JSON object found in AI response") from last_err


def extract_json_object(text: str) -> dict[str, Any]:
    payload = extract_json_value(text)
    if not isinstance(payload, dict):
        raise ValueError("AI JSON root is not an object")
    return payload


def parse_batch_payload(raw_response: str) -> tuple[list[dict[str, Any]], str]:
    """Return analysis objects from a batch or single-review AI response."""
    try:
        value = extract_json_value(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        return [], f"Malformed AI JSON: {exc}"

    items: list[Any]
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        if isinstance(value.get("results"), list):
            items = value["results"]
        elif {
            "relevance",
            "id",
            "problem",
            "wishlist_signal",
            "purchase_barrier",
            "uncertainty",
            "theme",
            "segment",
        }.intersection(value):
            items = [value]
        else:
            return [], "Malformed AI JSON: missing results[] array"
    else:
        return [], "Malformed AI JSON: root is not an object or array"

    objects = [item for item in items if isinstance(item, dict)]
    if not objects:
        return [], "Malformed AI JSON: results[] contained no objects"
    return objects, ""


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


def try_validate_payload(
    payload: dict[str, Any], original_text: str
) -> tuple[ReviewAnalysisSchema | None, str]:
    cleaned = {key: value for key, value in payload.items() if key not in {"id", "source_review_id"}}
    if cleaned.get("user_problem") and not cleaned.get("root_cause"):
        cleaned["root_cause"] = {"statement": str(cleaned.get("user_problem") or "")}
    if cleaned.get("problem") and not cleaned.get("root_cause"):
        cleaned["root_cause"] = {"statement": str(cleaned.get("problem") or "")}
    barrier = cleaned.get("purchase_barrier")
    if barrier and not cleaned.get("barriers"):
        cleaned["barriers"] = barrier if isinstance(barrier, list) else [str(barrier)]
    uncertainty = cleaned.get("uncertainty")
    if uncertainty and not cleaned.get("uncertainties"):
        cleaned["uncertainties"] = uncertainty if isinstance(uncertainty, list) else [str(uncertainty)]
    theme = str(cleaned.get("theme") or "").strip()
    if theme and not cleaned.get("intent"):
        cleaned["intent"] = [theme]
    segment = str(cleaned.get("segment") or "").strip()
    if segment:
        factors = list(cleaned.get("decision_factors") or [])
        if segment not in factors:
            factors.append(segment)
        cleaned["decision_factors"] = factors
    if "confidence" in cleaned and isinstance(cleaned.get("confidence"), float) and cleaned["confidence"] <= 1:
        cleaned["confidence"] = max(1, min(5, int(round(cleaned["confidence"] * 5)) or 1))
    if cleaned.get("severity") not in (None, "") and not cleaned.get("evidence_strength"):
        cleaned["evidence_strength"] = cleaned.get("severity")
    evidence_type = str(cleaned.get("evidence_type") or "").strip().lower()
    if not cleaned.get("relevance"):
        if evidence_type == "explicit":
            cleaned["relevance"] = "high"
        elif evidence_type == "inferred":
            cleaned["relevance"] = "medium"
        elif evidence_type == "none":
            cleaned["relevance"] = "none"
        elif cleaned.get("problem") or cleaned.get("purchase_barrier") or cleaned.get("uncertainty"):
            cleaned["relevance"] = "medium"
    if cleaned.get("purchase_barrier") and not cleaned.get("purchase_signal"):
        cleaned["purchase_signal"] = "hesitant"
        cleaned["purchase_hesitation"] = cleaned.get("purchase_hesitation") or "implicit"
    try:
        return validate_analysis_payload(cleaned, original_text), ""
    except (ValueError, ValidationError, json.JSONDecodeError, TypeError) as exc:
        return None, f"AI response failed schema validation: {exc}"


def try_validate_analysis(raw_response: str, original_text: str) -> tuple[ReviewAnalysisSchema | None, str]:
    try:
        payload = extract_json_object(raw_response)
        return try_validate_payload(payload, original_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"Malformed AI JSON: {exc}"
