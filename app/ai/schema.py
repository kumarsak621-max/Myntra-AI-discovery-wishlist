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
    STRING_LIST_FIELDS,
    coerce_string_list,
)
from app.pipeline.labels import is_placeholder_label, stored_category_text

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
OPEN_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    fenced = FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    if text.lower().startswith("```"):
        text = OPEN_FENCE_RE.sub("", text, count=1)
        if text.endswith("```"):
            text = text[: -3].strip()
    return text.strip()


def _jsonish_repairs(text: str) -> list[str]:
    """Progressive repairs for common model JSON mistakes. Original first."""
    variants = [text]
    smart = (
        text.replace("\ufeff", "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    if smart != text:
        variants.append(smart)
    no_commas = TRAILING_COMMA_RE.sub(r"\1", variants[-1])
    if no_commas not in variants:
        variants.append(no_commas)
    return variants


def _close_truncated_json(text: str) -> str | None:
    """Close unbalanced braces/brackets when the model hit max_tokens."""
    in_string = False
    escape = False
    stack: list[str] = []
    saw_key = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if stack and stack[-1] == char:
                stack.pop()
        elif char == ":":
            saw_key = True
    if in_string or not stack or not saw_key:
        return None
    if '"results"' not in text and '"id"' not in text and '"problem"' not in text and '"relevance"' not in text:
        return None
    return text + "".join(reversed(stack))


def _candidate_snippets(raw: str) -> list[str]:
    snippets: list[str] = []
    start_obj, end_obj = raw.find("{"), raw.rfind("}")
    start_arr, end_arr = raw.find("["), raw.rfind("]")
    if start_obj != -1 and end_obj > start_obj:
        snippets.append(raw[start_obj : end_obj + 1])
    if start_arr != -1 and end_arr > start_arr:
        snippets.append(raw[start_arr : end_arr + 1])
    if start_obj != -1 and (end_obj == -1 or end_obj < start_obj):
        closed = _close_truncated_json(raw[start_obj:])
        if closed:
            snippets.append(closed)
    # De-dupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in snippets:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _load_json(text: str) -> Any:
    return json.loads(text)


def extract_json_value(text: str) -> Any:
    if not text or not str(text).strip():
        raise ValueError("Empty AI response")
    raw = _strip_fences(str(text))
    last_err: Exception | None = None
    candidates = [raw] + _candidate_snippets(raw)
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for variant in _jsonish_repairs(candidate):
            try:
                payload = _load_json(variant)
            except json.JSONDecodeError as exc:
                last_err = exc
                closed = _close_truncated_json(variant)
                if closed and closed != variant:
                    try:
                        payload = _load_json(closed)
                    except json.JSONDecodeError as repair_exc:
                        last_err = repair_exc
                        continue
                else:
                    continue
            if isinstance(payload, (dict, list)):
                return payload
            if isinstance(payload, str) and payload.strip()[:1] in "{[":
                try:
                    nested = extract_json_value(payload)
                except ValueError as exc:
                    last_err = exc
                    continue
                return nested
            raise ValueError("AI JSON root is not an object or array")
    if last_err is not None:
        raise ValueError("No JSON object found in AI response") from last_err
    raise ValueError("No JSON object found in AI response") from None


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
        return RootCauseItem(
            observed=stored_category_text(value.observed),
            inferred=stored_category_text(value.inferred),
            hypothesized=stored_category_text(value.hypothesized),
            statement=stored_category_text(value.statement or value.hypothesized or value.inferred or value.observed),
        )
    if isinstance(value, dict):
        statement = stored_category_text(
            value.get("statement") or value.get("hypothesized") or value.get("inferred") or value.get("observed")
        )
        return RootCauseItem(
            observed=stored_category_text(value.get("observed") or ""),
            inferred=stored_category_text(value.get("inferred") or ""),
            hypothesized=stored_category_text(value.get("hypothesized") or ""),
            statement=statement,
        )
    statement = stored_category_text(value)
    return RootCauseItem(statement=statement, hypothesized=statement)


def validate_analysis_payload(payload: dict[str, Any], original_text: str) -> ReviewAnalysisSchema:
    parsed = ReviewAnalysisSchema.model_validate(payload)
    parsed.root_cause = normalize_root_cause(parsed.root_cause)
    return sanitize_quotes(parsed, original_text)


def try_validate_payload(
    payload: dict[str, Any], original_text: str
) -> tuple[ReviewAnalysisSchema | None, str]:
    cleaned = {key: value for key, value in payload.items() if key not in {"id", "source_review_id"}}

    def _meaningful(value: Any) -> str:
        return stored_category_text(value)

    problem_text = _meaningful(cleaned.get("user_problem")) or _meaningful(cleaned.get("problem"))
    if problem_text and not cleaned.get("root_cause"):
        cleaned["root_cause"] = {"statement": problem_text}
    barrier = cleaned.get("purchase_barrier")
    if barrier and not cleaned.get("barriers"):
        if isinstance(barrier, list):
            cleaned["barriers"] = [stored_category_text(x) for x in barrier if stored_category_text(x)]
        elif not is_placeholder_label(barrier):
            cleaned["barriers"] = [stored_category_text(barrier)]
    uncertainty = cleaned.get("uncertainty")
    if uncertainty and not cleaned.get("uncertainties"):
        if isinstance(uncertainty, list):
            cleaned["uncertainties"] = [stored_category_text(x) for x in uncertainty if stored_category_text(x)]
        elif not is_placeholder_label(uncertainty):
            cleaned["uncertainties"] = [stored_category_text(uncertainty)]
    theme = _meaningful(cleaned.get("theme"))
    if theme and not cleaned.get("intent"):
        cleaned["intent"] = [theme]
    segment = _meaningful(cleaned.get("segment"))
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

    for key in ("information_seeking", "behavioral_signals"):
        value = cleaned.get(key)
        if value in (None, "", False):
            cleaned[key] = []
        elif isinstance(value, dict):
            cleaned[key] = [value]
        elif isinstance(value, str):
            cleaned[key] = []

    for key in STRING_LIST_FIELDS:
        if key not in cleaned:
            continue
        try:
            cleaned[key] = coerce_string_list(cleaned.get(key))
        except (TypeError, ValueError) as exc:
            return None, f"AI response failed schema validation: {key}: {exc}"

    def _as_list(value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            items = [stored_category_text(item) for item in value]
        else:
            items = [stored_category_text(value)]
        return [text for text in items if text]

    def _merge(dest: str, extra: str) -> None:
        added = _as_list(cleaned.get(extra))
        if not added:
            return
        existing = _as_list(cleaned.get(dest))
        cleaned[dest] = list(dict.fromkeys(existing + added))

    _merge("barriers", "purchase_barriers")
    _merge("intent", "wishlist_behavior")
    _merge("intent", "themes")
    _merge("decision_factors", "segments")
    _merge("decision_factors", "comparison_factors")
    first_problem = (_as_list(cleaned.get("problems")) or [None])[0]
    if first_problem and not cleaned.get("root_cause"):
        cleaned["root_cause"] = {"statement": first_problem}
    seeking = list(cleaned.get("information_seeking") or [])
    for src in _as_list(cleaned.get("external_information_seeking")):
        if isinstance(seeking, list):
            seeking.append({"source": src, "basis": "inferred"})
    if seeking:
        cleaned["information_seeking"] = seeking
    signals = list(cleaned.get("behavioral_signals") or [])
    for sig in _as_list(cleaned.get("social_validation")):
        if isinstance(signals, list):
            signals.append({"signal": sig, "basis": "inferred"})
    if signals:
        cleaned["behavioral_signals"] = signals
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
