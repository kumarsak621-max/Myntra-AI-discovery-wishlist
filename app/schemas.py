"""Pydantic schemas for collectors, AI output, and API payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.pipeline.labels import is_placeholder_label, stored_category_text


class SourceValidation(BaseModel):
    platform: str
    app_id: str
    detected_app_name: str = ""
    detected_developer: str = ""
    region: str = ""
    expected_app: str = "Myntra"
    expected_id: str = ""
    expected_url: str = ""
    collection_date: datetime | None = None
    validation_status: str = "UNKNOWN"
    validation_result: str = "FAIL"
    is_valid_for_myntra: bool = False
    warning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def data_classification(self) -> str:
        if self.is_valid_for_myntra:
            return "MYNTRA EVIDENCE"
        if self.validation_status == "INVALID_FOR_MYNTRA_ANALYSIS":
            return "REFERENCE / NON-MYNTRA DATA"
        return "UNVALIDATED"


class NormalizedReview(BaseModel):
    source: str
    source_review_id: str = ""
    app_id: str = ""
    app_name: str = ""
    developer: str = ""
    region: str = ""
    rating: int | None = None
    title: str = ""
    text: str = ""
    review_date: datetime | None = None
    last_update_date: datetime | None = None
    app_version: str = ""
    source_url: str = ""
    developer_reply: str = ""
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    is_valid_source: bool = False
    data_classification: str = "UNVALIDATED"
    is_synthetic: bool = False


class CollectionStats(BaseModel):
    fetched: int = 0
    valid: int = 0
    rejected: int = 0
    duplicates: int = 0
    new: int = 0
    analyzed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)
    source_validations: list[SourceValidation] = Field(default_factory=list)
    by_source: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mode: str = ""
    window_start: datetime | None = None
    in_window: int = 0
    analysis_error: str = ""
    analysis_failed: int = 0
    pending_remaining: int = 0


class InformationSeekingItem(BaseModel):
    source: str = "unspecified"
    what: str = ""
    why: str = ""
    associated_with_hesitation: bool = False
    myntra_appears_to_lack_info: bool | None = None
    basis: Literal["explicit", "inferred"] = "inferred"
    quote: str = ""


class BehavioralSignalItem(BaseModel):
    signal: str
    basis: Literal["explicit", "inferred"] = "inferred"
    quote: str = ""


class RootCauseItem(BaseModel):
    observed: str = ""
    inferred: str = ""
    hypothesized: str = ""
    statement: str = ""


class ReviewAnalysisSchema(BaseModel):
    """Validated AI output. Quotes must later be checked against original text."""

    relevance: Literal["high", "medium", "low", "none"] = "none"
    wishlist_signal: Literal["explicit", "implicit", "none"] = "none"
    purchase_signal: Literal[
        "purchased", "intend_to_purchase", "hesitant", "abandoned", "none"
    ] = "none"
    purchase_hesitation: Literal["explicit", "implicit", "none"] = "none"
    intent: list[str] = Field(default_factory=list)
    barriers: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    information_seeking: list[InformationSeekingItem] = Field(default_factory=list)
    behavioral_signals: list[BehavioralSignalItem] = Field(default_factory=list)
    product_category: list[str] = Field(default_factory=list)
    decision_factors: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    wishlist_behavior: list[str] = Field(default_factory=list)
    purchase_barriers: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    comparison_factors: list[str] = Field(default_factory=list)
    external_information_seeking: list[str] = Field(default_factory=list)
    social_validation: list[str] = Field(default_factory=list)
    root_cause: RootCauseItem | str = ""
    sentiment: Literal["positive", "negative", "mixed", "neutral"] = "neutral"
    evidence_strength: int = Field(default=1, ge=1, le=5)
    confidence: int = Field(default=1, ge=1, le=5)

    @field_validator("wishlist_signal", mode="before")
    @classmethod
    def _wishlist_signal(cls, value: Any) -> str:
        if value in {False, 0, "0", "false", "False", "none", "None", None, ""}:
            return "none"
        if value in {True, 1, "1", "true", "True"}:
            return "implicit"
        return value

    @field_validator(
        "intent",
        "barriers",
        "uncertainties",
        "product_category",
        "decision_factors",
        "problems",
        "wishlist_behavior",
        "purchase_barriers",
        "themes",
        "segments",
        "comparison_factors",
        "external_information_seeking",
        "social_validation",
    )
    @classmethod
    def _strip_empty(cls, value: list[str]) -> list[str]:
        cleaned = []
        seen: set[str] = set()
        for item in value:
            text = stored_category_text(item)
            if not text or is_placeholder_label(text):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    @field_validator("evidence_strength", "confidence", mode="before")
    @classmethod
    def _clamp_int(cls, value: Any) -> int:
        if isinstance(value, float) and 0 <= value <= 1:
            return max(1, min(5, int(round(value * 5)) or 1))
        try:
            number = int(value)
        except (TypeError, ValueError):
            return 1
        return max(1, min(5, number))
