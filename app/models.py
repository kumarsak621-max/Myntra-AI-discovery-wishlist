"""Persistent models. Original review text is never overwritten."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    platform: Mapped[str] = mapped_column(String(64), index=True)
    app_id: Mapped[str] = mapped_column(String(255), index=True)
    detected_app_name: Mapped[str] = mapped_column(String(255), default="")
    detected_developer: Mapped[str] = mapped_column(String(255), default="")
    region: Mapped[str] = mapped_column(String(32), default="")
    expected_app: Mapped[str] = mapped_column(String(64), default="Myntra")
    validation_status: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    is_valid_for_myntra: Mapped[bool] = mapped_column(Boolean, default=False)
    warning: Mapped[str] = mapped_column(Text, default="")
    collection_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_collection_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    review_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("platform", "app_id", "region", name="uq_source_identity"),)


class CollectionRun(Base):
    __tablename__ = "collection_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    sources: Mapped[str] = mapped_column(String(255), default="")
    fetched: Mapped[int] = mapped_column(Integer, default=0)
    valid: Mapped[int] = mapped_column(Integer, default=0)
    rejected: Mapped[int] = mapped_column(Integer, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, default=0)
    new_count: Mapped[int] = mapped_column(Integer, default=0)
    analyzed: Mapped[int] = mapped_column(Integer, default=0)
    errors_json: Mapped[str] = mapped_column(Text, default="[]")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str] = mapped_column(Text, default="")
    mode: Mapped[str] = mapped_column(String(64), default="")

    reviews: Mapped[list["Review"]] = relationship(back_populates="collection_run")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    source_review_id: Mapped[str] = mapped_column(String(255), default="", index=True)
    app_id: Mapped[str] = mapped_column(String(255), index=True)
    app_name: Mapped[str] = mapped_column(String(255), default="")
    developer: Mapped[str] = mapped_column(String(255), default="")
    region: Mapped[str] = mapped_column(String(32), default="")
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text, default="")  # original; never overwrite
    review_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_update_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    app_version: Mapped[str] = mapped_column(String(64), default="")
    source_url: Mapped[str] = mapped_column(Text, default="")
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_valid_source: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    data_classification: Mapped[str] = mapped_column(String(64), default="UNVALIDATED")
    content_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    developer_reply: Mapped[str] = mapped_column(Text, default="")
    raw_payload_json: Mapped[str] = mapped_column(Text, default="{}")
    cleaned_text: Mapped[str] = mapped_column(Text, default="")
    is_spam: Mapped[bool] = mapped_column(Boolean, default=False)
    is_empty: Mapped[bool] = mapped_column(Boolean, default=False)
    is_promotional: Mapped[bool] = mapped_column(Boolean, default=False)
    is_short: Mapped[bool] = mapped_column(Boolean, default=False)
    is_long: Mapped[bool] = mapped_column(Boolean, default=False)
    language_notes: Mapped[str] = mapped_column(String(128), default="")
    collection_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("collection_runs.id"), nullable=True
    )

    collection_run: Mapped[CollectionRun | None] = relationship(back_populates="reviews")
    analysis: Mapped[Analysis | None] = relationship(back_populates="review", uselist=False)

    __table_args__ = (
        UniqueConstraint("source", "source_review_id", "app_id", name="uq_review_source_id"),
    )


class Analysis(Base):
    __tablename__ = "analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"), unique=True, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    relevance: Mapped[str] = mapped_column(String(32), default="none")
    wishlist_signal: Mapped[str] = mapped_column(String(32), default="none")
    purchase_signal: Mapped[str] = mapped_column(String(32), default="none")
    purchase_hesitation: Mapped[str] = mapped_column(String(32), default="none")
    intent_json: Mapped[str] = mapped_column(Text, default="[]")
    barriers_json: Mapped[str] = mapped_column(Text, default="[]")
    uncertainties_json: Mapped[str] = mapped_column(Text, default="[]")
    information_seeking_json: Mapped[str] = mapped_column(Text, default="[]")
    behavioral_signals_json: Mapped[str] = mapped_column(Text, default="[]")
    product_category_json: Mapped[str] = mapped_column(Text, default="[]")
    decision_factors_json: Mapped[str] = mapped_column(Text, default="[]")
    root_cause_observed: Mapped[str] = mapped_column(Text, default="")
    root_cause_inferred: Mapped[str] = mapped_column(Text, default="")
    root_cause_hypothesized: Mapped[str] = mapped_column(Text, default="")
    root_cause: Mapped[str] = mapped_column(Text, default="")
    sentiment: Mapped[str] = mapped_column(String(32), default="neutral")
    evidence_strength: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[int] = mapped_column(Integer, default=1)
    raw_response: Mapped[str] = mapped_column(Text, default="")
    parse_error: Mapped[str] = mapped_column(Text, default="")
    is_valid_json: Mapped[bool] = mapped_column(Boolean, default=False)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    analysis_version: Mapped[str] = mapped_column(String(32), default="1")

    review: Mapped[Review] = relationship(back_populates="analysis")


class Theme(Base):
    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    cluster_key: Mapped[str] = mapped_column(String(64), default="")
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    myntra_review_count: Mapped[int] = mapped_column(Integer, default=0)
    reference_review_count: Mapped[int] = mapped_column(Integer, default=0)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    evidence_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    is_emergent: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    basis: Mapped[str] = mapped_column(String(32), default="inferred")
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    myntra_review_count: Mapped[int] = mapped_column(Integer, default=0)
    evidence_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Opportunity(Base):
    __tablename__ = "opportunities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str] = mapped_column(String(255), default="")
    user_problem: Mapped[str] = mapped_column(Text, default="")
    problem_key: Mapped[str] = mapped_column(String(255), index=True, default="")
    reach: Mapped[int] = mapped_column(Integer, default=1)
    frequency: Mapped[int] = mapped_column(Integer, default=1)
    purchase_impact: Mapped[int] = mapped_column(Integer, default=1)
    severity: Mapped[int] = mapped_column(Integer, default=1)
    evidence_confidence: Mapped[int] = mapped_column(Integer, default=1)
    score: Mapped[int] = mapped_column(Integer, default=0)
    relevant_count: Mapped[int] = mapped_column(Integer, default=0)
    total_relevant: Mapped[int] = mapped_column(Integer, default=0)
    percentage: Mapped[float] = mapped_column(Float, default=0.0)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    evidence_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    quote_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    what_we_know: Mapped[str] = mapped_column(Text, default="")
    what_we_dont_know: Mapped[str] = mapped_column(Text, default="")
    why_investigate: Mapped[str] = mapped_column(Text, default="")
    cross_source_status: Mapped[str] = mapped_column(String(64), default="")
    includes_non_myntra: Mapped[bool] = mapped_column(Boolean, default=False)
    myntra_only_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
