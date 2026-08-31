"""Reusable collector interface, retries, and rate limiting."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Review, Source, utcnow
from app.pipeline.cleaning import clean_review
from app.pipeline.dedup import content_hash
from app.schemas import CollectionStats, NormalizedReview, SourceValidation

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


class RateLimiter:
    def __init__(self, min_interval: float = 1.0) -> None:
        self.min_interval = max(0.0, min_interval)
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


def with_retry(
    fn: Callable,
    *,
    attempts: int = 3,
    backoff: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    label: str = "operation",
):
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            last_error = exc
            logger.warning("%s failed (attempt %s/%s): %s", label, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(backoff ** (attempt - 1))
    assert last_error is not None
    raise last_error


class BaseCollector(ABC):
    platform: str = "unknown"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.rate_limiter = RateLimiter(self.settings.collection_rate_limit_seconds)
        self.errors: list[str] = []
        self.last_validation: SourceValidation | None = None

    @abstractmethod
    def validate_source(self, progress: ProgressCallback | None = None) -> SourceValidation:
        raise NotImplementedError

    @abstractmethod
    def collect(
        self,
        max_reviews: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[NormalizedReview]:
        raise NotImplementedError

    def save_raw(
        self,
        db: Session,
        reviews: list[NormalizedReview],
        validation: SourceValidation,
        collection_run_id: int | None = None,
    ) -> CollectionStats:
        stats = CollectionStats()
        stats.fetched = len(reviews)
        stats.source_validations = [validation]
        self._upsert_source(db, validation, review_count=0)

        for item in reviews:
            if not (item.text or "").strip() and not (item.title or "").strip():
                stats.rejected += 1
                continue

            digest = content_hash(item.source, item.source_review_id, item.text, item.app_id)
            existing = None
            if item.source_review_id:
                existing = (
                    db.query(Review)
                    .filter(
                        Review.source == item.source,
                        Review.source_review_id == item.source_review_id,
                        Review.app_id == item.app_id,
                    )
                    .one_or_none()
                )
            if existing is None:
                existing = (
                    db.query(Review)
                    .filter(Review.content_hash == digest, Review.source == item.source)
                    .one_or_none()
                )

            if existing is not None:
                stats.duplicates += 1
                continue

            flags = clean_review(item.title, item.text)
            if flags.is_empty:
                stats.rejected += 1
                continue

            row = Review(
                source=item.source,
                source_review_id=item.source_review_id or digest[:16],
                app_id=item.app_id,
                app_name=item.app_name,
                developer=item.developer,
                region=item.region,
                rating=item.rating,
                title=item.title or "",
                text=item.text or "",
                review_date=item.review_date,
                last_update_date=item.last_update_date,
                app_version=item.app_version or "",
                source_url=item.source_url or "",
                collected_at=utcnow(),
                is_valid_source=bool(item.is_valid_source),
                is_duplicate=False,
                is_synthetic=bool(item.is_synthetic),
                data_classification=item.data_classification,
                content_hash=digest,
                developer_reply=item.developer_reply or "",
                raw_payload_json=json.dumps(item.raw_payload, default=str)[:20000],
                cleaned_text=flags.cleaned_text,
                is_spam=flags.is_spam,
                is_empty=flags.is_empty,
                is_promotional=flags.is_promotional,
                is_short=flags.is_short,
                is_long=flags.is_long,
                language_notes=flags.language_notes,
                collection_run_id=collection_run_id,
            )
            db.add(row)
            stats.new += 1
            if item.is_valid_source:
                stats.valid += 1

        source_row = (
            db.query(Source)
            .filter(
                Source.platform == validation.platform,
                Source.app_id == validation.app_id,
                Source.region == (validation.region or ""),
            )
            .one_or_none()
        )
        if source_row is not None:
            source_row.review_count = (
                db.query(Review)
                .filter(Review.source == self.platform, Review.app_id == validation.app_id)
                .count()
            )

        db.commit()
        return stats

    def _upsert_source(self, db: Session, validation: SourceValidation, review_count: int) -> None:
        row = (
            db.query(Source)
            .filter(
                Source.platform == validation.platform,
                Source.app_id == validation.app_id,
                Source.region == (validation.region or ""),
            )
            .one_or_none()
        )
        now = datetime.now(timezone.utc)
        if row is None:
            row = Source(
                platform=validation.platform,
                app_id=validation.app_id,
                region=validation.region or "",
            )
            db.add(row)
        row.detected_app_name = validation.detected_app_name
        row.detected_developer = validation.detected_developer
        row.expected_app = validation.expected_app
        row.validation_status = validation.validation_status
        row.is_valid_for_myntra = validation.is_valid_for_myntra
        row.warning = validation.warning
        row.collection_date = now
        row.last_collection_at = now
        row.metadata_json = json.dumps(validation.metadata, default=str)[:20000]
        db.flush()
