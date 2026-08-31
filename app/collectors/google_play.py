"""Google Play review collector using google-play-scraper.

The configured package ID is never rewritten. Identity is validated against
the live Play Store metadata before reviews are stored.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.collectors.base_collector import BaseCollector, ProgressCallback, with_retry
from app.config import Settings
from app.pipeline.validation import validate_app_identity
from app.schemas import NormalizedReview, SourceValidation
from config.settings import is_banned_app_id, is_official_myntra_app_id

logger = logging.getLogger(__name__)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    return None


class GooglePlayCollector(BaseCollector):
    platform = "google_play"

    def __init__(self, settings: Settings | None = None, scraper=None) -> None:
        super().__init__(settings)
        self._scraper = scraper  # injectable for tests

    def _lib(self):
        if self._scraper is not None:
            return self._scraper
        import google_play_scraper as gps

        return gps

    def validate_source(self, progress: ProgressCallback | None = None) -> SourceValidation:
        app_id = self.settings.google_play_app_id
        expected = self.settings.expected_app_name
        if progress:
            progress(
                {
                    "stage": "google_play",
                    "status": "validating",
                    "app_id": app_id,
                    "message": f"Validating Google Play identity for {app_id}",
                }
            )

        def _fetch():
            self.rate_limiter.wait()
            gps = self._lib()
            return gps.app(
                app_id,
                lang=self.settings.google_play_language,
                country=self.settings.google_play_country,
            )

        try:
            meta = with_retry(
                _fetch,
                attempts=self.settings.collection_retry_attempts,
                label=f"google-play app({app_id})",
            )
        except Exception as exc:
            logger.exception("Google Play metadata fetch failed")
            validation = validate_app_identity(
                platform=self.platform,
                app_id=app_id,
                detected_app_name="",
                detected_developer="",
                region=self.settings.google_play_country,
                expected_app=expected,
                metadata={"error": str(exc)},
            )
            validation.warning = (
                f"Google Play metadata fetch failed for {app_id}: {exc}. "
                "The configured ID was not changed."
            )
            validation.validation_status = "ERROR"
            self.errors.append(str(exc))
            self.last_validation = validation
            return validation

        title = str(meta.get("title") or "")
        developer = str(meta.get("developer") or meta.get("developerId") or "")
        validation = validate_app_identity(
            platform=self.platform,
            app_id=app_id,
            detected_app_name=title,
            detected_developer=developer,
            region=self.settings.google_play_country,
            expected_app=expected,
            metadata={
                "score": meta.get("score"),
                "reviews": meta.get("reviews"),
                "installs": meta.get("installs"),
                "url": meta.get("url"),
                "genre": meta.get("genre"),
                "realInstalls": meta.get("realInstalls"),
                "summary": (meta.get("summary") or "")[:500],
            },
        )
        self.last_validation = validation
        if progress:
            progress(
                {
                    "stage": "google_play",
                    "status": "validated",
                    "app_id": app_id,
                    "detected_app_name": title,
                    "detected_developer": developer,
                    "validation_status": validation.validation_status,
                    "is_valid_for_myntra": validation.is_valid_for_myntra,
                    "warning": validation.warning,
                }
            )
        return validation

    def collect(
        self,
        max_reviews: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[NormalizedReview]:
        app_id = self.settings.google_play_app_id
        limit = max_reviews if max_reviews is not None else self.settings.google_play_max_reviews
        batch = max(1, self.settings.google_play_batch_size)
        validation = self.validate_source(progress=progress)
        if is_banned_app_id(app_id) or not is_official_myntra_app_id(app_id, self.platform):
            msg = validation.warning or (
                f"Refusing Myntra collection for non-official Google Play ID {app_id}."
            )
            logger.error(msg)
            self.errors.append(msg)
            if progress:
                progress(
                    {
                        "stage": "google_play",
                        "status": "error",
                        "message": msg,
                        "validation_result": "FAIL",
                    }
                )
            return []
        if not validation.is_valid_for_myntra:
            msg = validation.warning or "Google Play identity validation FAIL."
            logger.error(msg)
            self.errors.append(msg)
            if progress:
                progress(
                    {
                        "stage": "google_play",
                        "status": "error",
                        "message": msg,
                        "validation_result": "FAIL",
                    }
                )
            return []

        gps = self._lib()
        sort = getattr(gps, "Sort", None)
        sort_newest = getattr(sort, "NEWEST", None) if sort else None

        collected: list[NormalizedReview] = []
        continuation_token = None
        seen_ids: set[str] = set()

        while len(collected) < limit:
            count = min(batch, limit - len(collected))

            def _page(token=continuation_token, page_count=count):
                self.rate_limiter.wait()
                kwargs = {
                    "lang": self.settings.google_play_language,
                    "country": self.settings.google_play_country,
                    "count": page_count,
                    "continuation_token": token,
                }
                if sort_newest is not None:
                    kwargs["sort"] = sort_newest
                return gps.reviews(app_id, **kwargs)

            try:
                result, continuation_token = with_retry(
                    _page,
                    attempts=self.settings.collection_retry_attempts,
                    label="google-play reviews page",
                )
            except Exception as exc:
                logger.exception("Google Play review page failed")
                self.errors.append(str(exc))
                if progress:
                    progress(
                        {
                            "stage": "google_play",
                            "status": "error",
                            "message": str(exc),
                            "fetched": len(collected),
                        }
                    )
                break

            if not result:
                break

            new_on_page = 0
            for raw in result:
                review_id = str(raw.get("reviewId") or "")
                if review_id and review_id in seen_ids:
                    continue
                if review_id:
                    seen_ids.add(review_id)
                collected.append(self.normalize(raw, validation))
                new_on_page += 1
                if len(collected) >= limit:
                    break

            if progress:
                progress(
                    {
                        "stage": "google_play",
                        "status": "collecting",
                        "fetched": len(collected),
                        "target": limit,
                        "validation_status": validation.validation_status,
                    }
                )

            if new_on_page == 0 or continuation_token is None:
                break

        if progress:
            progress(
                {
                    "stage": "google_play",
                    "status": "complete",
                    "fetched": len(collected),
                    "errors": list(self.errors),
                    "validation": validation.model_dump(mode="json"),
                }
            )
        return collected[:limit]

    def normalize(self, raw: dict[str, Any], validation: SourceValidation) -> NormalizedReview:
        reply = raw.get("replyContent") or ""
        user_name = raw.get("userName") or ""
        # Do not persist raw usernames into the normalized public fields.
        payload = {
            "reviewId": raw.get("reviewId"),
            "score": raw.get("score"),
            "content": raw.get("content"),
            "at": str(raw.get("at")),
            "repliedAt": str(raw.get("repliedAt")),
            "reviewCreatedVersion": raw.get("reviewCreatedVersion"),
            "thumbsUpCount": raw.get("thumbsUpCount"),
            "has_user_name": bool(user_name),
        }
        return NormalizedReview(
            source=self.platform,
            source_review_id=str(raw.get("reviewId") or ""),
            app_id=self.settings.google_play_app_id,
            app_name=validation.detected_app_name,
            developer=validation.detected_developer,
            region=self.settings.google_play_country,
            rating=raw.get("score"),
            title="",
            text=str(raw.get("content") or ""),
            review_date=_parse_dt(raw.get("at")),
            last_update_date=_parse_dt(raw.get("at")),
            app_version=str(raw.get("reviewCreatedVersion") or ""),
            source_url=self.settings.google_play_url,
            developer_reply=str(reply or ""),
            raw_payload=payload,
            is_valid_source=validation.is_valid_for_myntra,
            data_classification=validation.data_classification,
            is_synthetic=False,
        )
