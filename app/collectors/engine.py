"""Orchestrates collection across sources, persistence, and optional analysis."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.collectors.app_store import (
    APPLE_FETCH_FAILED,
    APPLE_FETCH_SUCCESS_NO_NEW_REVIEWS,
    APPLE_NEW_REVIEWS_FOUND,
    PLAY_FETCH_FAILED,
    PLAY_FETCH_SUCCESS_NO_NEW_REVIEWS,
    PLAY_NEW_REVIEWS_FOUND,
    AppStoreCollector,
)
from app.collectors.google_play import GooglePlayCollector
from app.config import Settings, get_settings
from app.models import CollectionRun, utcnow
from app.pipeline.dates import filter_reviews_by_date, get_last_30_days_cutoff
from app.schemas import CollectionStats, SourceValidation

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


class CollectionEngine:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()

    def run(
        self,
        sources: list[str] | None = None,
        *,
        max_reviews: int | None = None,
        analyze: bool = True,
        mode: str = "latest",
        progress: ProgressCallback | None = None,
        analyze_limit: int | None = None,
    ) -> CollectionStats:
        wanted = sources or ["google_play", "apple_app_store"]
        cutoff = None
        if mode == "last_30_days":
            cutoff = get_last_30_days_cutoff(days=self.settings.collection_window_days)
        run = CollectionRun(
            status="running",
            sources=",".join(wanted),
            started_at=utcnow(),
            mode=mode,
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)

        combined = CollectionStats(mode=mode, window_start=cutoff)
        started = time.monotonic()
        validations: list[SourceValidation] = []

        def _collect_kwargs(source: str) -> dict[str, Any]:
            per_source = (
                int(self.settings.google_play_max_reviews)
                if source == "google_play"
                else int(self.settings.apple_max_reviews)
            )
            if mode == "last_30_days":
                safety = (
                    self.settings.google_play_window_safety_limit
                    if source == "google_play"
                    else self.settings.apple_window_safety_limit
                )
                return {"stop_when_older_than": cutoff, "safety_limit": min(int(safety), per_source)}
            limit = max_reviews if max_reviews is not None else self.settings.refresh_safety_limit
            return {"max_reviews": min(int(limit), per_source)}

        try:
            if "google_play" in wanted:
                gp = GooglePlayCollector(self.settings)
                if progress:
                    progress({"stage": "google_play", "status": "start", "run_id": run.id})
                reviews = gp.collect(progress=progress, **_collect_kwargs("google_play"))
                validation = gp.last_validation or gp.validate_source()
                stats = gp.save_raw(self.db, reviews, validation, collection_run_id=run.id)
                if gp.errors and stats.fetched == 0:
                    play_status = PLAY_FETCH_FAILED
                elif stats.new > 0:
                    play_status = PLAY_NEW_REVIEWS_FOUND
                else:
                    play_status = PLAY_FETCH_SUCCESS_NO_NEW_REVIEWS
                    if validation.is_valid_for_myntra and stats.fetched == 0:
                        logger.info(
                            "Google Play source checked successfully but returned 0 new reviews."
                        )
                combined.fetched += stats.fetched
                combined.valid += stats.valid
                combined.rejected += stats.rejected
                combined.duplicates += stats.duplicates
                combined.new += stats.new
                if play_status == PLAY_FETCH_FAILED:
                    combined.errors.extend(gp.errors)
                validations.extend(stats.source_validations)
                combined.by_source["google_play"] = {
                    "fetched": stats.fetched,
                    "new": stats.new,
                    "duplicates": stats.duplicates,
                    "rejected": stats.rejected,
                    "valid": stats.valid,
                    "errors": list(gp.errors),
                    "region": self.settings.google_play_country,
                    "app_id": self.settings.google_play_app_id,
                    "validation": validation.validation_result,
                    "detected_app": validation.detected_app_name,
                    "is_valid_for_myntra": validation.is_valid_for_myntra,
                    "warning": validation.warning,
                    "fetch_status": play_status,
                    "in_window": len(filter_reviews_by_date(reviews, cutoff)) if cutoff else stats.fetched,
                    "latest_review_at": (
                        max((r.review_date for r in reviews if r.review_date), default=None)
                    ).isoformat()
                    if any(r.review_date for r in reviews)
                    else None,
                }
                combined.in_window += combined.by_source["google_play"]["in_window"]
                if progress:
                    progress(
                        {
                            "stage": "google_play",
                            "status": "saved",
                            "fetched": stats.fetched,
                            "new": stats.new,
                            "duplicates": stats.duplicates,
                        }
                    )

            if "apple_app_store" in wanted:
                apple = AppStoreCollector(self.settings)
                if progress:
                    progress({"stage": "apple_app_store", "status": "start", "run_id": run.id})
                reviews = apple.collect(progress=progress, **_collect_kwargs("apple_app_store"))
                validation = apple.last_validation or apple.validate_source(region=apple.region_used)
                stats = apple.save_raw(self.db, reviews, validation, collection_run_id=run.id)
                if apple.fetch_status == APPLE_FETCH_FAILED:
                    apple_status = APPLE_FETCH_FAILED
                elif stats.new > 0:
                    apple_status = APPLE_NEW_REVIEWS_FOUND
                else:
                    apple_status = APPLE_FETCH_SUCCESS_NO_NEW_REVIEWS
                    logger.info(
                        "Apple App Store checked successfully (%s) but stored 0 new reviews "
                        "(fetched=%s duplicates=%s).",
                        apple.region_used,
                        stats.fetched,
                        stats.duplicates,
                    )
                apple.fetch_status = apple_status
                combined.fetched += stats.fetched
                combined.valid += stats.valid
                combined.rejected += stats.rejected
                combined.duplicates += stats.duplicates
                combined.new += stats.new
                if apple_status == APPLE_FETCH_FAILED:
                    combined.errors.extend(apple.errors)
                validations.extend(stats.source_validations)
                combined.by_source["apple_app_store"] = {
                    "fetched": stats.fetched,
                    "new": stats.new,
                    "duplicates": stats.duplicates,
                    "rejected": stats.rejected,
                    "valid": stats.valid,
                    "errors": list(apple.errors),
                    "region_attempted": self.settings.apple_primary_region,
                    "region_used": apple.region_used,
                    "fallback_used": apple.fallback_used,
                    "app_id": self.settings.apple_app_id,
                    "validation": validation.validation_result,
                    "detected_app": validation.detected_app_name,
                    "is_valid_for_myntra": validation.is_valid_for_myntra,
                    "warning": validation.warning,
                    "fetch_status": apple_status,
                    "in_window": len(filter_reviews_by_date(reviews, cutoff)) if cutoff else stats.fetched,
                    "latest_review_at": (
                        max((r.review_date for r in reviews if r.review_date), default=None)
                    ).isoformat()
                    if any(r.review_date for r in reviews)
                    else None,
                }
                combined.in_window += combined.by_source["apple_app_store"]["in_window"]
                if progress:
                    progress(
                        {
                            "stage": "apple_app_store",
                            "status": "saved",
                            "fetched": stats.fetched,
                            "new": stats.new,
                            "duplicates": stats.duplicates,
                            "region_used": apple.region_used,
                            "fallback_used": apple.fallback_used,
                            "fetch_status": apple_status,
                        }
                    )

            from app.pipeline.dataset import enforce_review_limit

            enforce_review_limit(self.db, prune=True)

            if analyze:
                if progress:
                    progress({"stage": "analysis", "status": "start"})
                from app.ai.provider import AIError
                from app.pipeline.orchestrator import run_analysis_pipeline

                try:
                    result = run_analysis_pipeline(
                        self.db,
                        progress=progress,
                        analyze_limit=analyze_limit,
                    )
                    combined.analyzed = result.analyzed
                    combined.analysis_failed = result.failed
                    if result.last_error:
                        combined.analysis_error = result.last_error
                    if result.failed and result.analyzed == 0:
                        combined.errors.append(result.last_error)
                except AIError as exc:
                    msg = str(exc)
                    logger.error(msg)
                    combined.errors.append(msg)
                    combined.analysis_error = msg
                except Exception as exc:
                    msg = f"OpenRouter analysis failed: {exc}"
                    logger.exception(msg)
                    combined.errors.append(msg)
                    combined.analysis_error = msg
            from app.database import get_database_diagnostics, get_review_count

            combined.pending_remaining = int(get_database_diagnostics(self.db).get("pending_reviews") or 0)

            combined.source_validations = validations
            combined.duration_seconds = round(time.monotonic() - started, 2)
            run.status = "completed" if not combined.errors else "completed_with_errors"
            run.mode = mode
            gp_info = combined.by_source.get("google_play") or {}
            apple_info = combined.by_source.get("apple_app_store") or {}
            run.notes = json.dumps(
                {
                    "window_start": cutoff.isoformat() if cutoff else None,
                    "google_play_new": gp_info.get("new", 0),
                    "apple_new": apple_info.get("new", 0),
                    "google_play_status": gp_info.get("fetch_status") or "",
                    "apple_status": apple_info.get("fetch_status") or "",
                    "stored": get_review_count(self.db, myntra_only=True),
                    "analyzed": combined.analyzed,
                    "errors": list(combined.errors)[:8],
                }
            )
            run.fetched = combined.fetched
            run.valid = combined.valid
            run.rejected = combined.rejected
            run.duplicates = combined.duplicates
            run.new_count = combined.new
            run.analyzed = combined.analyzed
            run.errors_json = json.dumps(combined.errors)
            run.duration_seconds = combined.duration_seconds
            run.finished_at = utcnow()
            self.db.commit()
            if progress:
                progress(
                    {
                        "stage": "complete",
                        "status": "complete",
                        "stats": combined.model_dump(mode="json"),
                        "run_id": run.id,
                    }
                )
            return combined
        except Exception as exc:
            logger.exception("Collection run failed")
            combined.errors.append(str(exc))
            combined.duration_seconds = round(time.monotonic() - started, 2)
            run.status = "failed"
            run.fetched = combined.fetched
            run.valid = combined.valid
            run.rejected = combined.rejected
            run.duplicates = combined.duplicates
            run.new_count = combined.new
            run.analyzed = combined.analyzed
            run.errors_json = json.dumps(combined.errors)
            run.duration_seconds = combined.duration_seconds
            run.finished_at = utcnow()
            run.notes = str(exc)
            self.db.commit()
            if progress:
                progress({"stage": "complete", "status": "failed", "message": str(exc), "run_id": run.id})
            raise
