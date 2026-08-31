"""Orchestrates collection across sources, persistence, and optional analysis."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.collectors.app_store import AppStoreCollector
from app.collectors.google_play import GooglePlayCollector
from app.config import Settings, get_settings
from app.models import CollectionRun, utcnow
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
        progress: ProgressCallback | None = None,
    ) -> CollectionStats:
        wanted = sources or ["google_play", "apple_app_store"]
        run = CollectionRun(
            status="running",
            sources=",".join(wanted),
            started_at=utcnow(),
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)

        combined = CollectionStats()
        started = time.monotonic()
        validations: list[SourceValidation] = []

        try:
            if "google_play" in wanted:
                gp = GooglePlayCollector(self.settings)
                if progress:
                    progress({"stage": "google_play", "status": "start", "run_id": run.id})
                reviews = gp.collect(max_reviews=max_reviews, progress=progress)
                validation = gp.last_validation or gp.validate_source()
                stats = gp.save_raw(self.db, reviews, validation, collection_run_id=run.id)
                combined.fetched += stats.fetched
                combined.valid += stats.valid
                combined.rejected += stats.rejected
                combined.duplicates += stats.duplicates
                combined.new += stats.new
                combined.errors.extend(gp.errors)
                validations.extend(stats.source_validations)
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
                reviews = apple.collect(max_reviews=max_reviews, progress=progress)
                validation = apple.last_validation or apple.validate_source(region=apple.region_used)
                stats = apple.save_raw(self.db, reviews, validation, collection_run_id=run.id)
                combined.fetched += stats.fetched
                combined.valid += stats.valid
                combined.rejected += stats.rejected
                combined.duplicates += stats.duplicates
                combined.new += stats.new
                combined.errors.extend(apple.errors)
                validations.extend(stats.source_validations)
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
                        }
                    )

            analyzed = 0
            if analyze:
                if progress:
                    progress({"stage": "analysis", "status": "start"})
                from app.pipeline.orchestrator import run_analysis_pipeline

                analyzed = run_analysis_pipeline(self.db, progress=progress)
                combined.analyzed = analyzed

            combined.source_validations = validations
            combined.duration_seconds = round(time.monotonic() - started, 2)
            run.status = "completed" if not combined.errors else "completed_with_errors"
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
