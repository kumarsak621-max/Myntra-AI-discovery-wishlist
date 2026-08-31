"""End-to-end analysis: classify new reviews, cluster, segment, score, report."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.pipeline.analysis import AnalysisRunResult, analyze_new_reviews
from app.pipeline.clustering import discover_themes
from app.pipeline.opportunities import rebuild_opportunities
from app.pipeline.segmentation import discover_segments

ProgressCallback = Callable[[dict[str, Any]], None]


def run_analysis_pipeline(
    db: Session,
    *,
    progress: ProgressCallback | None = None,
    analyze_limit: int | None = None,
    only_failed: bool = False,
    include_failed: bool = True,
) -> AnalysisRunResult:
    result = analyze_new_reviews(
        db,
        progress=progress,
        limit=analyze_limit,
        only_failed=only_failed,
        include_failed=include_failed,
    )
    if result.analyzed == 0:
        message = (
            result.last_error
            or "Discovery insights could not be generated because AI analysis failed."
        )
        if progress:
            progress({"stage": "insights", "status": "blocked", "message": message})
        return result
    if progress:
        progress({"stage": "themes", "status": "start"})
    discover_themes(db)
    if progress:
        progress({"stage": "segments", "status": "start"})
    discover_segments(db)
    if progress:
        progress({"stage": "opportunities", "status": "start"})
    rebuild_opportunities(db)
    if progress:
        progress(
            {
                "stage": "pipeline",
                "status": "complete",
                "analyzed": result.analyzed,
                "failed": result.failed,
                "message": result.last_error,
            }
        )
    return result
