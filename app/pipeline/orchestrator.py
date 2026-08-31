"""End-to-end analysis: classify new reviews, cluster, segment, score, report."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.pipeline.analysis import analyze_new_reviews
from app.pipeline.clustering import discover_themes
from app.pipeline.opportunities import rebuild_opportunities
from app.pipeline.segmentation import discover_segments

ProgressCallback = Callable[[dict[str, Any]], None]


def run_analysis_pipeline(
    db: Session,
    *,
    progress: ProgressCallback | None = None,
    analyze_limit: int | None = None,
) -> int:
    analyzed = analyze_new_reviews(db, progress=progress, limit=analyze_limit)
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
        progress({"stage": "pipeline", "status": "complete", "analyzed": analyzed})
    return analyzed
