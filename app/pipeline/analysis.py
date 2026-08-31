"""Per-review AI analysis with content-hash caching."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.ai.prompts import SYSTEM_PROMPT, analysis_user_prompt
from app.ai.provider import AIError, AIProvider
from app.ai.schema import try_validate_analysis
from app.config import get_settings
from app.models import Analysis, Review, utcnow
from app.schemas import RootCauseItem
from config.settings import official_ids

ANALYSIS_VERSION = "1"

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


def _original_blob(review: Review) -> str:
    return f"{review.title or ''}\n{review.text or ''}".strip()


def persist_analysis(db: Session, review: Review, parsed, raw: str, error: str, provider: AIProvider) -> Analysis:
    root = parsed.root_cause if parsed else RootCauseItem()
    if not isinstance(root, RootCauseItem):
        root = RootCauseItem(statement=str(root or ""))

    row = review.analysis
    if row is None:
        row = Analysis(review_id=review.id)
        db.add(row)

    row.content_hash = review.content_hash
    row.provider = provider.provider_name
    row.model = provider.model
    row.raw_response = (raw or "")[:20000]
    row.parse_error = error or ""
    row.is_valid_json = parsed is not None
    row.analyzed_at = utcnow()
    row.status = "analyzed" if parsed is not None else "failed"
    row.analysis_version = ANALYSIS_VERSION

    if parsed is not None:
        row.relevance = parsed.relevance
        row.wishlist_signal = parsed.wishlist_signal
        row.purchase_signal = parsed.purchase_signal
        row.purchase_hesitation = parsed.purchase_hesitation
        row.intent_json = json.dumps(parsed.intent)
        row.barriers_json = json.dumps(parsed.barriers)
        row.uncertainties_json = json.dumps(parsed.uncertainties)
        row.information_seeking_json = json.dumps(
            [i.model_dump() for i in parsed.information_seeking]
        )
        row.behavioral_signals_json = json.dumps(
            [i.model_dump() for i in parsed.behavioral_signals]
        )
        row.product_category_json = json.dumps(parsed.product_category)
        row.decision_factors_json = json.dumps(parsed.decision_factors)
        row.root_cause_observed = root.observed
        row.root_cause_inferred = root.inferred
        row.root_cause_hypothesized = root.hypothesized
        row.root_cause = root.statement or root.hypothesized or root.inferred or root.observed
        row.sentiment = parsed.sentiment
        row.evidence_strength = parsed.evidence_strength
        row.confidence = parsed.confidence
    db.flush()
    return row


def reviews_needing_analysis(db: Session) -> list[Review]:
    rows = (
        db.query(Review)
        .filter(
            Review.is_empty.is_(False),
            Review.is_valid_source.is_(True),
            Review.app_id.in_(list(official_ids())),
        )
        .all()
    )
    needed: list[Review] = []
    for review in rows:
        analysis = review.analysis
        if analysis is None or getattr(analysis, "status", "") in {"", "pending"}:
            needed.append(review)
            continue
        if analysis.status == "failed" and analysis.content_hash == review.content_hash:
            continue
        if analysis.content_hash != review.content_hash:
            needed.append(review)
            continue
        if analysis.status == "analyzed" and analysis.is_valid_json:
            continue
        if not analysis.is_valid_json:
            needed.append(review)
    return needed


def analyze_review(provider: AIProvider, review: Review) -> tuple[Any, str, str]:
    blob = _original_blob(review)
    user = analysis_user_prompt(
        source=review.source,
        app_name=review.app_name,
        data_classification=review.data_classification,
        rating=review.rating,
        title=review.title or "",
        text=(review.text or "")[: get_settings().ai_max_review_chars],
        region=review.region or "",
    )
    raw = provider.complete_json(system=SYSTEM_PROMPT, user=user)
    parsed, error = try_validate_analysis(raw, blob)
    return parsed, raw, error


def analyze_new_reviews(
    db: Session,
    *,
    progress: ProgressCallback | None = None,
    limit: int | None = None,
    provider: AIProvider | None = None,
) -> int:
    settings = get_settings()
    provider = provider or AIProvider(settings)
    pending = reviews_needing_analysis(db)
    total_pending = len(pending)
    if limit is not None:
        pending = pending[:limit]

    if not pending:
        from app.database import get_review_count

        message = (
            "Not enough real feedback collected for analysis."
            if get_review_count(db) == 0
            else "No new Myntra-valid reviews needed analysis (already analyzed)."
        )
        if progress:
            progress({"stage": "analysis", "status": "skipped", "message": message})
        return 0

    if not provider.available():
        message = (
            "AI analysis failed: OPENROUTER_API_KEY is not set in Streamlit Secrets or .env. "
            f"{total_pending} real Myntra-valid reviews are waiting for analysis."
        )
        logger.error(message)
        if progress:
            progress({"stage": "analysis", "status": "error", "message": message})
        raise AIError(message)

    analyzed = 0
    for index, review in enumerate(pending, start=1):
        try:
            time.sleep(settings.ai_rate_limit_seconds)
            parsed, raw, error = analyze_review(provider, review)
            persist_analysis(db, review, parsed, raw, error, provider)
            if parsed is not None:
                analyzed += 1
            elif progress:
                progress(
                    {
                        "stage": "analysis",
                        "status": "review_error",
                        "review_id": review.id,
                        "message": error,
                    }
                )
        except AIError as exc:
            logger.warning("AI failed for review %s: %s", review.id, exc)
            persist_analysis(db, review, None, "", str(exc), provider)
            if progress:
                progress(
                    {
                        "stage": "analysis",
                        "status": "review_error",
                        "review_id": review.id,
                        "message": str(exc),
                    }
                )
        except Exception as exc:  # never crash the pipeline on one review
            logger.exception("Unexpected analysis failure for review %s", review.id)
            persist_analysis(db, review, None, "", str(exc), provider)

        if index % 5 == 0:
            db.commit()
            if progress:
                progress(
                    {
                        "stage": "analysis",
                        "status": "progress",
                        "analyzed": analyzed,
                        "processed": index,
                        "total": len(pending),
                    }
                )

    db.commit()
    if progress:
        progress({"stage": "analysis", "status": "complete", "analyzed": analyzed, "total": len(pending)})
    return analyzed
