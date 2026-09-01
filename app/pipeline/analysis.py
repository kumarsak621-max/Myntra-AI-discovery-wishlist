"""Per-review AI analysis with batched OpenRouter calls and content-hash caching."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.ai.prompts import SYSTEM_PROMPT, analysis_batch_user_prompt, analysis_user_prompt
from app.ai.provider import AIError, AIProvider, redact_secrets
from app.ai.schema import parse_batch_payload, try_validate_analysis, try_validate_payload
from app.config import get_settings
from app.models import Analysis, Review, utcnow
from app.schemas import RootCauseItem
from config.settings import official_ids

ANALYSIS_VERSION = "1"

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class AnalysisRunResult:
    analyzed: int = 0
    failed: int = 0
    last_error: str = ""
    processed: int = 0
    last_http_status: int | None = None


def _original_blob(review: Review) -> str:
    return f"{review.title or ''}\n{review.text or ''}".strip()


def _request_batch_size(settings) -> int:
    from config.settings import clamp_batch_size

    alias = getattr(settings, "ai_batch_size", None)
    return clamp_batch_size(alias or getattr(settings, "ai_request_batch_size", 5) or 5)


def smoke_test_analyze_limit(db: Session, settings) -> int:
    """First run: 1 review. Next runs until 6 analyzed: 5. Then the configured cap."""
    from app.database import get_database_diagnostics

    configured = int(getattr(settings, "ai_analysis_batch_size", 60) or 60)
    configured = max(1, configured)
    diag = get_database_diagnostics(db)
    analyzed = int(diag.get("analyzed_reviews") or 0)
    if analyzed == 0:
        return 1
    if analyzed < 6:
        return min(configured, 5)
    return configured


def _chunks(items: Sequence[Review], size: int) -> list[list[Review]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def persist_analysis(
    db: Session,
    review: Review,
    parsed,
    raw: str,
    error: str,
    provider: AIProvider,
    *,
    http_status: int | None = None,
) -> Analysis:
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
    row.parse_error = redact_secrets(error or "")  # stored analysis_error; never contains the API key
    row.is_valid_json = parsed is not None
    row.analyzed_at = utcnow()
    row.status = "analyzed" if parsed is not None else "failed"
    row.analysis_version = ANALYSIS_VERSION
    if http_status and hasattr(row, "http_status"):
        row.http_status = int(http_status)
    usage = getattr(provider, "last_usage", None) or {}
    if parsed is not None and isinstance(usage, dict):
        if hasattr(row, "prompt_tokens"):
            row.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if hasattr(row, "completion_tokens"):
            row.completion_tokens = int(usage.get("completion_tokens") or 0)
        if hasattr(row, "total_tokens"):
            row.total_tokens = int(usage.get("total_tokens") or 0)

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


def reviews_needing_analysis(
    db: Session,
    *,
    only_failed: bool = False,
    include_failed: bool = True,
) -> list[Review]:
    """Select Myntra-valid reviews that still need OpenRouter analysis.

    Analyze Pending: pending / missing / stale-hash rows (include_failed=False).
    Retry Failed: only status=failed (only_failed=True).
    Default pipeline: pending and failed, never successfully analyzed rows.
    """
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
        status = getattr(analysis, "status", "") if analysis is not None else ""
        if only_failed:
            if analysis is not None and status == "failed":
                needed.append(review)
            continue
        if analysis is None or status in {"", "pending"}:
            needed.append(review)
            continue
        if analysis.content_hash != review.content_hash:
            needed.append(review)
            continue
        if status == "analyzed" and analysis.is_valid_json:
            continue
        if status == "failed" and not include_failed:
            continue
        needed.append(review)
    return needed


def _prompt_items(reviews: Sequence[Review], max_chars: int) -> list[dict[str, Any]]:
    items = []
    for review in reviews:
        items.append(
            {
                "id": str(review.id),
                "source": review.source,
                "app_name": review.app_name,
                "data_classification": review.data_classification,
                "region": review.region or "",
                "rating": review.rating,
                "title": review.title or "",
                "text": (review.text or "")[:max_chars],
            }
        )
    return items


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


def _analyze_batch(
    db: Session,
    provider: AIProvider,
    chunk: list[Review],
    *,
    max_chars: int,
) -> tuple[int, int, str, int | None]:
    user = analysis_batch_user_prompt(_prompt_items(chunk, max_chars))
    http_status: int | None = None
    try:
        raw = provider.complete_json(system=SYSTEM_PROMPT, user=user)
    except AIError as exc:
        error = redact_secrets(str(exc))
        http_status = getattr(exc, "http_status", None)
        logger.warning("AI batch failed (%s reviews): %s", len(chunk), error)
        if http_status == 402:
            return 0, 0, error, http_status
        for review in chunk:
            persist_analysis(db, review, None, "", error, provider, http_status=http_status)
        return 0, len(chunk), error, http_status
    except Exception as exc:  # never crash the pipeline on one batch
        error = redact_secrets(f"Unexpected analysis failure: {exc}")
        logger.exception("Unexpected analysis failure for batch of %s reviews", len(chunk))
        for review in chunk:
            persist_analysis(db, review, None, "", error, provider)
        return 0, len(chunk), error, None

    items, parse_error = parse_batch_payload(raw)
    if parse_error:
        error = redact_secrets(parse_error)
        for review in chunk:
            persist_analysis(db, review, None, raw, error, provider)
        return 0, len(chunk), error, None

    by_id: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for item in items:
        token = str(item.get("id") or item.get("source_review_id") or "").strip()
        if token:
            by_id[token] = item
        else:
            anonymous.append(item)

    analyzed = 0
    failed = 0
    last_error = ""
    for review in chunk:
        item = by_id.pop(str(review.id), None)
        source_id = str(review.source_review_id or "").strip()
        if item is None and source_id:
            item = by_id.pop(source_id, None)
        if item is None and anonymous:
            item = anonymous.pop(0)
        if item is None:
            error = "AI omitted this review from the batch response."
            persist_analysis(db, review, None, raw, error, provider)
            failed += 1
            last_error = error
            continue
        parsed, error = try_validate_payload(item, _original_blob(review))
        persist_analysis(db, review, parsed, raw, error, provider)
        if parsed is not None:
            analyzed += 1
        else:
            failed += 1
            last_error = error or "AI response failed schema validation."
    return analyzed, failed, last_error, None


def analyze_new_reviews(
    db: Session,
    *,
    progress: ProgressCallback | None = None,
    limit: int | None = None,
    provider: AIProvider | None = None,
    only_failed: bool = False,
    include_failed: bool = True,
) -> AnalysisRunResult:
    provider = provider or AIProvider(get_settings())
    settings = getattr(provider, "settings", None) or get_settings()
    from app.pipeline.dataset import enforce_review_limit

    enforce_review_limit(db, getattr(settings, "max_dataset_reviews", None))
    pending = reviews_needing_analysis(
        db, only_failed=only_failed, include_failed=include_failed
    )
    total_pending = len(pending)
    if limit is not None:
        pending = pending[:limit]

    result = AnalysisRunResult()
    if not pending:
        from app.database import get_review_count

        message = (
            "No real reviews available for analysis."
            if get_review_count(db) == 0
            else "No new Myntra-valid reviews needed analysis (already analyzed)."
        )
        if progress:
            progress({"stage": "analysis", "status": "skipped", "message": message})
        return result

    if not provider.available():
        message = (
            "OpenRouter API key is not configured. "
            "Add OPENROUTER_API_KEY to Streamlit Secrets or .env. "
            f"{total_pending} real Myntra-valid reviews are waiting for analysis."
        )
        logger.error(message)
        if progress:
            progress({"stage": "analysis", "status": "error", "message": message})
        raise AIError(message)

    batch_size = _request_batch_size(settings)
    rate = float(getattr(settings, "ai_rate_limit_seconds", 0) or 0)
    max_chars = int(getattr(settings, "ai_max_review_chars", 4000) or 4000)
    processed = 0

    try:
        for chunk in _chunks(pending, batch_size):
            if rate > 0:
                time.sleep(rate)
            analyzed, failed, error, http_status = _analyze_batch(db, provider, chunk, max_chars=max_chars)
            result.analyzed += analyzed
            result.failed += failed
            if http_status == 402:
                result.last_error = error
                result.last_http_status = http_status
                db.commit()
                break
            result.processed += len(chunk)
            processed += len(chunk)
            if error:
                result.last_error = error
            if http_status:
                result.last_http_status = http_status
            db.commit()
            if progress:
                progress(
                    {
                        "stage": "analysis",
                        "status": "progress",
                        "analyzed": result.analyzed,
                        "failed": result.failed,
                        "processed": processed,
                        "total": len(pending),
                        "message": error,
                    }
                )

        db.commit()
        if progress:
            status = "complete" if result.analyzed else ("error" if result.failed else "complete")
            progress(
                {
                    "stage": "analysis",
                    "status": status,
                    "analyzed": result.analyzed,
                    "failed": result.failed,
                    "total": len(pending),
                    "message": result.last_error,
                }
            )
        return result
    finally:
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()
