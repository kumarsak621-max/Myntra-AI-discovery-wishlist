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
from app.pipeline.labels import stored_category_text
from app.schemas import RootCauseItem

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
    selected: int = 0
    batches_processed: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    skipped_already_analyzed: int = 0


def _original_blob(review: Review) -> str:
    return f"{review.title or ''}\n{review.text or ''}".strip()


def _request_batch_size(settings) -> int:
    from config.settings import clamp_batch_size

    alias = getattr(settings, "analysis_batch_size", None) or getattr(settings, "ai_batch_size", None)
    return clamp_batch_size(alias or getattr(settings, "ai_request_batch_size", 10) or 10)


def remaining_analysis_limit(db: Session, settings) -> int:
    """Pending reviews in the selected analysis dataset. Not a smoke-test cap."""
    from app.pipeline.dataset import analysis_dataset_stats

    stats = analysis_dataset_stats(db)
    return max(0, int(stats.get("sample_pending") or 0) + int(stats.get("sample_failed") or 0))


def smoke_test_analyze_limit(db: Session, settings) -> int:
    """Deprecated alias kept for callers. Returns remaining selected reviews, never 1-then-5."""
    return remaining_analysis_limit(db, settings)


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
        intent_vals = [stored_category_text(x) for x in parsed.intent if stored_category_text(x)]
        for extra in parsed.wishlist_behavior:
            text = stored_category_text(extra)
            if text and text not in intent_vals:
                intent_vals.append(text)
        row.intent_json = json.dumps(intent_vals)
        barriers = [stored_category_text(x) for x in parsed.barriers if stored_category_text(x)]
        for extra in parsed.purchase_barriers:
            text = stored_category_text(extra)
            if text and text not in barriers:
                barriers.append(text)
        row.barriers_json = json.dumps(barriers)
        row.uncertainties_json = json.dumps(
            [stored_category_text(x) for x in parsed.uncertainties if stored_category_text(x)]
        )
        row.information_seeking_json = json.dumps(
            [i.model_dump() for i in parsed.information_seeking]
        )
        row.behavioral_signals_json = json.dumps(
            [i.model_dump() for i in parsed.behavioral_signals]
        )
        row.product_category_json = json.dumps(
            [stored_category_text(x) for x in parsed.product_category if stored_category_text(x)]
        )
        row.decision_factors_json = json.dumps(
            [stored_category_text(x) for x in parsed.decision_factors if stored_category_text(x)]
        )
        row.root_cause_observed = stored_category_text(root.observed)
        row.root_cause_inferred = stored_category_text(root.inferred)
        row.root_cause_hypothesized = stored_category_text(root.hypothesized)
        row.root_cause = stored_category_text(
            root.statement or root.hypothesized or root.inferred or root.observed
        )
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
    """Select Myntra-valid reviews in the analysis dataset that still need OpenRouter analysis.

    Already-analyzed rows with a matching content hash and analysis_version are skipped.
    """
    from app.pipeline.dataset import select_analysis_reviews

    selected = select_analysis_reviews(db)
    needed: list[Review] = []
    for review in selected:
        analysis = review.analysis
        status = getattr(analysis, "status", "") if analysis is not None else ""
        version = str(getattr(analysis, "analysis_version", "") or "") if analysis is not None else ""
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
        if (
            status == "analyzed"
            and analysis.is_valid_json
            and version == ANALYSIS_VERSION
        ):
            continue
        if status == "failed" and not include_failed:
            continue
        needed.append(review)
    needed.sort(key=lambda row: (int(row.id or 0),))
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


def _analyze_chunk(
    db: Session,
    provider: AIProvider,
    chunk: list[Review],
    *,
    max_chars: int,
    shrink_depth: int = 0,
) -> tuple[int, int, str, int | None, int, int]:
    """Returns analyzed, failed, error, http_status, successful_batches, failed_batches."""
    analyzed, failed, error, http_status = _analyze_batch(db, provider, chunk, max_chars=max_chars)
    if http_status == 402:
        return analyzed, failed, error, http_status, 0, 1
    if (
        analyzed == 0
        and failed == len(chunk)
        and http_status in {400, 413}
        and len(chunk) > 1
        and shrink_depth < 2
    ):
        mid = max(1, len(chunk) // 2)
        left = _analyze_chunk(db, provider, chunk[:mid], max_chars=max_chars, shrink_depth=shrink_depth + 1)
        if left[3] == 402:
            return left
        right = _analyze_chunk(db, provider, chunk[mid:], max_chars=max_chars, shrink_depth=shrink_depth + 1)
        return (
            left[0] + right[0],
            left[1] + right[1],
            right[2] or left[2],
            right[3] or left[3],
            left[4] + right[4],
            left[5] + right[5],
        )
    if analyzed:
        return analyzed, failed, error, http_status, 1, 1 if failed else 0
    return analyzed, failed, error, http_status, 0, 1


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
    from app.pipeline.dataset import analysis_dataset_stats, select_analysis_reviews

    selected = select_analysis_reviews(db)
    stats = analysis_dataset_stats(db)
    pending = reviews_needing_analysis(
        db, only_failed=only_failed, include_failed=include_failed
    )
    total_pending = len(pending)
    already_analyzed = int(stats.get("sample_analyzed") or stats.get("analyzed_reviews") or 0)
    if limit is not None:
        pending = pending[:limit]

    result = AnalysisRunResult(
        selected=len(selected),
        skipped_already_analyzed=already_analyzed,
    )
    if not pending:
        from app.database import get_review_count

        message = (
            "No real reviews available for analysis."
            if get_review_count(db) == 0
            else "No new Myntra-valid reviews needed analysis (already analyzed)."
        )
        if progress:
            progress(
                {
                    "stage": "analysis",
                    "status": "skipped",
                    "message": message,
                    "selected": result.selected,
                    "analyzed_total": already_analyzed,
                    "pending_total": 0,
                    "failed_total": int(stats.get("failed_reviews") or 0),
                }
            )
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
    if batch_size >= 5:
        max_chars = min(max_chars, 1500)
    from config.settings import clamp_max_tokens

    max_tokens = min(clamp_max_tokens(getattr(settings, "ai_max_tokens", 2000)), 2000)
    logger.info("OpenRouter model: %s", provider.model)
    logger.info("OpenRouter max_tokens: %s", max_tokens)
    logger.info("Batch size: %s", batch_size)
    chunks = _chunks(pending, batch_size)
    batch_total = len(chunks)
    processed = 0
    analyzed_total = already_analyzed
    failed_total = int(stats.get("failed_reviews") or 0)

    try:
        for batch_index, chunk in enumerate(chunks, start=1):
            if rate > 0:
                time.sleep(rate)
            analyzed, failed, error, http_status, ok_batches, bad_batches = _analyze_chunk(
                db, provider, chunk, max_chars=max_chars
            )
            result.analyzed += analyzed
            result.failed += failed
            result.batches_processed += max(1, ok_batches + bad_batches)
            result.successful_batches += ok_batches
            result.failed_batches += bad_batches
            if http_status == 402:
                result.last_error = error
                result.last_http_status = http_status
                db.commit()
                break
            result.processed += len(chunk)
            processed += len(chunk)
            analyzed_total += analyzed
            failed_total += failed
            if error:
                result.last_error = error
            if http_status:
                result.last_http_status = http_status
            db.commit()
            pending_left = max(0, result.selected - analyzed_total - failed_total)
            percent = int(round(100 * analyzed_total / max(1, result.selected)))
            if progress:
                progress(
                    {
                        "stage": "analysis",
                        "status": "progress",
                        "analyzed": result.analyzed,
                        "failed": result.failed,
                        "processed": processed,
                        "total": len(pending),
                        "selected": result.selected,
                        "analyzed_total": analyzed_total,
                        "pending_total": pending_left,
                        "failed_total": failed_total,
                        "batch_index": batch_index,
                        "batch_total": batch_total,
                        "percent": percent,
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
                    "selected": result.selected,
                    "analyzed_total": analyzed_total,
                    "batches_processed": result.batches_processed,
                    "successful_batches": result.successful_batches,
                    "failed_batches": result.failed_batches,
                    "message": result.last_error,
                }
            )
        return result
    finally:
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()
