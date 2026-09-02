"""Per-review AI analysis with batched OpenRouter calls and content-hash caching."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    omitted_after_retry: int = 0
    omitted_ids: list[int] = field(default_factory=list)


@dataclass
class BatchOutcome:
    analyzed: int = 0
    failed: int = 0
    error: str = ""
    http_status: int | None = None
    successful_batches: int = 0
    failed_batches: int = 0
    omitted_after_retry: int = 0
    omitted_ids: list[int] = field(default_factory=list)


def format_ai_analysis_summary(
    *,
    analyzed: int,
    failed: int = 0,
    omitted_after_retry: int = 0,
    selected: int | None = None,
) -> str:
    """Human-readable AI analysis counts. Never hides a genuine partial failure."""
    analyzed_n = int(analyzed or 0)
    failed_n = int(failed or 0)
    omitted_n = int(omitted_after_retry or 0)
    total = int(selected) if selected not in (None, 0) else analyzed_n + failed_n
    if total <= 0 and analyzed_n <= 0 and failed_n <= 0:
        return ""
    if total <= 0:
        total = analyzed_n + failed_n
    lines = [f"AI analysis: {analyzed_n} / {total} reviews analyzed"]
    if omitted_n:
        noun = "review" if omitted_n == 1 else "reviews"
        lines.append(f"{omitted_n} {noun} could not be analyzed after retry.")
    elif failed_n:
        noun = "review" if failed_n == 1 else "reviews"
        lines.append(f"{failed_n} {noun} could not be analyzed.")
    return "\n".join(lines)


def _result_id_token(item: dict[str, Any]) -> str:
    for key in ("id", "review_id", "source_review_id"):
        value = item.get(key)
        if value is None or value == "":
            continue
        token = str(value).strip()
        if token:
            return token
    return ""


def _index_batch_results(
    items: list[dict[str, Any]],
    chunk: list[Review],
) -> dict[int, dict[str, Any]]:
    """Map AI result objects onto reviews by stable ID. Never by array position."""
    by_db_id = {str(review.id): review for review in chunk}
    by_source = {
        str(review.source_review_id).strip(): review
        for review in chunk
        if str(review.source_review_id or "").strip()
    }
    assigned: dict[int, dict[str, Any]] = {}
    seen_tokens: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        token = _result_id_token(item)
        if not token:
            continue
        if token in seen_tokens:
            logger.warning("Ignoring duplicate AI result for review id %s", token)
            continue
        seen_tokens.add(token)
        review = by_db_id.get(token) or by_source.get(token)
        if review is None:
            logger.warning("Ignoring unknown AI review id %s", token)
            continue
        assigned[int(review.id)] = item
    if len(chunk) == 1 and not assigned and items:
        only = items[0]
        if isinstance(only, dict) and not _result_id_token(only):
            assigned[int(chunk[0].id)] = only
    return assigned


def _omit_message(review: Review, *, after_retry: bool) -> str:
    rid = int(review.id or 0)
    if after_retry:
        return f"failed_after_retry: review_id={rid}. AI omitted this review from the batch response."
    return f"omitted: review_id={rid}. AI omitted this review from the batch response."


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


def _merge_outcomes(left: BatchOutcome, right: BatchOutcome) -> BatchOutcome:
    return BatchOutcome(
        analyzed=left.analyzed + right.analyzed,
        failed=left.failed + right.failed,
        error=right.error or left.error,
        http_status=right.http_status or left.http_status,
        successful_batches=left.successful_batches + right.successful_batches,
        failed_batches=left.failed_batches + right.failed_batches,
        omitted_after_retry=left.omitted_after_retry + right.omitted_after_retry,
        omitted_ids=[*left.omitted_ids, *right.omitted_ids],
    )


def _analyze_chunk(
    db: Session,
    provider: AIProvider,
    chunk: list[Review],
    *,
    max_chars: int,
    shrink_depth: int = 0,
) -> BatchOutcome:
    outcome = _analyze_batch(db, provider, chunk, max_chars=max_chars, retry_missing=True)
    if outcome.http_status == 402:
        outcome.successful_batches = 0
        outcome.failed_batches = 1
        return outcome
    if (
        outcome.analyzed == 0
        and outcome.failed == len(chunk)
        and (
            outcome.http_status in {400, 413}
            or "Malformed AI JSON" in (outcome.error or "")
        )
        and len(chunk) > 1
        and shrink_depth < 2
    ):
        mid = max(1, len(chunk) // 2)
        left = _analyze_chunk(db, provider, chunk[:mid], max_chars=max_chars, shrink_depth=shrink_depth + 1)
        if left.http_status == 402:
            return left
        right = _analyze_chunk(db, provider, chunk[mid:], max_chars=max_chars, shrink_depth=shrink_depth + 1)
        return _merge_outcomes(left, right)
    if outcome.analyzed:
        outcome.successful_batches = 1
        outcome.failed_batches = 1 if outcome.failed else 0
        return outcome
    outcome.successful_batches = 0
    outcome.failed_batches = 1
    return outcome


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
    retry_missing: bool = True,
) -> BatchOutcome:
    user = analysis_batch_user_prompt(_prompt_items(chunk, max_chars))
    try:
        raw = provider.complete_json(system=SYSTEM_PROMPT, user=user)
    except AIError as exc:
        error = redact_secrets(str(exc))
        http_status = getattr(exc, "http_status", None)
        logger.warning("AI batch failed (%s reviews): %s", len(chunk), error)
        if http_status == 402:
            return BatchOutcome(error=error, http_status=http_status)
        for review in chunk:
            persist_analysis(db, review, None, "", error, provider, http_status=http_status)
        return BatchOutcome(failed=len(chunk), error=error, http_status=http_status)
    except Exception as exc:  # never crash the pipeline on one batch
        error = redact_secrets(f"Unexpected analysis failure: {exc}")
        logger.exception("Unexpected analysis failure for batch of %s reviews", len(chunk))
        for review in chunk:
            persist_analysis(db, review, None, "", error, provider)
        return BatchOutcome(failed=len(chunk), error=error)

    items, parse_error = parse_batch_payload(raw)
    if parse_error:
        retry_user = (
            user
            + "\n\nIMPORTANT: Your previous reply was not valid JSON. "
            "Reply with ONLY a JSON object. The first character must be {. "
            'Shape: {"results":[{...}]}. One results[] object per review. No markdown.'
        )
        try:
            raw = provider.complete_json(system=SYSTEM_PROMPT, user=retry_user)
            items, parse_error = parse_batch_payload(raw)
        except AIError as exc:
            if getattr(exc, "http_status", None) == 402:
                return BatchOutcome(error=redact_secrets(str(exc)), http_status=402)
            parse_error = parse_error or redact_secrets(str(exc))
        except Exception as exc:
            logger.exception("JSON repair retry failed for batch of %s reviews", len(chunk))
            parse_error = parse_error or redact_secrets(f"Unexpected analysis failure: {exc}")
    if parse_error:
        error = redact_secrets(parse_error)
        omitted_ids = []
        for review in chunk:
            stored_error = error
            if not retry_missing:
                stored_error = f"failed_after_retry: review_id={int(review.id)}. {error}"
                omitted_ids.append(int(review.id))
            persist_analysis(db, review, None, raw, stored_error, provider)
        return BatchOutcome(
            failed=len(chunk),
            error=stored_error if omitted_ids else error,
            omitted_after_retry=len(omitted_ids),
            omitted_ids=omitted_ids,
        )

    assigned = _index_batch_results(items, chunk)
    expected_ids = {int(review.id) for review in chunk}
    returned_ids = set(assigned.keys())
    missing = [review for review in chunk if int(review.id) not in returned_ids]
    logger.info(
        "AI batch matching expected=%s returned=%s missing=%s",
        len(expected_ids),
        len(returned_ids),
        [int(review.id) for review in missing],
    )

    analyzed = 0
    failed = 0
    last_error = ""
    omitted_after_retry = 0
    omitted_ids: list[int] = []
    for review in chunk:
        item = assigned.get(int(review.id))
        if item is None:
            continue
        parsed, error = try_validate_payload(item, _original_blob(review))
        persist_analysis(db, review, parsed, raw, error, provider)
        if parsed is not None:
            analyzed += 1
        else:
            failed += 1
            last_error = error or "AI response failed schema validation."

    if missing and retry_missing:
        for review in missing:
            inner = _analyze_batch(
                db, provider, [review], max_chars=max_chars, retry_missing=False
            )
            if inner.http_status == 402:
                return BatchOutcome(
                    analyzed=analyzed,
                    failed=failed,
                    error=inner.error,
                    http_status=402,
                    omitted_after_retry=omitted_after_retry,
                    omitted_ids=omitted_ids,
                )
            analyzed += inner.analyzed
            failed += inner.failed
            if inner.omitted_after_retry:
                omitted_after_retry += inner.omitted_after_retry
                omitted_ids.extend(inner.omitted_ids)
            elif inner.failed:
                omitted_after_retry += inner.failed
                omitted_ids.append(int(review.id))
            if inner.error:
                last_error = inner.error
    elif missing:
        for review in missing:
            error = _omit_message(review, after_retry=True)
            persist_analysis(db, review, None, raw, error, provider)
            failed += 1
            omitted_after_retry += 1
            omitted_ids.append(int(review.id))
            last_error = error

    return BatchOutcome(
        analyzed=analyzed,
        failed=failed,
        error=last_error,
        omitted_after_retry=omitted_after_retry,
        omitted_ids=omitted_ids,
    )


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
            outcome = _analyze_chunk(db, provider, chunk, max_chars=max_chars)
            result.analyzed += outcome.analyzed
            result.failed += outcome.failed
            result.omitted_after_retry += outcome.omitted_after_retry
            result.omitted_ids.extend(outcome.omitted_ids)
            result.batches_processed += max(1, outcome.successful_batches + outcome.failed_batches)
            result.successful_batches += outcome.successful_batches
            result.failed_batches += outcome.failed_batches
            if outcome.http_status == 402:
                result.last_error = outcome.error
                result.last_http_status = outcome.http_status
                db.commit()
                break
            result.processed += len(chunk)
            processed += len(chunk)
            analyzed_total += outcome.analyzed
            failed_total += outcome.failed
            if outcome.error:
                result.last_error = outcome.error
            if outcome.http_status:
                result.last_http_status = outcome.http_status
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
                        "omitted_after_retry": result.omitted_after_retry,
                        "batch_index": batch_index,
                        "batch_total": batch_total,
                        "percent": percent,
                        "message": outcome.error,
                    }
                )

        if result.omitted_after_retry:
            result.last_error = format_ai_analysis_summary(
                analyzed=result.analyzed,
                failed=result.failed,
                omitted_after_retry=result.omitted_after_retry,
            ) or result.last_error

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
                    "omitted_after_retry": result.omitted_after_retry,
                    "message": result.last_error,
                }
            )
        return result
    finally:
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()
