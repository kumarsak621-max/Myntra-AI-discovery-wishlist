"""HTTP API for collection, exploration, and reporting."""

from __future__ import annotations

import csv
import io
import json
import logging
import queue
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from app.collectors.engine import CollectionEngine
from app.config import get_settings
from app.database import SessionLocal, get_db
from app.models import CollectionRun, Opportunity, Review, Segment, Source, Theme
from app.pipeline.cleaning import clean_review
from app.pipeline.dedup import content_hash
from app.pipeline.orchestrator import run_analysis_pipeline
from app.pipeline.labels import merge_category_rows
from app.pipeline.quantification import (
    information_seeking,
    label_distribution,
    overview_metrics,
    signal_counts,
    time_trends,
)
from app.pipeline.report import build_report, evidence_cards
from app.pipeline.validation import validate_app_identity
from config.settings import (
    OFFICIAL_APPLE_APP_ID,
    OFFICIAL_APPLE_APP_NAME,
    OFFICIAL_APPLE_APP_URL,
    OFFICIAL_GOOGLE_PLAY_APP_ID,
    OFFICIAL_GOOGLE_PLAY_APP_NAME,
    OFFICIAL_GOOGLE_PLAY_URL,
    official_ids,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class CollectRequest(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["google_play", "apple_app_store"])
    max_reviews: int | None = None
    analyze: bool = True


def serialize_review(review: Review, include_analysis: bool = True) -> dict[str, Any]:
    analysis = None
    if include_analysis and review.analysis and review.analysis.is_valid_json:
        a = review.analysis
        analysis = {
            "relevance": a.relevance,
            "wishlist_signal": a.wishlist_signal,
            "purchase_signal": a.purchase_signal,
            "purchase_hesitation": a.purchase_hesitation,
            "intent": _loads(a.intent_json),
            "barriers": _loads(a.barriers_json),
            "uncertainties": _loads(a.uncertainties_json),
            "information_seeking": _loads(a.information_seeking_json),
            "behavioral_signals": _loads(a.behavioral_signals_json),
            "product_category": _loads(a.product_category_json),
            "decision_factors": _loads(a.decision_factors_json),
            "root_cause": {
                "observed": a.root_cause_observed,
                "inferred": a.root_cause_inferred,
                "hypothesized": a.root_cause_hypothesized,
                "statement": a.root_cause,
            },
            "sentiment": a.sentiment,
            "evidence_strength": a.evidence_strength,
            "confidence": a.confidence,
            "model": a.model,
            "provider": a.provider,
            "analyzed_at": a.analyzed_at.isoformat() if a.analyzed_at else None,
        }
    warning = ""
    if not review.is_valid_source:
        warning = (
            "WARNING: This record is REFERENCE / NON-MYNTRA DATA and must not be "
            "presented as Myntra evidence."
        )
    synthetic = ""
    if review.is_synthetic:
        synthetic = "SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA"
    return {
        "id": review.id,
        "source": review.source,
        "source_review_id": review.source_review_id,
        "app_id": review.app_id,
        "app_name": review.app_name,
        "developer": review.developer,
        "region": review.region,
        "rating": review.rating,
        "title": review.title,
        "text": review.text,
        "review_date": review.review_date.isoformat() if review.review_date else None,
        "last_update_date": review.last_update_date.isoformat() if review.last_update_date else None,
        "app_version": review.app_version,
        "source_url": review.source_url,
        "collected_at": review.collected_at.isoformat() if review.collected_at else None,
        "is_valid_source": review.is_valid_source,
        "is_duplicate": review.is_duplicate,
        "is_synthetic": review.is_synthetic,
        "data_classification": review.data_classification,
        "warning": warning,
        "synthetic_label": synthetic,
        "language_notes": review.language_notes,
        "is_spam": review.is_spam,
        "is_short": review.is_short,
        "is_promotional": review.is_promotional,
        "analysis": analysis,
        "analyzed": bool(review.analysis and review.analysis.is_valid_json),
    }


def _loads(raw: str) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def serialize_source(src: Source) -> dict[str, Any]:
    return {
        "platform": src.platform,
        "app_id": src.app_id,
        "detected_app_name": src.detected_app_name,
        "detected_developer": src.detected_developer,
        "region": src.region,
        "expected_app": src.expected_app,
        "validation_status": src.validation_status,
        "is_valid_for_myntra": src.is_valid_for_myntra,
        "warning": src.warning,
        "collection_date": src.collection_date.isoformat() if src.collection_date else None,
        "last_collection_at": src.last_collection_at.isoformat() if src.last_collection_at else None,
        "review_count": src.review_count,
        "data_classification": (
            "MYNTRA EVIDENCE" if src.is_valid_for_myntra else "REFERENCE / NON-MYNTRA DATA"
        ),
        "validation_result": "PASS" if src.is_valid_for_myntra else "FAIL",
        "expected_app": src.expected_app,
        "store_url": (
            OFFICIAL_GOOGLE_PLAY_URL
            if src.platform == "google_play"
            else OFFICIAL_APPLE_APP_URL
        ),
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config")
def config_public() -> dict[str, Any]:
    s = get_settings()
    return {
        "ai_provider": s.ai_provider,
        "ai_model": s.resolved_model,
        "ai_configured": s.has_ai_credentials,
        "expected_app_name": s.expected_app_name,
        "google_play": {
            "app": OFFICIAL_GOOGLE_PLAY_APP_NAME,
            "app_id": s.google_play_app_id,
            "package": s.google_play_app_id,
            "url": s.google_play_url,
            "expected_app": OFFICIAL_GOOGLE_PLAY_APP_NAME,
            "expected_id": OFFICIAL_GOOGLE_PLAY_APP_ID,
            "expected_url": OFFICIAL_GOOGLE_PLAY_URL,
            "max_reviews": s.google_play_max_reviews,
            "batch_size": s.google_play_batch_size,
            "language": s.google_play_language,
            "country": s.google_play_country,
        },
        "apple": {
            "app": OFFICIAL_APPLE_APP_NAME,
            "app_id": s.apple_app_id,
            "url": s.apple_app_store_url,
            "expected_app": OFFICIAL_APPLE_APP_NAME,
            "expected_id": OFFICIAL_APPLE_APP_ID,
            "expected_url": OFFICIAL_APPLE_APP_URL,
            "primary_region": s.apple_primary_region,
            "fallback_region": s.apple_fallback_region,
            "max_reviews": s.apple_max_reviews,
        },
    }


@router.get("/sources")
def list_sources(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(Source).all()
    s = get_settings()
    configured = [
        {
            "platform": "google_play",
            "app_id": s.google_play_app_id,
            "region": s.google_play_country,
        },
        {
            "platform": "apple_app_store",
            "app_id": s.apple_app_id,
            "primary_region": s.apple_primary_region,
            "fallback_region": s.apple_fallback_region,
        },
    ]
    return {"configured": configured, "collected": [serialize_source(r) for r in rows]}


@router.get("/collection-status")
def collection_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    s = get_settings()
    latest = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()

    def _source(platform: str, app_id: str) -> Source | None:
        row = (
            db.query(Source)
            .filter(Source.platform == platform, Source.app_id == app_id)
            .order_by(Source.last_collection_at.desc())
            .first()
        )
        if row:
            return row
        return (
            db.query(Source)
            .filter(Source.platform == platform)
            .order_by(Source.last_collection_at.desc())
            .first()
        )

    def _block(platform: str, app_id: str, extra: dict[str, Any]) -> dict[str, Any]:
        src = _source(platform, app_id)
        count = (
            db.query(Review)
            .filter(Review.source == platform, Review.app_id == app_id, Review.is_empty.is_(False))
            .count()
        )
        return {
            **extra,
            "validation": "PASS" if src and src.is_valid_for_myntra else "FAIL",
            "detected_app": src.detected_app_name if src else "",
            "detected_developer": src.detected_developer if src else "",
            "reviews_collected": count,
            "new_reviews": latest.new_count if latest else 0,
            "duplicates": latest.duplicates if latest else 0,
            "last_collection": src.last_collection_at.isoformat() if src and src.last_collection_at else None,
            "warning": src.warning if src else "",
        }

    return {
        "google_play": _block(
            "google_play",
            s.google_play_app_id,
            {
                "source": "Google Play",
                "app": OFFICIAL_GOOGLE_PLAY_APP_NAME,
                "package": s.google_play_app_id,
                "url": OFFICIAL_GOOGLE_PLAY_URL,
                "expected_app": OFFICIAL_GOOGLE_PLAY_APP_NAME,
                "configured_id": s.google_play_app_id,
            },
        ),
        "apple_app_store": _block(
            "apple_app_store",
            s.apple_app_id,
            {
                "source": "Apple App Store",
                "app": OFFICIAL_APPLE_APP_NAME,
                "app_id": s.apple_app_id,
                "url": OFFICIAL_APPLE_APP_URL,
                "expected_app": OFFICIAL_APPLE_APP_NAME,
                "configured_id": s.apple_app_id,
                "primary_region": "India",
                "fallback_region": "US",
            },
        ),
        "latest_run": {
            "fetched": latest.fetched if latest else 0,
            "valid": latest.valid if latest else 0,
            "rejected": latest.rejected if latest else 0,
            "duplicates": latest.duplicates if latest else 0,
            "new": latest.new_count if latest else 0,
            "analyzed": latest.analyzed if latest else 0,
            "status": latest.status if latest else None,
        }
        if latest
        else None,
    }


@router.get("/overview")
def overview(
    myntra_only: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    data = overview_metrics(db)
    data["signals"] = signal_counts(db, myntra_only=myntra_only)
    data["time_trends"] = time_trends(db, myntra_only=myntra_only)
    data["sources"] = [serialize_source(s) for s in db.query(Source).all()]
    latest = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
    data["latest_run"] = (
        {
            "id": latest.id,
            "status": latest.status,
            "fetched": latest.fetched,
            "valid": latest.valid,
            "rejected": latest.rejected,
            "duplicates": latest.duplicates,
            "new": latest.new_count,
            "analyzed": latest.analyzed,
            "duration_seconds": latest.duration_seconds,
            "errors": _loads(latest.errors_json),
            "started_at": latest.started_at.isoformat() if latest.started_at else None,
            "finished_at": latest.finished_at.isoformat() if latest.finished_at else None,
        }
        if latest
        else None
    )
    return data


@router.get("/collection-runs")
def collection_runs(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.query(CollectionRun).order_by(CollectionRun.id.desc()).limit(20).all()
    return [
        {
            "id": r.id,
            "status": r.status,
            "sources": r.sources,
            "fetched": r.fetched,
            "valid": r.valid,
            "rejected": r.rejected,
            "duplicates": r.duplicates,
            "new": r.new_count,
            "analyzed": r.analyzed,
            "duration_seconds": r.duration_seconds,
            "errors": _loads(r.errors_json),
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        }
        for r in rows
    ]


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/collect")
def collect(body: CollectRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    engine = CollectionEngine(db)
    try:
        stats = engine.run(
            sources=body.sources,
            max_reviews=body.max_reviews,
            analyze=body.analyze,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return stats.model_dump(mode="json")


@router.get("/collect/stream")
def collect_stream(
    sources: str = "google_play,apple_app_store",
    max_reviews: int | None = None,
    analyze: bool = True,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    wanted = [s.strip() for s in sources.split(",") if s.strip()]

    def generate() -> Iterator[str]:
        events: queue.Queue = queue.Queue()

        def progress(event: dict[str, Any]) -> None:
            events.put(event)

        def worker() -> None:
            session = SessionLocal()
            try:
                engine = CollectionEngine(session)
                stats = engine.run(
                    sources=wanted,
                    max_reviews=max_reviews,
                    analyze=analyze,
                    progress=progress,
                )
                events.put({"stage": "done", "status": "complete", "stats": stats.model_dump(mode="json")})
            except Exception as exc:
                events.put({"stage": "done", "status": "failed", "message": str(exc)})
            finally:
                session.close()
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()
        yield f"data: {json.dumps({'stage': 'start', 'status': 'running'})}\n\n"
        while True:
            item = events.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/analyze")
def analyze(db: Session = Depends(get_db)) -> dict[str, Any]:
    result = run_analysis_pipeline(db)
    return {
        "analyzed": result.analyzed,
        "failed": result.failed,
        "last_error": result.last_error or None,
    }


@router.get("/reviews")
def list_reviews(
    source: str | None = None,
    rating: int | None = None,
    theme: str | None = None,
    barrier: str | None = None,
    intent: str | None = None,
    uncertainty: str | None = None,
    category: str | None = None,
    confidence_min: int | None = None,
    purchase_signal: str | None = None,
    wishlist_signal: str | None = None,
    myntra_only: bool = False,
    q: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    query = db.query(Review).options(joinedload(Review.analysis)).filter(Review.is_empty.is_(False))
    if myntra_only:
        query = query.filter(
            Review.is_valid_source.is_(True),
            Review.app_id.in_(list(official_ids())),
        )
    if source:
        query = query.filter(Review.source == source)
    if rating is not None:
        query = query.filter(Review.rating == rating)
    if q:
        like = f"%{q}%"
        query = query.filter(Review.text.ilike(like) | Review.title.ilike(like))

    rows = query.order_by(Review.collected_at.desc()).all()
    filtered: list[Review] = []
    for review in rows:
        analysis = review.analysis
        if purchase_signal and (not analysis or analysis.purchase_signal != purchase_signal):
            continue
        if wishlist_signal and (not analysis or analysis.wishlist_signal != wishlist_signal):
            continue
        if confidence_min is not None and (not analysis or analysis.confidence < confidence_min):
            continue
        if barrier:
            labels = [str(x).lower() for x in _loads(analysis.barriers_json)] if analysis else []
            if barrier.lower() not in " ".join(labels):
                if not any(barrier.lower() in lab for lab in labels):
                    continue
        if intent:
            labels = [str(x).lower() for x in _loads(analysis.intent_json)] if analysis else []
            if not any(intent.lower() in lab for lab in labels):
                continue
        if uncertainty:
            labels = [str(x).lower() for x in _loads(analysis.uncertainties_json)] if analysis else []
            if not any(uncertainty.lower() in lab for lab in labels):
                continue
        if category:
            labels = [str(x).lower() for x in _loads(analysis.product_category_json)] if analysis else []
            if not any(category.lower() in lab for lab in labels):
                continue
        if theme:
            hay = " ".join(
                [
                    analysis.root_cause if analysis else "",
                    " ".join(_loads(analysis.barriers_json) if analysis else []),
                    " ".join(_loads(analysis.uncertainties_json) if analysis else []),
                ]
            ).lower()
            if theme.lower() not in hay:
                continue
        filtered.append(review)

    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [serialize_review(r) for r in page],
    }


@router.get("/reviews/{review_id}")
def get_review(review_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    review = (
        db.query(Review)
        .options(joinedload(Review.analysis))
        .filter(Review.id == review_id)
        .one_or_none()
    )
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return serialize_review(review)


@router.get("/intents")
def intents(myntra_only: bool = True, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return label_distribution(db, "intent", myntra_only=myntra_only)


@router.get("/barriers")
def barriers(myntra_only: bool = True, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return label_distribution(db, "barriers", myntra_only=myntra_only)


@router.get("/uncertainties")
def uncertainties(myntra_only: bool = True, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return label_distribution(db, "uncertainties", myntra_only=myntra_only)


@router.get("/themes")
def themes(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.query(Theme).order_by(Theme.review_count.desc()).all()
    return merge_category_rows(
        [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "review_count": t.review_count,
                "myntra_review_count": t.myntra_review_count,
                "reference_review_count": t.reference_review_count,
                "sources": _loads(t.sources_json),
                "evidence_ids": _loads(t.evidence_ids_json),
                "is_emergent": t.is_emergent,
            }
            for t in rows
        ],
        label_keys=("name", "label"),
        count_keys=("review_count", "count"),
        id_keys=("evidence_ids", "review_ids"),
    )


@router.get("/segments")
def segments(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.query(Segment).order_by(Segment.review_count.desc()).all()
    return merge_category_rows(
        [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "basis": s.basis,
                "review_count": s.review_count,
                "myntra_review_count": s.myntra_review_count,
                "sources": _loads(s.sources_json),
                "evidence_ids": _loads(s.evidence_ids_json),
            }
            for s in rows
        ],
        label_keys=("name", "label"),
        count_keys=("review_count", "count"),
        id_keys=("evidence_ids", "review_ids"),
    )


@router.get("/information-seeking")
def info_seeking(myntra_only: bool = True, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return information_seeking(db, myntra_only=myntra_only)


@router.get("/opportunities")
def opportunities(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
    return [
        {
            "id": o.id,
            "rank": o.rank,
            "name": o.name,
            "user_problem": o.user_problem,
            "reach": o.reach,
            "frequency": o.frequency,
            "purchase_impact": o.purchase_impact,
            "severity": o.severity,
            "evidence_confidence": o.evidence_confidence,
            "score": o.score,
            "relevant_count": o.relevant_count,
            "total_relevant": o.total_relevant,
            "percentage": o.percentage,
            "sources": _loads(o.sources_json),
            "evidence_ids": _loads(o.evidence_ids_json),
            "what_we_know": o.what_we_know,
            "what_we_dont_know": o.what_we_dont_know,
            "why_investigate": o.why_investigate,
            "cross_source_status": o.cross_source_status,
            "includes_non_myntra": o.includes_non_myntra,
            "myntra_only_count": o.myntra_only_count,
        }
        for o in rows
    ]


@router.get("/evidence/{kind}/{item_id}")
def evidence(kind: str, item_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    if kind == "opportunity":
        row = db.query(Opportunity).filter(Opportunity.id == item_id).one_or_none()
        if row is None:
            raise HTTPException(404, "Opportunity not found")
        ids = _loads(row.evidence_ids_json)
        return {
            "insight": row.user_problem,
            "kind": "opportunity",
            "quantitative_signal": {
                "count": row.relevant_count,
                "percentage": row.percentage,
                "denominator": row.total_relevant,
                "score": row.score,
            },
            "sources": _loads(row.sources_json),
            "confidence": row.evidence_confidence,
            "unknowns": row.what_we_dont_know,
            "business_relevance": row.why_investigate,
            "includes_non_myntra": row.includes_non_myntra,
            "evidence": evidence_cards(db, ids, limit=25),
        }
    if kind == "theme":
        row = db.query(Theme).filter(Theme.id == item_id).one_or_none()
        if row is None:
            raise HTTPException(404, "Theme not found")
        return {
            "insight": row.name,
            "kind": "theme",
            "evidence": evidence_cards(db, _loads(row.evidence_ids_json), limit=25),
        }
    if kind == "segment":
        row = db.query(Segment).filter(Segment.id == item_id).one_or_none()
        if row is None:
            raise HTTPException(404, "Segment not found")
        return {
            "insight": row.name,
            "kind": "segment",
            "basis": row.basis,
            "evidence": evidence_cards(db, _loads(row.evidence_ids_json), limit=25),
        }
    raise HTTPException(400, "kind must be opportunity, theme, or segment")


@router.get("/report")
def report(db: Session = Depends(get_db)) -> dict[str, Any]:
    return build_report(db)


@router.post("/upload")
async def upload_fallback(
    file: UploadFile = File(...),
    is_synthetic: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Secondary CSV/JSON fallback. Primary path remains automatic collection."""
    name = (file.filename or "").lower()
    raw = await file.read()
    records: list[dict[str, Any]] = []
    try:
        if name.endswith(".json"):
            payload = json.loads(raw.decode("utf-8"))
            records = payload if isinstance(payload, list) else payload.get("reviews") or []
        else:
            text = raw.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            records = list(reader)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse file: {exc}") from exc

    inserted = 0
    duplicates = 0
    rejected = 0
    for rec in records:
        text_val = str(rec.get("text") or rec.get("content") or rec.get("review") or "")
        title = str(rec.get("title") or "")
        if not text_val.strip() and not title.strip():
            rejected += 1
            continue
        source = str(rec.get("source") or "csv_upload")
        app_id = str(rec.get("app_id") or "uploaded")
        app_name = str(rec.get("app_name") or "")
        developer = str(rec.get("developer") or "")
        validation = validate_app_identity(
            platform=source,
            app_id=app_id,
            detected_app_name=app_name,
            detected_developer=developer,
        )
        digest = content_hash(source, str(rec.get("source_review_id") or ""), text_val, app_id)
        exists = db.query(Review).filter(Review.content_hash == digest).one_or_none()
        if exists:
            duplicates += 1
            continue
        flags = clean_review(title, text_val)
        rating = rec.get("rating")
        try:
            rating_i = int(rating) if rating not in (None, "") else None
        except ValueError:
            rating_i = None
        synthetic_flag = is_synthetic or str(rec.get("is_synthetic") or "").lower() in {"1", "true", "yes"}
        row = Review(
            source=source,
            source_review_id=str(rec.get("source_review_id") or digest[:16]),
            app_id=app_id,
            app_name=app_name,
            developer=developer,
            region=str(rec.get("region") or ""),
            rating=rating_i,
            title=title,
            text=text_val,
            source_url=str(rec.get("source_url") or ""),
            is_valid_source=validation.is_valid_for_myntra,
            data_classification=(
                "SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA"
                if synthetic_flag
                else validation.data_classification
            ),
            is_synthetic=synthetic_flag,
            content_hash=digest,
            cleaned_text=flags.cleaned_text,
            is_spam=flags.is_spam,
            is_empty=flags.is_empty,
            is_promotional=flags.is_promotional,
            is_short=flags.is_short,
            is_long=flags.is_long,
            language_notes=flags.language_notes,
            raw_payload_json=json.dumps(rec, default=str)[:15000],
        )
        db.add(row)
        inserted += 1
    db.commit()
    return {
        "inserted": inserted,
        "duplicates": duplicates,
        "rejected": rejected,
        "note": "CSV/JSON upload is a fallback. Prefer Collect New Data for public sources.",
        "synthetic": is_synthetic,
        "synthetic_label": "SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA" if is_synthetic else "",
    }
