"""SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import ROOT_DIR, get_settings


class Base(DeclarativeBase):
    pass


def resolve_database_url(url: str) -> str:
    """Pin relative SQLite files to the project root so collectors and the dashboard share one store."""
    if not url.startswith("sqlite"):
        return url
    rest = url.split(":///", 1)[-1] if ":///" in url else url
    if rest.startswith("./"):
        rest = rest[2:]
    path = Path(rest)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return "sqlite:///" + path.resolve().as_posix()


def sqlite_path() -> Path | None:
    url = resolve_database_url(get_settings().database_url)
    if not url.startswith("sqlite"):
        return None
    return Path(url.split(":///", 1)[-1])


def _make_engine():
    settings = get_settings()
    url = resolve_database_url(settings.database_url)
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    engine = create_engine(
        url,
        connect_args=connect_args,
        echo=False,
        future=True,
    )

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


def init_db() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    migrate_schema()
    quarantine_non_myntra_records()
    db = SessionLocal()
    try:
        from app.models import Review
        from app.pipeline.dataset import enforce_review_limit
        from config.settings import storage_review_limit

        if db.query(Review).count() > storage_review_limit():
            enforce_review_limit(db, prune=True)
    finally:
        db.close()
    ensure_pending_analysis_rows()


def migrate_schema() -> None:
    """Add columns on existing SQLite files without wiping collected reviews."""
    url = resolve_database_url(get_settings().database_url)
    if not url.startswith("sqlite"):
        return
    statements = [
        ("collection_runs", "mode", "ALTER TABLE collection_runs ADD COLUMN mode VARCHAR(64) DEFAULT ''"),
        ("analysis", "status", "ALTER TABLE analysis ADD COLUMN status VARCHAR(32) DEFAULT 'pending'"),
        (
            "analysis",
            "analysis_version",
            "ALTER TABLE analysis ADD COLUMN analysis_version VARCHAR(32) DEFAULT '1'",
        ),
        ("analysis", "http_status", "ALTER TABLE analysis ADD COLUMN http_status INTEGER DEFAULT 0"),
        ("analysis", "prompt_tokens", "ALTER TABLE analysis ADD COLUMN prompt_tokens INTEGER DEFAULT 0"),
        ("analysis", "completion_tokens", "ALTER TABLE analysis ADD COLUMN completion_tokens INTEGER DEFAULT 0"),
        ("analysis", "total_tokens", "ALTER TABLE analysis ADD COLUMN total_tokens INTEGER DEFAULT 0"),
    ]
    indexes = [
        "CREATE INDEX IF NOT EXISTS ix_reviews_review_date ON reviews(review_date)",
        "CREATE INDEX IF NOT EXISTS ix_analysis_status ON analysis(status)",
    ]
    with engine.begin() as conn:
        for table, column, ddl in statements:
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            names = {row[1] for row in rows}
            if column not in names:
                conn.execute(text(ddl))
        for ddl in indexes:
            conn.execute(text(ddl))


def quarantine_non_myntra_records() -> int:
    """Ensure Blinkit/Grofers and other unofficial IDs cannot enter Myntra analysis."""
    from app.models import Review
    from config.settings import BANNED_APP_IDS, official_ids

    db = SessionLocal()
    changed = 0
    try:
        allowed = official_ids()
        rows = db.query(Review).all()
        for row in rows:
            app_id = row.app_id or ""
            unofficial = app_id not in allowed or app_id in BANNED_APP_IDS
            if unofficial and (row.is_valid_source or row.data_classification != "REFERENCE / NON-MYNTRA DATA"):
                row.is_valid_source = False
                row.data_classification = "REFERENCE / NON-MYNTRA DATA"
                changed += 1
        if changed:
            db.commit()
    finally:
        db.close()
    return changed


def ensure_pending_analysis_rows() -> int:
    """Queue Myntra-valid reviews that were stored before pending Analysis rows existed."""
    from app.models import Analysis, Review
    from app.pipeline.analysis import ANALYSIS_VERSION
    from config.settings import official_ids

    db = SessionLocal()
    created = 0
    try:
        rows = (
            db.query(Review)
            .filter(
                Review.is_empty.is_(False),
                Review.is_valid_source.is_(True),
                Review.app_id.in_(list(official_ids())),
            )
            .all()
        )
        for row in rows:
            if row.analysis is not None:
                continue
            db.add(
                Analysis(
                    review_id=row.id,
                    content_hash=row.content_hash or "",
                    status="pending",
                    analysis_version=ANALYSIS_VERSION,
                )
            )
            created += 1
        if created:
            db.commit()
    finally:
        db.close()
    return created


def get_database_diagnostics(db: Session | None = None) -> dict[str, Any]:
    """Counts for collection vs analysis. Uses review timestamps for the 30-day window."""
    from app.ai.provider import redact_secrets
    from app.models import Analysis, Opportunity, Review, Segment, Theme
    from app.pipeline.dates import get_last_30_days_cutoff
    from config.settings import OFFICIAL_APPLE_APP_ID, OFFICIAL_GOOGLE_PLAY_APP_ID, official_ids

    owns = db is None
    session = db or SessionLocal()
    try:
        cutoff = get_last_30_days_cutoff()
        from app.pipeline.quantification import review_query

        all_q = review_query(session, myntra_only=False)
        myntra_q = review_query(session, myntra_only=True)
        window_q = review_query(session, myntra_only=True, since=cutoff)
        gp = (
            session.query(Review)
            .filter(
                Review.source == "google_play",
                Review.app_id == OFFICIAL_GOOGLE_PLAY_APP_ID,
                Review.is_empty.is_(False),
                Review.is_duplicate.is_(False),
            )
            .count()
        )
        apple = (
            session.query(Review)
            .filter(
                Review.source == "apple_app_store",
                Review.app_id == OFFICIAL_APPLE_APP_ID,
                Review.is_empty.is_(False),
                Review.is_duplicate.is_(False),
            )
            .count()
        )
        myntra_ids = list(official_ids())
        pending = (
            session.query(Analysis)
            .join(Review)
            .filter(
                Analysis.status == "pending",
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
            )
            .count()
        )
        no_row = (
            session.query(Review)
            .filter(
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
                Review.is_empty.is_(False),
                Review.analysis == None,  # noqa: E711
            )
            .count()
        )
        analyzed = (
            session.query(Analysis)
            .join(Review)
            .filter(
                Analysis.status == "analyzed",
                Analysis.is_valid_json.is_(True),
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
            )
            .count()
        )
        failed = (
            session.query(Analysis)
            .join(Review)
            .filter(
                Analysis.status == "failed",
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
            )
            .count()
        )
        last_ok = (
            session.query(Analysis.analyzed_at)
            .join(Review)
            .filter(
                Analysis.status == "analyzed",
                Analysis.is_valid_json.is_(True),
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
            )
            .order_by(Analysis.analyzed_at.desc())
            .limit(1)
            .scalar()
        )
        last_fail_row = (
            session.query(Analysis.parse_error)
            .join(Review)
            .filter(
                Analysis.status == "failed",
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
                Analysis.parse_error != "",
            )
            .order_by(Analysis.analyzed_at.desc())
            .limit(1)
            .first()
        )
        last_error = ""
        if last_fail_row and last_fail_row[0]:
            last_error = redact_secrets(str(last_fail_row[0]))
        settings = get_settings()
        path = sqlite_path()
        from app.pipeline.dataset import analysis_dataset_stats, dataset_integrity
        from config.settings import analysis_review_limit, storage_review_limit

        stats = analysis_dataset_stats(session)
        integrity = dataset_integrity(session)
        storage_cap = storage_review_limit(settings)
        sample_cap = analysis_review_limit(settings)
        total_stored = all_q.count()
        return {
            "database_path": str(path) if path else "",
            "total_reviews": total_stored,
            "myntra_reviews": myntra_q.count(),
            "available_reviews": stats.get("available_reviews") or myntra_q.count(),
            "selected_reviews": stats.get("selected_reviews") or 0,
            "google_play_reviews": gp,
            "apple_reviews": apple,
            "apple_app_store_reviews": apple,
            "last_30_day_reviews": window_q.count(),
            "pending_reviews": stats.get("pending_reviews") if stats.get("pending_reviews") is not None else pending,
            "analyzed_reviews": stats.get("analyzed_reviews") if stats.get("analyzed_reviews") is not None else analyzed,
            "failed_reviews": stats.get("failed_reviews") if stats.get("failed_reviews") is not None else failed,
            "sample_analyzed": stats.get("sample_analyzed") or 0,
            "sample_pending": stats.get("sample_pending") or 0,
            "sample_failed": stats.get("sample_failed") or 0,
            "max_dataset_reviews": storage_cap,
            "max_total_reviews": storage_cap,
            "max_analysis_reviews": sample_cap,
            "max_discovery_reviews": sample_cap,
            "google_play_selected": stats.get("google_play_selected") or 0,
            "apple_selected": stats.get("apple_selected") or 0,
            "analysis_batch_size": stats.get("batch_size") or 10,
            "analysis_batch_total": stats.get("batch_total") or 0,
            "dataset_limit_reached": total_stored >= storage_cap,
            "duplicate_source_ids": integrity.get("duplicate_source_ids") or 0,
            "orphaned_analysis": integrity.get("orphaned_analysis") or 0,
            "orphaned_evidence_ids": integrity.get("orphaned_evidence_ids") or 0,
            "themes": session.query(Theme).count(),
            "segments": session.query(Segment).count(),
            "opportunities": session.query(Opportunity).count(),
            "ai_provider": "OpenRouter",
            "ai_model": settings.resolved_model,
            "last_successful_analysis_at": last_ok.isoformat() if last_ok else None,
            "last_analysis_error": last_error or None,
        }
    finally:
        if owns:
            session.close()


def get_ai_diagnostics(db: Session | None = None) -> dict[str, Any]:
    """Safe AI diagnostics for Live Data. Never includes the API key."""
    from app.ai.provider import redact_secrets
    from app.models import Analysis, Review
    from config.settings import official_ids

    settings = get_settings()
    base = get_database_diagnostics(db)
    owns = db is None
    session = db or SessionLocal()
    try:
        myntra_ids = list(official_ids())
        last_fail_at = (
            session.query(Analysis.analyzed_at)
            .join(Review)
            .filter(
                Analysis.status == "failed",
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
            )
            .order_by(Analysis.analyzed_at.desc())
            .limit(1)
            .scalar()
        )
        last_http_row = (
            session.query(Analysis.http_status, Analysis.parse_error)
            .join(Review)
            .filter(
                Review.is_valid_source.is_(True),
                Review.app_id.in_(myntra_ids),
                Analysis.http_status > 0,
            )
            .order_by(Analysis.analyzed_at.desc())
            .limit(1)
            .first()
        )
        last_http = int(last_http_row[0]) if last_http_row and last_http_row[0] else None
        last_error = base.get("last_analysis_error")
        if last_http_row and last_http_row[1] and not last_error:
            last_error = redact_secrets(str(last_http_row[1]))
        return {
            "ai_provider": "OpenRouter",
            "ai_model": settings.resolved_model,
            "api_key_configured": "YES" if settings.has_ai_credentials else "NO",
            "max_tokens": int(getattr(settings, "ai_max_tokens", 2000) or 2000),
            "batch_size": int(
                getattr(settings, "analysis_batch_size", None)
                or getattr(settings, "ai_batch_size", None)
                or getattr(settings, "ai_request_batch_size", 10)
                or 10
            ),
            "pending_reviews": base.get("pending_reviews") or 0,
            "analyzed_reviews": base.get("analyzed_reviews") or 0,
            "failed_reviews": base.get("failed_reviews") or 0,
            "available_reviews": base.get("available_reviews") or base.get("myntra_reviews") or 0,
            "selected_reviews": base.get("selected_reviews") or 0,
            "max_analysis_reviews": base.get("max_analysis_reviews") or 150,
            "max_dataset_reviews": base.get("max_dataset_reviews") or 500,
            "max_discovery_reviews": base.get("max_discovery_reviews") or 150,
            "max_total_reviews": base.get("max_total_reviews") or 500,
            "dataset_limit_reached": bool(base.get("dataset_limit_reached")),
            "google_play_selected": base.get("google_play_selected") or 0,
            "apple_selected": base.get("apple_selected") or 0,
            "analysis_batch_size": base.get("analysis_batch_size") or 10,
            "analysis_batch_total": base.get("analysis_batch_total") or 0,
            "last_successful_analysis": base.get("last_successful_analysis_at"),
            "last_failed_analysis": last_fail_at.isoformat() if last_fail_at else None,
            "last_error": last_error or None,
            "last_http_status": last_http,
            "myntra_reviews": base.get("myntra_reviews") or 0,
            "total_reviews": base.get("total_reviews") or 0,
            "last_30_day_reviews": base.get("last_30_day_reviews") or 0,
        }
    finally:
        if owns:
            session.close()


def get_review_count(db: Session | None = None, *, since=None, myntra_only: bool = False) -> int:
    """Non-empty, non-duplicate review count. Optional review_date window."""
    from app.pipeline.quantification import review_query

    owns = db is None
    session = db or SessionLocal()
    try:
        return review_query(session, myntra_only=myntra_only, since=since).count()
    finally:
        if owns:
            session.close()


def get_review_stats(db: Session | None = None, *, since=None, myntra_only: bool = False) -> dict[str, Any]:
    from app.pipeline.quantification import overview_metrics

    owns = db is None
    session = db or SessionLocal()
    try:
        metrics = overview_metrics(session, since=since, myntra_only=myntra_only)
        metrics["all_time_count"] = get_review_count(session, myntra_only=myntra_only)
        path = sqlite_path()
        metrics["database_path"] = str(path) if path else resolve_database_url(get_settings().database_url)
        return metrics
    finally:
        if owns:
            session.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
