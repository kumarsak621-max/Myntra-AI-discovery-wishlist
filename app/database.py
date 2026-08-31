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
    ]
    with engine.begin() as conn:
        for table, column, ddl in statements:
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            names = {row[1] for row in rows}
            if column not in names:
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
        path = sqlite_path()
        return {
            "database_path": str(path) if path else "",
            "total_reviews": all_q.count(),
            "myntra_reviews": myntra_q.count(),
            "google_play_reviews": gp,
            "apple_reviews": apple,
            "last_30_day_reviews": window_q.count(),
            "pending_reviews": pending + no_row,
            "analyzed_reviews": analyzed,
            "failed_reviews": failed,
            "themes": session.query(Theme).count(),
            "segments": session.query(Segment).count(),
            "opportunities": session.query(Opportunity).count(),
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
