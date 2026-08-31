"""SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    settings = get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        echo=False,
        future=True,
    )

    if settings.database_url.startswith("sqlite"):

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
    quarantine_non_myntra_records()


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


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
