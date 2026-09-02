"""Operational diagnostics: python -m utils.diagnostics

Prints database, collector, AI, and discovery status. Never prints API keys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _iso(value) -> str:
    if value is None:
        return "None"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def run_diagnostics() -> dict:
    from app.ai.provider import test_openrouter_connection
    from app.database import SessionLocal, get_ai_diagnostics, get_database_diagnostics, init_db, sqlite_path
    from app.models import Analysis, CollectionRun, Opportunity, Segment, Source, Theme
    from app.pipeline.quantification import label_distribution, problem_rows
    from config.settings import OFFICIAL_APPLE_APP_ID, OFFICIAL_GOOGLE_PLAY_APP_ID, get_settings, reload_settings

    init_db()
    settings = reload_settings()
    db = SessionLocal()
    try:
        base = get_database_diagnostics(db)
        ai = get_ai_diagnostics(db)
        probe = test_openrouter_connection(settings)
        gp = (
            db.query(Source)
            .filter(Source.platform == "google_play", Source.app_id == OFFICIAL_GOOGLE_PLAY_APP_ID)
            .order_by(Source.last_collection_at.desc())
            .first()
        )
        apple = (
            db.query(Source)
            .filter(Source.platform == "apple_app_store", Source.app_id == OFFICIAL_APPLE_APP_ID)
            .order_by(Source.last_collection_at.desc())
            .first()
        )
        last_run = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
        problems = problem_rows(db, myntra_only=True)
        wishlist = label_distribution(db, "intent", myntra_only=True, relevant_only=False)
        barriers = label_distribution(db, "barriers", myntra_only=True, relevant_only=False)
        uncertainties = label_distribution(db, "uncertainties", myntra_only=True, relevant_only=False)
        evidence_n = (
            db.query(Analysis)
            .filter(Analysis.status == "analyzed", Analysis.is_valid_json.is_(True))
            .count()
        )
        report = {
            "database": {
                "path": str(sqlite_path() or ""),
                "total_reviews": base.get("total_reviews") or 0,
                "max_dataset_reviews": base.get("max_dataset_reviews") or 500,
                "max_total_reviews": base.get("max_total_reviews") or 500,
                "max_analysis_reviews": base.get("max_analysis_reviews") or 150,
                "dataset_limit_reached": bool(base.get("dataset_limit_reached")),
                "google_play_reviews": base.get("google_play_reviews") or 0,
                "apple_reviews": base.get("apple_reviews") or 0,
                "last_30_day_reviews": base.get("last_30_day_reviews") or 0,
                "pending_reviews": base.get("pending_reviews") or 0,
                "analyzed_reviews": base.get("analyzed_reviews") or 0,
                "failed_reviews": base.get("failed_reviews") or 0,
                "available_reviews": base.get("available_reviews") or base.get("myntra_reviews") or 0,
                "selected_reviews": base.get("selected_reviews") or 0,
            },
            "collectors": {
                "google_play_status": getattr(gp, "validation_status", None) or "not collected",
                "google_play_last_checked": _iso(getattr(gp, "last_collection_at", None)),
                "apple_status": getattr(apple, "validation_status", None) or "not collected",
                "apple_region": getattr(apple, "region", None) or "—",
                "apple_last_checked": _iso(getattr(apple, "last_collection_at", None)),
                "last_run_status": getattr(last_run, "status", None) or "none",
                "last_run_errors": getattr(last_run, "errors_json", None) or "[]",
            },
            "ai": {
                "provider": ai.get("ai_provider") or "OpenRouter",
                "model": ai.get("ai_model") or settings.resolved_model,
                "credentials_configured": ai.get("api_key_configured") or "NO",
                "connection_status": probe.get("status") or "FAILED",
                "http_status": probe.get("http_status"),
                "last_error": probe.get("error") or ai.get("last_error"),
            },
            "discovery": {
                "user_problems": len(problems),
                "wishlist_behavior": len(wishlist),
                "purchase_barriers": len(barriers),
                "uncertainties": len(uncertainties),
                "themes": db.query(Theme).count(),
                "segments": db.query(Segment).count(),
                "opportunities": db.query(Opportunity).count(),
                "evidence_records": evidence_n,
            },
        }
        return report
    finally:
        db.close()


def print_report(report: dict) -> None:
    db = report["database"]
    col = report["collectors"]
    ai = report["ai"]
    disc = report["discovery"]
    print("DATABASE")
    print("--------")
    print("Path:", db["path"])
    print("Total reviews:", db["total_reviews"])
    print("Max total:", db.get("max_total_reviews") or 500)
    print("Limit reached:", db.get("dataset_limit_reached"))
    print("Available:", db.get("available_reviews") or db["total_reviews"])
    print("AI sample:", db.get("selected_reviews") or "n/a")
    print("Google reviews:", db["google_play_reviews"])
    print("Apple reviews:", db["apple_reviews"])
    print("Last 30 days:", db["last_30_day_reviews"])
    print("Pending:", db["pending_reviews"])
    print("Analyzed:", db["analyzed_reviews"])
    print("Failed:", db["failed_reviews"])
    print()
    print("COLLECTORS")
    print("----------")
    print("Google Play status:", col["google_play_status"])
    print("Google Play last checked:", col["google_play_last_checked"])
    print("Apple status:", col["apple_status"])
    print("Apple region:", col["apple_region"])
    print("Apple last checked:", col["apple_last_checked"])
    print("Last run:", col["last_run_status"])
    errors = col["last_run_errors"]
    if errors and errors not in {"[]", "null"}:
        print("Last run errors:", errors[:500])
    print()
    print("AI")
    print("--")
    print("Provider:", ai["provider"])
    print("Model:", ai["model"])
    print("Credentials configured:", ai["credentials_configured"])
    print("Connection status:", ai["connection_status"])
    print("HTTP status:", ai["http_status"] if ai["http_status"] is not None else "N/A")
    print("Last error:", ai["last_error"] or "None")
    print()
    print("DISCOVERY")
    print("---------")
    print("User Problems:", disc["user_problems"])
    print("Wishlist Behavior:", disc["wishlist_behavior"])
    print("Purchase Barriers:", disc["purchase_barriers"])
    print("Uncertainties:", disc["uncertainties"])
    print("Themes:", disc["themes"])
    print("Segments:", disc["segments"])
    print("Opportunities:", disc["opportunities"])
    print("Evidence Records:", disc["evidence_records"])


def main() -> None:
    report = run_diagnostics()
    print_report(report)
    if "--json" in sys.argv:
        print()
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
