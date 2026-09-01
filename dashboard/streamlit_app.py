"""Streamlit dashboard that reuses collectors, analysis, and SQLite.

Launched by `streamlit run app.py`. Does not duplicate business logic.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timedelta, timezone

import streamlit as st
from sqlalchemy.orm import Session

from app.api.routes import serialize_review
from app.database import (
    SessionLocal,
    get_ai_diagnostics,
    get_database_diagnostics,
    get_review_count,
    get_review_stats,
    init_db,
    migrate_schema,
    sqlite_path,
)
from app.models import Analysis, CollectionRun, Opportunity, Review, Segment, Theme
from app.pipeline.dates import get_last_30_days_cutoff, humanize_ago, window_start
from app.pipeline.quantification import (
    daily_review_trends,
    label_distribution,
    label_window_momentum,
    problem_rows,
    source_live_status,
)
from app.pipeline.report import build_report, evidence_cards
from config.settings import (
    OFFICIAL_APPLE_APP_ID,
    OFFICIAL_APPLE_APP_NAME,
    OFFICIAL_APPLE_APP_URL,
    OFFICIAL_GOOGLE_PLAY_APP_ID,
    OFFICIAL_GOOGLE_PLAY_APP_NAME,
    OFFICIAL_GOOGLE_PLAY_URL,
    get_settings,
    reload_settings,
)

LOGGER = logging.getLogger("myntra.discovery")

PAGES = [
    "Overview",
    "Live Data",
    "Latest Reviews",
    "User Problems",
    "Wishlist Behavior",
    "Purchase Barriers",
    "Uncertainties",
    "Themes",
    "User Segments",
    "Opportunity Matrix",
    "Evidence Explorer",
    "Collection History",
    "Discovery Report",
]

EMPTY = "No reviews have been collected yet."
NEAR_REALTIME = "Near-real-time — refreshed from the public source"


def _analysis_blocker(
    *,
    stored: int,
    analyzed: int,
    pending: int,
    failed: int,
    ai_ok: bool,
    last_error: str = "",
) -> str | None:
    """Explain why discovery pages are empty. Never hide stored reviews behind a fake empty corpus."""
    if stored == 0:
        return EMPTY
    if analyzed > 0:
        return None
    if not ai_ok:
        return (
            f"{stored} real reviews are stored, but none have been analyzed. "
            "OpenRouter API key is not configured. "
            "Add OPENROUTER_API_KEY to Streamlit Secrets or .env, then click Test OpenRouter Connection."
        )
    if failed and not analyzed:
        extra = f" Last error: {last_error}." if last_error else ""
        return (
            f"OpenRouter analysis failed for {failed} reviews.{extra} "
            "See Live Data for the actual error. Click Retry Failed Analysis."
        )
    waiting = pending or stored
    return (
        f"{waiting} real reviews are awaiting AI analysis. "
        "Click Test OpenRouter Connection, then Analyze Pending Reviews or 🚀 Run Full Discovery Pipeline."
    )


@st.cache_resource
def _bootstrap() -> bool:
    init_db()
    return True


def _db() -> Session:
    return SessionLocal()


def _openrouter_error(exc: Exception | str) -> str:
    text = str(exc or "").strip()
    if not text:
        return "OpenRouter analysis failed."
    if text.lower().startswith("openrouter"):
        return text
    return f"OpenRouter analysis failed: {text}"


def _safe_error(exc: Exception) -> None:
    LOGGER.exception("Dashboard error")
    st.error(str(exc) or "Something went wrong.")
    if st.session_state.get("debug"):
        st.code(traceback.format_exc())


def _empty_notice(count: int, *, need_analysis: bool = False) -> bool:
    if count == 0:
        st.info(EMPTY)
        return True
    if need_analysis:
        diag = get_database_diagnostics()
        msg = _analysis_blocker(
            stored=count,
            analyzed=diag.get("analyzed_reviews") or 0,
            pending=diag.get("pending_reviews") or 0,
            failed=diag.get("failed_reviews") or 0,
            ai_ok=get_settings().has_ai_credentials,
            last_error=str(diag.get("last_analysis_error") or ""),
        )
        if msg:
            (st.info if msg == EMPTY else st.warning)(msg)
            return True
    return False


def render() -> None:
    st.set_page_config(page_title="Myntra Discovery Engine", layout="wide")
    _bootstrap()
    migrate_schema()
    reload_settings()
    settings = get_settings()

    st.sidebar.title("Wishlist → Purchase")
    st.sidebar.caption("Discovers the problem. Does not propose the feature.")
    period = st.sidebar.radio("Period", ["Last 30 Days", "All Time"], index=0)
    st.session_state["period"] = period
    page = st.sidebar.radio("Section", PAGES, index=0)
    myntra_only = st.sidebar.checkbox("Myntra-valid evidence only", value=True)
    st.session_state["debug"] = st.sidebar.checkbox("Show debug traces", value=False)

    ai_ok = settings.has_ai_credentials
    if "analyze_on_collect" not in st.session_state:
        st.session_state["analyze_on_collect"] = bool(ai_ok)
    cutoff = _period_since()
    stored_all = get_review_count(myntra_only=myntra_only)
    st.sidebar.markdown("**Stored reviews**")
    st.sidebar.write("All time:", stored_all)
    st.sidebar.write(
        "Last 30 days:",
        get_review_count(since=get_last_30_days_cutoff(), myntra_only=myntra_only),
    )
    st.sidebar.markdown("**AI**")
    st.sidebar.write("API key:", "Configured" if ai_ok else "Missing")
    st.sidebar.write("Model:", settings.resolved_model)
    diag = get_database_diagnostics()
    st.sidebar.markdown("**Analysis**")
    st.sidebar.write("Pending:", diag.get("pending_reviews", 0))
    st.sidebar.write("Analyzed:", diag.get("analyzed_reviews", 0))
    st.sidebar.write("Failed:", diag.get("failed_reviews", 0))
    path = sqlite_path()
    st.sidebar.caption("Storage: Local application storage")
    if path:
        st.sidebar.caption(f"`{path.name}` is ephemeral on Streamlit Cloud.")

    try:
        if page == "Overview":
            _overview(myntra_only, ai_ok)
        elif page == "Live Data":
            _live_data(ai_ok)
        elif page == "Latest Reviews":
            _latest_reviews(myntra_only)
        elif page == "User Problems":
            _root_causes(myntra_only)
        elif page == "Wishlist Behavior":
            _labels("intent", myntra_only, "Wishlist Behavior")
        elif page == "Purchase Barriers":
            _labels("barriers", myntra_only, "Purchase Barriers")
        elif page == "Uncertainties":
            _labels("uncertainties", myntra_only, "Uncertainties")
        elif page == "Themes":
            _named(Theme, "Themes")
        elif page == "User Segments":
            _named(Segment, "User Segments")
        elif page == "Opportunity Matrix":
            _opportunities()
        elif page == "Evidence Explorer":
            _evidence()
        elif page == "Collection History":
            _collection_history()
        elif page == "Discovery Report":
            _report()
    except Exception as exc:
        _safe_error(exc)


def _period_since():
    if st.session_state.get("period", "Last 30 Days") == "Last 30 Days":
        return get_last_30_days_cutoff()
    return None


def _collect_buttons(ai_ok: bool) -> None:
    c1, c2, c3 = st.columns(3)
    thirty = c1.button("Collect Last 30 Days")
    refresh = c2.button("🔄 Refresh Latest Reviews")
    full = c3.button("🚀 Run Full Discovery Pipeline", type="primary")
    analyze = bool(st.session_state.get("analyze_on_collect", ai_ok))
    if thirty:
        _run_collect(["google_play", "apple_app_store"], analyze=analyze, mode="last_30_days")
    if refresh:
        _run_collect(["google_play", "apple_app_store"], analyze=analyze, mode="latest")
    if full:
        _run_full_discovery(ai_ok)


def _overview(myntra_only: bool, ai_ok: bool) -> None:
    st.title("Overview")
    period = st.session_state.get("period", "Last 30 Days")
    cutoff = _period_since()
    st.caption(
        f"Period: **{period}**. "
        + (
            f"Window starts {cutoff.strftime('%Y-%m-%d %H:%M UTC')} (now minus 30 days). "
            "A review is included only if its review timestamp is inside this window."
            if cutoff
            else "Showing all stored reviews with a review timestamp or without a dated window."
        )
    )
    db = _db()
    try:
        data = get_review_stats(db, since=cutoff, myntra_only=myntra_only)
        trends = daily_review_trends(db, myntra_only=myntra_only, since=cutoff)
        last_run = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
        since_for_momentum = cutoff or get_last_30_days_cutoff()
        theme_momentum = label_window_momentum(
            db, "themes", since=since_for_momentum, myntra_only=myntra_only
        )
        barrier_momentum = label_window_momentum(
            db, "barriers", since=since_for_momentum, myntra_only=myntra_only
        )
        wishlist_momentum = label_window_momentum(
            db, "intent", since=since_for_momentum, myntra_only=myntra_only
        )
        opportunities = db.query(Opportunity).order_by(Opportunity.rank.asc()).limit(5).all()
    finally:
        db.close()

    count = data.get("total_reviews") or 0
    all_time = data.get("all_time_count") or 0
    if all_time == 0:
        st.info(EMPTY)
        _collect_buttons(ai_ok)
        return
    if count == 0 and cutoff is not None:
        st.info(
            "No reviews with timestamps in the last 30 days. "
            "Collect Last 30 Days, or switch Period to All Time."
        )
        _collect_buttons(ai_ok)
        return

    last = st.session_state.get("last_collection") or {}
    new_since = (last.get("stats") or {}).get("new", 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reviews", count)
    c2.metric("New reviews (last sync)", new_since)
    c3.metric("Average rating", data.get("average_rating") if data.get("average_rating") is not None else "—")
    c4.metric("Wishlist signals", data.get("wishlist_signals", 0))
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Google Play", data.get("google_play_reviews", 0))
    d2.metric("Apple App Store", data.get("apple_reviews", 0))
    d3.metric("Purchase barriers (analyzed)", data.get("purchase_hesitation", 0))
    top_opp = (data.get("opportunity_count") or 0)
    d4.metric("Opportunity areas", top_opp)
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("1-star", data.get("rating_1", 0))
    r2.metric("2-star", data.get("rating_2", 0))
    r3.metric("3-star", data.get("rating_3", 0))
    r4.metric("4-star", data.get("rating_4", 0))
    r5.metric("5-star", data.get("rating_5", 0))
    st.caption(
        f"Date range (review timestamps): {data.get('date_from') or '—'} → {data.get('date_to') or '—'}. "
        "Counts come from stored public reviews. The LLM does not invent these numbers."
    )
    if data.get("synthetic_count"):
        st.warning("SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA is present and must stay labelled.")

    last_insights = st.session_state.get("insight_snapshot") or {}
    current_themes = [x.get("label") for x in (data.get("top_barriers") or [])]
    prev_themes = last_insights.get("barriers") or []
    new_labels = [t for t in current_themes if t and t not in prev_themes]
    st.subheader("New vs existing insights")
    st.write("New reviews since last refresh:", new_since)
    if new_labels:
        st.write("Newly listed barrier(s):", ", ".join(new_labels))
    if prev_themes:
        st.write("Previously listed barrier(s):", ", ".join(str(x) for x in prev_themes[:5]))
    elif current_themes:
        st.write("Existing dominant barrier:", current_themes[0])
    if data.get("analyzed_reviews", 0) == 0:
        diag = get_database_diagnostics()
        msg = _analysis_blocker(
            stored=diag.get("myntra_reviews") or count,
            analyzed=diag.get("analyzed_reviews") or 0,
            pending=diag.get("pending_reviews") or 0,
            failed=diag.get("failed_reviews") or 0,
            ai_ok=ai_ok,
            last_error=str(diag.get("last_analysis_error") or ""),
        )
        if msg:
            st.warning(msg) if msg != EMPTY else st.info(msg)

    st.subheader("Top user problems")
    if data.get("top_problems"):
        st.dataframe(data["top_problems"], width="stretch")
    elif data.get("analyzed_reviews", 0) == 0:
        pass
    else:
        st.caption("No root-cause statements in this period.")

    st.subheader("Top purchase barriers")
    if data.get("top_barriers"):
        st.dataframe(data["top_barriers"], width="stretch")
    st.subheader("Top wishlist-related signals")
    if data.get("top_intents"):
        st.dataframe(data["top_intents"], width="stretch")
    st.subheader("Top uncertainties")
    if data.get("top_uncertainties"):
        st.dataframe(data["top_uncertainties"], width="stretch")

    st.subheader("Top emerging themes")
    if theme_momentum:
        st.dataframe(
            [
                {
                    "theme": x["label"],
                    "count": x["count"],
                    "momentum": x["momentum"],
                    "first_half": x["first_half"],
                    "second_half": x["second_half"],
                }
                for x in theme_momentum[:8]
            ],
            width="stretch",
        )
        st.caption("Momentum is a first-half vs second-half split of the selected window, not a significance test.")
    else:
        st.caption("Theme momentum appears after reviews in this window are analyzed.")

    st.subheader("Top opportunity areas")
    if opportunities:
        st.dataframe(
            [
                {"rank": o.rank, "problem": o.user_problem, "score": o.score, "pct": o.percentage}
                for o in opportunities
            ],
            width="stretch",
        )
        st.caption(
            "Opportunity scores are reach × frequency × purchase impact × severity × evidence confidence, "
            "calculated in Python from analyzed Myntra-valid reviews. They update when new reviews are analyzed."
        )

    st.subheader("Reviews by day (review date)")
    if trends:
        st.dataframe(trends, width="stretch")
        st.caption("Ratings by day use the review timestamp, not collection time.")
    else:
        st.caption("No dated reviews in this window.")

    st.subheader("Purchase-barrier frequency over time")
    if barrier_momentum:
        st.dataframe(
            [
                {
                    "barrier": x["label"],
                    "count": x["count"],
                    "momentum": x["momentum"],
                    "first_half": x["first_half"],
                    "second_half": x["second_half"],
                }
                for x in barrier_momentum[:8]
            ],
            width="stretch",
        )
    st.subheader("Wishlist-related signal frequency over time")
    if wishlist_momentum:
        st.dataframe(
            [
                {
                    "signal": x["label"],
                    "count": x["count"],
                    "momentum": x["momentum"],
                    "first_half": x["first_half"],
                    "second_half": x["second_half"],
                }
                for x in wishlist_momentum[:8]
            ],
            width="stretch",
        )

    _collect_buttons(ai_ok)
    if last_run:
        st.caption(
            f"Last collection run #{last_run.id}: {last_run.status} · "
            f"fetched {last_run.fetched} · new {last_run.new_count} · mode {last_run.mode or '—'}"
        )


def _show_last_collection() -> None:
    last = st.session_state.get("last_collection")
    if not last:
        return
    stats = last.get("stats") or {}
    by_source = stats.get("by_source") or {}
    st.subheader("Last collection run")
    st.caption(f"Finished at {last.get('at') or '—'}")
    gp = by_source.get("google_play") or {}
    ap = by_source.get("apple_app_store") or {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Google Play**")
        if gp:
            st.write("Reviews fetched:", gp.get("fetched", 0))
            st.write("New reviews:", gp.get("new", 0))
            st.write("Duplicates skipped:", gp.get("duplicates", 0))
            for err in gp.get("errors") or []:
                st.error(f"Google Play collection failed: {err}")
        else:
            st.write("Not run in the last collection.")
    with col2:
        st.markdown("**Apple App Store**")
        if ap:
            st.write("Region attempted:", "India" if ap.get("region_attempted") == "in" else ap.get("region_attempted"))
            st.write("Region stored:", ap.get("region_used") or "—")
            if ap.get("fallback_used"):
                st.write("India feed had no written reviews → US fallback used.")
            st.write("Reviews fetched:", ap.get("fetched", 0))
            st.write("New reviews:", ap.get("new", 0))
            st.write("Duplicates skipped:", ap.get("duplicates", 0))
            for err in ap.get("errors") or []:
                st.error(f"Apple App Store collection failed: {err}")
        else:
            st.write("Not run in the last collection.")
    total = get_review_count()
    st.write("**Total reviews available:**", total)
    for err in stats.get("errors") or []:
        st.error(err)


def _diagnostics() -> None:
    st.subheader("LIVE DATA STATUS")
    st.caption(NEAR_REALTIME + ". Status is Connected only after a recent successful poll of that source.")
    settings = get_settings()
    db = _db()
    try:
        freshness = source_live_status(db)
        gp_stored = (
            db.query(Review)
            .filter(
                Review.source == "google_play",
                Review.app_id == OFFICIAL_GOOGLE_PLAY_APP_ID,
                Review.is_empty.is_(False),
            )
            .count()
        )
        ap_stored = (
            db.query(Review)
            .filter(
                Review.source == "apple_app_store",
                Review.app_id == OFFICIAL_APPLE_APP_ID,
                Review.is_empty.is_(False),
            )
            .count()
        )
    finally:
        db.close()

    last = st.session_state.get("last_collection") or {}
    by_source = (last.get("stats") or {}).get("by_source") or {}
    gp_run = by_source.get("google_play") or {}
    ap_run = by_source.get("apple_app_store") or {}
    gp_src = freshness["google_play"]["source"]
    ap_src = freshness["apple_app_store"]["source"]
    gp_latest = freshness["google_play"]["latest_review_at"]
    ap_latest = freshness["apple_app_store"]["latest_review_at"]
    last_ok = freshness["last_successful_run"]

    def _status(src, run: dict) -> str:
        if run.get("errors"):
            return "Failed"
        if last.get("at") and src and src.is_valid_for_myntra:
            return "Connected"
        if src and src.validation_status == "ERROR":
            return "Failed"
        if src and src.last_collection_at and src.is_valid_for_myntra:
            return "Connected"
        return "Not collected"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Google Play**")
        st.write("App ID:", OFFICIAL_GOOGLE_PLAY_APP_ID)
        st.write("Status:", _status(gp_src, gp_run))
        checked = gp_src.last_collection_at if gp_src else None
        st.write("Last checked:", humanize_ago(checked) if checked else "never")
        if checked:
            st.caption(checked.isoformat())
        st.write("Latest review found:", gp_latest.isoformat() if gp_latest else "—")
        st.write("New reviews:", gp_run.get("new") if "fetched" in gp_run else "—")
        st.write("Reviews stored:", gp_stored)
        if gp_src and gp_src.warning:
            st.error(gp_src.warning)
    with col2:
        st.markdown("**Apple App Store**")
        st.write("App ID:", OFFICIAL_APPLE_APP_ID)
        region = (ap_run.get("region_used") or settings.apple_primary_region or "").lower()
        st.write("Region:", "India" if region == "in" else ("US" if region == "us" else region or "—"))
        if ap_run.get("fallback_used"):
            st.write("India feed had no written reviews → US fallback used. Stored region is US, not India.")
        st.write("Status:", _status(ap_src, ap_run))
        checked = ap_src.last_collection_at if ap_src else None
        st.write("Last checked:", humanize_ago(checked) if checked else "never")
        if checked:
            st.caption(checked.isoformat())
        st.write("Latest review found:", ap_latest.isoformat() if ap_latest else "—")
        st.write("New reviews:", ap_run.get("new") if "fetched" in ap_run else "—")
        st.write("Reviews stored:", ap_stored)
        if ap_src and ap_src.warning:
            st.error(ap_src.warning)
    with col3:
        st.markdown("**Overall**")
        st.write(
            "Last successful sync:",
            humanize_ago(last_ok.finished_at) if last_ok and last_ok.finished_at else "never",
        )
        if last_ok and last_ok.finished_at:
            st.caption(last_ok.finished_at.isoformat())
        st.write("New reviews since previous sync:", (last.get("stats") or {}).get("new", last_ok.new_count if last_ok else 0))
        st.write("Provider: OpenRouter")
        st.write("Model:", settings.resolved_model)
        last_ai = st.session_state.get("last_analysis")
        if settings.has_ai_credentials:
            st.write("AI:", (last_ai or {}).get("status", "Configured"))
        else:
            st.write("AI: Not configured")
        if last_ai and last_ai.get("message"):
            st.caption(last_ai["message"])
        st.caption("API key is never displayed. This is not a live stream.")
        st.caption("Per-source new counts update after a poll in this session. Overall uses the last stored sync.")

    latest = freshness["last_run"]
    if latest and latest.status == "failed" and latest.notes:
        st.error(f"Last collection run failed: {latest.notes}")


def _render_ai_probe(probe: dict) -> None:
    connected = bool(probe.get("ok") and probe.get("status") in {"SUCCESS", "Connected"})
    http_status = probe.get("http_status")
    st.subheader("AI CONNECTION TEST")
    st.write("Provider:")
    st.write(probe.get("provider") or "OpenRouter")
    st.write("Model:")
    st.write(probe.get("model") or "")
    st.write("Secret source:")
    st.write(probe.get("secret_source") or "Missing")
    st.write("Key configured:")
    st.write("YES" if probe.get("credentials") == "Configured" else "NO")
    st.write("Key format:")
    st.write(probe.get("key_format") or "MISSING")
    st.write("Key prefix:")
    st.write(probe.get("key_prefix") or "none")
    st.write("Connection:")
    if connected:
        st.write("PASS")
    elif http_status:
        st.write(f"FAIL — HTTP {http_status}")
    else:
        st.write("FAIL")
    st.write("HTTP status:")
    st.write(http_status if http_status is not None else "N/A")
    st.write("Error:")
    st.write(probe.get("error") or "None")
    st.caption("The full API key is never displayed.")
    if connected:
        st.success("OpenRouter accepted a live test request.")
    else:
        st.error(probe.get("error") or "OpenRouter connection test failed.")


def _ai_analysis_status_panel(ai_ok: bool) -> None:
    from app.ai.provider import test_openrouter_connection
    from config.settings import get_ai_config

    reload_settings()
    ai = get_ai_diagnostics()
    cfg = get_ai_config()
    key_label = "Configured" if cfg["configured"] else "Missing"
    probe = st.session_state.get("ai_connection_test") or {}
    if probe.get("ok") and probe.get("status") in {"SUCCESS", "Connected"}:
        connection = "Connected"
    elif probe.get("http_status"):
        connection = f"Failed — HTTP {probe.get('http_status')}"
    elif probe:
        connection = "Failed"
    else:
        connection = "Not tested"
    st.subheader("AI ANALYSIS STATUS")
    st.write("Provider:")
    st.write(cfg["provider_label"])
    st.write("Model:")
    st.write(cfg["model"])
    st.write("Reviews stored:")
    st.write(ai.get("myntra_reviews") if ai.get("myntra_reviews") is not None else ai.get("total_reviews"))
    st.write("Pending:")
    st.write(ai.get("pending_reviews"))
    st.write("Analyzed:")
    st.write(ai.get("analyzed_reviews"))
    st.write("Failed:")
    st.write(ai.get("failed_reviews"))
    st.write("Last analysis:")
    st.write(ai.get("last_successful_analysis") or "None")
    st.write("Last error:")
    st.write(ai.get("last_error") or "None")
    st.write("API KEY")
    st.write(key_label)
    st.write("Key format:")
    st.write(cfg.get("key_format") or "MISSING")
    st.write("Secret source:")
    st.write(cfg.get("secret_source") or "Missing")
    st.write("Connection")
    st.write(connection)
    st.caption("Configured means a secret value exists. Connection is a real OpenRouter request.")

    test_col, analyze_col, retry_col = st.columns(3)
    with test_col:
        run_test = st.button("Test OpenRouter Connection")
    with analyze_col:
        run_pending = st.button("Analyze Pending Reviews", key="analyze_pending_status")
    with retry_col:
        run_retry = st.button("Retry Failed Analysis")
    if run_test:
        with st.spinner("Testing OpenRouter connection…"):
            probe = test_openrouter_connection()
        st.session_state["ai_connection_test"] = probe
        _render_ai_probe(probe)
    elif st.session_state.get("ai_connection_test"):
        _render_ai_probe(st.session_state["ai_connection_test"])
    if run_pending or run_retry:
        reload_settings()
        if not get_ai_config()["configured"]:
            st.error(get_ai_config()["missing_key_message"])
        elif get_review_count() == 0:
            st.info(EMPTY)
        else:
            _run_analyze(only_failed=bool(run_retry and not run_pending))


def _live_data(ai_ok: bool) -> None:
    st.title("Live Data")
    st.caption(NEAR_REALTIME)
    db = _db()
    try:
        freshness = source_live_status(db)
    finally:
        db.close()
    last_ok = freshness.get("last_successful_run")
    checked = last_ok.finished_at if last_ok else None
    last_session = st.session_state.get("last_collection") or {}
    new_found = (last_session.get("stats") or {}).get("new")
    if new_found is None and last_ok is not None:
        new_found = last_ok.new_count
    st.write("Last checked:", f"{humanize_ago(checked)} ({checked.isoformat()})" if checked else "never")
    st.write("New reviews found:", new_found if new_found is not None else "—")
    st.write("Do not treat this as a live stream. Public store feeds are polled on demand.")
    st.write("Google Play package:", OFFICIAL_GOOGLE_PLAY_APP_ID)
    st.write("Apple App ID:", OFFICIAL_APPLE_APP_ID)
    st.write("**All-time Myntra-valid reviews:**", get_review_count(myntra_only=True))
    st.write(
        "**Last 30-day Myntra-valid reviews:**",
        get_review_count(since=get_last_30_days_cutoff(), myntra_only=True),
    )
    st.caption("Storage: Local application storage. Streamlit Cloud restarts can wipe stored reviews.")

    _show_last_collection()
    _diagnostics()
    _pipeline_status_panel()
    _ai_analysis_status_panel(ai_ok)
    diag = get_database_diagnostics()
    st.subheader("DATABASE DIAGNOSTICS")
    st.write("Total reviews:", diag.get("total_reviews"))
    st.write("Google Play reviews:", diag.get("google_play_reviews"))
    st.write("Apple reviews:", diag.get("apple_reviews"))
    st.write("Last 30-day reviews:", diag.get("last_30_day_reviews"))
    st.write("Pending reviews:", diag.get("pending_reviews"))
    st.write("Analyzed reviews:", diag.get("analyzed_reviews"))
    st.write("Failed reviews:", diag.get("failed_reviews"))
    _collect_buttons(ai_ok)

    analyze = st.checkbox("Analyze new reviews after each poll", value=ai_ok)
    if analyze and not ai_ok:
        st.warning(
            "OpenRouter API key is not configured. "
            "Add OPENROUTER_API_KEY to Streamlit Secrets or .env. Collection will run; analysis will be skipped."
        )
        analyze = False
    st.session_state["analyze_on_collect"] = analyze

    auto = st.selectbox("Auto-refresh", ["OFF", "5 minutes", "15 minutes", "30 minutes", "60 minutes"], index=0)
    st.caption(
        "Auto-refresh is a near-real-time poll of public store feeds, not a live stream. "
        "Streamlit Cloud often pauses idle apps, so manual Refresh Latest Reviews is the reliable mechanism."
    )
    if auto != "OFF":
        seconds = {"5 minutes": 300, "15 minutes": 900, "30 minutes": 1800, "60 minutes": 3600}[auto]
        if st.session_state.get("auto_refresh_interval") != auto:
            st.session_state["auto_refresh_interval"] = auto
            st.session_state["auto_refresh_armed_at"] = datetime.now(timezone.utc).isoformat()
        armed_at = datetime.fromisoformat(st.session_state["auto_refresh_armed_at"])
        last_auto = st.session_state.get("last_auto_poll_at")
        baseline = datetime.fromisoformat(last_auto) if last_auto else armed_at
        remaining = max(0, seconds - (datetime.now(timezone.utc) - baseline).total_seconds())
        st.write(f"Next automatic poll in about {int(remaining)} seconds.")
        tick = min(60, seconds)

        @st.fragment(run_every=timedelta(seconds=tick))
        def _auto_refresh_watchdog() -> None:
            last_poll = st.session_state.get("last_auto_poll_at")
            start = datetime.fromisoformat(last_poll) if last_poll else datetime.fromisoformat(
                st.session_state["auto_refresh_armed_at"]
            )
            now = datetime.now(timezone.utc)
            if (now - start).total_seconds() < seconds:
                left = int(max(0, seconds - (now - start).total_seconds()))
                st.caption(f"Auto-poll waiting · {left}s remaining · last checked {humanize_ago(start)}")
                return
            st.session_state["last_auto_poll_at"] = now.isoformat()
            _run_collect(["google_play", "apple_app_store"], analyze=analyze, mode="latest")

        _auto_refresh_watchdog()
        if st.button("Run due auto-poll now"):
            st.session_state["last_auto_poll_at"] = datetime.now(timezone.utc).isoformat()
            _run_collect(["google_play", "apple_app_store"], analyze=analyze, mode="latest")

    play_only = st.button("Collect Google Play only")
    apple_only = st.button("Collect Apple App Store only")
    if play_only:
        _run_collect(["google_play"], analyze=analyze, mode="latest")
    if apple_only:
        _run_collect(["apple_app_store"], analyze=analyze, mode="latest")
    if st.button("Analyze Pending Reviews"):
        reload_settings()
        if not get_settings().has_ai_credentials:
            st.error(
                "OpenRouter API key is not configured. "
                "Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
            )
        elif get_review_count() == 0:
            st.info(EMPTY)
        else:
            _run_analyze()


def _run_full_discovery(ai_ok: bool) -> None:
    from app.collectors.engine import CollectionEngine
    from app.pipeline.orchestrator import run_analysis_pipeline

    reload_settings()
    settings = get_settings()
    steps = st.session_state.setdefault(
        "pipeline_steps",
        {
            "play": "pending",
            "apple": "pending",
            "save": "pending",
            "analyze": "pending",
            "insights": "pending",
            "dashboard": "pending",
        },
    )
    db = _db()
    try:
        with st.status("Full discovery pipeline", expanded=True) as box:
            stored = get_review_count(db, myntra_only=True)
            if stored > 0:
                box.write(
                    f"STEP 1 — Collecting Google Play skipped "
                    f"({stored} stored Myntra-valid reviews already in the database)"
                )
                steps["play"] = "done"
                box.write("STEP 2 — Collecting Apple App Store skipped (using existing database)")
                steps["apple"] = "done"
                box.write(f"STEP 3 — Saving reviews ✓  using {stored} stored reviews")
                steps["save"] = "done"
            else:
                box.write("STEP 1 — Collecting Google Play")
                steps["play"] = "running"
                engine = CollectionEngine(db)
                gp = engine.run(["google_play"], analyze=False, mode="last_30_days")
                if gp.errors and gp.fetched == 0:
                    steps["play"] = "failed"
                    st.error("Google Play collection failed: " + "; ".join(gp.errors))
                else:
                    steps["play"] = "done"
                    box.write(f"STEP 1 — Collecting Google Play ✓  fetched {gp.fetched}, new {gp.new}")

                box.write("STEP 2 — Collecting Apple App Store")
                steps["apple"] = "running"
                apple = engine.run(["apple_app_store"], analyze=False, mode="last_30_days")
                if apple.errors and apple.fetched == 0:
                    steps["apple"] = "failed"
                    st.error("Apple App Store collection failed: " + "; ".join(apple.errors))
                else:
                    steps["apple"] = "done"
                    region = (apple.by_source.get("apple_app_store") or {}).get("region_used") or "—"
                    box.write(
                        f"STEP 2 — Collecting Apple App Store ✓  region={region} "
                        f"fetched {apple.fetched}, new {apple.new}"
                    )
                box.write("STEP 3 — Saving reviews ✓")
                steps["save"] = "done"

            box.write("STEP 4 — Analyzing reviews")
            steps["analyze"] = "running"
            from app.pipeline.analysis import AnalysisRunResult, smoke_test_analyze_limit

            result = AnalysisRunResult()
            model_name = get_settings().resolved_model
            if not get_settings().has_ai_credentials:
                steps["analyze"] = "failed"
                msg = (
                    "OpenRouter API key is not configured. "
                    "Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
                )
                result.last_error = msg
                st.session_state["step4_error"] = {
                    "provider": "OpenRouter",
                    "model": model_name,
                    "error": msg,
                    "http_status": None,
                }
                st.session_state["last_analysis"] = {"status": "Failed", "message": msg}
            else:
                try:
                    result = run_analysis_pipeline(
                        db, analyze_limit=smoke_test_analyze_limit(db, settings)
                    )
                except Exception as exc:
                    steps["analyze"] = "failed"
                    msg = _openrouter_error(exc)
                    http_status = getattr(exc, "http_status", None)
                    result = AnalysisRunResult(
                        failed=1, last_error=msg, last_http_status=http_status
                    )
                    st.session_state["step4_error"] = {
                        "provider": "OpenRouter",
                        "model": model_name,
                        "error": msg,
                        "http_status": http_status,
                    }
                    st.session_state["last_analysis"] = {"status": "Failed", "message": msg}
            if result.analyzed == 0 and (result.failed or result.last_error):
                steps["analyze"] = "failed"
                box.write("STEP 4 — Analyzing reviews FAILED")
                st.session_state["step4_error"] = {
                    "provider": "OpenRouter",
                    "model": model_name,
                    "error": result.last_error,
                    "http_status": result.last_http_status,
                }
                _render_step4_failure(
                    model=model_name,
                    error=result.last_error,
                    http_status=result.last_http_status,
                )
            else:
                steps["analyze"] = "done"
                st.session_state.pop("step4_error", None)
                box.write(
                    f"STEP 4 — Analyzing reviews ✓  analyzed {result.analyzed}, failed {result.failed}"
                )
            if result.analyzed:
                box.write("STEP 5 — Generating discovery insights ✓")
                steps["insights"] = "done"
            else:
                steps["insights"] = "failed"
                box.write(
                    "STEP 5 — Generating discovery insights FAILED. "
                    "Discovery insights could not be generated because AI analysis failed."
                )
            box.write("STEP 6 — Updating dashboard ✓")
            steps["dashboard"] = "done"
            if result.analyzed:
                box.update(label="Discovery pipeline complete", state="complete")
            else:
                box.update(label="AI analysis failed", state="error")
            st.session_state["last_analysis"] = {
                "status": "Connected" if result.analyzed else ("Failed" if result.failed or result.last_error else "Configured"),
                "message": (
                    f"Analyzed {result.analyzed} reviews, failed {result.failed}. "
                    + (result.last_error or "Themes, segments, and opportunity scores were rebuilt.")
                ),
            }
            st.session_state["pipeline_steps"] = steps
    except Exception as exc:
        LOGGER.exception("Full discovery failed")
        st.error(f"Discovery pipeline failed: {exc}")
        if st.session_state.get("debug"):
            st.code(traceback.format_exc())
        return
    finally:
        db.close()
    st.rerun()


def _render_step4_failure(*, model: str, error: str, http_status: int | None = None) -> None:
    st.error("AI analysis failed")
    st.write("Provider:")
    st.write("OpenRouter")
    st.write("Model:")
    st.write(model or "")
    st.write("Error:")
    st.write(error or "OpenRouter analysis failed.")
    st.write("HTTP status:")
    st.write(http_status if http_status is not None else "N/A")


def _pipeline_status_panel() -> None:
    steps = st.session_state.get("pipeline_steps") or {}
    if not steps:
        return
    st.subheader("PIPELINE STATUS")
    labels = [
        ("play", "STEP 1  Collecting Google Play"),
        ("apple", "STEP 2  Collecting Apple App Store"),
        ("save", "STEP 3  Saving reviews"),
        ("analyze", "STEP 4  Analyzing reviews"),
        ("insights", "STEP 5  Generating discovery insights"),
        ("dashboard", "STEP 6  Updating dashboard"),
    ]
    for key, title in labels:
        state = steps.get(key, "pending")
        mark = "✓" if state == "done" else ("FAILED" if state == "failed" else ("…" if state == "running" else "—"))
        st.write(f"{title}  {mark}")
    step4 = st.session_state.get("step4_error") or {}
    if steps.get("analyze") == "failed" and step4:
        _render_step4_failure(
            model=str(step4.get("model") or ""),
            error=str(step4.get("error") or ""),
            http_status=step4.get("http_status"),
        )
    if steps.get("insights") == "failed":
        st.warning(
            "Discovery insights could not be generated because AI analysis failed."
        )


def _run_collect(sources: list[str], analyze: bool, mode: str = "latest") -> None:
    from app.collectors.engine import CollectionEngine
    from app.pipeline.quantification import overview_metrics

    reload_settings()
    db = _db()
    try:
        prior = overview_metrics(db, since=get_last_30_days_cutoff(), myntra_only=True)
        st.session_state["insight_snapshot"] = {
            "barriers": [x.get("label") for x in (prior.get("top_barriers") or [])]
        }
        spinner = (
            "Collecting last-30-day public reviews (paginated until older than cutoff)…"
            if mode == "last_30_days"
            else "Polling public sources for newly available reviews…"
        )
        with st.spinner(spinner):
            engine = CollectionEngine(db)
            stats = engine.run(sources, analyze=analyze, mode=mode)
        payload = stats.model_dump(mode="json")
        st.session_state["last_collection"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "mode": mode,
            "stats": payload,
        }
        if analyze:
            if stats.analysis_error and stats.analyzed == 0:
                st.error("Reviews collected successfully, but OpenRouter analysis failed.")
                st.error(stats.analysis_error)
                st.session_state["last_analysis"] = {"status": "Failed", "message": stats.analysis_error}
            else:
                st.session_state["last_analysis"] = {
                    "status": "Connected" if stats.analyzed else "Skipped",
                    "message": (
                        f"Analyzed {stats.analyzed} reviews"
                        + (f", failed {stats.analysis_failed}" if stats.analysis_failed else "")
                        + f". {stats.pending_remaining} still pending."
                    ),
                }
    except Exception as exc:
        LOGGER.exception("Collection failed")
        st.session_state["last_collection"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "mode": mode,
            "stats": {"fetched": 0, "new": 0, "duplicates": 0, "errors": [str(exc)], "by_source": {}},
        }
        st.error(f"Collection failed: {exc}")
        if st.session_state.get("debug"):
            st.code(traceback.format_exc())
        return
    finally:
        db.close()
    st.rerun()


def _latest_reviews(myntra_only: bool) -> None:
    st.title("Latest Reviews")
    st.caption(NEAR_REALTIME + ". Sorted by review date, not collection time.")
    source = st.selectbox("Source", ["All", "Google Play", "Apple"])
    rating = st.selectbox("Rating", ["All", "1", "2", "3", "4", "5"])
    date_filter = st.selectbox(
        "Date",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom", "All Time"],
        index=2,
    )
    since = None
    until = None
    if date_filter == "Last 24 hours":
        since = window_start(hours=24)
    elif date_filter == "Last 7 days":
        since = window_start(days=7)
    elif date_filter == "Last 30 days":
        since = get_last_30_days_cutoff()
    elif date_filter == "Custom":
        c1, c2 = st.columns(2)
        start = c1.date_input("From (review date)")
        end = c2.date_input("To (review date)")
        since = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        until = datetime.combine(end, datetime.max.time(), tzinfo=timezone.utc)
    db = _db()
    try:
        query = db.query(Review).filter(Review.is_empty.is_(False), Review.is_duplicate.is_(False))
        if myntra_only:
            query = query.filter(Review.is_valid_source.is_(True))
        if source == "Google Play":
            query = query.filter(Review.source == "google_play")
        elif source == "Apple":
            query = query.filter(Review.source == "apple_app_store")
        if rating != "All":
            query = query.filter(Review.rating == int(rating))
        if since is not None:
            query = query.filter(Review.review_date.isnot(None), Review.review_date >= since)
        if until is not None:
            query = query.filter(Review.review_date.isnot(None), Review.review_date <= until)
        rows = query.order_by(Review.review_date.desc(), Review.collected_at.desc()).limit(80).all()
        if not rows:
            if get_review_count(db) == 0:
                st.info(EMPTY)
                _collect_buttons(get_settings().has_ai_credentials)
            else:
                st.info("No reviews match these filters.")
            return
        for review in rows:
            st.markdown(f"**{review.source}** · rating {review.rating} · region {review.region or '—'}")
            st.write(review.text or review.title)
            st.caption(
                f"Review date: {review.review_date} · collected: {review.collected_at} · "
                f"id {review.source_review_id} · {review.source_url}"
            )
    finally:
        db.close()


def _collection_history() -> None:
    st.title("Collection History")
    st.caption("Storage: Local application storage. Streamlit Cloud restarts can wipe this history.")
    db = _db()
    try:
        rows = db.query(CollectionRun).order_by(CollectionRun.id.desc()).limit(25).all()
        if not rows:
            st.info("No collection runs recorded yet.")
            _collect_buttons(get_settings().has_ai_credentials)
            return
        st.dataframe(
            [
                {
                    "collection_id": r.id,
                    "source": r.sources,
                    "mode": r.mode or "",
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.finished_at.isoformat() if r.finished_at else None,
                    "reviews_fetched": r.fetched,
                    "new_reviews": r.new_count,
                    "duplicates": r.duplicates,
                    "errors": r.errors_json,
                    "status": r.status,
                }
                for r in rows
            ],
            width="stretch",
        )
    finally:
        db.close()


def _run_analyze(*, only_failed: bool = False) -> None:
    from app.pipeline.orchestrator import run_analysis_pipeline

    reload_settings()
    db = _db()
    try:
        if not get_settings().has_ai_credentials:
            st.error(
                "OpenRouter API key is not configured. "
                "Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
            )
            st.session_state["last_analysis"] = {
                "status": "Failed",
                "message": (
                    "OpenRouter API key is not configured. "
                    "Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
                ),
            }
            return
        spinner = (
            "Retrying failed OpenRouter analysis…"
            if only_failed
            else "Analyzing stored Myntra-valid reviews with OpenRouter…"
        )
        with st.spinner(spinner):
            from app.pipeline.analysis import smoke_test_analyze_limit

            settings = get_settings()
            result = run_analysis_pipeline(
                db,
                analyze_limit=smoke_test_analyze_limit(db, settings),
                only_failed=only_failed,
                include_failed=only_failed,
            )
        if get_review_count(db) == 0:
            st.session_state["last_analysis"] = {"status": "Failed", "message": EMPTY}
            st.info(EMPTY)
            return
        if result.analyzed == 0 and result.failed:
            msg = _openrouter_error(
                result.last_error or f"OpenRouter analysis failed for {result.failed} reviews."
            )
            st.session_state["last_analysis"] = {"status": "Failed", "message": msg}
            st.error(msg)
        elif result.analyzed == 0:
            st.session_state["last_analysis"] = {
                "status": "Configured",
                "message": (
                    "No failed reviews to retry."
                    if only_failed
                    else "No new Myntra-valid reviews needed analysis (already analyzed)."
                ),
            }
        else:
            st.session_state["last_analysis"] = {
                "status": "Connected",
                "message": (
                    f"Analyzed {result.analyzed} reviews"
                    + (f", failed {result.failed}" if result.failed else "")
                    + ". Themes, segments, and opportunity scores were rebuilt."
                ),
            }
    except Exception as exc:
        LOGGER.exception("Analysis failed")
        msg = _openrouter_error(exc)
        st.session_state["last_analysis"] = {"status": "Failed", "message": msg}
        st.error(msg)
        return
    finally:
        db.close()
    st.rerun()


def _explorer(myntra_only: bool) -> None:
    st.title("Feedback Explorer")
    q = st.text_input("Search original text")
    db = _db()
    try:
        query = db.query(Review).filter(Review.is_empty.is_(False))
        if myntra_only:
            query = query.filter(Review.is_valid_source.is_(True))
        rows = query.order_by(Review.collected_at.desc()).limit(80).all()
        if q:
            needle = q.lower()
            rows = [r for r in rows if needle in (r.text or "").lower() or needle in (r.title or "").lower()]
        if not rows:
            _empty_notice(get_review_count(db))
            return
        for review in rows:
            payload = serialize_review(review)
            with st.expander(f"{payload['app_name']} · {payload['source']} · {payload['data_classification']}"):
                if payload["is_synthetic"]:
                    st.warning("SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA")
                st.write(payload["text"])
                st.caption(
                    f"ID {payload['source_review_id']} · rating {payload['rating']} · "
                    f"{payload['review_date']} · {payload['source_url']} · region {payload['region']}"
                )
                if payload["warning"]:
                    st.error(payload["warning"])
                analysis = payload.get("analysis")
                if analysis:
                    st.write("Root cause:", analysis["root_cause"])
                    st.write("Barriers:", analysis["barriers"])
                    st.write("Uncertainties:", analysis["uncertainties"])
                else:
                    st.caption("Not yet analyzed.")
    finally:
        db.close()


def _labels(field: str, myntra_only: bool, title: str) -> None:
    st.title(title)
    since = _period_since()
    db = _db()
    try:
        count = get_review_count(db, since=since, myntra_only=myntra_only)
        all_time = get_review_count(db, myntra_only=myntra_only)
        rows = label_distribution(
            db, field, myntra_only=myntra_only, relevant_only=False, since=since
        )
        diag = get_database_diagnostics(db)
        analyzed = diag.get("analyzed_reviews") or 0
    finally:
        db.close()
    ai_ok = get_settings().has_ai_credentials
    if all_time == 0:
        st.info(EMPTY)
        _collect_buttons(ai_ok)
        return
    if count == 0:
        st.info("No reviews with timestamps in the selected period.")
        return
    if not rows:
        msg = _analysis_blocker(
            stored=all_time,
            analyzed=analyzed,
            pending=diag.get("pending_reviews") or 0,
            failed=diag.get("failed_reviews") or 0,
            ai_ok=ai_ok,
            last_error=str(diag.get("last_analysis_error") or ""),
        )
        if msg:
            (st.info if msg == EMPTY else st.warning)(msg)
            _collect_buttons(ai_ok)
            return
        st.info(
            f"{analyzed} reviews were analyzed; none contained this signal in the selected period. "
            "That is a sample-size statement about analyzed reviews, not about all Myntra users."
        )
        return
    st.caption(
        f"Sample size: {rows[0].get('denominator', analyzed)} analyzed reviews in this period. "
        "Percentages are of analyzed reviews, not of the Myntra user base."
    )
    st.dataframe(
        [
            {
                "signal": r["label"],
                "frequency": r["count"],
                "percentage": r["percentage"],
                "supporting_reviews": r["count"],
                "sources": ", ".join(r.get("sources") or []),
                "review_ids": r.get("review_ids") or [],
            }
            for r in rows
        ],
        width="stretch",
    )
    if rows and rows[0].get("review_ids"):
        st.subheader("Supporting evidence")
        db = _db()
        try:
            cards = evidence_cards(db, rows[0]["review_ids"], limit=8)
        finally:
            db.close()
        for card in cards:
            st.write(f"“{card['quote']}”")
            st.caption(
                f"{card['source']} · rating {card.get('rating')} · region {card.get('region') or '—'} · "
                f"review date {card['date']} · id {card['source_review_id']}"
            )


def _root_causes(myntra_only: bool = True) -> None:
    st.title("User Problems")
    since = _period_since()
    ai_ok = get_settings().has_ai_credentials
    db = _db()
    try:
        all_time = get_review_count(db, myntra_only=myntra_only)
        count = get_review_count(db, since=since, myntra_only=myntra_only)
        rows = problem_rows(db, myntra_only=myntra_only, since=since)
        diag = get_database_diagnostics(db)
        cards_by_problem = {}
        if rows:
            cards_by_problem = {
                rows[0]["problem"]: evidence_cards(db, rows[0].get("review_ids") or [], limit=8)
            }
    finally:
        db.close()
    if all_time == 0:
        st.info(EMPTY)
        _collect_buttons(ai_ok)
        return
    if count == 0:
        st.info("No reviews with timestamps in the selected period.")
        return
    if not rows:
        msg = _analysis_blocker(
            stored=all_time,
            analyzed=diag.get("analyzed_reviews") or 0,
            pending=diag.get("pending_reviews") or 0,
            failed=diag.get("failed_reviews") or 0,
            ai_ok=ai_ok,
            last_error=str(diag.get("last_analysis_error") or ""),
        )
        if msg:
            (st.info if msg == EMPTY else st.warning)(msg)
            _collect_buttons(ai_ok)
            return
        st.info(
            f"{diag.get('analyzed_reviews', 0)} reviews were analyzed; "
            "no root-cause statements were extracted in this period."
        )
        return
    st.caption(
        f"Sample size: {rows[0].get('denominator')} analyzed reviews. "
        "Severity and purchase impact are programmatic 1–5 scores, not LLM arithmetic."
    )
    st.dataframe(
        [
            {
                "problem": r["problem"],
                "frequency": r["frequency"],
                "% of analyzed": r["percentage"],
                "severity": f"{r['severity']}/5",
                "purchase_impact": f"{r['purchase_impact']}/5",
                "supporting_reviews": r["supporting_reviews"],
                "confidence": r["confidence"],
                "model": r["model"],
                "analysis_version": r["analysis_version"],
            }
            for r in rows
        ],
        width="stretch",
    )
    if cards_by_problem:
        st.subheader("Supporting evidence (top problem)")
        for card in next(iter(cards_by_problem.values())):
            st.write(f"“{card['quote']}”")
            st.caption(
                f"{card['source']} · rating {card.get('rating')} · region {card.get('region') or '—'} · "
                f"review date {card['date']} · id {card['source_review_id']}"
            )


def _opportunities() -> None:
    st.title("Opportunity Matrix")
    st.caption("Score = reach × frequency × purchase impact × severity × evidence confidence (each 1–5). Calculated in Python.")
    ai_ok = get_settings().has_ai_credentials
    db = _db()
    try:
        count = get_review_count(db, myntra_only=True)
        rows = db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
        diag = get_database_diagnostics(db)
    finally:
        db.close()
    if count == 0:
        st.info(EMPTY)
        _collect_buttons(ai_ok)
        return
    if not rows:
        msg = _analysis_blocker(
            stored=count,
            analyzed=diag.get("analyzed_reviews") or 0,
            pending=diag.get("pending_reviews") or 0,
            failed=diag.get("failed_reviews") or 0,
            ai_ok=ai_ok,
            last_error=str(diag.get("last_analysis_error") or ""),
        )
        if msg:
            (st.info if msg == EMPTY else st.warning)(msg)
            _collect_buttons(ai_ok)
            return
        st.info("Reviews were analyzed, but no opportunity groups could be scored from extracted barriers yet.")
        return
    st.dataframe(
        [
            {
                "rank": o.rank,
                "problem": o.user_problem,
                "score": o.score,
                "reach": o.reach,
                "frequency": o.frequency,
                "impact": o.purchase_impact,
                "severity": o.severity,
                "confidence": o.evidence_confidence,
                "pct": o.percentage,
                "sources": o.sources_json,
                "cross_source": o.cross_source_status,
                "includes_non_myntra": o.includes_non_myntra,
            }
            for o in rows
        ],
        width="stretch",
    )


def _named(model, title: str) -> None:
    st.title(title)
    since = _period_since()
    db = _db()
    try:
        count = get_review_count(db, since=since, myntra_only=True)
        all_time = get_review_count(db, myntra_only=True)
        rows = db.query(model).all()
        momentum = []
        if model is Theme and since is not None:
            momentum = label_window_momentum(db, "themes", since=since, myntra_only=True)
    finally:
        db.close()
    if all_time == 0:
        st.info(EMPTY)
        _collect_buttons(get_settings().has_ai_credentials)
        return
    if count == 0:
        st.info("No reviews with timestamps in the selected period.")
        return
    if not rows:
        diag = get_database_diagnostics()
        msg = _analysis_blocker(
            stored=all_time,
            analyzed=diag.get("analyzed_reviews") or 0,
            pending=diag.get("pending_reviews") or 0,
            failed=diag.get("failed_reviews") or 0,
            ai_ok=get_settings().has_ai_credentials,
            last_error=str(diag.get("last_analysis_error") or ""),
        )
        if msg:
            (st.info if msg == EMPTY else st.warning)(msg)
            _collect_buttons(get_settings().has_ai_credentials)
            return
        st.info(
            f"{diag.get('analyzed_reviews', 0)} reviews were analyzed; "
            "no evidence-backed items of this type were generated."
        )
        return
    if momentum:
        st.subheader("Emerging / stable / declining (selected period)")
        st.dataframe(
            [
                {
                    "theme": x["label"],
                    "count": x["count"],
                    "momentum": x["momentum"],
                    "first_half": x["first_half"],
                    "second_half": x["second_half"],
                }
                for x in momentum[:15]
            ],
            width="stretch",
        )
        st.caption("Descriptive split of this window, not a statistical significance test.")
    for row in rows:
        st.subheader(row.name)
        st.write(getattr(row, "description", "") or "")
        st.caption(f"{row.review_count} reviews · Myntra {getattr(row, 'myntra_review_count', 0)}")


def _evidence() -> None:
    st.title("Evidence Explorer")
    db = _db()
    try:
        count = get_review_count(db)
        rows = db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
        if count == 0:
            st.info(EMPTY)
            _collect_buttons(get_settings().has_ai_credentials)
            return
        if not rows:
            diag = get_database_diagnostics(db)
            msg = _analysis_blocker(
                stored=count,
                analyzed=diag.get("analyzed_reviews") or 0,
                pending=diag.get("pending_reviews") or 0,
                failed=diag.get("failed_reviews") or 0,
                ai_ok=get_settings().has_ai_credentials,
                last_error=str(diag.get("last_analysis_error") or ""),
            )
            if msg:
                (st.info if msg == EMPTY else st.warning)(msg)
                _collect_buttons(get_settings().has_ai_credentials)
                return
            st.info("No opportunity evidence groups are linked yet.")
            return
        choice = st.selectbox("Opportunity", [f"{o.rank}. {o.user_problem}" for o in rows])
        rank = int(str(choice).split(".", 1)[0])
        selected = next((o for o in rows if o.rank == rank), rows[0])
        ids = json.loads(selected.evidence_ids_json or "[]")
        cards = evidence_cards(db, ids, limit=15)
        if not cards:
            st.info("No linked source records.")
            return
        for card in cards:
            if card.get("is_synthetic"):
                st.warning("SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA")
            st.write(f"“{card['quote']}”")
            st.caption(
                f"{card['source']} · rating {card.get('rating')} · region {card.get('region') or '—'} · "
                f"{card['app_name']} · {card['data_classification']} · "
                f"{card['source_review_id']} · review date {card['date']} · {card['source_url']}"
            )
    finally:
        db.close()


def _report() -> None:
    st.title("Discovery Report")
    db = _db()
    try:
        count = get_review_count(db)
        report = build_report(db)
    finally:
        db.close()
    if count == 0:
        st.info(EMPTY)
        _collect_buttons(get_settings().has_ai_credentials)
        return
    st.write(report["research_question"])
    st.write(report["business_goal"])
    st.caption(report["anti_solution_note"])
    diag = get_database_diagnostics()
    st.subheader("Executive summary")
    st.write(
        f"Stored Myntra-valid reviews: {diag.get('myntra_reviews')}. "
        f"Analyzed: {diag.get('analyzed_reviews')}. Pending: {diag.get('pending_reviews')}."
    )
    if (diag.get("analyzed_reviews") or 0) == 0:
        msg = _analysis_blocker(
            stored=diag.get("myntra_reviews") or count,
            analyzed=0,
            pending=diag.get("pending_reviews") or 0,
            failed=diag.get("failed_reviews") or 0,
            ai_ok=get_settings().has_ai_credentials,
            last_error=str(diag.get("last_analysis_error") or ""),
        )
        if msg:
            (st.info if msg == EMPTY else st.warning)(msg)
            _collect_buttons(get_settings().has_ai_credentials)
            return
    st.subheader("Key user problems")
    for item in (report["sections"].get("7_purchase_barriers") or [])[:8]:
        st.write(f"- {item.get('label')} ({item.get('count')} reviews, {item.get('percentage')}%)")
    st.subheader("Why users wishlist")
    for item in (report["sections"].get("6_wishlist_motivations") or [])[:8]:
        st.write(f"- {item.get('label')} ({item.get('count')})")
    st.subheader("Why users do not purchase")
    for item in (report["sections"].get("7_purchase_barriers") or [])[:8]:
        st.write(f"- {item.get('label')} ({item.get('count')})")
    st.subheader("Purchase uncertainties")
    for item in (report["sections"].get("8_uncertainties") or [])[:8]:
        st.write(f"- {item.get('label')} ({item.get('count')})")
    st.subheader("Top themes")
    for theme in report["sections"].get("13_emergent_themes") or []:
        st.write(f"- {theme.get('name')} ({theme.get('review_count')} reviews)")
    st.subheader("User segments")
    for seg in report["sections"].get("11_user_segments") or []:
        st.write(f"- {seg.get('name')} ({seg.get('review_count')} reviews)")
    st.subheader("What we know")
    for item in report["sections"]["19_what_we_know"]:
        st.write("-", item)
    st.subheader("What we don’t know")
    for item in report["sections"]["20_what_we_dont_know"]:
        st.write("-", item)
    st.subheader("Data limitations")
    st.write(
        "Public app-store reviews are not a conversion dataset. "
        "They over-represent extreme ratings and cannot prove 30-day wishlist conversion. "
        "Storage: Local application storage (ephemeral on Streamlit Cloud)."
    )
    primary = report.get("single_most_promising_problem") or {}
    st.subheader("Opportunity areas")
    st.write(primary.get("why_first") or "No scored opportunity yet — analyze pending reviews first.")
    top = report.get("top_opportunities") or []
    if not top:
        st.caption("Opportunity scores appear after barriers are extracted from analyzed reviews.")
        return
    st.subheader("Evidence")
    for item in top:
        st.markdown(f"**#{item['rank']} {item['opportunity']}** — score {item['opportunity_score']}")
        for ev in item.get("evidence") or []:
            st.write(f"“{ev['quote']}”")
            st.caption(f"{ev['source']} · {ev['data_classification']} · {ev['source_url']}")
