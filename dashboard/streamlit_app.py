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
from app.database import SessionLocal, get_review_count, get_review_stats, init_db, sqlite_path
from app.models import Analysis, CollectionRun, Opportunity, Review, Segment, Theme
from app.pipeline.dates import get_last_30_days_cutoff, humanize_ago, window_start
from app.pipeline.quantification import (
    daily_review_trends,
    label_distribution,
    label_window_momentum,
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
NO_ANALYSIS = "Not enough real feedback collected for analysis."
NEAR_REALTIME = "Near-real-time — refreshed from the public source"


@st.cache_resource
def _bootstrap() -> bool:
    init_db()
    return True


def _db() -> Session:
    return SessionLocal()


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
        st.info(NO_ANALYSIS)
        return True
    return False


def render() -> None:
    st.set_page_config(page_title="Myntra Discovery Engine", layout="wide")
    _bootstrap()
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
    cutoff = _period_since()
    stored_all = get_review_count(myntra_only=myntra_only)
    st.sidebar.markdown("**Stored reviews**")
    st.sidebar.write("All time:", stored_all)
    st.sidebar.write(
        "Last 30 days:",
        get_review_count(since=get_last_30_days_cutoff(), myntra_only=myntra_only),
    )
    st.sidebar.markdown("**AI**")
    st.sidebar.write("OpenRouter key:", "configured" if ai_ok else "missing")
    st.sidebar.write("Model:", settings.resolved_model)
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
    c1, c2 = st.columns(2)
    thirty = c1.button("Collect Last 30 Days", type="primary")
    refresh = c2.button("🔄 Refresh Latest Reviews", type="primary")
    analyze = bool(st.session_state.get("analyze_on_collect", False))
    if thirty:
        _run_collect(["google_play", "apple_app_store"], analyze=analyze, mode="last_30_days")
    if refresh:
        _run_collect(["google_play", "apple_app_store"], analyze=analyze, mode="latest")


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
        st.info(NO_ANALYSIS)

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
    _collect_buttons(ai_ok)

    analyze = st.checkbox("Analyze new reviews after each poll", value=ai_ok)
    if analyze and not ai_ok:
        st.warning("OPENROUTER_API_KEY is missing. Collection will run; analysis will be skipped.")
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
    if st.button("Analyze pending Myntra-valid reviews"):
        if not ai_ok:
            st.error("OPENROUTER_API_KEY is not set in Streamlit Secrets or .env.")
        elif get_review_count() == 0:
            st.info(NO_ANALYSIS)
        else:
            _run_analyze()


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
            st.session_state["last_analysis"] = {
                "status": "Connected" if stats.analyzed else "Skipped",
                "message": f"Analyzed {stats.analyzed} new/pending reviews.",
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


def _run_analyze() -> None:
    from app.pipeline.orchestrator import run_analysis_pipeline

    reload_settings()
    db = _db()
    try:
        with st.spinner("Analyzing stored Myntra-valid reviews…"):
            analyzed = run_analysis_pipeline(db)
        if get_review_count(db) == 0:
            st.session_state["last_analysis"] = {"status": "Failed", "message": NO_ANALYSIS}
            st.info(NO_ANALYSIS)
            return
        if analyzed == 0:
            st.session_state["last_analysis"] = {
                "status": "Configured",
                "message": "No new Myntra-valid reviews needed analysis (already analyzed).",
            }
        else:
            st.session_state["last_analysis"] = {
                "status": "Connected",
                "message": f"Analyzed {analyzed} reviews. Themes, segments, and opportunity scores were rebuilt.",
            }
    except Exception as exc:
        LOGGER.exception("Analysis failed")
        st.session_state["last_analysis"] = {"status": "Failed", "message": str(exc)}
        st.error(f"AI analysis failed: {exc}")
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
        count = get_review_count(db, since=since)
        all_time = get_review_count(db)
        rows = label_distribution(db, field, myntra_only=myntra_only, since=since)
    finally:
        db.close()
    if all_time == 0:
        st.info(EMPTY)
        return
    if count == 0:
        st.info("No reviews with timestamps in the selected period.")
        return
    if not rows:
        st.info(NO_ANALYSIS)
        return
    st.dataframe(rows, width="stretch")


def _root_causes(myntra_only: bool = True) -> None:
    st.title("User Problems")
    since = _period_since()
    db = _db()
    try:
        all_time = get_review_count(db)
        count = get_review_count(db, since=since)
        query = db.query(Analysis).join(Review).filter(Analysis.is_valid_json.is_(True))
        if myntra_only:
            query = query.filter(Review.is_valid_source.is_(True))
        if since is not None:
            query = query.filter(Review.review_date.isnot(None), Review.review_date >= since)
        rows = query.all()
        items = [r for r in rows if (r.root_cause or "").strip()]
    finally:
        db.close()
    if all_time == 0:
        st.info(EMPTY)
        return
    if count == 0:
        st.info("No reviews with timestamps in the selected period.")
        return
    if not items:
        st.info(NO_ANALYSIS)
        return
    for row in items[:80]:
        st.write(row.root_cause)
        st.caption(
            f"observed: {row.root_cause_observed or '—'} · "
            f"inferred: {row.root_cause_inferred or '—'} · "
            f"hypothesized: {row.root_cause_hypothesized or '—'} · "
            f"analyzed_at: {row.analyzed_at} · model: {row.model} · status: {getattr(row, 'status', '')}"
        )


def _opportunities() -> None:
    st.title("Opportunity Matrix")
    st.caption("Score = reach × frequency × purchase impact × severity × evidence confidence (each 1–5). Calculated in Python.")
    db = _db()
    try:
        count = get_review_count(db)
        rows = db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
    finally:
        db.close()
    if count == 0:
        st.info(EMPTY)
        return
    if not rows:
        st.info(NO_ANALYSIS)
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
        count = get_review_count(db, since=since)
        all_time = get_review_count(db)
        rows = db.query(model).all()
        momentum = []
        if model is Theme and since is not None:
            momentum = label_window_momentum(db, "themes", since=since, myntra_only=True)
    finally:
        db.close()
    if all_time == 0:
        st.info(EMPTY)
        return
    if count == 0:
        st.info("No reviews with timestamps in the selected period.")
        return
    if not rows:
        st.info(NO_ANALYSIS)
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
            return
        if not rows:
            st.info(NO_ANALYSIS)
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
        return
    if report["sections"]["4_data_volume"]["total_reviews"] == 0:
        st.info(EMPTY)
        return
    st.write(report["research_question"])
    st.write(report["business_goal"])
    st.caption(report["anti_solution_note"])
    st.subheader("What we know")
    for item in report["sections"]["19_what_we_know"]:
        st.write("-", item)
    st.subheader("What we don’t know")
    for item in report["sections"]["20_what_we_dont_know"]:
        st.write("-", item)
    primary = report.get("single_most_promising_problem") or {}
    st.subheader("Single most promising user problem")
    st.write(primary.get("why_first") or NO_ANALYSIS)
    top = report.get("top_opportunities") or []
    if not top:
        st.info(NO_ANALYSIS)
        return
    for item in top:
        st.markdown(f"**#{item['rank']} {item['opportunity']}** — score {item['opportunity_score']}")
        for ev in item.get("evidence") or []:
            st.write(f"“{ev['quote']}”")
            st.caption(f"{ev['source']} · {ev['data_classification']} · {ev['source_url']}")
