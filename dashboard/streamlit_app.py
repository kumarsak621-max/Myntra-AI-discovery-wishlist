"""Single-page Streamlit discovery dashboard. Collectors and analysis stay unchanged."""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session

from app.database import (
    SessionLocal,
    get_database_diagnostics,
    get_review_count,
    init_db,
    migrate_schema,
)
from app.models import CollectionRun, Opportunity, Review, Segment, Theme
from app.pipeline.dates import ensure_aware, get_last_30_days_cutoff, humanize_ago, utcnow
from app.pipeline.labels import merge_category_rows, normalize_category_label, normalize_label, normalize_label_list
from app.pipeline.quantification import (
    BARRIER_TERMS,
    COMPARISON_METHOD_TERMS,
    COMPARISON_TERMS,
    EXTERNAL_TERMS,
    SEGMENT_TERMS,
    SOCIAL_TERMS,
    UNCERTAINTY_TERMS,
    WISHLIST_BEHAVIOR_TERMS,
    PURCHASE_BEHAVIOR_TERMS,
    daily_review_trends,
    explicit_age_mentions,
    label_distribution,
    label_window_momentum,
    latest_review_cards,
    overview_metrics,
    problem_rows,
    hesitation_split,
    purchase_signal_counts,
    rating_distribution,
    root_cause_hierarchy,
    signal_counts,
    source_live_status,
    taxonomy_counts,
    wishlist_intent_split,
)
from app.pipeline.report import build_report, evidence_cards
from config.settings import (
    analysis_review_limit,
    get_ai_config,
    get_settings,
    reload_settings,
    storage_review_limit,
)
from dashboard.charts import bar_chart, donut_chart, heatmap_impact_frequency, scatter_chart, trend_frame
from dashboard.chat import ask_product_assistant
from dashboard.insights import (
    derive_root_cause,
    funnel_stages,
    pm_insight,
    pm_insight_card,
    why_this_matters,
    wishlist_conversion_copy,
)
from dashboard.pipeline_status import derive_failed_reason, insights_status_for_analyze
from dashboard.questions import DISCOVERY_QUESTIONS

LOGGER = logging.getLogger("myntra.discovery")

EMPTY = "No real reviews have been collected yet."
PERIODIC_REFRESH = "Periodic public-source refresh"
AUTO_REFRESH_INTERVAL = timedelta(minutes=5)
AUTO_REFRESH_SECONDS = int(AUTO_REFRESH_INTERVAL.total_seconds())
INSUFFICIENT_ROOT = "Insufficient evidence to establish a reliable root cause."

SUGGESTED_QUESTIONS = list(DISCOVERY_QUESTIONS)
LIMITATIONS_NOTICE = (
    "Reviews are public feedback, not the complete Myntra customer base. "
    "Public reviews do not directly expose actual wishlist → purchase conversion events. "
    "Demographic information is generally unavailable. Behavioral segments are inferred only "
    "from textual evidence. Near-real-time means periodic public-source refresh, not live "
    "event streaming. Small samples should not be generalized to the full user base."
)
DEMOGRAPHIC_NOTICE = "Demographic attributes are not reliably available from public review text."


def _analysis_blocker(*, stored: int, analyzed: int, pending: int, failed: int, ai_ok: bool, last_error: str = "") -> str | None:
    if stored == 0:
        return EMPTY
    if analyzed > 0:
        return None
    if not ai_ok:
        return (
            f"{stored} real reviews are stored, but none have been analyzed. "
            "OpenRouter API key is not configured. Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
        )
    if failed and not analyzed:
        extra = f" Last error: {last_error}." if last_error else ""
        return f"{stored} reviews collected, but AI analysis failed.{extra}"
    return f"{pending or stored} real reviews are awaiting AI analysis."


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


def _dataset_limit() -> int:
    return storage_review_limit(get_settings())


def _analysis_limit() -> int:
    return analysis_review_limit(get_settings())


def _period_since():
    if st.session_state.get("period") == "All Time":
        return None
    return get_last_30_days_cutoff()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #e8eef8; }
        [data-testid="stSidebar"] { background: #101827; }
        [data-testid="stMetric"] {
            background: #151d2e; border: 1px solid #243047; border-radius: 14px;
            padding: 10px 12px;
        }
        [data-testid="stMetricLabel"] { color: #9fb0c8 !important; }
        .hero h1 { font-size: 1.7rem; margin-bottom: 0.15rem; }
        .hero p { color: #9fb0c8; margin-top: 0; }
        .section-h { font-size: 1.15rem; letter-spacing: 0.02em; margin: 1.4rem 0 0.4rem; }
        .pm-card {
            background: #132033; border-left: 3px solid #3d8bfd; border-radius: 10px;
            padding: 12px 14px; margin: 8px 0 16px;
        }
        .quote {
            background: #151d2e; border-radius: 10px; padding: 10px 12px; margin: 6px 0;
            border: 1px solid #243047;
        }
        .limit-card {
            background: #1a1520; border: 1px solid #3a2a3a; border-radius: 10px;
            padding: 12px 14px; color: #c9b8c8;
        }
        .pm-insight-card {
            background: #15243a; border: 1px solid #3d8bfd; border-radius: 12px;
            padding: 16px 18px; margin: 8px 0 16px;
        }
        .pipeline-row { font-size: 1.05rem; margin: 4px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _bar(df: pd.DataFrame, *, empty: str = "Insufficient evidence for this visualization.") -> None:
    if df is None or df.empty:
        st.info(empty)
        return
    bar_chart(df.to_dict("records"), empty=empty)


def _scatter(df: pd.DataFrame) -> None:
    scatter_chart(df.to_dict("records") if df is not None and not df.empty else [])


def _insight(text: str) -> None:
    st.markdown(f'<div class="pm-card"><strong>💡 PM INSIGHT</strong><br/>{text}</div>', unsafe_allow_html=True)


def _quotes(cards: list[dict]) -> None:
    if not cards:
        st.caption("No stored review quotes for this selection.")
        return
    for card in cards:
        st.markdown(
            f'<div class="quote">“{card["quote"]}”<br/>'
            f'<small>{card["source"]} · rating {card.get("rating")} · '
            f'{card.get("date") or "—"} · {card.get("region") or "—"} · id {card.get("source_review_id")}</small></div>',
            unsafe_allow_html=True,
        )


def _example_quote(review_ids: list[int] | None) -> str:
    ids = [int(i) for i in (review_ids or [])]
    if not ids:
        return ""
    db = _db()
    try:
        cards = evidence_cards(db, ids, limit=1)
    finally:
        db.close()
    if not cards:
        return ""
    return str(cards[0].get("quote") or "")


def _view_evidence(review_ids: list[int] | None, *, title: str = "View supporting reviews") -> None:
    ids = [int(i) for i in (review_ids or [])]
    if not ids:
        return
    with st.expander(f"{title} ({len(ids)} ids)"):
        st.caption("Review IDs: " + ", ".join(str(i) for i in ids[:30]))
        db = _db()
        try:
            _quotes(evidence_cards(db, ids, limit=8))
        finally:
            db.close()


def _seconds_since_last_collection() -> float | None:
    db = _db()
    try:
        last = (
            db.query(CollectionRun)
            .filter(CollectionRun.status.in_(["completed", "completed_with_errors"]))
            .order_by(CollectionRun.id.desc())
            .first()
        )
        if not last or not last.finished_at:
            return None
        stamp = ensure_aware(last.finished_at)
        if stamp is None:
            return None
        return (utcnow() - stamp).total_seconds()
    finally:
        db.close()


def _try_auto_collect() -> None:
    """Collect latest public reviews; analyze only newly pending reviews if credentials exist."""
    last_attempt = float(st.session_state.get("_auto_collect_attempt") or 0)
    if last_attempt and (time.time() - last_attempt) < AUTO_REFRESH_SECONDS:
        return
    elapsed = _seconds_since_last_collection()
    if elapsed is not None and elapsed < AUTO_REFRESH_SECONDS:
        return
    st.session_state["_auto_collect_attempt"] = time.time()
    _run_collect(
        ["google_play", "apple_app_store"],
        analyze=bool(get_settings().has_ai_credentials),
        mode="latest",
    )


def _auto_refresh_fragment() -> None:
    st.caption(
        f"{PERIODIC_REFRESH}. Automatic source check every 5 minutes. "
        "This is not real-time streaming."
    )
    now = time.time()
    if "_page_started_at" not in st.session_state:
        st.session_state["_page_started_at"] = now
        return
    if now - float(st.session_state["_page_started_at"]) < AUTO_REFRESH_SECONDS:
        return
    _try_auto_collect()


if hasattr(st, "fragment"):
    _auto_sync_ui = st.fragment(run_every="5m")(_auto_refresh_fragment)
else:
    _auto_sync_ui = _auto_refresh_fragment


def _source_key(source: str) -> str | None:
    return {"Google Play": "google_play", "Apple App Store": "apple_app_store"}.get(source)


def _cache_token() -> str:
    db = _db()
    try:
        diag = get_database_diagnostics(db)
        last = db.query(CollectionRun.id).order_by(CollectionRun.id.desc()).first()
        return (
            f"{diag.get('analyzed_reviews')}:{diag.get('pending_reviews')}:"
            f"{diag.get('failed_reviews')}:{getattr(last, 'id', 0)}"
        )
    finally:
        db.close()


def _json_ids(raw: str) -> list[int]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[int] = []
    for item in data:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


@st.cache_data(ttl=45, show_spinner=False)
def _load_bundle(since_iso: str | None, source: str, cache_token: str) -> dict:
    since = datetime.fromisoformat(since_iso) if since_iso else None
    src = _source_key(source)
    db = _db()
    try:
        metrics = overview_metrics(db, since=since, myntra_only=True, source=src)
        diag = get_database_diagnostics(db)
        signals = signal_counts(db, myntra_only=True, since=since, source=src)
        problems = merge_category_rows(
            problem_rows(db, myntra_only=True, since=since, source=src),
            label_keys=("problem", "label"),
            count_keys=("frequency", "count", "supporting_reviews"),
        )
        intents = merge_category_rows(
            label_distribution(db, "intent", myntra_only=True, relevant_only=False, since=since, source=src)
        )
        barriers = merge_category_rows(
            label_distribution(db, "barriers", myntra_only=True, relevant_only=False, since=since, source=src)
        )
        uncertainties = merge_category_rows(
            label_distribution(
                db, "uncertainties", myntra_only=True, relevant_only=False, since=since, source=src
            )
        )
        wishlist_beh = taxonomy_counts(db, WISHLIST_BEHAVIOR_TERMS, since=since, source=src)
        barrier_tax = taxonomy_counts(db, BARRIER_TERMS, since=since, source=src)
        unc_tax = taxonomy_counts(db, UNCERTAINTY_TERMS, since=since, source=src)
        compare = taxonomy_counts(db, COMPARISON_TERMS, since=since, source=src)
        compare_how = taxonomy_counts(db, COMPARISON_METHOD_TERMS, since=since, source=src)
        external = taxonomy_counts(db, EXTERNAL_TERMS, since=since, source=src)
        social = taxonomy_counts(db, SOCIAL_TERMS, since=since, source=src)
        purchase_beh = taxonomy_counts(db, PURCHASE_BEHAVIOR_TERMS, since=since, source=src)
        purchase_signals = purchase_signal_counts(db, since=since, source=src)
        hesitation = hesitation_split(db, since=since, source=src)
        root_causes = merge_category_rows(
            root_cause_hierarchy(db, since=since, source=src),
            label_keys=("root_cause", "problem", "label"),
            count_keys=("count", "frequency"),
        )
        wishlist_intent = wishlist_intent_split(db, since=since, source=src)
        segment_tax = taxonomy_counts(db, SEGMENT_TERMS, since=since, source=src)
        ratings = rating_distribution(db, since=since, source=src)
        latest = latest_review_cards(db, since=since, source=src, limit=8)
        daily = daily_review_trends(db, myntra_only=True, since=since)
        themes = merge_category_rows(
            [
                {
                    "name": t.name,
                    "review_count": t.review_count,
                    "evidence_ids": _json_ids(t.evidence_ids_json),
                }
                for t in db.query(Theme).order_by(Theme.review_count.desc()).all()
            ],
            label_keys=("name", "label"),
            count_keys=("review_count", "count"),
            id_keys=("evidence_ids", "review_ids"),
        )
        segments = merge_category_rows(
            [
                {
                    "name": s.name,
                    "review_count": s.review_count,
                    "basis": s.basis,
                    "evidence_ids": _json_ids(s.evidence_ids_json),
                }
                for s in db.query(Segment).order_by(Segment.review_count.desc()).all()
            ],
            label_keys=("name", "label"),
            count_keys=("review_count", "count"),
            id_keys=("evidence_ids", "review_ids"),
        )
        opps = merge_category_rows(
            [
                {
                    "rank": o.rank,
                    "name": o.name,
                    "reach": o.reach,
                    "frequency": o.frequency,
                    "purchase_impact": o.purchase_impact,
                    "severity": o.severity,
                    "evidence_confidence": o.evidence_confidence,
                    "score": o.score,
                    "relevant_count": o.relevant_count,
                    "percentage": o.percentage,
                    "why_investigate": o.why_investigate,
                    "evidence_ids": _json_ids(o.evidence_ids_json),
                }
                for o in db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
            ],
            label_keys=("name", "label"),
            count_keys=("relevant_count", "count"),
            id_keys=("evidence_ids", "review_ids"),
        )
        theme_momentum = label_window_momentum(
            db, "themes", since=since or get_last_30_days_cutoff(), source=src
        )
        ages = explicit_age_mentions(db, since=since, source=src)
        live = source_live_status(db)
        last_ok = live.get("last_successful_run")
        checked = last_ok.finished_at if last_ok else None
        report = build_report(db) if (diag.get("analyzed_reviews") or 0) else {}
        return {
            "metrics": metrics,
            "diag": diag,
            "signals": signals,
            "problems": problems,
            "intents": intents,
            "barriers": barriers,
            "uncertainties": uncertainties,
            "wishlist_beh": wishlist_beh,
            "barrier_tax": barrier_tax,
            "unc_tax": unc_tax,
            "compare": compare,
            "compare_how": compare_how,
            "external": external,
            "social": social,
            "purchase_beh": purchase_beh,
            "purchase_signals": purchase_signals,
            "hesitation": hesitation,
            "root_causes": root_causes,
            "wishlist_intent": wishlist_intent,
            "segment_tax": segment_tax,
            "ratings": ratings,
            "latest": latest,
            "daily": daily,
            "themes": themes,
            "segments": segments,
            "opps": opps,
            "theme_momentum": theme_momentum,
            "ages": ages,
            "freshness": {
                "last_checked": checked.isoformat() if checked else None,
            },
            "report": report,
            "source_filter": src,
            "cache_token": cache_token,
        }
    finally:
        db.close()


def render() -> None:
    st.set_page_config(page_title="Myntra AI Discovery Engine", layout="wide")
    _bootstrap()
    migrate_schema()
    reload_settings()
    _inject_css()
    settings = get_settings()
    cfg = get_ai_config()
    ai_ok = settings.has_ai_credentials
    source = "All"

    with st.sidebar:
        st.markdown("**Myntra AI Discovery Engine**")
        period = st.selectbox("Period", ["Last 30 Days", "All Time"], index=0)
        st.session_state["period"] = period
        if st.button("🔄 Refresh Latest Reviews"):
            _run_collect(
                ["google_play", "apple_app_store"],
                analyze=bool(get_settings().has_ai_credentials),
                mode="latest",
            )
        st.session_state["analyze_on_collect"] = False
        diag_side = get_database_diagnostics()
        freshness_side = _seconds_since_last_collection()
        last_checked = "never"
        if freshness_side is not None:
            last_checked = humanize_ago(utcnow() - timedelta(seconds=freshness_side))
        st.caption(f"Last checked: {last_checked}")
        st.caption(f"Stored reviews: {diag_side.get('total_reviews') or 0} / {diag_side.get('max_total_reviews') or _dataset_limit()}")
        st.caption(f"AI analyzed: {diag_side.get('analyzed_reviews') or 0} / {diag_side.get('max_analysis_reviews') or _analysis_limit()}")
        st.caption(PERIODIC_REFRESH)
        with st.expander("More"):
            if st.button("Analyze pending"):
                _run_analyze()
            if st.button("Retry failed analysis"):
                _run_analyze(only_failed=True)
            if st.button("Enforce 500 Review Limit"):
                from app.pipeline.dataset import enforce_review_limit

                prune_db = _db()
                try:
                    result = enforce_review_limit(prune_db, prune=True)
                    st.session_state["prune_result"] = result
                    _load_bundle.clear()
                finally:
                    prune_db.close()
                st.rerun()
            prune_result = st.session_state.get("prune_result")
            if prune_result:
                st.caption(
                    f"Last prune: kept {prune_result.get('kept')} / {prune_result.get('max_reviews')}, "
                    f"deleted {prune_result.get('deleted')}."
                )
            st.caption(
                f"Storage cap: {_dataset_limit()} combined. "
                f"AI sample: {_analysis_limit()}. "
                "API key is never displayed."
            )

    since = _period_since()
    cache_token = f"{_cache_token()}:{source}:{period}"
    data = _load_bundle(since.isoformat() if since else None, source, cache_token)
    diag = data["diag"]
    metrics = data["metrics"]
    analyzed = int(diag.get("analyzed_reviews") or 0)
    stored = int(diag.get("total_reviews") or 0)
    blocker = _analysis_blocker(
        stored=stored,
        analyzed=analyzed,
        pending=int(diag.get("pending_reviews") or 0),
        failed=int(diag.get("failed_reviews") or 0),
        ai_ok=ai_ok,
        last_error=str(diag.get("last_analysis_error") or ""),
    )

    _header(cfg, data["freshness"], period, diag)
    _actions(ai_ok)
    _pipeline_status_panel()
    _auto_sync_ui()

    if stored == 0:
        st.info(EMPTY)
        _limitations()
        return
    if blocker and analyzed == 0:
        st.warning(blocker)

    _overview(diag, metrics, data, period)
    _live_status(data, diag)
    _latest_reviews(data)
    _assistant(analyzed)
    _wishlist_intelligence(data, analyzed)
    _problems(data, analyzed)
    _wishlist(data, analyzed)
    _barriers(data, analyzed)
    _uncertainties(data, analyzed)
    _themes(data, analyzed)
    _segments(data, analyzed)
    _external(data, analyzed)
    _opportunities(data, analyzed)
    _pm_insight_section(data, analyzed)
    _root_causes(data, analyzed)
    _discovery_report(data, analyzed)
    _evidence_explorer("All", "All", since, data)
    _collection_history()
    _limitations()


def _header(cfg: dict, freshness: dict, period: str, diag: dict | None = None) -> None:
    raw = freshness.get("last_checked")
    checked = datetime.fromisoformat(raw) if raw else None
    st.markdown(
        '<div class="hero"><h1>MYNTRA AI DISCOVERY ENGINE</h1>'
        "<p>Wishlist → Purchase Growth Intelligence</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.write("**Data period**")
    c1.write(period)
    c2.write("**Data sources**")
    c2.write("Google Play + Apple App Store")
    c3.write("**AI provider**")
    c3.write("OpenRouter")
    c4.write("**Model**")
    c4.write(cfg.get("model") or "")
    c5.write("**Data status**")
    c5.write(PERIODIC_REFRESH)
    diag = diag or {}
    stored = int(diag.get("total_reviews") or 0)
    analyzed = int(diag.get("analyzed_reviews") or 0)
    pending = int(diag.get("pending_reviews") or 0)
    failed = int(diag.get("failed_reviews") or 0)
    storage_cap = int(diag.get("max_total_reviews") or _dataset_limit())
    sample_cap = int(diag.get("max_analysis_reviews") or _analysis_limit())
    st.caption(
        f"Stored reviews: {stored} / {storage_cap} · "
        f"Analyzed reviews: {analyzed} / {sample_cap} · "
        f"Pending reviews: {pending} · "
        f"Failed reviews: {failed}"
    )
    st.caption(
        f"Last checked: {humanize_ago(checked)} ({checked.isoformat() if checked else 'never'}). "
        "Never a live stream of in-app events. Periodic public-source refresh only."
    )


def _actions(ai_ok: bool) -> None:
    a, b, c = st.columns([1, 1, 2])
    with a:
        if st.button("🚀 Run full discovery", type="primary"):
            _run_full_discovery(ai_ok)
    with b:
        if st.button("🔄 Refresh latest reviews"):
            _run_collect(
                ["google_play", "apple_app_store"],
                analyze=bool(get_settings().has_ai_credentials),
                mode="latest",
            )
    with c:
        from app.ai.provider import test_openrouter_connection

        if st.button("Test OpenRouter connection"):
            st.session_state["ai_connection_test"] = test_openrouter_connection()
        probe = st.session_state.get("ai_connection_test")
        if probe:
            model = probe.get("model") or ""
            max_tokens = probe.get("max_tokens") or ""
            http_status = probe.get("http_status")
            if probe.get("ok"):
                st.success(
                    f"OpenRouter connection: SUCCESS  \n"
                    f"Model: {model}  \n"
                    f"max_tokens: {max_tokens}"
                )
            else:
                reason = probe.get("error") or "OpenRouter connection test failed."
                http_line = f"HTTP {http_status}" if http_status else "HTTP n/a"
                st.error(
                    f"OpenRouter connection: FAILED  \n"
                    f"{http_line}  \n"
                    f"Reason: {reason}"
                )


def _kpis(diag: dict, metrics: dict, opps: list, themes: list, period: str) -> None:
    top_score = opps[0]["score"] if opps else 0
    storage_cap = int(diag.get("max_total_reviews") or diag.get("max_dataset_reviews") or _dataset_limit())
    sample_cap = int(diag.get("max_analysis_reviews") or _analysis_limit())
    stored = int(diag.get("total_reviews") or 0)
    google = int(diag.get("google_play_reviews") or 0)
    apple = int(diag.get("apple_reviews") or 0)
    analyzed = int(diag.get("analyzed_reviews") or 0)
    pending = int(diag.get("pending_reviews") or 0)
    failed = int(diag.get("failed_reviews") or 0)
    selected = int(diag.get("selected_reviews") or 0)
    st.markdown("**REVIEW DATASET**")
    cols = st.columns(5)
    cols[0].metric("Total reviews", f"{stored} / {storage_cap}")
    cols[1].metric("Google Play", google)
    cols[2].metric("Apple App Store", apple)
    cols[3].metric("AI analyzed", analyzed)
    cols[4].metric("Pending", pending)
    cols2 = st.columns(5)
    cols2[0].metric("Failed", failed)
    cols2[1].metric("AI analysis sample", f"{selected} / {sample_cap}")
    cols2[2].metric("Last 30 days", diag.get("last_30_day_reviews") or 0)
    cols2[3].metric("Wishlist-related", metrics.get("wishlist_signals") or 0)
    cols2[4].metric("Unique themes / top score", f"{len(themes)} / {top_score}")
    if diag.get("dataset_limit_reached"):
        st.info(f"Dataset limit reached: {stored} stored reviews (maximum {storage_cap}).")
    st.caption(
        f"Storage holds {stored} real public reviews (maximum {storage_cap}). "
        f"AI analyzes a sample of {selected} (maximum {sample_cap}), not the full stored set. "
        f"Period filter: {period}. This sample is not the entire Myntra customer base. "
        "Public reviews do not expose Myntra's actual wishlist-to-purchase conversion events. "
        "Conversion insights are evidence-based opportunity indicators, not Myntra's actual conversion rate. "
        "Counts come from the database, not the LLM."
    )


def _overview(diag: dict, metrics: dict, data: dict, period: str) -> None:
    st.markdown('<div class="section-h">DATASET OVERVIEW</div>', unsafe_allow_html=True)
    _kpis(diag, metrics, data["opps"], data["themes"], period)
    left, right = st.columns(2)
    with left:
        st.caption("Review rating distribution")
        donut_chart(data.get("ratings") or [])
    with right:
        st.caption("Last 30 Days vs All Time is selected in the sidebar.")
        st.metric("Active period", period)
        st.caption("Default business analysis is Last 30 Days. Switch to All Time to explore history.")


def _live_status(data: dict, diag: dict) -> None:
    st.markdown('<div class="section-h">LIVE DATA STATUS</div>', unsafe_allow_html=True)
    raw = (data.get("freshness") or {}).get("last_checked")
    checked = datetime.fromisoformat(raw) if raw else None
    a, b, c, d = st.columns(4)
    a.metric("Status", PERIODIC_REFRESH)
    b.metric("Last checked", humanize_ago(checked))
    c.metric("Pending analysis", diag.get("pending_reviews") or 0)
    d.metric("Failed analysis", diag.get("failed_reviews") or 0)
    stored = int(diag.get("total_reviews") or 0)
    storage_cap = int(diag.get("max_total_reviews") or _dataset_limit())
    sample_cap = int(diag.get("max_analysis_reviews") or _analysis_limit())
    selected = int(diag.get("selected_reviews") or 0)
    analyzed = int(diag.get("analyzed_reviews") or 0)
    pending = int(diag.get("pending_reviews") or 0)
    failed = int(diag.get("failed_reviews") or 0)
    sample_analyzed = int(diag.get("sample_analyzed") or 0)
    st.markdown("**DATASET LIMIT**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Maximum total reviews", storage_cap)
    d2.metric("Current stored reviews", stored)
    d3.metric("Google Play", diag.get("google_play_reviews") or 0)
    d4.metric("Apple App Store", diag.get("apple_reviews") or 0)
    d5, d6, d7, d8 = st.columns(4)
    d5.metric("Available analysis sample", f"{selected} / {sample_cap}")
    d6.metric("AI analyzed", analyzed)
    d7.metric("Pending", pending)
    d8.metric("Failed", failed)
    if diag.get("dataset_limit_reached"):
        st.warning(f"Storage limit reached ({stored} / {storage_cap}). New collection will prune the oldest excess reviews.")
    batch_size = int(diag.get("analysis_batch_size") or get_ai_config().get("batch_size") or 10)
    batch_total = int(diag.get("analysis_batch_total") or ((selected + batch_size - 1) // batch_size if selected else 0))
    st.markdown("**AI ANALYSIS PROGRESS**")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Sample selected", f"{selected} / {sample_cap}")
    p2.metric("Batch size", batch_size)
    p3.metric("Total batches", batch_total)
    p4.metric("Sample analyzed", f"{sample_analyzed} / {selected or sample_cap}")
    last = st.session_state.get("analysis_progress") or {}
    completed_batches = int(last.get("batch_index") or 0)
    failed_batches = int((st.session_state.get("last_analysis") or {}).get("failed_batches") or 0)
    b1, b2, b3 = st.columns(3)
    b1.metric("Completed batches", completed_batches)
    b2.metric("Failed batches", failed_batches)
    b3.metric("Failed reviews", failed)
    percent = int(round(100 * sample_analyzed / selected)) if selected else 0
    st.progress(min(100, max(0, percent)) / 100, text=f"Progress: {percent}%")
    if last.get("batch_index") and last.get("batch_total"):
        st.caption(f"Current batch: {last.get('batch_index')} / {last.get('batch_total')}")
    st.caption(
        "AI analyzes the selected sample in batches of 10. "
        "Pending stored reviews outside the sample are not treated as analyzed. "
        "Progress updates after each completed batch."
    )
    if diag.get("last_analysis_error"):
        st.error(diag.get("last_analysis_error"))
    daily = data.get("daily") or []
    if daily:
        st.caption("Reviews by day (review timestamps, not collection time)")
        st.bar_chart(pd.DataFrame(daily), x="day", y="reviews", height=220)
    else:
        st.info("Insufficient evidence for this visualization.")
    st.caption("Never a live stream of in-app events. Sources are checked at most every 5 minutes.")


def _latest_reviews(data: dict) -> None:
    st.markdown('<div class="section-h">LATEST REVIEWS</div>', unsafe_allow_html=True)
    rows = data.get("latest") or []
    if not rows:
        st.info("No real reviews have been collected yet.")
        return
    for row in rows:
        with st.expander(f"{row.get('source')} · {row.get('rating')}★ · {row.get('status')} · {row.get('date')}"):
            st.write(row.get("text") or "")
            st.caption(
                f"id {row.get('source_review_id')} · region {row.get('region') or '—'} · review_id {row.get('id')}"
            )


def _assistant(analyzed: int) -> None:
    st.markdown('<div class="section-h">1. AI PRODUCT MANAGER ASSISTANT</div>', unsafe_allow_html=True)
    st.caption("Answers use retrieved database evidence only.")
    pick = st.selectbox("Suggested questions", SUGGESTED_QUESTIONS)
    typed = st.text_input("Question", value="", placeholder="Ask using stored review evidence…")
    question = typed.strip() or pick
    if st.button("Ask", type="primary") and question:
        db = _db()
        try:
            result = ask_product_assistant(db, question, analyzed=analyzed)
        finally:
            db.close()
        st.markdown("**Answer**")
        st.write(result.get("answer") or "Insufficient evidence.")
        st.markdown("**Evidence summary**")
        st.write(result.get("evidence_summary") or "No matching stored evidence.")
        st.write("**Supporting review count:**", result.get("supporting_review_count") or 0)
        st.markdown("**Example evidence**")
        quotes = result.get("quotes") or []
        if quotes:
            for quote in quotes[:5]:
                st.markdown(f"“{quote.get('text') or ''}”")
                st.caption(
                    f"{quote.get('source')} · rating {quote.get('rating')} · "
                    f"{quote.get('date') or '—'} · id {quote.get('id')}"
                )
        else:
            st.caption("No stored quotes matched this question.")
        caveat = result.get("caveat") or ""
        if caveat or (result.get("supporting_review_count") or 0) < 5:
            st.markdown("**Important caveat**")
            st.write(
                caveat
                or "Evidence is weak in this sample. Do not generalize to the full Myntra user base."
            )


def _wishlist_intelligence(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">2. WISHLIST → PURCHASE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.caption(
        "Public reviews are proxy evidence for consideration, uncertainty, hesitation, "
        "comparison, and purchase language. They do not measure actual wishlist-to-purchase conversion."
    )
    signals = data.get("signals") or {}
    a, b, c, d = st.columns(4)
    a.metric("Wishlist language", f"{signals.get('wishlist_pct', 0)}%")
    b.metric("Purchase hesitation", f"{signals.get('hesitation_pct', 0)}%")
    c.metric("Analyzed reviews", analyzed)
    d.metric("High-impact barriers", sum(1 for row in (data.get("barriers") or []) if (row.get("hesitant_count") or 0) >= 2))
    st.caption(wishlist_conversion_copy(signals))
    hesitation = data.get("hesitation") or []
    compare = data.get("compare") or []
    left, mid, right = st.columns(3)
    with left:
        st.caption("Purchase hesitation")
        bar_chart(hesitation, empty="No explicit or implicit hesitation labels in this sample.")
    with mid:
        st.caption("Comparison")
        bar_chart(compare, empty="No comparison evidence in this sample.")
    with right:
        st.caption("Purchase signals")
        bar_chart(data.get("purchase_signals") or [], empty="No purchase-signal labels in this sample.")
    abandoned = next((r["count"] for r in (data.get("purchase_signals") or []) if r["label"] == "abandoned"), 0)
    intent = next(
        (r["count"] for r in (data.get("purchase_signals") or []) if r["label"] in {"intend_to_purchase", "purchased"}),
        0,
    )
    stages = funnel_stages(
        analyzed=signals.get("analyzed") or 0,
        wishlist=signals.get("wishlist_signal") or 0,
        intent=intent,
        hesitation=signals.get("purchase_hesitation") or 0,
        barriers=(data.get("barriers") or [{}])[0].get("count", 0) if data.get("barriers") else 0,
        uncertainties=(data.get("uncertainties") or [{}])[0].get("count", 0) if data.get("uncertainties") else 0,
        abandoned=abandoned,
        comparison=(compare[0].get("count", 0) if compare else 0),
    )
    present = [s for s in stages if s["count"]]
    if present:
        st.caption("Consideration path (counts from stored analysis, not conversion events)")
        bar_chart([{"label": s["stage"], "count": s["count"]} for s in present])


def _root_causes(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">ROOT CAUSE ANALYSIS</div>', unsafe_allow_html=True)
    st.caption(
        "User Behavior → User Problem → Uncertainty / Barrier → Purchase Hesitation → "
        "Wishlist → Purchase Impact. Derived from stored review evidence, not hardcoded. "
        "Public reviews do not directly measure wishlist-to-purchase conversion."
    )
    derived = derive_root_cause(
        analyzed=analyzed,
        problems=data.get("problems") or [],
        barriers=data.get("barriers") or data.get("barrier_tax") or [],
        uncertainties=data.get("uncertainties") or data.get("unc_tax") or [],
        wishlist=data.get("wishlist_beh") or data.get("intents") or [],
        hesitation_count=int((data.get("signals") or {}).get("purchase_hesitation") or 0),
    )
    st.markdown(f"**{derived['statement']}**")
    chain = derived.get("chain") or {}
    if derived.get("supported") and chain:
        st.markdown(
            f"**User Behavior:** {chain.get('user_behavior')}  \n"
            f"**User Problem:** {chain.get('problem')}  \n"
            f"**Uncertainty / Barrier:** {chain.get('uncertainty_or_barrier')}  \n"
            f"**Purchase Hesitation:** {chain.get('purchase_hesitation')} supporting reviews  \n"
            f"**Wishlist → Purchase Impact:** {chain.get('business_metric')}"
        )
        evidence_ids = []
        if data.get("problems"):
            evidence_ids = data["problems"][0].get("review_ids") or []
        elif data.get("barriers"):
            evidence_ids = data["barriers"][0].get("review_ids") or []
        if evidence_ids:
            _view_evidence(evidence_ids, title="View supporting reviews")
    rows = data.get("root_causes") or []
    if not rows:
        if not derived.get("supported"):
            st.info(INSUFFICIENT_ROOT)
        return
    left, right = st.columns(2)
    with left:
        bar_chart([{"label": r["root_cause"], "count": r["count"]} for r in rows[:8]])
    with right:
        scatter_chart(
            [{"frequency": r["count"], "impact": r["purchase_impact"], "count": r["count"]} for r in rows[:12]]
        )
    for i, row in enumerate(rows[:6], 1):
        st.markdown(
            f"**{i}. {row['root_cause']}**  \n"
            f"Evidence {row['count']} · {row['percentage']}% · "
            f"Severity {row['severity']}/5 · Purchase impact {row['purchase_impact']}/5"
        )
        st.caption(why_this_matters(row, analyzed=analyzed))
        _view_evidence(row.get("review_ids"), title="View supporting reviews")


def _problems(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">3. USER PROBLEMS</div>', unsafe_allow_html=True)
    rows = data["problems"]
    if not rows:
        st.info("Insufficient evidence for this visualization." if analyzed else "No analyzed reviews yet.")
        return
    if len(rows) < 4:
        st.caption(f"Only {len(rows)} problems had sufficient evidence in the analyzed sample.")
    shown = rows[:8]
    table = []
    for row in shown:
        table.append(
            {
                "Problem": row.get("problem"),
                "Frequency": row.get("frequency"),
                "Share": f"{row.get('percentage')}%",
                "Severity": row.get("severity"),
                "Purchase impact": row.get("purchase_impact"),
                "Evidence confidence": row.get("confidence"),
                "Example review evidence": _example_quote(row.get("review_ids")),
            }
        )
    st.dataframe(table, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        bar_chart([{"label": r["problem"], "count": r["frequency"]} for r in shown])
    with right:
        scatter_chart(
            [{"frequency": r["frequency"], "impact": r["purchase_impact"], "count": r["frequency"]} for r in shown]
        )
    if shown:
        _view_evidence(shown[0].get("review_ids"), title="View supporting reviews")


def _wishlist(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">4. WHY USERS WISHLIST</div>', unsafe_allow_html=True)
    st.caption("Only evidence-backed wishlist behavior is shown. A generic price comment is not treated as wishlist behavior.")
    rows = data["wishlist_beh"] or data["intents"]
    split = data.get("wishlist_intent") or {}
    split_rows = split.get("rows") or []
    if split.get("limited") or not split_rows:
        st.info(
            "Public reviews contain limited direct evidence about wishlist "
            "behavior. This cannot be used to calculate actual wishlist conversion."
        )
    if not rows and not split_rows:
        st.info("Insufficient evidence for this visualization.")
        return
    left, right = st.columns(2)
    with left:
        bar_chart(rows[:8] if rows else split_rows)
    with right:
        st.caption("Purchase intent vs bookmarking vs unclear")
        donut_chart(split_rows)
    if rows:
        st.dataframe(
            [{"behavior": r["label"], "frequency": r["count"], "percentage": r["percentage"]} for r in rows[:10]],
            hide_index=True,
            width="stretch",
        )
        _view_evidence(rows[0].get("review_ids"), title="View supporting reviews")


def _barriers(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">5. PURCHASE BARRIERS</div>', unsafe_allow_html=True)
    rows = [r for r in (data["barriers"] or data["barrier_tax"] or []) if int(r.get("count") or 0) > 0]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        return
    from app.pipeline.scoring import purchase_impact_from_hesitation

    for row in rows:
        if row.get("purchase_impact") is None:
            row["purchase_impact"] = purchase_impact_from_hesitation(
                int(row.get("hesitant_count") or 0), int(row.get("count") or 0)
            )
    shown = rows[:10]
    table = []
    for row in shown:
        denom = int(row.get("denominator") or analyzed or 0)
        share = row.get("percentage")
        if share is None and denom:
            share = round(100.0 * int(row.get("count") or 0) / denom, 1)
        table.append(
            {
                "Barrier": row.get("label"),
                "Frequency": row.get("count"),
                "% of analyzed reviews": share,
                "Example evidence": _example_quote(row.get("review_ids")),
            }
        )
    st.dataframe(table, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        bar_chart(shown)
    with right:
        heatmap_impact_frequency(shown)
    _view_evidence(shown[0].get("review_ids"), title="View supporting reviews")


def _uncertainties(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">6. PURCHASE UNCERTAINTIES</div>', unsafe_allow_html=True)
    rows = data["uncertainties"] or data["unc_tax"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        return
    left, right = st.columns(2)
    with left:
        bar_chart(rows[:10])
    with right:
        donut_chart(rows[:10])
    st.dataframe(
        [
            {
                "uncertainty": r["label"],
                "frequency": r["count"],
                "percentage": r["percentage"],
            }
            for r in rows[:12]
        ],
        hide_index=True,
        width="stretch",
    )
    _view_evidence(rows[0].get("review_ids"), title="View supporting reviews")


def _themes(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">7. KEY THEMES</div>', unsafe_allow_html=True)
    themes = data["themes"]
    momentum = {m["label"]: m for m in data["theme_momentum"]}
    if not themes:
        st.info("Insufficient evidence for this visualization.")
        return
    cards = st.columns(min(4, len(themes)))
    for i, theme in enumerate(themes[:4]):
        trend = (momentum.get(theme["name"]) or {}).get("momentum") or "insufficient data"
        cards[i].metric(theme["name"][:40], f"{theme['review_count']} reviews")
        cards[i].caption(f"Trend: {trend} (descriptive split, not a significance test)")
    left, right = st.columns(2)
    with left:
        bar_chart([{"label": t["name"], "count": t["review_count"]} for t in themes[:10]])
    with right:
        donut_chart([{"label": t["name"], "count": t["review_count"]} for t in themes[:10]])
    if data["theme_momentum"]:
        st.dataframe(
            [
                {
                    "theme": m["label"],
                    "count": m["count"],
                    "trend": m["momentum"],
                }
                for m in data["theme_momentum"][:10]
            ],
            hide_index=True,
            width="stretch",
        )
        daily_rows = []
        for m in data["theme_momentum"][:5]:
            for day, count in (m.get("by_day") or {}).items():
                daily_rows.append({"day": day, "theme": m["label"], "count": count})
        if daily_rows:
            st.caption("Theme mentions by review date")
            trend_df = trend_frame(daily_rows, label_key="theme")
            if not trend_df.empty:
                st.bar_chart(trend_df, x="day", y="count", color="theme", height=240)
    if themes:
        _view_evidence(themes[0].get("evidence_ids"), title="View supporting reviews")


def _segments(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">8. USER SEGMENTS</div>', unsafe_allow_html=True)
    st.caption("Evidence-based behavioral segments")
    st.info(DEMOGRAPHIC_NOTICE)
    segs = data["segments"] or [
        {"name": r["label"], "review_count": r["count"], "basis": "Evidence-based inferred segment", "evidence_ids": r.get("review_ids") or []}
        for r in (data.get("segment_tax") or [])
    ]
    if not segs:
        st.info("Insufficient evidence for reliable segmentation.")
        return
    relevant = max(analyzed, 1)
    left, right = st.columns(2)
    with left:
        donut_chart([{"label": s["name"], "count": s["review_count"]} for s in segs[:8]])
    with right:
        bar_chart([{"label": s["name"], "count": s["review_count"]} for s in segs[:8]])
    table = []
    for s in segs:
        table.append(
            {
                "segment": s["name"],
                "evidence": s["review_count"],
                "pct_analyzed": round(100 * s["review_count"] / relevant, 2) if analyzed else 0,
                "basis": s.get("basis") or "Textual evidence only",
            }
        )
    st.dataframe(table, hide_index=True, width="stretch")
    _view_evidence(segs[0].get("evidence_ids"), title="View supporting reviews")
    if data["ages"]:
        st.caption(f"Explicit age evidence: {len(data['ages'])} reviews mention an age number. Survey-derived demographic data is not connected.")
    else:
        st.caption(DEMOGRAPHIC_NOTICE)


def _purchase_behavior(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">7. PURCHASING & DECISION BEHAVIOR</div>', unsafe_allow_html=True)
    rows = data["purchase_beh"] or data["purchase_signals"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="purchase behavior", rows=[], analyzed=analyzed))
        return
    _bar(pd.DataFrame([{"label": r["label"][:40], "count": r["count"]} for r in rows[:10]]))
    st.dataframe(
        [{"behavior": r["label"], "frequency": r["count"], "percentage": r["percentage"]} for r in rows],
        hide_index=True,
        width="stretch",
    )
    _insight(pm_insight(topic="purchase / decision behavior", rows=rows, analyzed=analyzed))


def _comparison(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">11. COMPARISON BEHAVIOR</div>', unsafe_allow_html=True)
    rows = data["compare"]
    methods = data["compare_how"]
    if not rows and not methods:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="comparison behavior", rows=[], analyzed=analyzed))
        return
    left, right = st.columns(2)
    with left:
        bar_chart(rows[:8] if rows else None)
    with right:
        if methods:
            st.caption("Comparison methods found in review text")
            bar_chart(methods)
        else:
            st.info("No explicit comparison-method mentions in the stored reviews.")
    if rows:
        st.dataframe(
            [{"factor": r["label"], "frequency": r["count"], "percentage": r["percentage"]} for r in rows],
            hide_index=True,
            width="stretch",
        )
        _view_evidence(rows[0].get("review_ids"), title="View supporting reviews")
    _insight(pm_insight(topic="comparison factor", rows=rows or methods, analyzed=analyzed))
    social = data.get("social") or []
    if social:
        st.caption("Trust & social validation (evidence-supported only)")
        donut_chart(social)


def _external(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">9. EXTERNAL INFORMATION</div>', unsafe_allow_html=True)
    rows = data["external"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        return
    left, right = st.columns(2)
    with left:
        bar_chart(rows)
    with right:
        donut_chart(rows)
    st.dataframe(
        [{"source": r["label"], "frequency": r["count"], "percentage": r["percentage"]} for r in rows],
        hide_index=True,
        width="stretch",
    )
    _view_evidence(rows[0].get("review_ids"), title="View supporting reviews")


def _social(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">10. TRUST & SOCIAL VALIDATION</div>', unsafe_allow_html=True)
    rows = data["social"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="trust / social validation", rows=[], analyzed=analyzed))
        return
    _bar(pd.DataFrame([{"label": r["label"], "count": r["count"]} for r in rows]))
    _insight(pm_insight(topic="trust / social validation", rows=rows, analyzed=analyzed))


def _opportunities(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">10. TOP OPPORTUNITIES</div>', unsafe_allow_html=True)
    st.caption("Opportunity Score = Reach × Frequency × Purchase Impact × Severity × Evidence Confidence. Calculated in Python, not by the LLM.")
    opps = data["opps"]
    if not opps:
        st.info("Insufficient evidence for this visualization.")
        return
    st.dataframe(
        [
            {
                "Opportunity": o["name"],
                "Reach": o["reach"],
                "Frequency": o["frequency"],
                "Purchase Impact": o["purchase_impact"],
                "Severity": o["severity"],
                "Evidence Confidence": o["evidence_confidence"],
                "Opportunity Score": o["score"],
                "Supporting evidence": o["relevant_count"],
            }
            for o in opps[:15]
        ],
        hide_index=True,
        width="stretch",
    )
    scatter_chart(
        [
            {"frequency": o["frequency"], "impact": o["purchase_impact"], "count": o["relevant_count"]}
            for o in opps[:20]
        ]
    )
    bar_chart([{"label": o["name"], "count": o["score"]} for o in opps[:10]], empty="No scored opportunities.")
    _view_evidence(opps[0].get("evidence_ids"), title="View supporting reviews")


def _pm_insight_section(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">11. PM INSIGHT</div>', unsafe_allow_html=True)
    st.caption("What should a Product Manager learn from this evidence?")
    example = ""
    problems = data.get("problems") or []
    opps = data.get("opps") or []
    if problems:
        example = _example_quote(problems[0].get("review_ids"))
    elif opps:
        example = _example_quote(opps[0].get("evidence_ids") or opps[0].get("review_ids"))
    card = pm_insight_card(
        analyzed=analyzed,
        problems=problems,
        opportunities=opps,
        evidence_count=(problems[0].get("frequency") if problems else 0),
        example=example,
        confidence=(problems[0].get("confidence") if problems else None),
    )
    st.markdown(
        f'<div class="pm-insight-card">'
        f"<strong>Strongest signal:</strong><br/>{card['strongest_signal']}<br/><br/>"
        f"<strong>Why it matters:</strong><br/>{card['why_it_matters']}<br/><br/>"
        f"<strong>Evidence:</strong><br/>{card['evidence']}<br/><br/>"
        f"<strong>Confidence:</strong><br/>{card['confidence']}<br/><br/>"
        f"<strong>Caveat:</strong><br/>{card['caveat']}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _decomposition(data: dict) -> None:
    st.markdown('<div class="section-h">14. WISHLIST → PURCHASE METRIC DECOMPOSITION</div>', unsafe_allow_html=True)
    st.caption(
        "Public reviews cannot measure actual wishlist-to-purchase conversion. "
        "First-party event data is required for the actual business KPI."
    )
    signals = data["signals"]
    abandoned = next((r["count"] for r in data["purchase_signals"] if r["label"] == "abandoned"), 0)
    intent = next(
        (r["count"] for r in data["purchase_signals"] if r["label"] in {"intend_to_purchase", "purchased"}),
        0,
    )
    stages = funnel_stages(
        analyzed=signals.get("analyzed") or 0,
        wishlist=signals.get("wishlist_signal") or 0,
        intent=intent,
        hesitation=signals.get("purchase_hesitation") or 0,
        barriers=data["barriers"][0]["count"] if data["barriers"] else 0,
        uncertainties=data["uncertainties"][0]["count"] if data["uncertainties"] else 0,
        abandoned=abandoned,
        comparison=(data.get("compare") or [{}])[0].get("count", 0) if data.get("compare") else 0,
    )
    funnel = [
        {
            "stage": "Wishlist Intent",
            "behavior": "Save / wishlist language in reviews",
            "friction": "Unclear whether bookmarking or purchase intent",
            "evidence": signals.get("wishlist_signal") or 0,
            "first_party_needed": "wishlist_add events",
        },
        {
            "stage": "Product Revisit",
            "behavior": "Delayed / later language",
            "friction": "No session revisit telemetry in public reviews",
            "evidence": next((r["count"] for r in (data.get("purchase_beh") or []) if "Delayed" in r["label"]), 0),
            "first_party_needed": "product_view after wishlist",
        },
        {
            "stage": "Information Confidence",
            "behavior": "Uncertainties extracted from analysis",
            "friction": "Fit, size, quality, returns questions",
            "evidence": data["uncertainties"][0]["count"] if data["uncertainties"] else 0,
            "first_party_needed": "PDP help / size-guide interactions",
        },
        {
            "stage": "Purchase Consideration",
            "behavior": "Named barriers + comparison",
            "friction": "Price, fit, quality, alternatives",
            "evidence": (data["barriers"][0]["count"] if data["barriers"] else 0),
            "first_party_needed": "add_to_bag / compare events",
        },
        {
            "stage": "Decision",
            "behavior": "Purchase hesitation labels",
            "friction": "Postponement / abandoned intent",
            "evidence": signals.get("purchase_hesitation") or 0,
            "first_party_needed": "checkout start",
        },
        {
            "stage": "Purchase",
            "behavior": "purchase_signal purchased / abandoned",
            "friction": "Public reviews are not orders",
            "evidence": next((r["count"] for r in data["purchase_signals"] if r["label"] == "purchased"), 0),
            "first_party_needed": "order_completed + wishlist attribution",
        },
    ]
    present = [s for s in stages if s["count"]]
    if present:
        bar_chart([{"label": s["stage"], "count": s["count"]} for s in present])
    else:
        st.info("Insufficient evidence for this visualization.")
    st.dataframe(funnel, hide_index=True, width="stretch")
    st.dataframe(stages, hide_index=True, width="stretch")


def _named_list(items: list, key: str = "label") -> list[str]:
    names = []
    for item in items or []:
        name = normalize_label(item.get(key) or item.get("problem") or item.get("name"))
        if name:
            names.append(name)
    return names


def _discovery_report(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">12. DISCOVERY REPORT</div>', unsafe_allow_html=True)
    st.caption("Executive report for a Product Manager. Readable in under 3 minutes. All findings come from stored analysis.")
    if analyzed <= 0:
        st.info("Discovery report requires stored analysis records.")
        return
    problems = data.get("problems") or []
    wishlist = data.get("wishlist_beh") or data.get("intents") or []
    barriers = data.get("barriers") or []
    uncertainties = data.get("uncertainties") or []
    themes = data.get("themes") or []
    segments = data.get("segments") or []
    external = data.get("external") or []
    opps = data.get("opps") or []
    derived = derive_root_cause(
        analyzed=analyzed,
        problems=problems,
        barriers=barriers,
        uncertainties=uncertainties,
        wishlist=wishlist,
        hesitation_count=int((data.get("signals") or {}).get("purchase_hesitation") or 0),
    )
    card = pm_insight_card(analyzed=analyzed, problems=problems, opportunities=opps)
    st.markdown("**Executive Summary**")
    st.write(
        f"{analyzed} public reviews were analyzed in this sample. "
        f"Strongest evidenced problem: {problems[0]['problem'] if problems else 'none extracted'}. "
        f"Top scored opportunity: {opps[0]['name'] if opps else 'none scored'}. "
        "Public reviews do not measure actual wishlist-to-purchase conversion."
    )
    blocks = [
        ("Key User Problems", _named_list(problems, "problem")),
        ("Why Users Wishlist", _named_list(wishlist)),
        ("Why Users Do Not Purchase", _named_list(barriers)),
        ("Purchase Uncertainties", _named_list(uncertainties)),
        ("Key Themes", _named_list(themes, "name")),
        ("User Segments", _named_list(segments, "name")),
        ("External Information", _named_list(external)),
        ("Top Opportunities", _named_list(opps, "name")),
    ]
    for title, items in blocks:
        st.markdown(f"**{title}**")
        st.write(", ".join(items) if items else "Insufficient evidence in the current dataset.")
    st.markdown("**Root Cause**")
    st.write(derived["statement"])
    st.markdown("**PM Insight**")
    st.write(
        f"{card['strongest_signal']}. {card['why_it_matters']} "
        f"Evidence: {card['evidence']} Confidence: {card['confidence']}. Caveat: {card['caveat']}"
    )


def _evidence_explorer(source: str, rating: str, since, data: dict) -> None:
    st.markdown('<div class="section-h">13. EVIDENCE EXPLORER</div>', unsafe_allow_html=True)
    st.caption("Original stored review text only. Quotes are never fabricated.")
    q = st.text_input("Search original review text")
    theme_names = _named_list(data["themes"], "name")
    problem_names = _named_list(data["problems"], "problem")
    barrier_names = _named_list(data["barriers"] or data["barrier_tax"])
    segment_names = _named_list(data["segments"], "name")
    f1, f2, f3, f4 = st.columns(4)
    theme_f = f1.selectbox("Theme", ["All"] + theme_names)
    problem_f = f2.selectbox("Problem", ["All"] + problem_names)
    barrier_f = f3.selectbox("Barrier", ["All"] + barrier_names)
    segment_f = f4.selectbox("Segment", ["All"] + segment_names)
    allowed: set[int] | None = None

    def _intersect(ids: list[int]) -> None:
        nonlocal allowed
        incoming = set(ids)
        allowed = incoming if allowed is None else allowed & incoming

    if theme_f != "All":
        match = next((t for t in data["themes"] if t["name"] == theme_f), None)
        _intersect(match["evidence_ids"] if match else [])
    if problem_f != "All":
        match = next((p for p in data["problems"] if p["problem"] == problem_f), None)
        _intersect(match.get("review_ids") if match else [])
    if barrier_f != "All":
        match = next((r for r in (data["barriers"] or data["barrier_tax"]) if r["label"] == barrier_f), None)
        _intersect(match.get("review_ids") if match else [])
    if segment_f != "All":
        match = next((s for s in data["segments"] if s["name"] == segment_f), None)
        _intersect(match["evidence_ids"] if match else [])

    db = _db()
    try:
        query = db.query(Review).filter(Review.is_empty.is_(False), Review.is_duplicate.is_(False))
        if source == "Google Play":
            query = query.filter(Review.source == "google_play")
        elif source == "Apple App Store":
            query = query.filter(Review.source == "apple_app_store")
        if rating != "All":
            query = query.filter(Review.rating == int(rating))
        if since is not None:
            query = query.filter(Review.review_date.isnot(None), Review.review_date >= since)
        if q:
            needle = f"%{q}%"
            query = query.filter((Review.text.ilike(needle)) | (Review.title.ilike(needle)))
        if allowed is not None:
            if not allowed:
                st.info("No reviews match these filters.")
                return
            query = query.filter(Review.id.in_(list(allowed)))
        rows = query.order_by(Review.review_date.desc()).limit(40).all()
        if not rows:
            st.info("No reviews match these filters.")
            return
        for review in rows:
            status = getattr(review.analysis, "status", "none") if review.analysis else "unanalyzed"
            date = review.review_date.isoformat() if review.review_date else "—"
            with st.expander(
                f"{review.source} · rating {review.rating} · {date} · review ID {review.source_review_id}"
            ):
                st.write(review.text or review.title or "")
                st.caption(
                    f"source {review.source} · rating {review.rating} · date {date} · "
                    f"review ID {review.source_review_id} · db id {review.id} · analysis={status}"
                )
                if review.analysis and review.analysis.is_valid_json:
                    st.write("Problem:", normalize_category_label(review.analysis.root_cause) or "—")
                    barriers = []
                    try:
                        barriers = json.loads(review.analysis.barriers_json or "[]")
                    except json.JSONDecodeError:
                        barriers = []
                    uncs = []
                    try:
                        uncs = json.loads(review.analysis.uncertainties_json or "[]")
                    except json.JSONDecodeError:
                        uncs = []
                    st.write("Barriers:", ", ".join(normalize_label_list(barriers)) or "—")
                    st.write("Uncertainties:", ", ".join(normalize_label_list(uncs)) or "—")
    finally:
        db.close()


def _limitations() -> None:
    st.markdown('<div class="section-h">DATA LIMITATIONS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="limit-card">{LIMITATIONS_NOTICE}</div>', unsafe_allow_html=True)


def _parse_run_notes(raw: str | None) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    notes = {}
    if text.startswith("window_start="):
        notes["window_start"] = text.split("=", 1)[-1]
    return notes


def _collection_history() -> None:
    st.markdown('<div class="section-h">COLLECTION HISTORY</div>', unsafe_allow_html=True)
    db = _db()
    try:
        rows = db.query(CollectionRun).order_by(CollectionRun.id.desc()).limit(15).all()
        if not rows:
            st.info("No collection runs recorded yet.")
            return
        table = []
        for r in rows:
            notes = _parse_run_notes(r.notes)
            errors = notes.get("errors")
            if not errors:
                try:
                    errors = json.loads(r.errors_json or "[]")
                except json.JSONDecodeError:
                    errors = [r.errors_json] if r.errors_json else []
            table.append(
                {
                    "timestamp": r.finished_at.isoformat() if r.finished_at else None,
                    "Google Play new reviews": notes.get("google_play_new", r.new_count if "google" in (r.sources or "") else 0),
                    "Apple new reviews": notes.get("apple_new", r.new_count if "apple" in (r.sources or "") else 0),
                    "total stored": notes.get("stored"),
                    "analysis count": notes.get("analyzed", r.analyzed),
                    "errors": "; ".join(str(e) for e in (errors or [])[:4]) or "",
                }
            )
        st.dataframe(table, hide_index=True, width="stretch")
    finally:
        db.close()


def _pipeline_step_label(value: str) -> str:
    return {
        "done": "SUCCESS",
        "no_new": "NO NEW REVIEWS",
        "failed": "FAILED",
        "partial": "PARTIAL",
        "insufficient": "INSUFFICIENT DATA",
        "pending": "—",
    }.get(value or "", "—")


def _source_pipeline_step(stats, source: str) -> str:
    info = (getattr(stats, "by_source", None) or {}).get(source) or {}
    status = str(info.get("fetch_status") or "")
    if status.endswith("FAILED"):
        return "failed"
    if status.endswith("NO_NEW_REVIEWS"):
        return "no_new"
    if status.endswith("NEW_REVIEWS_FOUND"):
        return "done"
    if getattr(stats, "errors", None) and getattr(stats, "fetched", 0) == 0:
        return "failed"
    if getattr(stats, "new", 0) == 0:
        return "no_new"
    return "done"


def _pipeline_mark(value: str) -> str:
    if value == "done":
        return "✓ SUCCESS"
    if value == "no_new":
        return "✓ NO NEW REVIEWS"
    if value == "partial":
        return "PARTIAL"
    if value == "failed":
        return "FAILED"
    if value == "insufficient":
        return "INSUFFICIENT DATA"
    return "—"


def _pipeline_from_last_run() -> dict:
    db = _db()
    try:
        last = db.query(CollectionRun).order_by(CollectionRun.id.desc()).first()
        if last is None:
            return {}
        notes = _parse_run_notes(last.notes)
        steps = {"play": "pending", "apple": "pending", "save": "pending", "analyze": "pending", "insights": "pending"}
        sources = (last.sources or "").lower()
        gp = str(notes.get("google_play_status") or "")
        apple = str(notes.get("apple_status") or "")
        if "google" in sources or gp:
            if gp.endswith("FAILED"):
                steps["play"] = "failed"
            elif gp.endswith("NO_NEW_REVIEWS") or (last.new_count == 0 and "google" in sources and not gp.endswith("NEW_REVIEWS_FOUND")):
                steps["play"] = "no_new"
            else:
                steps["play"] = "done"
        if "apple" in sources or apple:
            if apple.endswith("FAILED"):
                steps["apple"] = "failed"
            elif apple.endswith("NO_NEW_REVIEWS"):
                steps["apple"] = "no_new"
            elif apple.endswith("NEW_REVIEWS_FOUND"):
                steps["apple"] = "done"
            elif last.status in {"completed", "completed_with_errors"}:
                steps["apple"] = "no_new"
        if last.status in {"completed", "completed_with_errors"}:
            steps["save"] = "done"
        analyzed = int(notes.get("analyzed") or last.analyzed or 0)
        errors = notes.get("errors") or []
        if analyzed > 0 and errors:
            steps["analyze"] = "partial"
            steps["insights"] = insights_status_for_analyze("partial", analyzed)
        elif analyzed > 0:
            steps["analyze"] = "done"
            steps["insights"] = insights_status_for_analyze("done", analyzed)
        elif errors:
            steps["analyze"] = "failed"
            steps["insights"] = insights_status_for_analyze("failed", analyzed)
        else:
            diag = get_database_diagnostics(db)
            stored_analyzed = int(diag.get("analyzed_reviews") or 0)
            if stored_analyzed > 0:
                steps["analyze"] = "done"
                steps["insights"] = insights_status_for_analyze("done", stored_analyzed)
            else:
                steps["analyze"] = "insufficient"
                steps["insights"] = insights_status_for_analyze("insufficient", stored_analyzed)
        return steps
    finally:
        db.close()


def _pipeline_status_panel() -> None:
    failed_reason = None
    pipeline_result = st.session_state.get("pipeline_result") or {}
    steps = (
        st.session_state.get("pipeline_steps")
        or pipeline_result.get("steps")
        or _pipeline_from_last_run()
    )
    if not steps:
        return
    failed_reason = derive_failed_reason(
        steps=steps,
        last_analysis=st.session_state.get("last_analysis"),
        last_collection=st.session_state.get("last_collection"),
        step4_error=st.session_state.get("step4_error"),
        pipeline_result=pipeline_result,
    )
    st.markdown('<div class="section-h">PIPELINE STATUS</div>', unsafe_allow_html=True)
    for key, label in (
        ("play", "Google Play"),
        ("apple", "Apple"),
        ("save", "Save"),
        ("analyze", "Analyze"),
        ("insights", "Insights"),
    ):
        mark = _pipeline_mark(steps.get(key, "pending"))
        st.markdown(f'<div class="pipeline-row">{label} {mark}</div>', unsafe_allow_html=True)
    if failed_reason and steps.get("analyze") in {"failed", "partial"}:
        st.error(failed_reason)
    elif failed_reason and (steps.get("play") == "failed" or steps.get("apple") == "failed"):
        st.error(failed_reason)


def _run_full_discovery(ai_ok: bool) -> None:
    from app.collectors.engine import CollectionEngine
    from app.pipeline.analysis import AnalysisRunResult
    from app.pipeline.orchestrator import run_analysis_pipeline

    reload_settings()
    settings = get_settings()
    steps = {"play": "pending", "apple": "pending", "save": "pending", "analyze": "pending", "insights": "pending", "dashboard": "pending"}
    db = _db()
    try:
        with st.status("Full discovery pipeline", expanded=True) as box:
            stored = get_review_count(db, myntra_only=True)
            from app.pipeline.dataset import analysis_dataset_stats, enforce_review_limit

            if stored > _dataset_limit():
                prune = enforce_review_limit(db, prune=True)
                box.write(
                    f"Pruned storage to {prune.get('kept')} / {prune.get('max_reviews')} "
                    f"(deleted {prune.get('deleted')} excess reviews)"
                )
                stored = get_review_count(db, myntra_only=True)
            stats = analysis_dataset_stats(db)
            storage_cap = int(stats.get("max_total_reviews") or _dataset_limit())
            if stored >= storage_cap:
                box.write(
                    f"Collection skipped — {stored} stored Myntra-valid reviews already at the "
                    f"{storage_cap} combined limit. AI sample: {stats.get('selected_reviews')} / "
                    f"{stats.get('max_analysis_reviews')}."
                )
                steps["play"] = steps["apple"] = "no_new"
                steps["save"] = "done"
            else:
                engine = CollectionEngine(db)
                box.write("Collecting Google Play")
                gp = engine.run(["google_play"], analyze=False, mode="last_30_days")
                steps["play"] = _source_pipeline_step(gp, "google_play")
                box.write("Collecting Apple App Store")
                apple = engine.run(["apple_app_store"], analyze=False, mode="last_30_days")
                steps["apple"] = _source_pipeline_step(apple, "apple_app_store")
                steps["save"] = "done"
            result = AnalysisRunResult()
            if not get_settings().has_ai_credentials:
                steps["analyze"] = "failed"
                result.last_error = "OpenRouter API key is not configured. Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
                st.error(result.last_error)
            else:
                try:
                    def _progress(event: dict) -> None:
                        st.session_state["analysis_progress"] = event
                        if event.get("batch_index"):
                            box.write(
                                f"Batch {event.get('batch_index')}/{event.get('batch_total')} · "
                                f"analyzed {event.get('analyzed_total')} / selected {event.get('selected')} · "
                                f"{event.get('percent')}%"
                            )

                    result = run_analysis_pipeline(db, progress=_progress)
                    if result.analyzed > 0 and (result.failed or result.last_error):
                        steps["analyze"] = "partial"
                    elif result.analyzed > 0 or not result.last_error:
                        steps["analyze"] = "done"
                    else:
                        steps["analyze"] = "failed"
                except Exception as exc:
                    steps["analyze"] = "failed"
                    result.last_error = _openrouter_error(exc)
                    st.error(result.last_error)
            analyzed_now = int(get_database_diagnostics(db).get("analyzed_reviews") or 0)
            steps["insights"] = insights_status_for_analyze(steps["analyze"], analyzed_now)
            steps["dashboard"] = "done"
            st.session_state["pipeline_steps"] = steps
            st.session_state["pipeline_result"] = {
                "steps": dict(steps),
                "failed_reason": result.last_error or None,
                "analyzed": result.analyzed,
                "failed": result.failed,
            }
            _load_bundle.clear()
            st.session_state["last_analysis"] = {
                "status": "Connected" if result.analyzed else "Failed",
                "message": result.last_error or f"Analyzed {result.analyzed}, failed {result.failed}.",
                "batches_processed": result.batches_processed,
                "successful_batches": result.successful_batches,
                "failed_batches": result.failed_batches,
            }
            if result.last_error:
                st.session_state["step4_error"] = {"error": result.last_error, "model": settings.resolved_model}
    finally:
        db.close()
    st.rerun()


def _run_collect(sources: list[str], analyze: bool, mode: str = "latest") -> None:
    from app.collectors.engine import CollectionEngine

    reload_settings()
    db = _db()
    try:
        engine = CollectionEngine(db)
        stats = engine.run(sources, analyze=analyze, mode=mode)
        st.session_state["last_collection"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "stats": stats.model_dump(mode="json"),
        }
        steps = st.session_state.get("pipeline_steps") or {
            "play": "pending",
            "apple": "pending",
            "save": "pending",
            "analyze": "pending",
            "insights": "pending",
        }
        if "google_play" in sources:
            steps["play"] = _source_pipeline_step(stats, "google_play")
        if "apple_app_store" in sources:
            steps["apple"] = _source_pipeline_step(stats, "apple_app_store")
        steps["save"] = "done" if not stats.errors or stats.new or stats.fetched else "failed"
        if analyze:
            if stats.analysis_error and stats.analyzed == 0:
                steps["analyze"] = "failed"
            elif stats.analyzed > 0 and (stats.analysis_error or stats.analysis_failed):
                steps["analyze"] = "partial"
            elif stats.analyzed > 0:
                steps["analyze"] = "done"
            else:
                steps["analyze"] = "no_new"
            analyzed_now = int(get_database_diagnostics(db).get("analyzed_reviews") or 0)
            steps["insights"] = insights_status_for_analyze(steps["analyze"], analyzed_now)
        st.session_state["pipeline_steps"] = steps
        st.session_state["pipeline_result"] = {
            "steps": dict(steps),
            "failed_reason": stats.analysis_error or ("; ".join(stats.errors[:3]) if stats.errors else None),
            "analyzed": stats.analyzed,
            "failed": stats.analysis_failed,
        }
        _load_bundle.clear()
        if analyze and stats.analysis_error and stats.analyzed == 0:
            st.error("Reviews collected successfully, but OpenRouter analysis failed.")
            st.error(stats.analysis_error)
    except Exception as exc:
        st.error(f"Collection failed: {exc}")
        if st.session_state.get("debug"):
            st.code(traceback.format_exc())
        return
    finally:
        db.close()
    st.rerun()


def _run_analyze(*, only_failed: bool = False) -> None:
    from app.pipeline.orchestrator import run_analysis_pipeline

    reload_settings()
    if not get_settings().has_ai_credentials:
        st.error("OpenRouter API key is not configured. Add OPENROUTER_API_KEY to Streamlit Secrets or .env.")
        return
    db = _db()
    try:
        def _progress(event: dict) -> None:
            st.session_state["analysis_progress"] = event

        result = run_analysis_pipeline(
            db,
            progress=_progress,
            only_failed=only_failed,
            include_failed=only_failed,
        )
        if result.analyzed == 0 and result.failed:
            st.error(_openrouter_error(result.last_error or f"OpenRouter analysis failed for {result.failed} reviews."))
        analyze_step = "pending"
        if result.analyzed > 0 and (result.failed or result.last_error):
            analyze_step = "partial"
        elif result.analyzed > 0:
            analyze_step = "done"
        elif result.failed or result.last_error:
            analyze_step = "failed"
        else:
            analyze_step = "insufficient"
        analyzed_now = int(get_database_diagnostics(db).get("analyzed_reviews") or 0)
        steps = st.session_state.get("pipeline_steps") or {}
        steps["analyze"] = analyze_step
        steps["insights"] = insights_status_for_analyze(analyze_step, analyzed_now)
        st.session_state["pipeline_steps"] = steps
        st.session_state["pipeline_result"] = {
            "steps": dict(steps),
            "failed_reason": result.last_error or None,
            "analyzed": result.analyzed,
            "failed": result.failed,
        }
        st.session_state["last_analysis"] = {
            "status": "Connected" if result.analyzed else "Configured",
            "message": result.last_error or f"Analyzed {result.analyzed}, failed {result.failed}.",
            "batches_processed": result.batches_processed,
            "successful_batches": result.successful_batches,
            "failed_batches": result.failed_batches,
        }
        _load_bundle.clear()
    except Exception as exc:
        st.error(_openrouter_error(exc))
        return
    finally:
        db.close()
    st.rerun()
