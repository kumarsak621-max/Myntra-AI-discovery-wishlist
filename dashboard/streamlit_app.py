"""Single-page Streamlit discovery dashboard. Collectors and analysis stay unchanged."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone

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
from app.pipeline.dates import get_last_30_days_cutoff, humanize_ago
from app.pipeline.quantification import (
    BARRIER_TERMS,
    COMPARISON_METHOD_TERMS,
    COMPARISON_TERMS,
    EXTERNAL_TERMS,
    SOCIAL_TERMS,
    UNCERTAINTY_TERMS,
    WISHLIST_BEHAVIOR_TERMS,
    PURCHASE_BEHAVIOR_TERMS,
    explicit_age_mentions,
    label_distribution,
    label_window_momentum,
    overview_metrics,
    problem_rows,
    purchase_signal_counts,
    signal_counts,
    source_live_status,
    taxonomy_counts,
)
from app.pipeline.report import build_report, evidence_cards
from config.settings import (
    clamp_max_dataset_reviews,
    get_ai_config,
    get_settings,
    reload_settings,
)
from dashboard.chat import ask_product_assistant
from dashboard.insights import funnel_stages, pm_insight, wishlist_conversion_copy

LOGGER = logging.getLogger("myntra.discovery")

EMPTY = "No real reviews have been collected yet."
NEAR_REALTIME = "Near-real-time — refreshed from public sources"

SUGGESTED_QUESTIONS = [
    "Why do users add fashion products to their wishlist?",
    "What prevents wishlisted products from eventually being purchased?",
    "What uncertainties remain after users identify a product?",
    "What causes users to postpone a purchase?",
    "How do users compare multiple shortlisted products?",
    "What information do users seek outside Myntra before purchasing?",
    "What role do fit, size, styling, price, reviews, occasion and social validation play?",
    "When is wishlist behavior genuine purchase intent versus bookmarking?",
    "How do these behaviors differ across user segments?",
    "What unmet needs emerge consistently?",
    "What is the highest-impact opportunity?",
    "Which problem has the strongest evidence?",
    "Show me evidence for size uncertainty.",
    "Which barriers have the highest impact?",
    "What should a Product Manager investigate next?",
]


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
    return clamp_max_dataset_reviews(get_settings().max_dataset_reviews)


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
        </style>
        """,
        unsafe_allow_html=True,
    )


def _bar(df: pd.DataFrame, *, empty: str = "Insufficient evidence for this visualization.") -> None:
    if df is None or df.empty:
        st.info(empty)
        return
    try:
        st.bar_chart(df, x="label", y="count", horizontal=True, height=260)
    except TypeError:
        st.bar_chart(df.set_index("label")["count"], height=260)


def _scatter(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("Insufficient evidence for this visualization.")
        return
    st.scatter_chart(df, x="frequency", y="impact", size="count", height=260)


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
        problems = problem_rows(db, myntra_only=True, since=since, source=src)
        intents = label_distribution(db, "intent", myntra_only=True, relevant_only=False, since=since, source=src)
        barriers = label_distribution(db, "barriers", myntra_only=True, relevant_only=False, since=since, source=src)
        uncertainties = label_distribution(
            db, "uncertainties", myntra_only=True, relevant_only=False, since=since, source=src
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
        themes = [
            {
                "name": t.name,
                "review_count": t.review_count,
                "evidence_ids": _json_ids(t.evidence_ids_json),
            }
            for t in db.query(Theme).order_by(Theme.review_count.desc()).all()
        ]
        segments = [
            {
                "name": s.name,
                "review_count": s.review_count,
                "basis": s.basis,
                "evidence_ids": _json_ids(s.evidence_ids_json),
            }
            for s in db.query(Segment).order_by(Segment.review_count.desc()).all()
        ]
        opps = [
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
        ]
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

    with st.sidebar:
        st.markdown("**Controls**")
        period = st.radio("Date range", ["Last 30 Days", "All Time"], index=0)
        st.session_state["period"] = period
        source = st.selectbox("Source", ["All", "Google Play", "Apple App Store"])
        rating = st.selectbox("Rating filter", ["All", "1", "2", "3", "4", "5"])
        auto = st.selectbox("Auto-refresh", ["OFF", "5 minutes", "15 minutes"], index=0)
        st.session_state["analyze_on_collect"] = st.checkbox("Analyze after refresh", value=ai_ok and True)
        st.caption(f"Dataset limit: {_dataset_limit()} (maximum)")
        st.caption("API key is never displayed.")
        if st.button("Analyze pending"):
            _run_analyze()
        if st.button("Retry failed analysis"):
            _run_analyze(only_failed=True)

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

    _header(cfg, data["freshness"], period)
    _actions(ai_ok)
    _pipeline_status_panel()
    _kpis(diag, metrics, data["opps"], data["themes"], period)
    _wishlist_indicator(data["signals"], data["barriers"])

    if stored == 0:
        st.info(EMPTY)
        return
    if blocker and analyzed == 0:
        st.warning(blocker)

    _problems(data, analyzed)
    _wishlist(data, analyzed)
    _barriers(data, analyzed)
    _uncertainties(data, analyzed)
    _themes(data, analyzed)
    _segments(data, analyzed)
    _purchase_behavior(data, analyzed)
    _comparison(data, analyzed)
    _external(data, analyzed)
    _social(data, analyzed)
    _opportunities(data, analyzed)
    _decomposition(data)
    _discovery_report(data, analyzed)
    _evidence_explorer(source, rating, since, data)
    _chatbot(analyzed)
    _limitations()
    _collection_history()

    if auto != "OFF":
        st.caption("Auto-refresh polls public feeds periodically. It is not a live event stream.")


def _header(cfg: dict, freshness: dict, period: str) -> None:
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
    c2.write("**Data source**")
    c2.write("Google Play + Apple App Store")
    c3.write("**AI provider**")
    c3.write("OpenRouter")
    c4.write("**Model**")
    c4.write(cfg.get("model") or "")
    c5.write("**Data status**")
    c5.write(NEAR_REALTIME)
    st.caption(
        f"Last checked: {humanize_ago(checked)} ({checked.isoformat() if checked else 'never'}). "
        "Never a live stream of in-app events."
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
                analyze=bool(st.session_state.get("analyze_on_collect")),
                mode="latest",
            )
    with c:
        from app.ai.provider import test_openrouter_connection

        if st.button("Test OpenRouter connection"):
            st.session_state["ai_connection_test"] = test_openrouter_connection()
        probe = st.session_state.get("ai_connection_test")
        if probe:
            if probe.get("ok"):
                st.success("OpenRouter accepted a live test request.")
            else:
                st.error(probe.get("error") or "OpenRouter connection test failed.")


def _kpis(diag: dict, metrics: dict, opps: list, themes: list, period: str) -> None:
    top_score = opps[0]["score"] if opps else 0
    cols = st.columns(5)
    cols[0].metric("Total reviews", diag.get("total_reviews") or 0)
    cols[1].metric("Dataset limit", _dataset_limit())
    cols[2].metric("Google Play", diag.get("google_play_reviews") or 0)
    cols[3].metric("Apple App Store", diag.get("apple_reviews") or 0)
    cols[4].metric("Last 30 days", diag.get("last_30_day_reviews") or 0)
    cols2 = st.columns(5)
    cols2[0].metric("Analyzed", diag.get("analyzed_reviews") or 0)
    cols2[1].metric("Pending", diag.get("pending_reviews") or 0)
    cols2[2].metric("Failed", diag.get("failed_reviews") or 0)
    cols2[3].metric("Wishlist-related", metrics.get("wishlist_signals") or 0)
    cols2[4].metric("Unique themes / top score", f"{len(themes)} / {top_score}")
    st.caption(
        f"{diag.get('last_30_day_reviews') or 0} real reviews available in the last 30 days. "
        f"Period filter: {period}. Counts come from the database, not the LLM."
    )


def _wishlist_indicator(signals: dict, barriers: list) -> None:
    st.markdown('<div class="section-h">WISHLIST → PURCHASE</div>', unsafe_allow_html=True)
    st.caption("Evidence-based opportunity indicator — not an actual Myntra conversion rate.")
    a, b, c = st.columns(3)
    a.metric("Wishlist purchase-intent signals", f"{signals.get('wishlist_pct', 0)}%")
    b.metric("Purchase hesitation signals", f"{signals.get('hesitation_pct', 0)}%")
    high = sum(1 for row in barriers if (row.get("hesitant_count") or 0) >= max(2, (row.get("count") or 0) // 2))
    c.metric("High-impact barriers", high)
    st.caption(
        f"Sample: {signals.get('analyzed', 0)} analyzed reviews. "
        f"Wishlist mentions: {signals.get('wishlist_signal', 0)}. "
        f"Hesitation mentions: {signals.get('purchase_hesitation', 0)}."
    )
    _insight(wishlist_conversion_copy(signals))


def _problems(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">1. USER PROBLEMS</div>', unsafe_allow_html=True)
    rows = data["problems"]
    if not rows:
        st.info("Insufficient evidence for this visualization." if analyzed else "No analyzed reviews yet.")
        _insight(pm_insight(topic="user problem", rows=[], analyzed=analyzed))
        return
    shown = rows[:6]
    left, mid, right = st.columns([1.2, 1, 1])
    with left:
        for i, row in enumerate(shown, 1):
            st.markdown(
                f"**{i}. {row['problem']}**  \n"
                f"{row['frequency']} reviews · {row['percentage']}%  \n"
                f"Severity {row['severity']}/5 · Purchase impact {row['purchase_impact']}/5 · "
                f"Confidence {row.get('confidence') or '—'}"
            )
    with mid:
        _bar(pd.DataFrame([{"label": r["problem"][:40], "count": r["frequency"]} for r in shown]))
    with right:
        _scatter(
            pd.DataFrame(
                [
                    {
                        "frequency": r["frequency"],
                        "impact": r["purchase_impact"],
                        "count": r["frequency"],
                    }
                    for r in shown
                ]
            )
        )
    _insight(pm_insight(topic="user problem", rows=[{"label": r["problem"], "count": r["frequency"], "percentage": r["percentage"]} for r in shown], analyzed=analyzed))
    db = _db()
    try:
        _quotes(evidence_cards(db, shown[0].get("review_ids") or [], limit=3))
    finally:
        db.close()


def _wishlist(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">2. WISHLIST BEHAVIOR</div>', unsafe_allow_html=True)
    rows = data["wishlist_beh"] or data["intents"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="wishlist behavior", rows=[], analyzed=analyzed))
        return
    left, right = st.columns(2)
    with left:
        _bar(pd.DataFrame([{"label": r["label"][:40], "count": r["count"]} for r in rows[:8]]))
    with right:
        intent = next((r for r in rows if "intent" in r["label"].lower() or "purchase" in r["label"].lower()), None)
        bookmark = next((r for r in rows if "bookmark" in r["label"].lower() or "save" in r["label"].lower()), None)
        donut = []
        if intent:
            donut.append({"label": intent["label"], "count": intent["count"]})
        if bookmark:
            donut.append({"label": bookmark["label"], "count": bookmark["count"]})
        if donut:
            _bar(pd.DataFrame(donut))
        else:
            st.info("Insufficient evidence to split purchase intent vs bookmarking.")
        st.dataframe(
            [{"behavior": r["label"], "frequency": r["count"], "percentage": r["percentage"]} for r in rows[:10]],
            hide_index=True,
            width="stretch",
        )
    _insight(pm_insight(topic="wishlist behavior", rows=rows, analyzed=analyzed, extra="Wishlist language in public reviews can mean delayed purchase, price watching, or bookmarking — not a completed conversion."))
    db = _db()
    try:
        st.caption("Wishlist quotes — copied from stored review text.")
        _quotes(evidence_cards(db, rows[0].get("review_ids") or [], limit=3))
    finally:
        db.close()


def _barriers(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">3. PURCHASE BARRIERS</div>', unsafe_allow_html=True)
    rows = data["barriers"] or data["barrier_tax"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="purchase barrier", rows=[], analyzed=analyzed))
        return
    left, right = st.columns(2)
    with left:
        _bar(pd.DataFrame([{"label": r["label"][:40], "count": r["count"]} for r in rows[:8]]))
    with right:
        heat = pd.DataFrame(
            [
                {
                    "barrier": r["label"][:32],
                    "frequency": r["count"],
                    "hesitation": r.get("hesitant_count") or 0,
                }
                for r in rows[:8]
            ]
        )
        st.dataframe(heat, hide_index=True, width="stretch")
    _insight(pm_insight(topic="purchase barrier", rows=rows, analyzed=analyzed))


def _uncertainties(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">4. PURCHASE UNCERTAINTIES</div>', unsafe_allow_html=True)
    rows = data["uncertainties"] or data["unc_tax"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="purchase uncertainty", rows=[], analyzed=analyzed))
        return
    left, right = st.columns(2)
    with left:
        _bar(pd.DataFrame([{"label": r["label"][:40], "count": r["count"]} for r in rows[:8]]))
    with right:
        st.dataframe(
            [{"uncertainty": r["label"], "frequency": r["count"], "percentage": r["percentage"]} for r in rows[:10]],
            hide_index=True,
            width="stretch",
        )
    _insight(pm_insight(topic="purchase uncertainty", rows=rows, analyzed=analyzed))
    db = _db()
    try:
        st.caption("Top uncertainty quotes — copied from stored review text.")
        _quotes(evidence_cards(db, rows[0].get("review_ids") or [], limit=4))
    finally:
        db.close()


def _themes(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">5. KEY THEMES</div>', unsafe_allow_html=True)
    themes = data["themes"]
    momentum = {m["label"]: m for m in data["theme_momentum"]}
    if not themes:
        st.info("Insufficient evidence for this visualization.")
        return
    cards = st.columns(min(4, len(themes)))
    for i, theme in enumerate(themes[:4]):
        trend = (momentum.get(theme["name"]) or {}).get("momentum") or "insufficient data"
        cards[i].metric(theme["name"][:40], f"{theme['review_count']} reviews")
        cards[i].caption(f"Trend: {trend}")
    _bar(pd.DataFrame([{"label": t["name"][:40], "count": t["review_count"]} for t in themes[:10]]))
    if data["theme_momentum"]:
        st.dataframe(
            [{"theme": m["label"], "count": m["count"], "trend": m["momentum"]} for m in data["theme_momentum"][:10]],
            hide_index=True,
            width="stretch",
        )
    _insight(pm_insight(topic="theme", rows=[{"label": t["name"], "count": t["review_count"]} for t in themes], analyzed=analyzed))


def _segments(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">6. USER SEGMENTS</div>', unsafe_allow_html=True)
    segs = data["segments"]
    if not segs:
        st.info("Insufficient evidence for reliable segmentation.")
    else:
        left, right = st.columns(2)
        with left:
            _bar(pd.DataFrame([{"label": s["name"][:40], "count": s["review_count"]} for s in segs[:8]]))
        with right:
            st.dataframe(
                [{"segment": s["name"], "reviews": s["review_count"], "basis": s["basis"]} for s in segs],
                hide_index=True,
                width="stretch",
            )
        _insight(pm_insight(topic="behavioral segment", rows=[{"label": s["name"], "count": s["review_count"]} for s in segs], analyzed=analyzed))
    if data["ages"]:
        st.caption(f"Explicit age evidence: {len(data['ages'])} reviews mention an age number.")
    else:
        st.info(
            "Age segmentation is not reliably available from public review data. "
            "Behavioral segmentation is used instead. Age data unavailable from public review evidence."
        )


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
    st.markdown('<div class="section-h">8. HOW USERS COMPARE PRODUCTS</div>', unsafe_allow_html=True)
    rows = data["compare"]
    methods = data["compare_how"]
    if not rows and not methods:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="comparison behavior", rows=[], analyzed=analyzed))
        return
    left, right = st.columns(2)
    with left:
        _bar(pd.DataFrame([{"label": r["label"], "count": r["count"]} for r in rows[:8]]) if rows else None)
    with right:
        if methods:
            st.caption("Comparison methods found in review text")
            _bar(pd.DataFrame([{"label": r["label"], "count": r["count"]} for r in methods]))
        else:
            st.info("No explicit comparison-method mentions in the stored reviews.")
    _insight(pm_insight(topic="comparison factor", rows=rows or methods, analyzed=analyzed))


def _external(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">9. EXTERNAL INFORMATION SEEKING</div>', unsafe_allow_html=True)
    rows = data["external"]
    if not rows:
        st.info("Insufficient evidence for this visualization.")
        _insight(pm_insight(topic="external information seeking", rows=[], analyzed=analyzed))
        return
    _bar(pd.DataFrame([{"label": r["label"], "count": r["count"]} for r in rows]))
    _insight(pm_insight(topic="external information seeking", rows=rows, analyzed=analyzed))


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
    st.markdown('<div class="section-h">11. OPPORTUNITY MATRIX</div>', unsafe_allow_html=True)
    st.caption("Score = reach × frequency × purchase impact × severity × evidence confidence. Calculated in Python.")
    opps = data["opps"]
    if not opps:
        st.info("Insufficient evidence for this visualization.")
        return
    top = opps[0]
    st.markdown(f"**Top opportunity:** {top['name']} · score {top['score']} · {top['relevant_count']} reviews")
    left, right = st.columns(2)
    with left:
        _scatter(
            pd.DataFrame(
                [
                    {"frequency": o["frequency"], "impact": o["purchase_impact"], "count": o["relevant_count"]}
                    for o in opps[:20]
                ]
            )
        )
    with right:
        st.dataframe(
            [
                {
                    "rank": o["rank"],
                    "opportunity": o["name"],
                    "reach": o["reach"],
                    "frequency": o["frequency"],
                    "impact": o["purchase_impact"],
                    "severity": o["severity"],
                    "confidence": o["evidence_confidence"],
                    "score": o["score"],
                    "reviews": o["relevant_count"],
                }
                for o in opps[:15]
            ],
            hide_index=True,
            width="stretch",
        )
    _insight(
        pm_insight(
            topic="opportunity",
            rows=[{"label": o["name"], "count": o["relevant_count"], "percentage": o["percentage"]} for o in opps],
            analyzed=analyzed,
            extra=top.get("why_investigate") or "",
        )
    )


def _decomposition(data: dict) -> None:
    st.markdown('<div class="section-h">12. WISHLIST → PURCHASE METRIC DECOMPOSITION</div>', unsafe_allow_html=True)
    st.caption("Evidence-based behavioral decomposition from public reviews — not an internal Myntra funnel.")
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
    )
    present = [s for s in stages if s["count"]]
    if present:
        _bar(pd.DataFrame([{"label": s["stage"], "count": s["count"]} for s in present]))
    else:
        st.info("Insufficient evidence for this visualization.")
    st.dataframe(stages, hide_index=True, width="stretch")


def _discovery_report(data: dict, analyzed: int) -> None:
    st.markdown('<div class="section-h">13. DISCOVERY REPORT</div>', unsafe_allow_html=True)
    if analyzed <= 0:
        st.info("Discovery report requires stored analysis records.")
        return
    report = data.get("report") or {}
    primary = ((report.get("16_recommendation") or {}).get("primary_opportunity")) or {}
    st.markdown("**Executive summary**")
    st.write(
        f"{analyzed} reviews analyzed. "
        f"Top problem: {(data['problems'][0]['problem'] if data['problems'] else 'none extracted')}. "
        f"Top opportunity: {primary.get('opportunity') or (data['opps'][0]['name'] if data['opps'] else 'none scored')}."
    )
    blocks = [
        ("Key user problems", [p["problem"] for p in data["problems"][:5]]),
        ("Why users wishlist", [r["label"] for r in (data["wishlist_beh"] or data["intents"])[:5]]),
        ("Why users do not purchase", [r["label"] for r in data["barriers"][:5]]),
        ("Purchase uncertainties", [r["label"] for r in data["uncertainties"][:5]]),
        ("Key themes", [t["name"] for t in data["themes"][:5]]),
        ("User segments", [s["name"] for s in data["segments"][:5]]),
        ("Comparison behavior", [r["label"] for r in data["compare"][:5]]),
        ("External information", [r["label"] for r in data["external"][:5]]),
        ("Top opportunities", [o["name"] for o in data["opps"][:5]]),
    ]
    for title, items in blocks:
        st.markdown(f"**{title}**")
        st.write(", ".join(items) if items else "Insufficient evidence in the current dataset.")


def _evidence_explorer(source: str, rating: str, since, data: dict) -> None:
    st.markdown('<div class="section-h">14. EVIDENCE EXPLORER</div>', unsafe_allow_html=True)
    q = st.text_input("Search original review text")
    theme_names = [t["name"] for t in data["themes"]]
    problem_names = [p["problem"] for p in data["problems"]]
    barrier_names = [r["label"] for r in (data["barriers"] or data["barrier_tax"])]
    segment_names = [s["name"] for s in data["segments"]]
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
            status = getattr(review.analysis, "status", "none") if review.analysis else "none"
            with st.expander(f"{review.source} · {review.rating}★ · {status} · {review.review_date}"):
                st.write(review.text or review.title)
                st.caption(
                    f"id {review.source_review_id} · region {review.region or '—'} · "
                    f"{review.source_url} · analysis={status}"
                )
                if review.analysis and review.analysis.is_valid_json:
                    st.write("Problem:", review.analysis.root_cause or "—")
                    st.write("Barriers:", review.analysis.barriers_json)
                    st.write("Uncertainties:", review.analysis.uncertainties_json)
    finally:
        db.close()


def _chatbot(analyzed: int) -> None:
    st.markdown('<div class="section-h">🤖 AI PRODUCT MANAGER ASSISTANT</div>', unsafe_allow_html=True)
    st.caption("Answers use retrieved database evidence only. The API key is never shown.")
    pick = st.selectbox("Suggested questions", ["(type your own)"] + SUGGESTED_QUESTIONS)
    question = st.text_input("Ask about wishlist → purchase evidence", value="" if pick == "(type your own)" else pick)
    if st.button("Ask") and question.strip():
        db = _db()
        try:
            result = ask_product_assistant(db, question.strip(), analyzed=analyzed)
        finally:
            db.close()
        st.write(result["answer"])
        st.caption(result["evidence_summary"])
        st.write("Supporting reviews:", result["supporting_review_count"])
        if result.get("themes"):
            st.write("Themes:", ", ".join(result["themes"]))
        st.write("PM implication:", result["pm_implication"])
        for quote in result.get("quotes") or []:
            st.markdown(f"“{quote['text']}”")
            st.caption(f"{quote['source']} · {quote.get('rating')} · {quote.get('date')} · id {quote['id']}")


def _limitations() -> None:
    st.markdown('<div class="section-h">DATA LIMITATIONS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="limit-card">
        Reviews are public feedback, not the complete Myntra customer base.
        Public reviews do not directly expose actual wishlist → purchase conversion events.
        Demographic information is generally unavailable.
        Behavioral segments are inferred only from textual evidence.
        Apple/Google availability depends on public source feeds.
        Near-real-time means periodic refresh, not true event streaming.
        Small samples should not be generalized to the full user base.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _collection_history() -> None:
    with st.expander("Collection history"):
        db = _db()
        try:
            rows = db.query(CollectionRun).order_by(CollectionRun.id.desc()).limit(15).all()
            if not rows:
                st.info("No collection runs recorded yet.")
                return
            st.dataframe(
                [
                    {
                        "id": r.id,
                        "sources": r.sources,
                        "mode": r.mode,
                        "new": r.new_count,
                        "status": r.status,
                        "finished": r.finished_at.isoformat() if r.finished_at else None,
                    }
                    for r in rows
                ],
                hide_index=True,
                width="stretch",
            )
        finally:
            db.close()


def _pipeline_status_panel() -> None:
    steps = st.session_state.get("pipeline_steps") or {}
    if not steps:
        return
    st.caption(
        "Pipeline: "
        + " · ".join(
            f"{label}={'✓' if steps.get(key)=='done' else ('FAILED' if steps.get(key)=='failed' else '—')}"
            for key, label in (
                ("play", "Play"),
                ("apple", "Apple"),
                ("save", "Save"),
                ("analyze", "Analyze"),
                ("insights", "Insights"),
            )
        )
    )
    if steps.get("analyze") == "failed":
        err = (st.session_state.get("step4_error") or {}).get("error")
        if err:
            st.error(err)


def _run_full_discovery(ai_ok: bool) -> None:
    from app.collectors.engine import CollectionEngine
    from app.pipeline.analysis import AnalysisRunResult, smoke_test_analyze_limit
    from app.pipeline.orchestrator import run_analysis_pipeline

    reload_settings()
    settings = get_settings()
    steps = {"play": "pending", "apple": "pending", "save": "pending", "analyze": "pending", "insights": "pending", "dashboard": "pending"}
    db = _db()
    try:
        with st.status("Full discovery pipeline", expanded=True) as box:
            stored = get_review_count(db, myntra_only=True)
            if stored > 0:
                box.write(f"Collection skipped — {stored} stored Myntra-valid reviews already in the database")
                steps["play"] = steps["apple"] = steps["save"] = "done"
            else:
                engine = CollectionEngine(db)
                box.write("Collecting Google Play")
                gp = engine.run(["google_play"], analyze=False, mode="last_30_days")
                steps["play"] = "failed" if gp.errors and gp.fetched == 0 else "done"
                box.write("Collecting Apple App Store")
                apple = engine.run(["apple_app_store"], analyze=False, mode="last_30_days")
                steps["apple"] = "failed" if apple.errors and apple.fetched == 0 else "done"
                steps["save"] = "done"
            result = AnalysisRunResult()
            if not get_settings().has_ai_credentials:
                steps["analyze"] = "failed"
                result.last_error = "OpenRouter API key is not configured. Add OPENROUTER_API_KEY to Streamlit Secrets or .env."
                st.error(result.last_error)
            else:
                try:
                    result = run_analysis_pipeline(db, analyze_limit=smoke_test_analyze_limit(db, settings))
                    steps["analyze"] = "done" if result.analyzed or not result.last_error else "failed"
                except Exception as exc:
                    steps["analyze"] = "failed"
                    result.last_error = _openrouter_error(exc)
                    st.error(result.last_error)
            steps["insights"] = "done" if result.analyzed else "failed"
            steps["dashboard"] = "done"
            st.session_state["pipeline_steps"] = steps
            _load_bundle.clear()
            st.session_state["last_analysis"] = {
                "status": "Connected" if result.analyzed else "Failed",
                "message": result.last_error or f"Analyzed {result.analyzed}, failed {result.failed}.",
            }
            if result.last_error and not result.analyzed:
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
    from app.pipeline.analysis import smoke_test_analyze_limit
    from app.pipeline.orchestrator import run_analysis_pipeline

    reload_settings()
    if not get_settings().has_ai_credentials:
        st.error("OpenRouter API key is not configured. Add OPENROUTER_API_KEY to Streamlit Secrets or .env.")
        return
    db = _db()
    try:
        result = run_analysis_pipeline(
            db,
            analyze_limit=smoke_test_analyze_limit(db, get_settings()),
            only_failed=only_failed,
            include_failed=only_failed,
        )
        if result.analyzed == 0 and result.failed:
            st.error(_openrouter_error(result.last_error or f"OpenRouter analysis failed for {result.failed} reviews."))
        st.session_state["last_analysis"] = {
            "status": "Connected" if result.analyzed else "Configured",
            "message": result.last_error or f"Analyzed {result.analyzed}, failed {result.failed}.",
        }
        _load_bundle.clear()
    except Exception as exc:
        st.error(_openrouter_error(exc))
        return
    finally:
        db.close()
    st.rerun()
