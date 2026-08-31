"""Streamlit dashboard that reuses collectors, analysis, and SQLite.

Launched by `streamlit run app.py`. Does not duplicate business logic.
"""

from __future__ import annotations

import traceback

import streamlit as st
from sqlalchemy.orm import Session

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
from app.database import SessionLocal, init_db
from app.models import Analysis, Opportunity, Review, Segment, Theme
from app.api.routes import serialize_review
from app.pipeline.quantification import label_distribution, overview_metrics
from app.pipeline.report import build_report, evidence_cards


PAGES = [
    "Overview",
    "Data Collection",
    "Feedback Explorer",
    "Wishlist Motivations",
    "Purchase Barriers",
    "Uncertainties",
    "Root Causes",
    "Themes",
    "User Segments",
    "Opportunity Matrix",
    "Evidence Explorer",
    "Discovery Report",
]

EMPTY = "No data collected yet. Run the data collection pipeline."


@st.cache_resource
def _bootstrap() -> bool:
    init_db()
    return True


def _db() -> Session:
    return SessionLocal()


def _safe_error(exc: Exception) -> None:
    st.error(str(exc) or "Something went wrong.")
    if st.session_state.get("debug"):
        st.code(traceback.format_exc())


def render() -> None:
    st.set_page_config(page_title="Myntra Discovery Engine", layout="wide")
    _bootstrap()
    reload_settings()
    settings = get_settings()

    st.sidebar.title("Wishlist → Purchase")
    st.sidebar.caption("Discovers the problem. Does not propose the feature.")
    page = st.sidebar.radio("Section", PAGES, index=0)
    myntra_only = st.sidebar.checkbox("Myntra-valid evidence only", value=True)
    st.session_state["debug"] = st.sidebar.checkbox("Show debug traces", value=False)

    ai_ok = settings.has_ai_credentials
    st.sidebar.markdown("**AI**")
    st.sidebar.write("OpenRouter key:", "configured" if ai_ok else "missing")
    st.sidebar.write("Model:", settings.resolved_model)
    st.sidebar.caption("SQLite storage is ephemeral on Streamlit Cloud.")

    try:
        if page == "Overview":
            _overview(myntra_only)
        elif page == "Data Collection":
            _collection(settings, ai_ok)
        elif page == "Feedback Explorer":
            _explorer(myntra_only)
        elif page == "Wishlist Motivations":
            _labels("intent", myntra_only, "Wishlist Motivations")
        elif page == "Purchase Barriers":
            _labels("barriers", myntra_only, "Purchase Barriers")
        elif page == "Uncertainties":
            _labels("uncertainties", myntra_only, "Uncertainties")
        elif page == "Root Causes":
            _root_causes()
        elif page == "Themes":
            _named(Theme, "Themes")
        elif page == "User Segments":
            _named(Segment, "User Segments")
        elif page == "Opportunity Matrix":
            _opportunities()
        elif page == "Evidence Explorer":
            _evidence()
        elif page == "Discovery Report":
            _report()
    except Exception as exc:
        _safe_error(exc)


def _validation_display(block: dict) -> str:
    if not block.get("last_collection"):
        return "not collected"
    return str(block.get("validation") or "unknown")


def _overview(myntra_only: bool) -> None:
    st.title("Overview")
    st.caption("Filter: Myntra-valid evidence only" if myntra_only else "Filter: all stored records")
    db = _db()
    try:
        from app.models import Source

        data = overview_metrics(db)
        rows = db.query(Source).all()
    finally:
        db.close()
    if data["total_reviews"] == 0:
        st.info(EMPTY)
        if not rows:
            return
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total reviews", data["total_reviews"])
        c2.metric("Myntra-valid", data["myntra_reviews"])
        c3.metric("Reference / non-Myntra", data["reference_non_myntra_reviews"])
        c4.metric("Relevant analyzed", data["relevant_reviews"])
        st.caption("Percentages use count / appropriate denominator. The LLM does not compute them.")
        if data["synthetic_count"]:
            st.warning(
                "SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA is present and must stay labelled."
            )
    st.subheader("Sources")
    if not rows:
        st.info(EMPTY)
        return
    st.dataframe(
        [
            {
                "platform": s.platform,
                "app_id": s.app_id,
                "detected_app": s.detected_app_name,
                "status": s.validation_status,
                "myntra_valid": s.is_valid_for_myntra,
                "reviews": s.review_count,
            }
            for s in rows
        ],
        width="stretch",
    )


def _collection(settings, ai_ok: bool) -> None:
    st.title("Data Collection")
    st.write("Google Play package:", OFFICIAL_GOOGLE_PLAY_APP_ID)
    st.write("Google Play URL:", OFFICIAL_GOOGLE_PLAY_URL)
    st.write("Apple App ID:", OFFICIAL_APPLE_APP_ID)
    st.write("Apple URL:", OFFICIAL_APPLE_APP_URL)
    st.write("Apple regions: India first, US fallback. US reviews stay labelled as US.")

    db = _db()
    try:
        from app.api.routes import collection_status

        status = collection_status(db)
    except Exception as exc:
        _safe_error(exc)
        db.close()
        return
    db.close()

    gp = status["google_play"]
    ap = status["apple_app_store"]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Google Play")
        st.write("App:", OFFICIAL_GOOGLE_PLAY_APP_NAME)
        st.write("Detected:", gp.get("detected_app") or "—")
        st.write("Validation:", _validation_display(gp))
        st.write("Reviews collected:", gp.get("reviews_collected", 0))
        st.write("Last collection:", gp.get("last_collection") or "never")
        if gp.get("warning"):
            st.error(gp["warning"])
        play_btn = st.button("Collect Google Play Reviews")
    with col2:
        st.subheader("Apple App Store")
        st.write("App:", OFFICIAL_APPLE_APP_NAME)
        st.write("Detected:", ap.get("detected_app") or "—")
        st.write("Validation:", _validation_display(ap))
        st.write("Reviews collected:", ap.get("reviews_collected", 0))
        st.write("Last collection:", ap.get("last_collection") or "never")
        if ap.get("warning"):
            st.error(ap["warning"])
        apple_btn = st.button("Collect Apple App Store Reviews")

    max_reviews = st.number_input("Max reviews per source", min_value=10, max_value=200, value=25, step=5)
    analyze = st.checkbox("Run AI analysis after collection", value=False)
    if analyze and not ai_ok:
        st.warning(
            "OPENROUTER_API_KEY is not set in Streamlit Secrets or .env. "
            "Collection will still run; analysis will be skipped."
        )
        analyze = False
    all_btn = st.button("Collect All")
    analyze_btn = st.button("Analyze stored Myntra-valid reviews")
    st.caption(
        "On Streamlit Cloud keep this number small; large runs can time out. "
        "Dedup uses source review IDs and content hashes, so reruns do not re-analyze unchanged text. "
        "AI is only called from these buttons, not on every page rerun."
    )

    sources = None
    if play_btn:
        sources = ["google_play"]
    elif apple_btn:
        sources = ["apple_app_store"]
    elif all_btn:
        sources = ["google_play", "apple_app_store"]
    if sources:
        _run_collect(sources, int(max_reviews), analyze)
    if analyze_btn:
        if not ai_ok:
            st.error("OPENROUTER_API_KEY is not set in Streamlit Secrets or .env.")
        else:
            _run_analyze()


def _run_collect(sources: list[str], max_reviews: int, analyze: bool) -> None:
    from app.collectors.engine import CollectionEngine
    from config.settings import reload_settings

    reload_settings()
    db = _db()
    try:
        with st.spinner("Collecting public reviews and validating Myntra identity…"):
            engine = CollectionEngine(db)
            stats = engine.run(sources, max_reviews=max_reviews, analyze=analyze)
        st.success(
            f"Fetched {stats.fetched} · Myntra-valid {stats.valid} · new {stats.new} · "
            f"duplicates {stats.duplicates} · rejected {stats.rejected} · analyzed {stats.analyzed}"
        )
        if stats.errors:
            for err in stats.errors:
                st.error(err)
        for item in stats.source_validations:
            label = "PASS" if item.is_valid_for_myntra else "FAIL"
            st.write(f"{item.platform}: {item.detected_app_name} · {label}")
            if not item.is_valid_for_myntra and item.warning:
                st.error(item.warning)
        if analyze and stats.new and stats.analyzed == 0 and not stats.errors:
            st.warning("Collection finished but no reviews were analyzed. Check the OpenRouter key or retry Analyze.")
    except Exception as exc:
        _safe_error(exc)
    finally:
        db.close()


def _run_analyze() -> None:
    from app.pipeline.orchestrator import run_analysis_pipeline
    from config.settings import reload_settings

    reload_settings()
    db = _db()
    try:
        with st.spinner("Analyzing stored Myntra-valid reviews…"):
            analyzed = run_analysis_pipeline(db)
        if analyzed == 0:
            st.info("No new Myntra-valid reviews needed analysis (already analyzed, or none stored).")
        else:
            st.success(
                f"Analyzed {analyzed} reviews. Themes, segments, and opportunity scores were rebuilt."
            )
    except Exception as exc:
        _safe_error(exc)
    finally:
        db.close()


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
            st.info(EMPTY)
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
    db = _db()
    try:
        rows = label_distribution(db, field, myntra_only=myntra_only)
    finally:
        db.close()
    if not rows:
        st.info(EMPTY)
        return
    st.dataframe(rows, width="stretch")


def _root_causes() -> None:
    st.title("Root Causes")
    db = _db()
    try:
        rows = (
            db.query(Analysis)
            .join(Review)
            .filter(
                Review.is_valid_source.is_(True),
                Analysis.is_valid_json.is_(True),
            )
            .all()
        )
        items = [r for r in rows if (r.root_cause or "").strip()]
        if not items:
            st.info(EMPTY)
            return
        for row in items[:80]:
            st.write(row.root_cause)
            st.caption(
                f"observed: {row.root_cause_observed or '—'} · "
                f"inferred: {row.root_cause_inferred or '—'} · "
                f"hypothesized: {row.root_cause_hypothesized or '—'}"
            )
    finally:
        db.close()


def _opportunities() -> None:
    st.title("Opportunity Matrix")
    st.caption("Score = reach × frequency × purchase impact × severity × evidence confidence (each 1–5). Calculated in Python.")
    db = _db()
    try:
        rows = db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
        if not rows:
            st.info(EMPTY)
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
    finally:
        db.close()


def _named(model, title: str) -> None:
    st.title(title)
    db = _db()
    try:
        rows = db.query(model).all()
        if not rows:
            st.info(EMPTY)
            return
        for row in rows:
            st.subheader(row.name)
            st.write(getattr(row, "description", "") or "")
            st.caption(f"{row.review_count} reviews · Myntra {getattr(row, 'myntra_review_count', 0)}")
    finally:
        db.close()


def _evidence() -> None:
    st.title("Evidence Explorer")
    db = _db()
    try:
        rows = db.query(Opportunity).order_by(Opportunity.rank.asc()).all()
        if not rows:
            st.info(EMPTY)
            return
        choice = st.selectbox("Opportunity", [f"{o.rank}. {o.user_problem}" for o in rows])
        rank = int(str(choice).split(".", 1)[0])
        selected = next((o for o in rows if o.rank == rank), rows[0])
        import json

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
                f"{card['source']} · {card['app_name']} · {card['data_classification']} · "
                f"{card['source_review_id']} · {card['date']} · {card['source_url']}"
            )
    finally:
        db.close()


def _report() -> None:
    st.title("Discovery Report")
    db = _db()
    try:
        report = build_report(db)
    finally:
        db.close()
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
    st.write(primary.get("why_first") or "Collect and analyze Myntra-valid reviews first.")
    top = report.get("top_opportunities") or []
    for item in top:
        st.markdown(f"**#{item['rank']} {item['opportunity']}** — score {item['opportunity_score']}")
        for ev in item.get("evidence") or []:
            st.write(f"“{ev['quote']}”")
            st.caption(f"{ev['source']} · {ev['data_classification']} · {ev['source_url']}")
