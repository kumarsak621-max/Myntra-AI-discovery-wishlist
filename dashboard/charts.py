"""Streamlit charts from real aggregate rows. Empty input never invents values."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any

import pandas as pd
import streamlit as st

from app.pipeline.labels import merge_category_rows, normalize_category_label, validate_chart_categories

logger = logging.getLogger(__name__)
EMPTY = "Insufficient evidence for this visualization."


def _assert_clean(labels: list[str]) -> None:
    issues = validate_chart_categories(labels)
    if issues:
        logger.warning("Chart still has duplicate missing labels: %s", issues)


def _frame(rows: list[dict[str, Any]] | None, *, label="label", count="count") -> pd.DataFrame:
    prepared = []
    for row in rows or []:
        name = normalize_category_label(row.get(label) or row.get("problem") or row.get("name") or row.get("root_cause"))
        value = row.get(count) or row.get("frequency") or row.get("review_count") or 0
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            prepared.append({"label": name, "count": value, "review_ids": row.get("review_ids") or row.get("evidence_ids") or []})
    merged = merge_category_rows(prepared)
    data = [{"label": item["label"], "count": int(item.get("count") or 0)} for item in merged if int(item.get("count") or 0) > 0]
    _assert_clean([item["label"] for item in data])
    return pd.DataFrame(data)


def bar_chart(rows: list[dict[str, Any]] | None, *, empty: str = EMPTY, height: int = 260) -> None:
    df = _frame(rows)
    if df.empty:
        st.info(empty)
        return
    try:
        st.bar_chart(df, x="label", y="count", horizontal=True, height=height)
    except TypeError:
        st.bar_chart(df.set_index("label")["count"], height=height)


def scatter_chart(rows: list[dict[str, Any]] | None, *, empty: str = EMPTY) -> None:
    data = []
    for row in rows or []:
        freq = row.get("frequency") or row.get("count") or 0
        impact = row.get("impact") or row.get("purchase_impact") or 0
        size = row.get("count") or row.get("frequency") or freq
        if freq and impact:
            data.append({"frequency": int(freq), "impact": int(impact), "count": int(size)})
    df = pd.DataFrame(data)
    if df.empty:
        st.info(empty)
        return
    st.scatter_chart(df, x="frequency", y="impact", size="count", height=260)


def donut_chart(rows: list[dict[str, Any]] | None, *, empty: str = EMPTY) -> None:
    df = _frame(rows)
    if df.empty:
        st.info(empty)
        return
    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("label:N", legend=alt.Legend(orient="bottom")),
                tooltip=["label", "count"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(df.set_index("label")["count"], height=260)


def heatmap_impact_frequency(rows: list[dict[str, Any]] | None, *, empty: str = EMPTY) -> None:
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"frequency": 0, "impact": 0})
    for row in rows or []:
        label = normalize_category_label(row.get("label") or row.get("barrier") or row.get("problem") or "")
        freq = int(row.get("count") or row.get("frequency") or 0)
        impact = int(row.get("purchase_impact") or row.get("impact") or row.get("hesitant_count") or 0)
        if freq:
            buckets[label]["frequency"] += freq
            buckets[label]["impact"] += impact
    data = [
        {"label": name, "frequency": vals["frequency"], "impact": vals["impact"]}
        for name, vals in buckets.items()
        if vals["frequency"] > 0
    ]
    _assert_clean([item["label"] for item in data])
    df = pd.DataFrame(data)
    if df.empty:
        st.info(empty)
        return
    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("impact:O", title="Purchase impact"),
                y=alt.Y("label:N", title=""),
                color=alt.Color("frequency:Q", title="Frequency"),
                tooltip=["label", "frequency", "impact"],
            )
            .properties(height=max(180, 28 * len(df)))
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(df, hide_index=True, width="stretch")


def trend_frame(rows: list[dict[str, Any]] | None, *, label_key: str = "theme") -> pd.DataFrame:
    """Normalize series names then sum identical (day, label) pairs before a trend chart."""
    buckets: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows or []:
        day = str(row.get("day") or "").strip()
        name = normalize_category_label(row.get(label_key) or row.get("label") or row.get("theme"))
        try:
            value = int(row.get("count") or 0)
        except (TypeError, ValueError):
            value = 0
        if day and value > 0:
            buckets[(day, name)] += value
    data = [{"day": day, label_key: name, "count": count} for (day, name), count in sorted(buckets.items())]
    _assert_clean([item[label_key] for item in data])
    return pd.DataFrame(data)
