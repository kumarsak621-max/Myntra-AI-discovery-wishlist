"""Streamlit charts from real aggregate rows. Empty input never invents values."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

EMPTY = "Insufficient evidence for this visualization."


def _frame(rows: list[dict[str, Any]] | None, *, label="label", count="count") -> pd.DataFrame:
    data = []
    for row in rows or []:
        name = str(row.get(label) or row.get("problem") or row.get("name") or "").strip()
        value = row.get(count) or row.get("frequency") or row.get("review_count") or 0
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        if name and value > 0:
            data.append({"label": name[:48], "count": value})
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
    data = []
    for row in rows or []:
        label = str(row.get("label") or row.get("barrier") or row.get("problem") or "").strip()
        freq = int(row.get("count") or row.get("frequency") or 0)
        impact = int(row.get("purchase_impact") or row.get("impact") or row.get("hesitant_count") or 0)
        if label and freq:
            data.append({"label": label[:40], "frequency": freq, "impact": impact})
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
