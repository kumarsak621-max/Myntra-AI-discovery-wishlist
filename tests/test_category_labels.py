from __future__ import annotations

from datetime import datetime, timezone

from app.models import Analysis, Review
from app.pipeline.labels import (
    UNCATEGORIZED,
    merge_category_rows,
    normalize_category_label,
    normalize_label_list,
    validate_chart_categories,
)


def test_normalize_category_label_maps_placeholders():
    samples = [
        None,
        "",
        " ",
        "none",
        "None",
        "NONE",
        "null",
        "NULL",
        "N/A",
        "n/a",
        "NA",
        "na",
        "unknown",
        "Unknown",
        "undefined",
        "none (2)",
        "none (3)",
        "none (4)",
        "None (2)",
    ]
    for value in samples:
        assert normalize_category_label(value) == UNCATEGORIZED, value


def test_normalize_keeps_real_labels():
    assert normalize_category_label("Size & Fit") == "Size & Fit"
    assert normalize_category_label("  Price  ") == "Price"
    assert normalize_category_label("Delivery") == "Delivery"
    assert normalize_category_label("Returns") == "Returns"
    assert normalize_category_label("Quality") == "Quality"


def test_normalize_label_list_drops_dummy_none_when_real_exists():
    assert normalize_label_list(["Price", "none", "none (2)"]) == ["Price"]
    assert normalize_label_list(["none", "None", "none (3)"]) == [UNCATEGORIZED]
    assert normalize_label_list([]) == []


def test_merge_collapses_duplicate_missing_and_keeps_ids():
    rows = [
        {"label": "none", "count": 5, "review_ids": [1, 2]},
        {"label": "none (2)", "count": 3, "review_ids": [3]},
        {"label": "none (3)", "count": 2, "review_ids": [4, 5]},
        {"label": "Size & Fit", "count": 4, "review_ids": [6]},
    ]
    merged = merge_category_rows(rows)
    labels = [r["label"] for r in merged]
    assert labels.count(UNCATEGORIZED) == 1
    assert "none (2)" not in labels
    uncat = next(r for r in merged if r["label"] == UNCATEGORIZED)
    assert uncat["count"] == 10
    assert uncat["review_ids"] == [1, 2, 3, 4, 5]


def test_validate_chart_categories_flags_raw_missing():
    issues = validate_chart_categories(["none", "none (2)", "Size & Fit", UNCATEGORIZED])
    assert "none" in issues
    assert "none (2)" in issues
    assert UNCATEGORIZED not in issues
    assert not validate_chart_categories([UNCATEGORIZED, "Size & Fit"])


def test_chart_frame_aggregates_before_render():
    from dashboard.charts import _frame, trend_frame

    df = _frame(
        [
            {"label": "none", "count": 14},
            {"label": "none (2)", "count": 13},
            {"label": "none (3)", "count": 9},
            {"label": "none (4)", "count": 6},
        ]
    )
    assert list(df["label"]) == [UNCATEGORIZED]
    assert int(df["count"].sum()) == 42
    assert validate_chart_categories(list(df["label"])) == []

    trend = trend_frame(
        [
            {"day": "2026-08-01", "theme": "none", "count": 2},
            {"day": "2026-08-01", "theme": "none (2)", "count": 3},
            {"day": "2026-08-02", "theme": "None", "count": 1},
        ]
    )
    assert set(trend["theme"]) == {UNCATEGORIZED}
    assert int(trend.loc[trend["day"] == "2026-08-01", "count"].sum()) == 5


def _analyzed_review(db, source_id: str, *, barriers: str, root_cause: str = "") -> Review:
    row = Review(
        source="google_play",
        source_review_id=source_id,
        app_id="com.myntra.android",
        app_name="Myntra",
        text="Wishlisted this dress but did not buy because of sizing.",
        title="",
        rating=3,
        review_date=datetime(2026, 8, 20, tzinfo=timezone.utc),
        is_valid_source=True,
        is_empty=False,
        is_synthetic=False,
        data_classification="MYNTRA EVIDENCE",
        content_hash=source_id,
    )
    db.add(row)
    db.flush()
    db.add(
        Analysis(
            review_id=row.id,
            content_hash=source_id,
            status="analyzed",
            is_valid_json=True,
            relevance="high",
            root_cause=root_cause,
            barriers_json=barriers,
            uncertainties_json="[]",
            intent_json="[]",
        )
    )
    db.commit()
    return row


def test_label_distribution_merges_placeholder_barriers(db):
    from app.pipeline.quantification import label_distribution, root_cause_distribution

    _analyzed_review(db, "n1", barriers='["none"]', root_cause="none")
    _analyzed_review(db, "n2", barriers='["none (2)"]', root_cause="None")
    _analyzed_review(db, "n3", barriers='["null"]', root_cause="")
    _analyzed_review(db, "n4", barriers='["Size & Fit"]', root_cause="size chart missing")
    barriers = label_distribution(db, "barriers", myntra_only=True, relevant_only=False)
    labels = [r["label"] for r in barriers]
    assert labels.count(UNCATEGORIZED) == 1
    assert "none (2)" not in labels
    uncat = next(r for r in barriers if r["label"] == UNCATEGORIZED)
    assert uncat["count"] == 3
    assert len(uncat["review_ids"]) == 3
    assert any(r["label"] == "Size & Fit" for r in barriers)

    roots = root_cause_distribution(db, myntra_only=True)
    root_labels = [r["label"] for r in roots]
    assert root_labels.count(UNCATEGORIZED) == 1
    assert "none (2)" not in root_labels
    assert "none" not in root_labels
    assert any(r["label"] == "size chart missing" for r in roots)


def test_clustering_merges_none_theme_names(db):
    from app.pipeline.clustering import discover_themes

    for i in range(2):
        _analyzed_review(db, f"c{i}", barriers='["none"]' if i == 0 else '["none (2)"]', root_cause="none")
    themes = discover_themes(db)
    names = [t.name for t in themes]
    assert "none (2)" not in names
    assert "none (3)" not in names
    assert names.count("none") == 0
    assert names.count(UNCATEGORIZED) == 1
    assert themes[0].review_count == 2
    assert themes[0].evidence_ids_json


def test_compact_none_problem_does_not_invent_root_cause():
    from app.ai.schema import try_validate_payload

    parsed, error = try_validate_payload(
        {"problem": "none", "theme": "none", "purchase_barrier": "none", "evidence_type": "none"},
        "Nice app",
    )
    assert error == ""
    assert parsed is not None
    assert parsed.root_cause.statement == ""
    assert parsed.barriers == []
    assert parsed.intent == []
