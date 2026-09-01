from __future__ import annotations

from app.ai.schema import extract_json_object, try_validate_analysis, validate_analysis_payload
from app.pipeline.scoring import (
    evidence_confidence_from_sources,
    frequency_from_share,
    opportunity_score,
    purchase_impact_from_hesitation,
)


def test_extract_json_from_fences():
    raw = '```json\n{"relevance": "high", "intent": ["bookmarking"]}\n```'
    payload = extract_json_object(raw)
    assert payload["relevance"] == "high"


def test_malformed_json_is_handled():
    parsed, error = try_validate_analysis("not json at all", "hello")
    assert parsed is None
    assert "Malformed AI JSON" in error


def test_batch_payload_extracts_results():
    from app.ai.schema import parse_batch_payload

    raw = '{"results": [{"id": "1", "relevance": "low"}, {"id": "2", "relevance": "high"}]}'
    items, error = parse_batch_payload(raw)
    assert error == ""
    assert [x["id"] for x in items] == ["1", "2"]


def test_batch_payload_records_parse_error():
    from app.ai.schema import parse_batch_payload

    items, error = parse_batch_payload("definitely not json {")
    assert items == []
    assert "Malformed AI JSON" in error


def test_schema_defaults_and_quote_sanitization():
    payload = {
        "relevance": "medium",
        "wishlist_signal": "explicit",
        "purchase_signal": "hesitant",
        "purchase_hesitation": "explicit",
        "intent": ["future purchase", "future purchase", ""],
        "barriers": ["fit"],
        "uncertainties": ["Will it fit?"],
        "information_seeking": [],
        "behavioral_signals": [
            {
                "signal": "size_checking",
                "basis": "explicit",
                "quote": "this quote is fabricated and not in the review",
            }
        ],
        "root_cause": {
            "observed": "size chart confusing",
            "inferred": "fit confidence is low",
            "hypothesized": "lack of confidence the product will fit",
            "statement": "lack of fit confidence",
        },
        "evidence_strength": 9,
        "confidence": 0,
    }
    original = "The size chart is confusing so I did not order."
    parsed = validate_analysis_payload(payload, original)
    assert parsed.intent == ["future purchase"]
    assert parsed.evidence_strength == 5
    assert parsed.confidence == 1
    assert parsed.behavioral_signals[0].basis == "inferred"
    assert parsed.behavioral_signals[0].quote == ""


def test_wishlist_false_coerces_to_none():
    parsed = validate_analysis_payload({"wishlist_signal": False, "relevance": "low"}, "Nice app")
    assert parsed.wishlist_signal == "none"


def test_compact_json_maps_to_dashboard_schema():
    from app.ai.schema import try_validate_payload

    payload = {
        "problem": "size chart is missing",
        "wishlist_signal": True,
        "purchase_barrier": "size",
        "uncertainty": "Will it fit?",
        "theme": "fit confidence",
        "segment": "apparel shoppers",
        "severity": 4,
        "purchase_impact": 4,
        "evidence_type": "explicit",
        "confidence": 0.8,
    }
    parsed, error = try_validate_payload(payload, "Wishlisted a dress but the size chart is missing.")
    assert error == ""
    assert parsed is not None
    assert parsed.root_cause.statement == "size chart is missing"
    assert parsed.wishlist_signal == "implicit"
    assert parsed.barriers == ["size"]
    assert parsed.uncertainties == ["Will it fit?"]
    assert parsed.intent == ["fit confidence"]
    assert "apparel shoppers" in parsed.decision_factors
    assert parsed.evidence_strength == 4
    assert parsed.confidence == 4
    assert parsed.relevance == "high"
    assert parsed.purchase_signal == "hesitant"


def test_compact_single_object_parses_without_results_array():
    from app.ai.schema import parse_batch_payload

    raw = '{"problem":"price too high","wishlist_signal":true,"purchase_barrier":"price"}'
    items, error = parse_batch_payload(raw)
    assert error == ""
    assert len(items) == 1
    assert items[0]["problem"] == "price too high"


def test_opportunity_score_is_deterministic():
    assert opportunity_score(5, 5, 5, 5, 5) == 3125
    assert opportunity_score(1, 1, 1, 1, 1) == 1
    assert opportunity_score(2, 3, 4, 1, 5) == 2 * 3 * 4 * 1 * 5


def test_frequency_and_confidence_mappings():
    assert frequency_from_share(1, 100) == 1
    assert frequency_from_share(25, 100) == 5
    # Non-Myntra-only evidence cannot get high confidence
    assert evidence_confidence_from_sources(2, 50, has_myntra=False, has_non_myntra_only=True) <= 2
    assert evidence_confidence_from_sources(2, 20, has_myntra=True, has_non_myntra_only=False) >= 4
    assert purchase_impact_from_hesitation(8, 10) == 5
