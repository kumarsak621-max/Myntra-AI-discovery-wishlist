from dashboard.pipeline_status import derive_failed_reason, insights_status_for_analyze


def test_insights_partial_when_analyze_partial():
    assert insights_status_for_analyze("partial", 40) == "partial"
    assert insights_status_for_analyze("partial", 0) == "insufficient"


def test_insights_failed_when_analyze_failed_with_no_data():
    assert insights_status_for_analyze("failed", 0) == "failed"
    assert insights_status_for_analyze("failed", 12) == "partial"


def test_insights_insufficient_without_analyzed_reviews():
    assert insights_status_for_analyze("done", 0) == "insufficient"
    assert insights_status_for_analyze("insufficient", 0) == "insufficient"
    assert insights_status_for_analyze("done", 20) == "done"


def test_failed_reason_is_none_on_success():
    assert (
        derive_failed_reason(
            steps={"play": "done", "apple": "no_new", "save": "done", "analyze": "done", "insights": "done"},
            last_analysis={"message": "Analyzed 10, failed 0."},
        )
        is None
    )


def test_failed_reason_comes_from_pipeline_result():
    reason = derive_failed_reason(
        steps={"analyze": "partial"},
        pipeline_result={"failed_reason": "Malformed AI JSON: No JSON object found in AI response"},
    )
    assert "Malformed AI JSON" in reason


def test_failed_reason_defined_when_no_state():
    assert derive_failed_reason(steps=None) is None
    assert derive_failed_reason(steps={}) is None
