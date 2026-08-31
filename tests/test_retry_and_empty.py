from __future__ import annotations

from app.collectors.base_collector import with_retry
from app.pipeline.validation import validate_app_identity


def test_retry_eventually_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("nope")
        return "ok"

    assert with_retry(flaky, attempts=3, backoff=0.01, retry_on=(TimeoutError,), label="flaky") == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_attempts():
    def always():
        raise ConnectionError("down")

    try:
        with_retry(always, attempts=2, backoff=0.01, retry_on=(ConnectionError,), label="down")
        raise AssertionError("should have failed")
    except ConnectionError as exc:
        assert "down" in str(exc)


def test_unknown_app_is_not_silently_myntra():
    result = validate_app_identity(
        platform="apple_app_store",
        app_id="1",
        detected_app_name="Random Shopping",
        detected_developer="Acme",
    )
    assert result.is_valid_for_myntra is False
    assert result.validation_status == "INVALID_FOR_MYNTRA_ANALYSIS"
