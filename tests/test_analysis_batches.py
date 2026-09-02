from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from app.ai.provider import AIError
from app.collectors.google_play import GooglePlayCollector
from app.config import Settings
from app.models import Analysis, Review
from app.pipeline.analysis import analyze_new_reviews, reviews_needing_analysis
from app.pipeline.validation import validate_app_identity
from app.schemas import NormalizedReview


def _settings(**kwargs) -> Settings:
    values = {
        "collection_rate_limit_seconds": 0,
        "openrouter_api_key": "unit-test-key",
        "openrouter_model": "google/gemini-2.5-flash",
        "ai_rate_limit_seconds": 0,
        "ai_request_batch_size": 2,
        "ai_batch_size": 2,
        "ai_retry_attempts": 2,
        "ai_analysis_batch_size": 60,
        "ai_max_tokens": 2000,
    }
    values.update(kwargs)
    if "ai_request_batch_size" in values and "ai_batch_size" not in kwargs:
        values["ai_batch_size"] = values["ai_request_batch_size"]
    return Settings(**values)


def _insert(db, source_id: str, text: str) -> Review:
    collector = GooglePlayCollector(settings=_settings())
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.myntra.android",
        detected_app_name="Myntra",
        detected_developer="Myntra Designs Private Limited",
    )
    collector.save_raw(
        db,
        [
            NormalizedReview(
                source="google_play",
                source_review_id=source_id,
                app_id="com.myntra.android",
                app_name="Myntra",
                text=text,
                title="Fit worry",
                rating=3,
                review_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
                is_valid_source=True,
                data_classification="MYNTRA EVIDENCE",
            )
        ],
        validation,
    )
    return db.query(Review).filter(Review.source_review_id == source_id).one()


def _ok_item(review_id: str) -> dict:
    return {
        "id": str(review_id),
        "relevance": "high",
        "wishlist_signal": "implicit",
        "purchase_signal": "hesitant",
        "purchase_hesitation": "explicit",
        "intent": ["future purchase"],
        "barriers": ["size"],
        "uncertainties": ["Will it fit?"],
        "root_cause": {
            "observed": "size chart missing",
            "inferred": "fit confidence is low",
            "hypothesized": "lack of fit confidence",
            "statement": "lack of fit confidence",
        },
        "sentiment": "mixed",
        "evidence_strength": 3,
        "confidence": 3,
    }


class FakeProvider:
    def __init__(self, complete, settings: Settings | None = None) -> None:
        self.settings = settings or _settings()
        self._complete = complete
        self.calls: list[str] = []
        self.last_usage: dict = {}

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model(self) -> str:
        return self.settings.resolved_model

    def available(self) -> bool:
        return True

    def complete_json(self, *, system: str, user: str) -> str:
        self.calls.append(user)
        return self._complete(system=system, user=user)


def test_one_failed_batch_does_not_fail_other_batches(db):
    first = _insert(db, "b1", "Saved to wishlist but size is unclear so I did not buy.")
    second = _insert(db, "b2", "I liked the kurta but did not purchase because returns feel risky.")
    third = _insert(db, "b3", "Bookmarked a dress. Waiting for a better price.")
    fourth = _insert(db, "b4", "Want this jacket but the colour looks different in photos.")

    def complete(*, system, user):
        ids = re.findall(r"REVIEW ID: (\S+)", user)
        if str(first.id) in ids:
            raise AIError("AI provider error HTTP 503. Retrying.")
        return json.dumps({"results": [_ok_item(i) for i in ids]})

    provider = FakeProvider(complete, _settings(ai_request_batch_size=2))
    result = analyze_new_reviews(db, provider=provider, limit=4)
    db.commit()

    assert len(provider.calls) == 2
    assert result.failed == 2
    assert result.analyzed == 2
    assert "503" in result.last_error
    assert first.analysis.status == "failed"
    assert second.analysis.status == "failed"
    assert third.analysis.status == "analyzed"
    assert fourth.analysis.status == "analyzed"
    assert first.analysis.parse_error
    assert "sk-or-" not in (first.analysis.parse_error or "")


def test_malformed_batch_json_marks_only_that_batch_failed(db):
    a = _insert(db, "m1", "Wishlisted sandals but sizing is confusing so I did not order.")
    b = _insert(db, "m2", "Kept a saree in wishlist. Not sure about the fabric.")
    c = _insert(db, "m3", "Added sneakers to wishlist. Delivery time is unclear.")

    def complete(*, system, user):
        ids = re.findall(r"REVIEW ID: (\S+)", user)
        if str(a.id) in ids:
            return "this is not json at all"
        return json.dumps({"results": [_ok_item(i) for i in ids]})

    provider = FakeProvider(complete, _settings(ai_request_batch_size=2))
    result = analyze_new_reviews(db, provider=provider, limit=3)
    db.commit()

    assert a.analysis.status == "failed"
    assert b.analysis.status == "failed"
    assert "Malformed AI JSON" in (a.analysis.parse_error or "")
    assert c.analysis.status == "analyzed"
    assert result.analyzed == 1
    assert result.failed == 2


def test_analyzed_reviews_are_skipped(db):
    row = _insert(db, "skip-me", "Saved a shirt but did not buy because of size doubts.")
    row.analysis.status = "analyzed"
    row.analysis.is_valid_json = True
    row.analysis.content_hash = row.content_hash
    row.analysis.analysis_version = "1"
    db.commit()
    pending = _insert(db, "need-me", "Wishlisted jeans. Not sure they will fit.")

    calls = {"n": 0}

    def complete(*, system, user):
        calls["n"] += 1
        ids = re.findall(r"REVIEW ID: (\S+)", user)
        assert str(row.id) not in ids
        return json.dumps({"results": [_ok_item(i) for i in ids]})

    provider = FakeProvider(complete, _settings(ai_request_batch_size=8))
    result = analyze_new_reviews(db, provider=provider)
    db.commit()
    assert calls["n"] == 1
    assert result.analyzed == 1
    assert result.skipped_already_analyzed >= 1
    assert pending.analysis.status == "analyzed"
    assert row.analysis.status == "analyzed"


def test_smoke_limit_is_remaining_pending_not_one(db):
    from app.pipeline.analysis import smoke_test_analyze_limit

    for i in range(8):
        _insert(db, f"pending-{i}", "Wishlisted a dress. Size chart is missing so I did not buy.")
    db.commit()
    settings = _settings()
    limit = smoke_test_analyze_limit(db, settings)
    assert limit == 8
    assert limit != 1
    assert limit != 5


def test_failed_reviews_are_retried(db):
    row = _insert(db, "retry-me", "Kept a dress in wishlist. Size chart is missing so I did not buy.")
    row.analysis.status = "failed"
    row.analysis.is_valid_json = False
    row.analysis.parse_error = "previous timeout"
    row.analysis.content_hash = row.content_hash
    db.commit()
    assert row.id in {r.id for r in reviews_needing_analysis(db)}

    def complete(*, system, user):
        ids = re.findall(r"REVIEW ID: (\S+)", user)
        return json.dumps({"results": [_ok_item(i) for i in ids]})

    result = analyze_new_reviews(db, provider=FakeProvider(complete, _settings()))
    db.commit()
    assert result.analyzed == 1
    assert row.analysis.status == "analyzed"
    assert db.query(Analysis).filter(Analysis.status == "analyzed").count() == 1


def test_only_failed_skips_pending(db):
    pending = _insert(db, "still-pending", "Wishlisted a kurta. Waiting for a sale.")
    failed = _insert(db, "was-failed", "Saved shoes but size chart is missing.")
    failed.analysis.status = "failed"
    failed.analysis.is_valid_json = False
    failed.analysis.parse_error = "previous timeout"
    failed.analysis.content_hash = failed.content_hash
    db.commit()

    assert pending.id not in {r.id for r in reviews_needing_analysis(db, only_failed=True)}
    assert failed.id in {r.id for r in reviews_needing_analysis(db, only_failed=True)}
    assert failed.id not in {r.id for r in reviews_needing_analysis(db, include_failed=False)}
    assert pending.id in {r.id for r in reviews_needing_analysis(db, include_failed=False)}


def test_http_402_leaves_reviews_pending(db):
    from app.ai.provider import CREDIT_402_MESSAGE

    row = _insert(db, "credit-limit", "Wishlisted a dress but the size chart is missing.")
    db.commit()

    def complete(*, system, user):
        raise AIError(CREDIT_402_MESSAGE, http_status=402)

    result = analyze_new_reviews(db, provider=FakeProvider(complete, _settings()), limit=1)
    db.commit()
    assert result.analyzed == 0
    assert result.failed == 0
    assert result.last_http_status == 402
    assert "credits" in (result.last_error or "").lower() or "max_tokens" in (result.last_error or "").lower()
    assert row.analysis.status == "pending"


def test_insights_are_blocked_when_analysis_fails(db, monkeypatch):
    from app.pipeline.analysis import AnalysisRunResult
    from app.pipeline.orchestrator import run_analysis_pipeline

    called = {"themes": 0}

    def boom(_session):
        called["themes"] += 1
        raise AssertionError("themes must not run when analysis failed")

    monkeypatch.setattr("app.pipeline.orchestrator.discover_themes", boom)
    monkeypatch.setattr(
        "app.pipeline.orchestrator.analyze_new_reviews",
        lambda *a, **k: AnalysisRunResult(
            analyzed=0,
            failed=5,
            last_error="OpenRouter analysis failed: invalid request (HTTP 400).",
        ),
    )
    result = run_analysis_pipeline(db, analyze_limit=5)
    assert result.analyzed == 0
    assert result.failed == 5
    assert called["themes"] == 0
