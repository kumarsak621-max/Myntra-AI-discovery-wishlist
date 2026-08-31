from __future__ import annotations

from datetime import datetime, timezone

from app.collectors.google_play import GooglePlayCollector
from app.config import Settings


class FakeGPS:
    class Sort:
        NEWEST = "newest"

    def __init__(self, pages):
        self.pages = list(pages)
        self.app_calls = []
        self.review_calls = []

    def app(self, app_id, lang=None, country=None):
        self.app_calls.append({"app_id": app_id, "lang": lang, "country": country})
        return {
            "title": "Myntra",
            "developer": "Myntra Designs Private Limited",
            "score": 4.2,
            "url": "https://play.google.com/store/apps/details?id=" + app_id,
        }

    def reviews(self, app_id, **kwargs):
        self.review_calls.append({"app_id": app_id, **kwargs})
        if not self.pages:
            return [], None
        page = self.pages.pop(0)
        token = "next" if self.pages else None
        return page, token


def test_google_play_paginates_and_normalizes(db):
    page1 = [
        {
            "reviewId": "r1",
            "content": "I saved items but did not buy because size is unclear.",
            "score": 2,
            "at": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "reviewCreatedVersion": "1.2",
            "userName": "should-not-appear",
        }
    ]
    page2 = [
        {
            "reviewId": "r2",
            "content": "Wishlist is full of dresses I might buy later.",
            "score": 4,
            "at": datetime(2024, 2, 2, tzinfo=timezone.utc),
        }
    ]
    fake = FakeGPS([page1, page2])
    settings = Settings(
        google_play_app_id="com.myntra.android",
        google_play_batch_size=1,
        google_play_max_reviews=10,
        collection_rate_limit_seconds=0,
        collection_retry_attempts=1,
    )
    collector = GooglePlayCollector(settings=settings, scraper=fake)
    reviews = collector.collect(max_reviews=2)
    assert len(reviews) == 2
    assert len(fake.review_calls) == 2
    assert reviews[0].source_review_id == "r1"
    assert reviews[0].app_id == "com.myntra.android"
    assert reviews[0].app_name == "Myntra"
    assert reviews[0].is_valid_source is True
    assert reviews[0].data_classification == "MYNTRA EVIDENCE"
    assert "should-not-appear" not in str(reviews[0].raw_payload)
    stats = collector.save_raw(db, reviews, collector.last_validation)
    assert stats.fetched == 2
    assert stats.new == 2
    assert stats.valid == 2


def test_banned_blinkit_id_is_not_collected():
    fake = FakeGPS(
        [
            [
                {
                    "reviewId": "r1",
                    "content": "should not be stored as myntra",
                    "score": 1,
                }
            ]
        ]
    )
    settings = Settings(
        google_play_app_id="com.grofers.customerapp",
        collection_rate_limit_seconds=0,
        collection_retry_attempts=1,
    )
    collector = GooglePlayCollector(settings=settings, scraper=fake)
    reviews = collector.collect(max_reviews=5)
    assert reviews == []
    assert collector.last_validation is not None
    assert collector.last_validation.is_valid_for_myntra is False
    assert collector.last_validation.validation_result == "FAIL"
    assert fake.review_calls == []


def test_google_play_metadata_failure_does_not_crash():
    class Boom:
        class Sort:
            NEWEST = 1

        def app(self, *args, **kwargs):
            raise ConnectionError("play down")

        def reviews(self, *args, **kwargs):
            raise AssertionError("should not fetch reviews if we only validate")

    settings = Settings(google_play_app_id="com.grofers.customerapp", collection_retry_attempts=1)
    collector = GooglePlayCollector(settings=settings, scraper=Boom())
    validation = collector.validate_source()
    assert validation.validation_status == "ERROR"
    assert "play down" in validation.warning
    assert validation.app_id == "com.grofers.customerapp"


def test_google_play_stops_after_page_older_than_cutoff():
    from datetime import timedelta

    now = datetime(2026, 8, 31, tzinfo=timezone.utc)
    cutoff = now - timedelta(days=30)
    recent = {
        "reviewId": "recent",
        "content": "size chart is confusing so I did not buy",
        "score": 2,
        "at": now - timedelta(days=3),
    }
    old = {
        "reviewId": "old",
        "content": "old review outside window",
        "score": 5,
        "at": now - timedelta(days=45),
    }
    extra = {
        "reviewId": "should-not-fetch",
        "content": "should not be requested",
        "score": 4,
        "at": now - timedelta(days=50),
    }
    fake = FakeGPS([[recent], [old], [extra]])
    settings = Settings(
        google_play_app_id="com.myntra.android",
        google_play_batch_size=1,
        collection_rate_limit_seconds=0,
        collection_retry_attempts=1,
    )
    collector = GooglePlayCollector(settings=settings, scraper=fake)
    reviews = collector.collect(safety_limit=50, stop_when_older_than=cutoff)
    ids = [r.source_review_id for r in reviews]
    assert "recent" in ids
    assert "old" in ids
    assert "should-not-fetch" not in ids
    assert len(fake.review_calls) == 2

