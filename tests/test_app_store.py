from __future__ import annotations

import httpx

from app.collectors.app_store import AppStoreCollector
from app.config import Settings


def _feed(reviews):
    entries = [{"im:name": {"label": "App"}}]
    for rev in reviews:
        entries.append(
            {
                "id": {"label": rev["id"]},
                "title": {"label": rev.get("title", "title")},
                "content": {"label": rev.get("content", "")},
                "im:rating": {"label": str(rev.get("rating", 5))},
                "im:version": {"label": "1.0"},
                "updated": {"label": "2024-05-01T10:00:00Z"},
                "author": {"name": {"label": "hidden"}},
                "link": {"attributes": {"href": f"https://example.test/{rev['id']}"}},
            }
        )
    return {"feed": {"entry": entries}}


def _lookup(name="Myntra Fashion Shopping App"):
    return {"results": [{"trackName": name, "artistName": "Myntra Designs Private Limited", "bundleId": "com.myntra.android"}]}


def test_india_then_us_fallback_stores_us_region():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "lookup" in url and "country=in" in url:
            return httpx.Response(200, json=_lookup())
        if "lookup" in url and "country=us" in url:
            return httpx.Response(200, json=_lookup())
        if "/in/rss/" in url:
            # Ratings only — no written body
            return httpx.Response(
                200,
                json=_feed([{"id": "in1", "title": "", "content": "", "rating": 5}]),
            )
        if "/us/rss/" in url:
            return httpx.Response(
                200,
                json=_feed(
                    [
                        {
                            "id": "us1",
                            "title": "Size chart",
                            "content": "I added to wishlist but size chart is confusing so I did not buy.",
                            "rating": 2,
                        }
                    ]
                ),
            )
        return httpx.Response(404, json={"error": url})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    settings = Settings(
        apple_app_id="907394059",
        apple_primary_region="in",
        apple_fallback_region="us",
        collection_rate_limit_seconds=0,
        collection_retry_attempts=1,
        apple_max_reviews=50,
    )
    collector = AppStoreCollector(settings=settings, client=client)
    reviews = collector.collect(max_reviews=10)
    assert collector.fallback_used is True
    assert collector.region_used == "us"
    assert len(reviews) == 1
    assert reviews[0].region == "us"
    assert reviews[0].region != "in"
    assert reviews[0].is_valid_source is True
    assert "size chart" in reviews[0].text.lower()


def test_written_india_reviews_do_not_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "lookup" in url:
            return httpx.Response(200, json=_lookup())
        if "/in/rss/" in url:
            return httpx.Response(
                200,
                json=_feed(
                    [{"id": "in9", "title": "Fit", "content": "Will this kurta fit me? Saved for later.", "rating": 3}]
                ),
            )
        if "/us/rss/" in url:
            raise AssertionError("US feed should not be requested")
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(collection_rate_limit_seconds=0, collection_retry_attempts=1)
    collector = AppStoreCollector(settings=settings, client=client)
    reviews = collector.collect(max_reviews=5)
    assert collector.fallback_used is False
    assert reviews[0].region == "in"


def test_empty_json_feed_falls_back_to_xml_same_region():
    atom = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:im="http://itunes.apple.com/rss">
  <entry>
    <id>app-meta</id>
    <title>Myntra Fashion Shopping App</title>
  </entry>
  <entry>
    <id>xml-in-1</id>
    <title>Size</title>
    <content type="text">Wishlisted but size is uncertain so I did not buy.</content>
    <updated>2026-08-20T10:00:00Z</updated>
    <im:rating>2</im:rating>
    <im:version>4.0</im:version>
    <author><name>hidden</name></author>
    <link href="https://example.test/xml-in-1"/>
  </entry>
</feed>
"""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "lookup" in url:
            return httpx.Response(200, json=_lookup())
        if "/us/rss/" in url:
            raise AssertionError("US feed should not be requested when India XML has written reviews")
        if "/in/rss/" in url and url.rstrip("/").endswith("json"):
            return httpx.Response(200, json={"feed": {"entry": [{"im:name": {"label": "Myntra"}}]}})
        if "/in/rss/" in url and url.rstrip("/").endswith("xml"):
            return httpx.Response(200, text=atom)
        return httpx.Response(404, text=url)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(collection_rate_limit_seconds=0, collection_retry_attempts=1)
    collector = AppStoreCollector(settings=settings, client=client)
    reviews = collector.collect(max_reviews=5)
    assert collector.fallback_used is False
    assert collector.region_used == "in"
    assert len(reviews) == 1
    assert reviews[0].source_review_id == "xml-in-1"
    assert reviews[0].region == "in"
    assert "size is uncertain" in reviews[0].text.lower()


def test_json_list_content_is_normalized():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "lookup" in url:
            return httpx.Response(200, json=_lookup())
        if "/in/rss/" in url and url.rstrip("/").endswith("json"):
            return httpx.Response(
                200,
                json={
                    "feed": {
                        "entry": [
                            {"im:name": {"label": "Myntra"}},
                            {
                                "id": {"label": "list-1"},
                                "title": {"label": "Fit"},
                                "content": [{"label": "Added to wishlist. Need a size chart.", "attributes": {"type": "text"}}],
                                "im:rating": {"label": "3"},
                                "im:version": {"label": "1.0"},
                                "updated": {"label": "2026-08-21T10:00:00Z"},
                                "author": {"name": {"label": "hidden"}},
                                "link": {"attributes": {"href": "https://example.test/list-1"}},
                            },
                        ]
                    }
                },
            )
        if "/us/rss/" in url:
            raise AssertionError("US feed should not be requested")
        return httpx.Response(404, text=url)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(collection_rate_limit_seconds=0, collection_retry_attempts=1)
    collector = AppStoreCollector(settings=settings, client=client)
    reviews = collector.collect(max_reviews=5)
    assert collector.fallback_used is False
    assert len(reviews) == 1
    assert "size chart" in reviews[0].text.lower()
    assert reviews[0].region == "in"


def test_empty_feed_returns_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "lookup" in url:
            return httpx.Response(200, json=_lookup("Unknown App"))
        return httpx.Response(200, json={"feed": {"entry": []}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(collection_rate_limit_seconds=0, collection_retry_attempts=1)
    collector = AppStoreCollector(settings=settings, client=client)
    reviews = collector.collect(max_reviews=5)
    assert reviews == []


def test_app_store_http_error_is_recorded():
    def handler(request: httpx.Request) -> httpx.Response:
        if "lookup" in str(request.url):
            return httpx.Response(200, json=_lookup())
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(collection_rate_limit_seconds=0, collection_retry_attempts=1)
    collector = AppStoreCollector(settings=settings, client=client)
    reviews = collector.collect(max_reviews=5)
    assert reviews == []
    assert collector.errors
