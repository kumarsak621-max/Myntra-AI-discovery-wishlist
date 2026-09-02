"""Apple App Store collector using the public iTunes RSS review feed.

India is requested first. If that region yields no written reviews, the
configured fallback region (default: US) is used. Region is stored on every
review; US reviews are never labelled as Indian.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree

import httpx

from app.collectors.base_collector import BaseCollector, ProgressCallback, with_retry
from app.config import Settings
from app.pipeline.validation import validate_app_identity
from app.schemas import NormalizedReview, SourceValidation
from config.settings import is_banned_app_id, is_official_myntra_app_id

logger = logging.getLogger(__name__)

LOOKUP_URL = "https://itunes.apple.com/lookup"
RSS_JSON = (
    "https://itunes.apple.com/{region}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
)
RSS_XML = (
    "https://itunes.apple.com/{region}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/xml"
)
MAX_RSS_PAGES = 10
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "im": "http://itunes.apple.com/rss"}
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, application/xml, text/xml, */*",
}

APPLE_FETCH_FAILED = "APPLE_FETCH_FAILED"
APPLE_FETCH_SUCCESS_NO_NEW_REVIEWS = "APPLE_FETCH_SUCCESS_NO_NEW_REVIEWS"
APPLE_NEW_REVIEWS_FOUND = "APPLE_NEW_REVIEWS_FOUND"
PLAY_FETCH_FAILED = "PLAY_FETCH_FAILED"
PLAY_FETCH_SUCCESS_NO_NEW_REVIEWS = "PLAY_FETCH_SUCCESS_NO_NEW_REVIEWS"
PLAY_NEW_REVIEWS_FOUND = "PLAY_NEW_REVIEWS_FOUND"


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    try:
        iso = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text[:19] if fmt != "%Y-%m-%d" else text[:10], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _label_from_entry(entry: dict[str, Any], key: str) -> str:
    node = entry.get(key)
    if node is None:
        return ""
    if isinstance(node, list) and node:
        node = node[0]
    if isinstance(node, dict):
        return str(node.get("label") or node.get("text") or "")
    return str(node or "")


def _has_written_body(title: str, body: str) -> bool:
    body = (body or "").strip()
    title = (title or "").strip()
    if not body and not title:
        return False
    if body.lower() in {"", "none", "n/a", "na"}:
        return False
    return len(body) >= 2 or len(title) >= 2


class AppStoreCollector(BaseCollector):
    platform = "apple_app_store"

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        super().__init__(settings)
        self._client = client
        self.region_used: str = self.settings.apple_primary_region
        self.fallback_used: bool = False
        self.fetch_status: str = APPLE_FETCH_FAILED
        self._last_fetch_ok: bool = False

    def _http(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=30.0, follow_redirects=True, headers=HTTP_HEADERS)

    def _get_json(self, url: str) -> dict[str, Any]:
        self.rate_limiter.wait()
        client = self._http()
        owns = self._client is None
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        finally:
            if owns:
                client.close()

    def _get_text(self, url: str) -> str:
        self.rate_limiter.wait()
        client = self._http()
        owns = self._client is None
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.text
        finally:
            if owns:
                client.close()

    def lookup_metadata(self, region: str) -> dict[str, Any]:
        url = f"{LOOKUP_URL}?id={self.settings.apple_app_id}&country={region}"

        def _fetch():
            data = self._get_json(url)
            results = data.get("results") or []
            return results[0] if results else {}

        return with_retry(
            _fetch,
            attempts=self.settings.collection_retry_attempts,
            label=f"itunes lookup {self.settings.apple_app_id} {region}",
        )

    def validate_source(
        self,
        progress: ProgressCallback | None = None,
        region: str | None = None,
    ) -> SourceValidation:
        region = region or self.settings.apple_primary_region
        app_id = self.settings.apple_app_id
        if progress:
            progress(
                {
                    "stage": "apple_app_store",
                    "status": "validating",
                    "app_id": app_id,
                    "region": region,
                    "message": f"Validating App Store identity for {app_id} ({region})",
                }
            )
        try:
            meta = self.lookup_metadata(region)
        except Exception as exc:
            logger.warning("App Store lookup failed for %s (%s): %s", app_id, region, exc)
            if is_official_myntra_app_id(app_id, self.platform):
                validation = validate_app_identity(
                    platform=self.platform,
                    app_id=app_id,
                    detected_app_name=self.settings.expected_apple_app_name,
                    detected_developer="Myntra Designs Private Limited",
                    region=region,
                    expected_app=self.settings.expected_apple_app_name,
                    metadata={"error": str(exc), "lookup_failed": True},
                )
                validation.warning = (
                    f"App Store lookup failed for {app_id} ({region}): {exc}. "
                    "Continuing with the official Myntra App Store ID."
                )
            else:
                validation = validate_app_identity(
                    platform=self.platform,
                    app_id=app_id,
                    region=region,
                    expected_app=self.settings.expected_apple_app_name,
                    metadata={"error": str(exc)},
                )
                validation.validation_status = "ERROR"
                validation.warning = f"App Store lookup failed for {app_id} ({region}): {exc}"
            self.last_validation = validation
            return validation

        name = str(meta.get("trackName") or "")
        developer = str(meta.get("artistName") or "")
        validation = validate_app_identity(
            platform=self.platform,
            app_id=app_id,
            detected_app_name=name,
            detected_developer=developer,
            region=region,
            expected_app=self.settings.expected_apple_app_name,
            metadata={
                "bundleId": meta.get("bundleId"),
                "averageUserRating": meta.get("averageUserRating"),
                "userRatingCount": meta.get("userRatingCount"),
                "trackViewUrl": meta.get("trackViewUrl"),
                "primaryGenreName": meta.get("primaryGenreName"),
            },
        )
        if progress:
            progress(
                {
                    "stage": "apple_app_store",
                    "status": "validated",
                    "app_id": app_id,
                    "region": region,
                    "detected_app_name": name,
                    "detected_developer": developer,
                    "validation_status": validation.validation_status,
                    "is_valid_for_myntra": validation.is_valid_for_myntra,
                    "warning": validation.warning,
                }
            )
        self.last_validation = validation
        return validation

    def _parse_json_feed(self, payload: dict[str, Any], region: str) -> list[dict[str, Any]]:
        feed = payload.get("feed") or {}
        entries = feed.get("entry") or []
        if isinstance(entries, dict):
            entries = [entries]
        reviews: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "im:rating" not in entry:
                continue
            author = entry.get("author") or {}
            author_name = ""
            if isinstance(author, dict):
                author_name = _label_from_entry(author, "name")
            reviews.append(
                {
                    "id": _label_from_entry(entry, "id"),
                    "title": _label_from_entry(entry, "title"),
                    "content": _label_from_entry(entry, "content"),
                    "rating": _label_from_entry(entry, "im:rating"),
                    "version": _label_from_entry(entry, "im:version"),
                    "updated": _label_from_entry(entry, "updated"),
                    "author": author_name,
                    "link": ((entry.get("link") or {}).get("attributes") or {}).get("href")
                    or _label_from_entry(entry, "link"),
                    "region": region,
                }
            )
        return reviews

    def _parse_xml_feed(self, xml_text: str, region: str) -> list[dict[str, Any]]:
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as exc:
            logger.warning("App Store XML parse error: %s", exc)
            return []
        reviews: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            rating_el = entry.find("im:rating", ATOM_NS)
            if rating_el is None:
                continue
            review_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            content = ""
            for content_el in entry.findall("atom:content", ATOM_NS):
                if (content_el.get("type") or "text") in {"text", "html"}:
                    content = (content_el.text or "").strip()
                    break
            author = entry.find("atom:author", ATOM_NS)
            author_name = ""
            if author is not None:
                author_name = (author.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
            link = ""
            link_el = entry.find("atom:link", ATOM_NS)
            if link_el is not None:
                link = link_el.get("href") or ""
            version_el = entry.find("im:version", ATOM_NS)
            updated = (entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or "").strip()
            reviews.append(
                {
                    "id": review_id,
                    "title": title,
                    "content": content,
                    "rating": (rating_el.text or "").strip(),
                    "version": (version_el.text or "").strip() if version_el is not None else "",
                    "updated": updated,
                    "author": author_name,
                    "link": link,
                    "region": region,
                }
            )
        return reviews

    def _fetch_region_pages(
        self,
        region: str,
        limit: int,
        progress: ProgressCallback | None,
        stop_when_older_than: datetime | None = None,
    ) -> list[dict[str, Any]]:
        from app.pipeline.dates import ensure_aware

        gathered: list[dict[str, Any]] = []
        seen: set[str] = set()
        cutoff = ensure_aware(stop_when_older_than)
        app_id = self.settings.apple_app_id
        self._last_fetch_ok = False
        for page in range(1, MAX_RSS_PAGES + 1):
            if len(gathered) >= limit:
                break
            url = RSS_JSON.format(region=region, page=page, app_id=app_id)
            page_rows: list[dict[str, Any]] = []
            json_error: Exception | None = None
            page_http_ok = False
            try:
                payload = with_retry(
                    lambda u=url: self._get_json(u),
                    attempts=self.settings.collection_retry_attempts,
                    label=f"app-store rss json {region} p{page}",
                )
                page_rows = self._parse_json_feed(payload, region)
                page_http_ok = True
            except Exception as exc:
                json_error = exc
                logger.warning("JSON RSS failed %s page %s: %s — trying XML", region, page, exc)
            if not page_rows:
                try:
                    xml_text = with_retry(
                        lambda p=page: self._get_text(RSS_XML.format(region=region, page=p, app_id=app_id)),
                        attempts=self.settings.collection_retry_attempts,
                        label=f"app-store rss xml {region} p{page}",
                    )
                    page_rows = self._parse_xml_feed(xml_text, region)
                    page_http_ok = True
                except Exception as xml_exc:
                    detail = f"{region} page {page}: {xml_exc}"
                    if json_error:
                        detail = f"{region} page {page}: json={json_error}; xml={xml_exc}"
                    logger.error("Apple App Store fetch failed: %s", detail)
                    self.errors.append(detail)
                    if progress:
                        progress(
                            {
                                "stage": "apple_app_store",
                                "status": "error",
                                "region": region,
                                "page": page,
                                "message": detail,
                            }
                        )
                    break

            if page_http_ok:
                self._last_fetch_ok = True
            if not page_rows:
                break
            new_on_page = 0
            added_this_page: list[dict[str, Any]] = []
            for row in page_rows:
                rid = str(row.get("id") or "")
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                gathered.append(row)
                added_this_page.append(row)
                new_on_page += 1
                if len(gathered) >= limit:
                    break
            if progress:
                progress(
                    {
                        "stage": "apple_app_store",
                        "status": "collecting",
                        "region": region,
                        "page": page,
                        "fetched": len(gathered),
                        "target": limit,
                    }
                )
            if cutoff is not None and added_this_page:
                dated = []
                for row in added_this_page:
                    stamp = ensure_aware(_parse_dt(row.get("updated")))
                    if stamp is not None:
                        dated.append(stamp)
                if dated and all(stamp < cutoff for stamp in dated):
                    break
            if new_on_page == 0:
                break
        return gathered[:limit]

    def collect(
        self,
        max_reviews: int | None = None,
        progress: ProgressCallback | None = None,
        *,
        stop_when_older_than: datetime | None = None,
        safety_limit: int | None = None,
    ) -> list[NormalizedReview]:
        if safety_limit is not None:
            limit = safety_limit
        elif max_reviews is not None:
            limit = max_reviews
        else:
            limit = self.settings.apple_max_reviews
        primary = self.settings.apple_primary_region
        fallback = self.settings.apple_fallback_region
        self.fetch_status = APPLE_FETCH_FAILED
        self.fallback_used = False
        self.region_used = primary

        app_id = self.settings.apple_app_id
        if is_banned_app_id(app_id) or not is_official_myntra_app_id(app_id, self.platform):
            msg = f"Refusing Myntra collection for non-official App Store ID {app_id}."
            logger.error(msg)
            self.errors.append(msg)
            self.last_validation = self.validate_source(progress=progress, region=primary)
            if progress:
                progress(
                    {
                        "stage": "apple_app_store",
                        "status": "error",
                        "message": msg,
                        "validation_result": "FAIL",
                    }
                )
            return []

        def _written(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                row
                for row in rows
                if _has_written_body(str(row.get("title") or ""), str(row.get("content") or ""))
            ]

        def _try_region(region: str) -> tuple[SourceValidation, list[dict[str, Any]], bool]:
            prior_errors = list(self.errors)
            validation = self.validate_source(progress=progress, region=region)
            raw = self._fetch_region_pages(
                region, limit, progress, stop_when_older_than=stop_when_older_than
            )
            fetch_ok = bool(self._last_fetch_ok)
            if fetch_ok:
                self.errors = [item for item in self.errors if item in prior_errors]
            return validation, raw, fetch_ok

        primary_validation, raw_primary, primary_ok = _try_region(primary)
        written_primary = _written(raw_primary)

        raw_rows = raw_primary
        validation = primary_validation
        self.region_used = primary
        region_ok = primary_ok

        need_fallback = (not primary_validation.is_valid_for_myntra) or (not written_primary) or (not primary_ok)
        if need_fallback and fallback and fallback != primary:
            if progress:
                progress(
                    {
                        "stage": "apple_app_store",
                        "status": "fallback",
                        "message": (
                            f"India region '{primary}' had no usable reviews "
                            f"(identity_ok={primary_validation.is_valid_for_myntra}, "
                            f"written={len(written_primary)}, fetch_ok={primary_ok}). "
                            f"Falling back to '{fallback}'."
                        ),
                    }
                )
            logger.warning(
                "Apple India fetch insufficient (identity_ok=%s written=%s fetch_ok=%s); trying %s",
                primary_validation.is_valid_for_myntra,
                len(written_primary),
                primary_ok,
                fallback,
            )
            fallback_validation, raw_fallback, fallback_ok = _try_region(fallback)
            written_fallback = _written(raw_fallback)
            if written_fallback or (fallback_ok and fallback_validation.is_valid_for_myntra):
                raw_rows = raw_fallback
                validation = fallback_validation
                self.region_used = fallback
                self.fallback_used = True
                region_ok = fallback_ok
            elif not written_primary and written_fallback:
                raw_rows = raw_fallback
                validation = fallback_validation
                self.region_used = fallback
                self.fallback_used = True
                region_ok = fallback_ok

        self.last_validation = validation
        if not validation.is_valid_for_myntra and not _written(raw_rows):
            msg = validation.warning or (
                f"Apple App Store identity validation FAIL for {app_id} "
                f"(region={self.region_used})."
            )
            logger.error(msg)
            if msg not in self.errors:
                self.errors.append(msg)
            self.fetch_status = APPLE_FETCH_FAILED
            if progress:
                progress(
                    {
                        "stage": "apple_app_store",
                        "status": "error",
                        "message": msg,
                        "validation_result": "FAIL",
                    }
                )
            return []

        normalized = [self.normalize(row, validation) for row in raw_rows]
        if region_ok:
            self.fetch_status = (
                APPLE_NEW_REVIEWS_FOUND if normalized else APPLE_FETCH_SUCCESS_NO_NEW_REVIEWS
            )
            if not normalized:
                logger.info(
                    "Apple App Store checked %s successfully but returned 0 written reviews "
                    "(fallback_used=%s). Not treating this as a fetch failure.",
                    self.region_used,
                    self.fallback_used,
                )
        else:
            self.fetch_status = APPLE_FETCH_FAILED
            msg = (
                f"Apple App Store fetch failed for {app_id} "
                f"(region={self.region_used}, fallback_used={self.fallback_used})."
            )
            logger.error("%s errors=%s", msg, list(self.errors))
            if msg not in self.errors:
                self.errors.append(msg)
        if self.fetch_status != APPLE_FETCH_FAILED:
            self.errors = []
        if progress:
            progress(
                {
                    "stage": "apple_app_store",
                    "status": "complete",
                    "fetched": len(normalized),
                    "region_used": self.region_used,
                    "fallback_used": self.fallback_used,
                    "fetch_status": self.fetch_status,
                    "errors": list(self.errors),
                    "validation": validation.model_dump(mode="json"),
                }
            )
        return normalized[:limit]

    def normalize(self, raw: dict[str, Any], validation: SourceValidation) -> NormalizedReview:
        rating_raw = raw.get("rating")
        try:
            rating = int(str(rating_raw).strip()) if rating_raw not in (None, "") else None
        except ValueError:
            rating = None
        region = str(raw.get("region") or validation.region or "")
        review_id = str(raw.get("id") or "")
        link = str(raw.get("link") or "")
        payload = {
            "id": review_id,
            "title": raw.get("title"),
            "content": raw.get("content"),
            "rating": rating,
            "version": raw.get("version"),
            "updated": raw.get("updated"),
            "region": region,
            "has_author": bool(raw.get("author")),
        }
        return NormalizedReview(
            source=self.platform,
            source_review_id=review_id,
            app_id=self.settings.apple_app_id,
            app_name=validation.detected_app_name,
            developer=validation.detected_developer,
            region=region,
            rating=rating,
            title=str(raw.get("title") or ""),
            text=str(raw.get("content") or ""),
            review_date=_parse_dt(raw.get("updated")),
            last_update_date=_parse_dt(raw.get("updated")),
            app_version=str(raw.get("version") or ""),
            source_url=link or self.settings.apple_app_store_url,
            developer_reply="",
            raw_payload=payload,
            is_valid_source=validation.is_valid_for_myntra,
            data_classification=validation.data_classification,
            is_synthetic=False,
        )
