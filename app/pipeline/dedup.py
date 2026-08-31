"""Deduplication and content hashing. Original text is never mutated."""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def content_hash(
    source: str,
    source_review_id: str,
    text: str,
    app_id: str = "",
) -> str:
    payload = "|".join(
        [
            source or "",
            app_id or "",
            source_review_id or "",
            normalize_for_hash(text),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()
