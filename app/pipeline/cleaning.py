"""Light preprocessing that preserves behavioral meaning.

Original review text is never overwritten. Flags and a lightly cleaned copy
are produced for indexing and analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROMO_PATTERNS = (
    re.compile(r"\bwhatsapp\b", re.I),
    re.compile(r"\bclick here\b", re.I),
    re.compile(r"\bfree followers\b", re.I),
    re.compile(r"https?://\S+", re.I),
    re.compile(r"\bpromo code\b", re.I),
    re.compile(r"\buse code\b", re.I),
)
REPEATED_CHAR = re.compile(r"(.)\1{8,}")
REPEATED_WORD = re.compile(r"\b(\w+)(?:\s+\1){4,}\b", re.I)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


@dataclass
class CleanResult:
    cleaned_text: str
    is_empty: bool
    is_spam: bool
    is_promotional: bool
    is_short: bool
    is_long: bool
    language_notes: str


def clean_review(title: str, text: str) -> CleanResult:
    original = text or ""
    combined = f"{title or ''} {original}".strip()
    collapsed = re.sub(r"\s+", " ", original).strip()

    is_empty = len(collapsed) == 0
    url_count = len(re.findall(r"https?://", collapsed, re.I))
    promo = any(p.search(combined) for p in PROMO_PATTERNS)
    repeated = bool(REPEATED_CHAR.search(collapsed) or REPEATED_WORD.search(collapsed))
    is_spam = (url_count >= 3) or repeated
    is_short = 0 < len(collapsed) < 15
    is_long = len(collapsed) > 2500

    has_hi = bool(DEVANAGARI.search(collapsed))
    has_en = bool(LATIN.search(collapsed))
    if has_hi and has_en:
        language_notes = "hinglish_or_mixed"
    elif has_hi:
        language_notes = "devanagari"
    elif has_en:
        language_notes = "latin"
    else:
        language_notes = "other_or_empty"

    # Keep emojis and mixed language. Only collapse whitespace.
    return CleanResult(
        cleaned_text=collapsed,
        is_empty=is_empty,
        is_spam=is_spam,
        is_promotional=promo and url_count >= 1,
        is_short=is_short,
        is_long=is_long,
        language_notes=language_notes,
    )
