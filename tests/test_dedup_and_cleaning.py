from __future__ import annotations

from app.collectors.google_play import GooglePlayCollector as GP
from app.config import Settings
from app.pipeline.cleaning import clean_review
from app.pipeline.dedup import content_hash
from app.pipeline.validation import validate_app_identity
from app.schemas import NormalizedReview


def test_content_hash_stable_and_case_insensitive():
    a = content_hash("google_play", "1", "Size will NOT fit", "com.x")
    b = content_hash("google_play", "1", "size will not fit", "com.x")
    c = content_hash("google_play", "2", "size will not fit", "com.x")
    assert a == b
    assert a != c


def test_duplicate_second_insert_is_counted(db):
    settings = Settings(google_play_app_id="com.grofers.customerapp", collection_rate_limit_seconds=0)
    collector = GP(settings=settings)
    validation = validate_app_identity(
        platform="google_play",
        app_id="com.grofers.customerapp",
        detected_app_name="Blinkit",
        detected_developer="Blink Commerce",
    )
    item = NormalizedReview(
        source="google_play",
        source_review_id="same",
        app_id="com.grofers.customerapp",
        app_name="Blinkit",
        text="I love this dress but did not buy because I don't know if the size will fit.",
        is_valid_source=False,
        data_classification=validation.data_classification,
    )
    first = collector.save_raw(db, [item], validation)
    second = collector.save_raw(db, [item], validation)
    assert first.new == 1
    assert second.duplicates == 1
    assert second.new == 0
    from app.models import Review

    rows = db.query(Review).all()
    assert len(rows) == 1
    assert rows[0].is_duplicate is False
    assert rows[0].text.endswith("fit.")


def test_cleaning_preserves_interest_and_hesitation():
    text = "I love this dress but didn't buy because I don't know if the size will fit."
    result = clean_review("", text)
    assert "love this dress" in result.cleaned_text
    assert "didn't buy" in result.cleaned_text
    assert "size will fit" in result.cleaned_text
    assert result.is_empty is False


def test_empty_and_spam_flags():
    empty = clean_review("", "   ")
    assert empty.is_empty is True
    spam = clean_review("", "http://a.com http://b.com http://c.com buy now")
    assert spam.is_spam is True
    mixed = clean_review("", "Yeh dress achhi hai but size ka tension hai")
    assert mixed.language_notes in {"hinglish_or_mixed", "latin"}
