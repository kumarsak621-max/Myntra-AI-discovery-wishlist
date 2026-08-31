"""Emergent theme clustering from analysis labels + review text.

Uses TF-IDF + KMeans when enough documents exist. Cluster names come from
the most common AI labels in each cluster, not from a fixed taxonomy.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict

from sqlalchemy.orm import Session

from app.models import Review, Theme, utcnow

logger = logging.getLogger(__name__)


def _loads(raw: str) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}|\u0900-\u097F+", (text or "").lower())


def _label_bag(review: Review) -> list[str]:
    analysis = review.analysis
    if not analysis or not analysis.is_valid_json:
        return []
    labels = []
    for field in ("barriers_json", "uncertainties_json", "intent_json"):
        labels.extend(str(x) for x in _loads(getattr(analysis, field)))
    if analysis.root_cause:
        labels.append(analysis.root_cause)
    return [x.strip() for x in labels if str(x).strip()]


def discover_themes(db: Session) -> list[Theme]:
    reviews = (
        db.query(Review)
        .filter(Review.is_duplicate.is_(False), Review.is_empty.is_(False))
        .all()
    )
    analyzed = [
        r
        for r in reviews
        if r.analysis and r.analysis.is_valid_json and r.is_valid_source
    ]
    db.query(Theme).delete()
    if len(analyzed) < 3:
        # Fall back to raw label frequency as singleton "themes"
        freq: Counter[str] = Counter()
        owners: dict[str, list[int]] = defaultdict(list)
        sources: dict[str, set[str]] = defaultdict(set)
        myntra_c: Counter[str] = Counter()
        for review in analyzed:
            for label in _label_bag(review):
                key = label[:120]
                freq[key] += 1
                owners[key].append(review.id)
                sources[key].add(review.source)
                if review.is_valid_source:
                    myntra_c[key] += 1
        themes: list[Theme] = []
        for name, count in freq.most_common(20):
            theme = Theme(
                name=name[:255],
                description="Emergent label cluster (small corpus — frequency grouping).",
                cluster_key="label-freq",
                review_count=count,
                myntra_review_count=myntra_c[name],
                reference_review_count=count - myntra_c[name],
                sources_json=json.dumps(sorted(sources[name])),
                evidence_ids_json=json.dumps(owners[name][:50]),
                is_emergent=True,
                updated_at=utcnow(),
            )
            db.add(theme)
            themes.append(theme)
        db.commit()
        return themes

    documents = []
    for review in analyzed:
        labels = " ".join(_label_bag(review))
        body = review.cleaned_text or review.text or ""
        documents.append(f"{labels} {body}"[:4000])

    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        n_clusters = max(2, min(8, len(analyzed) // 8 or 2))
        n_clusters = min(n_clusters, len(analyzed))
        vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1, stop_words="english")
        matrix = vectorizer.fit_transform(documents)
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = model.fit_predict(matrix)
        terms = vectorizer.get_feature_names_out()
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    except Exception as exc:
        logger.warning("Clustering fallback to labels: %s", exc)
        labels = [0] * len(analyzed)
        n_clusters = 1
        terms = []
        order_centroids = []

    groups: dict[int, list[Review]] = defaultdict(list)
    for review, cid in zip(analyzed, labels):
        groups[int(cid)].append(review)

    themes: list[Theme] = []
    for cid, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        label_freq: Counter[str] = Counter()
        sources: set[str] = set()
        myntra_n = 0
        ids: list[int] = []
        for review in members:
            for lab in _label_bag(review):
                label_freq[lab[:120]] += 1
            sources.add(review.source)
            ids.append(review.id)
            if review.is_valid_source:
                myntra_n += 1
        name = label_freq.most_common(1)[0][0] if label_freq else f"Cluster {cid + 1}"
        top_terms = []
        if len(order_centroids):
            top_terms = [str(terms[i]) for i in order_centroids[cid][:8]]
        description = (
            "Emergent theme from clustering. Top terms: " + ", ".join(top_terms)
            if top_terms
            else "Emergent theme from label grouping."
        )
        # Avoid duplicate names
        existing_names = {t.name.lower() for t in themes}
        final_name = name
        suffix = 2
        while final_name.lower() in existing_names:
            final_name = f"{name} ({suffix})"
            suffix += 1
        theme = Theme(
            name=final_name[:255],
            description=description[:2000],
            cluster_key=str(cid),
            review_count=len(members),
            myntra_review_count=myntra_n,
            reference_review_count=len(members) - myntra_n,
            sources_json=json.dumps(sorted(sources)),
            evidence_ids_json=json.dumps(ids[:80]),
            is_emergent=True,
            updated_at=utcnow(),
        )
        db.add(theme)
        themes.append(theme)

    db.commit()
    return themes
