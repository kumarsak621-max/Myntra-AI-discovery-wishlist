"""Prompts for structured discovery analysis.

Hypotheses listed here are examples only. The model must not assume they apply.
Keep outputs short so OpenRouter stays within a small max_tokens budget.
"""

SYSTEM_PROMPT = """You analyze public app reviews for wishlist-to-purchase research.

Return concise JSON only.
Do not include explanations outside the JSON.
Keep each field concise.
Do not invent quotes, review IDs, or facts.
Analyze only the supplied review text.
Do not write essays, reasoning traces, or reports.
If the review is unrelated to shopping or purchase hesitation, set evidence_type to "none" and leave the other fields empty.
"""


ANALYSIS_ITEM_SCHEMA = """{
  "id": "supplied review id",
  "problem": "",
  "wishlist_signal": false,
  "purchase_barrier": "",
  "uncertainty": "",
  "theme": "",
  "segment": "",
  "severity": 1,
  "purchase_impact": 1,
  "evidence_type": "explicit|inferred|none",
  "confidence": 0.0
}"""


def analysis_user_prompt(
    *,
    source: str,
    app_name: str,
    data_classification: str,
    rating: int | None,
    title: str,
    text: str,
    region: str,
) -> str:
    return f"""Return concise JSON only. Do not include explanations outside the JSON. Keep each field concise.

SOURCE: {source}
APP: {app_name}
CLASSIFICATION: {data_classification}
REGION: {region}
RATING: {rating if rating is not None else "unknown"}
TITLE: {title or "(none)"}
TEXT:
{text}

Schema:
{ANALYSIS_ITEM_SCHEMA}

Use short phrases. Empty strings if unsupported. Do not assume wishlist unless the text supports it.
"""


def analysis_batch_user_prompt(items: list[dict]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"REVIEW ID: {item['id']}",
                    f"SOURCE: {item.get('source') or ''}",
                    f"APP: {item.get('app_name') or ''}",
                    f"CLASSIFICATION: {item.get('data_classification') or ''}",
                    f"REGION: {item.get('region') or ''}",
                    f"RATING: {item.get('rating') if item.get('rating') is not None else 'unknown'}",
                    f"TITLE: {item.get('title') or '(none)'}",
                    "TEXT:",
                    item.get("text") or "",
                ]
            )
        )
    joined = "\n\n-----\n\n".join(blocks)
    return f"""Return concise JSON only. Do not include explanations outside the JSON. Keep each field concise.

{{
  "results": [
    {ANALYSIS_ITEM_SCHEMA}
  ]
}}

One results[] object per review. Use the supplied REVIEW ID. Short phrases only.

REVIEWS:

{joined}
"""
