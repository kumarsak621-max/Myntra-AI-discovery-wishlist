"""Prompts for structured discovery analysis.

Hypotheses listed here are examples only. The model must not assume they apply.
Keep outputs short so OpenRouter stays within a small max_tokens budget.
"""

SYSTEM_PROMPT = """You analyze public app reviews as proxy evidence for wishlist-to-purchase research.

The business objective is to increase the percentage of users who purchase at least one wishlist item within 30 days.
Public reviews do NOT measure actual wishlist-to-purchase conversion. Treat them as proxy evidence only.
Do not claim that a review proves in-app conversion.

Look specifically for evidence related to:
- saving, bookmarking, wishlist, saved items, intent to purchase later
- hesitation, postponement, price concerns, comparison, reviews
- size/fit, availability, styling uncertainty, occasion
- social validation, external research, purchase confidence

Return ONLY valid JSON. The first non-whitespace character MUST be `{`.
No markdown, no prose, no code fences, no explanations outside JSON.
Do not invent quotes, review IDs, or facts.
Analyze only the supplied review text.
If there is no evidence for a field, return an empty list [] rather than None, empty string, or a placeholder.
Never use "none", "None", "N/A", "unknown", or similar placeholders as labels.
Do not infer wishlist behavior from a generic price compliment. Use the actual review text.
Distinguish explicit hesitation from implicit hesitation when the schema asks.

List fields must ALWAYS be JSON arrays of strings.
If there is no evidence, return [].
Never return an empty string for a list field.
Never return null for a list field unless the schema explicitly allows it.
"""


ANALYSIS_ITEM_SCHEMA = """{
  "id": "supplied review id",
  "problem": "",
  "root_cause": "",
  "wishlist_signal": false,
  "wishlist_behavior": [],
  "purchase_barriers": [],
  "uncertainties": [],
  "themes": [],
  "segments": [],
  "purchase_barrier": "",
  "uncertainty": "",
  "theme": "",
  "segment": "",
  "severity": 1,
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
    return f"""Return ONLY valid JSON matching this schema. The first character must be {{. No markdown. No prose.

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

Rules:
- wishlist_behavior must ALWAYS be a JSON array of strings. If there is no evidence, return []. Never return an empty string.
- purchase_barriers, uncertainties, themes, and segments must ALWAYS be JSON arrays of strings. Use [] when evidence is absent.
- Do not invent. Do not assume wishlist unless the text supports it.
- Public reviews are proxy evidence only, not conversion events.
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
    return f"""Return ONLY valid JSON. The first character must be {{. No markdown. No prose. No explanations outside JSON.

{{
  "results": [
    {ANALYSIS_ITEM_SCHEMA}
  ]
}}

One results[] object per review. Use the supplied REVIEW ID.

Rules:
- wishlist_behavior must ALWAYS be a JSON array of strings. If there is no evidence, return []. Never return an empty string.
- purchase_barriers, uncertainties, themes, and segments must ALWAYS be JSON arrays of strings. Use [] when evidence is absent.
- Never return "" or null for a list field.
- Return exactly one results[] object for every REVIEW ID supplied below.
- Copy the supplied REVIEW ID into each object's "id" field. Do not skip reviews. Do not invent IDs.
- Short phrases only. Do not invent. Do not assume wishlist unless the text supports it.
- Public reviews are proxy evidence only, not conversion events.

REVIEWS:

{joined}
"""
