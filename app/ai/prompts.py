"""Prompts for structured discovery analysis.

Hypotheses listed here are examples only. The model must not assume they apply.
"""

SYSTEM_PROMPT = """You are a qualitative research analyst helping a product manager discover WHY users add fashion products to a wishlist but do not purchase them within 30 days.

You do NOT propose product solutions, discounts, notifications, AI recommendations, or features.
You do NOT invent quotes, statistics, or facts that are not in the review.
You do NOT assume price, fit, reviews, or any other factor is the main problem.

Your job is discovery: extract what the user actually said, then separately mark inferences and hypotheses.

Always distinguish:
- OBSERVED: explicitly present in the review text
- INFERRED: a reasonable reading of the text, labelled as inferred
- HYPOTHESIZED: a possible underlying problem, labelled as hypothesized

If the review is not about fashion, wishlist, purchase hesitation, or shopping decisions, set relevance to "none" or "low" and leave arrays empty rather than forcing categories.

The review may come from a NON-MYNTRA app (for example Blinkit/Grofers grocery delivery). If so:
- Do not pretend it is Myntra evidence
- Still extract shopping/purchase barriers if present
- Set product_category accordingly (e.g. grocery) and relevance low for fashion-wishlist research unless the text actually discusses fashion/wishlist behavior

Return ONLY valid JSON matching the schema. No markdown. No preamble.
"""


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
    return f"""Analyze this public app review for wishlist-to-purchase discovery research.

SOURCE: {source}
APP NAME: {app_name}
DATA CLASSIFICATION: {data_classification}
REGION: {region}
RATING: {rating if rating is not None else "unknown"}

REVIEW TITLE:
{title or "(none)"}

REVIEW TEXT (original, unmodified):
{text}

JSON schema to return:
{{
  "relevance": "high|medium|low|none",
  "wishlist_signal": "explicit|implicit|none",
  "purchase_signal": "purchased|intend_to_purchase|hesitant|abandoned|none",
  "purchase_hesitation": "explicit|implicit|none",
  "intent": ["emergent labels for why they save/browse/delay — only if evidenced"],
  "barriers": ["emergent purchase-barrier labels — only if evidenced"],
  "uncertainties": ["unanswered questions the user expresses or clearly implies"],
  "information_seeking": [
    {{
      "source": "reddit|youtube|google|instagram|other_ecommerce|brand_site|influencer|friends_family|fashion_community|unspecified",
      "what": "",
      "why": "",
      "associated_with_hesitation": false,
      "myntra_appears_to_lack_info": null,
      "basis": "explicit|inferred",
      "quote": "verbatim substring of the review or empty"
    }}
  ],
  "behavioral_signals": [
    {{
      "signal": "wishlist_save|purchase|delayed_purchase|abandonment|comparison|external_search|review_checking|photo_checking|size_checking|asking_friends|social_validation|return_concern|delivery_concern|availability_concern|revisit|waiting|other",
      "basis": "explicit|inferred",
      "quote": "verbatim substring or empty"
    }}
  ],
  "product_category": [],
  "decision_factors": [],
  "root_cause": {{
    "observed": "what the user said",
    "inferred": "interpretation, labelled as inference",
    "hypothesized": "possible underlying problem, labelled as hypothesis",
    "statement": "one-sentence root problem hypothesis"
  }},
  "sentiment": "positive|negative|mixed|neutral",
  "evidence_strength": 1,
  "confidence": 1
}}

Rules:
- intent/barriers/uncertainties must be discovered from THIS text. Do not dump a default taxonomy.
- Example intent hypotheses (do NOT assume): genuine purchase intent, bookmarking, future purchase, comparison, occasion planning, outfit planning, price consideration, size uncertainty, sharing, inspiration, fear of losing the product, product tracking.
- Example barriers (do NOT assume they apply): fit, size, material, quality, color, appearance, reviews, ratings, user photos, returns, exchanges, delivery, availability, trust, styling, occasion, social validation, decision fatigue, comparison, price/value, waiting, lack of urgency, better alternatives, preference change.
- If the review contains no wishlist evidence, set wishlist_signal to "none". Do not assume every Myntra review is a wishlist review.
- Do not force every review into every category. Leave arrays empty when the text does not support them.
- evidence_strength and confidence are integers 1-5.
- quote fields MUST be exact substrings of the review text, or empty.
- Sentiment is secondary. Preserve interest + hesitation + uncertainty when all are present. Never collapse a mixed review to sentiment=positive.
"""


ANALYSIS_ITEM_SCHEMA = """{
  "id": "the review id supplied below",
  "relevance": "high|medium|low|none",
  "wishlist_signal": "explicit|implicit|none",
  "purchase_signal": "purchased|intend_to_purchase|hesitant|abandoned|none",
  "purchase_hesitation": "explicit|implicit|none",
  "intent": [],
  "barriers": [],
  "uncertainties": [],
  "information_seeking": [],
  "behavioral_signals": [],
  "product_category": [],
  "decision_factors": [],
  "root_cause": {"observed": "", "inferred": "", "hypothesized": "", "statement": ""},
  "sentiment": "positive|negative|mixed|neutral",
  "evidence_strength": 1,
  "confidence": 1
}"""


def analysis_batch_user_prompt(items: list[dict]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"REVIEW ID: {item['id']}",
                    f"SOURCE: {item.get('source') or ''}",
                    f"APP NAME: {item.get('app_name') or ''}",
                    f"DATA CLASSIFICATION: {item.get('data_classification') or ''}",
                    f"REGION: {item.get('region') or ''}",
                    f"RATING: {item.get('rating') if item.get('rating') is not None else 'unknown'}",
                    f"TITLE: {item.get('title') or '(none)'}",
                    "TEXT:",
                    item.get("text") or "",
                ]
            )
        )
    joined = "\n\n-----\n\n".join(blocks)
    return f"""Analyze EACH public app review below for wishlist-to-purchase discovery research.

Return ONLY JSON in this shape:
{{
  "results": [
    {ANALYSIS_ITEM_SCHEMA}
  ]
}}

Rules:
- Include one results[] object per review, using the supplied REVIEW ID as "id".
- intent/barriers/uncertainties must come from that review's text. Do not invent quotes.
- If a review is unrelated to shopping/wishlist/purchase, set relevance to "none" or "low" and leave arrays empty.
- If a review contains no wishlist evidence, set wishlist_signal to "none". Do not assume every Myntra review is a wishlist review.
- Do not force every review into every category.
- evidence_strength and confidence are integers 1-5.
- quote fields MUST be exact substrings of that review, or empty.

REVIEWS:

{joined}
"""
