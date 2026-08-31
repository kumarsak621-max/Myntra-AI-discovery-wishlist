# Myntra AI Discovery Engine

Public-feedback discovery system for Myntra wishlist-to-purchase research.

## Project Objective

Increase the percentage of users who purchase at least one item from their wishlist within 30 days of adding it.

The engine does **not** start with a product solution. It discovers **why wishlisted fashion products fail to become purchases within 30 days** from real public app-store reviews, then ranks problems with evidence.

## Architecture

```
Public sources
    → automatic collection (Google Play + App Store)
    → source identity validation
    → cleaning + deduplication
    → AI analysis (intent, barriers, uncertainty, root cause)
    → theme discovery
    → behavioral segmentation
    → programmatic quantification
    → opportunity scoring
    → dashboard + discovery report
```

`python app.py` is the entry point. It loads `.env`, validates official Myntra app identities, initializes SQLite, collectors, the AI provider, and the dashboard.

## Data Sources

**Google Play**

- Package: `com.myntra.android`
- URL: https://play.google.com/store/apps/details?id=com.myntra.android
- Collector: `google-play-scraper` with pagination

**Apple App Store**

- App ID: `907394059`
- URL: https://apps.apple.com/in/app/myntra-fashion-shopping-app/id907394059
- Collector: iTunes RSS review feed
- Region: try `in` first; if there are no written reviews, fall back to `us`
- US reviews are stored with `region=us` and are never labelled as Indian

Blinkit/Grofers IDs (`com.grofers.customerapp`, `960335206`) are banned. They cannot be used for Myntra collection or presented as Myntra evidence.

## Setup

```bash
git clone https://github.com/kumarsak621-max/Myntra-AI-discovery-wishlist.git
cd Myntra-AI-discovery-wishlist

python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Python 3.12 is recommended.

## Environment Variables

Copy the template and fill in secrets locally:

```powershell
copy .env.example .env
```

```bash
cp .env.example .env
```

Open `.env` and set:

```
OPENROUTER_API_KEY=your_key_here
```

Leave `GOOGLE_PLAY_APP_ID=com.myntra.android` and `APPLE_APP_ID=907394059`.

Never put a real API key in this README. Never commit `.env`.

Collection still works without an AI key. Analysis is skipped until `OPENROUTER_API_KEY` is set.

## Run

```bash
python app.py
```

Then open http://127.0.0.1:8000

## Data Collection

On **Data Collection**:

- **Collect Google Play Reviews** — live Play Store reviews for `com.myntra.android`
- **Collect Apple App Store Reviews** — iTunes RSS for `907394059` (India first)
- **Collect All** — both sources, then the analysis pipeline for new Myntra-valid reviews

Raw reviews are stored in local SQLite (`myntra_discovery.db`), which is gitignored. Collect after clone; do not commit the database.

## AI Analysis

OpenRouter is the default gateway (`AI_PROVIDER=openrouter`). The model is set with `OPENROUTER_MODEL` / `AI_MODEL`.

The LLM extracts structured JSON: relevance, wishlist/purchase signals, intents, barriers, uncertainties, information-seeking, behavioral signals, and observed / inferred / hypothesized root causes.

Python (not the LLM) calculates counts, percentages, clustering inputs, and opportunity scores:

```
score = reach × frequency × purchase_impact × severity × evidence_confidence
```

Each dimension is 1–5.

## Dashboard

The dashboard includes Overview, Data Collection, Feedback Explorer, Wishlist Motivations, Purchase Barriers, Uncertainties, Root Causes, Themes, Segments, External Information Seeking, Opportunity Matrix, Evidence Explorer, and Discovery Report.

Every insight opens the original review text, source, URL, review ID, date, and classification. Quotes are never generated.

Default filter: **Myntra-valid evidence only**.

## Tests

```bash
pytest -q
```

## Security

- `.env` and API keys must never be committed
- Author names are not shown in the UI
- Only publicly available store reviews are collected
- Original review text is never overwritten
