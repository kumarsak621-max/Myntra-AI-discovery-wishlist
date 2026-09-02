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
    → OpenRouter AI analysis (intent, barriers, uncertainty, root cause)
    → theme discovery
    → behavioral segmentation
    → programmatic quantification
    → opportunity scoring
    → dashboard + discovery report
```

`python app.py` starts the FastAPI dashboard. `streamlit run app.py` starts the Streamlit dashboard used on Streamlit Cloud. Both reuse the same collectors, validation, analysis, and SQLite store.

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
OPENROUTER_MODEL=google/gemini-2.5-flash
```

Leave `GOOGLE_PLAY_APP_ID=com.myntra.android` and `APPLE_APP_ID=907394059`.

Never put a real API key in this README. Never commit `.env` or `.streamlit/secrets.toml`.

The previous direct Google Gemini provider (`GEMINI_API_KEY` / `google-genai`) is not used. Production analysis goes only through OpenRouter. There is no silent Gemini fallback.

Collection still works without an AI key. Analysis is skipped until `OPENROUTER_API_KEY` is set. If the key is missing, the app shows `OpenRouter API key is not configured.` — not a fake empty corpus.

**Configuration priority**

1. Streamlit Cloud: when the app runs inside Streamlit, `st.secrets` (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`) overlay local `.env` values.
2. Local development: pydantic-settings loads `.env`.

Never hard-code the key. Never print it. Never show it in the UI.

## Run

Local FastAPI dashboard:

```bash
python app.py
```

Then open http://127.0.0.1:8000

Local Streamlit dashboard (same entry file Streamlit Cloud uses):

```bash
streamlit run app.py
```

Then open http://127.0.0.1:8501

Main file for Streamlit Cloud: `app.py`

## Data Collection

The default analysis window is **last 30 days**, computed as `now − 30 days` in UTC. Calendar dates are never hard-coded.

On **Live Data** / Overview:

- **Collect Last 30 Days** — paginated Google Play + Apple RSS until reviews are older than the cutoff, the source is exhausted, or a safety limit is reached. Filtering uses the **review timestamp**, not collection time.
- **Refresh Latest Reviews** — incremental near-real-time poll of newly available public reviews. Existing `review_id` / content hashes are skipped. Only **pending** (or changed) reviews are sent to the AI pipeline.
- Per-source buttons still exist for Google Play (`com.myntra.android`) and Apple (`907394059`, India first, US fallback).

This is **not** a live stream. Public store feeds are polled on demand. The UI labels the feature:

`Near-real-time — refreshed from the public source`

and shows `Last checked: <timestamp>` (for example “Last checked 2 minutes ago”). It never claims “Live” unless a source was actually polled.

Raw reviews are stored in local SQLite (`myntra_discovery.db`), which is gitignored. Collect after clone; do not commit the database.

The dashboard shows **Storage: Local application storage**. On Streamlit Cloud the filesystem is **ephemeral**. SQLite is available for the running session and is wiped on reboot or redeploy. This app does not provide permanent historical continuity across restarts.

## AI Analysis

**AI Provider:** OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)

**Default model:** `google/gemini-2.5-flash`

**Required secret:** `OPENROUTER_API_KEY`

**Configured model:** `OPENROUTER_MODEL` (optional; default `google/gemini-2.5-flash`)

Set `OPENROUTER_MODEL` to any OpenRouter model id your account can use. Do not leave an unavailable model hard-coded.

Locally the key and model come from `.env`. On Streamlit Cloud they come from **App settings → Secrets**. When Streamlit is running, `st.secrets` overlays `.env` values. The API key is never shown in the UI or printed in logs.

OpenRouter usage is subject to OpenRouter’s quota and rate limits. If those are hit, the app reports: `OpenRouter API quota/rate limit reached. Please try again later.`

The LLM extracts structured JSON: relevance, wishlist/purchase signals, intents, barriers, uncertainties, information-seeking, behavioral signals, and observed / inferred / hypothesized root causes. It analyzes only the supplied review text and supplied review IDs. It must not invent quotes, IDs, or assume every review is about wishlist behavior.

Python (not the LLM) calculates counts, percentages, clustering inputs, and opportunity scores:

```
score = reach × frequency × purchase_impact × severity × evidence_confidence
```

Each dimension is 1–5.

AI analysis is cached by review content hash. Unchanged `analyzed` reviews are not sent to the model again. New reviews are stored as `pending`. Failed reviews are retried and store the error. OpenRouter requests use `max_tokens` from `AI_MAX_TOKENS` (default **2000**, never the 65535 model default) and batches of `AI_REQUEST_BATCH_SIZE` / `AI_BATCH_SIZE` (default **5** reviews per request). The first analysis run is capped at **1** review, then **5**, so the OpenRouter path can be verified before processing the rest. A later run processes up to `AI_ANALYSIS_BATCH_SIZE` (default 60) pending reviews so Streamlit Cloud requests do not time out. Click **Analyze Pending Reviews** or **Run Full Discovery Pipeline** again to continue. **Retry Failed Analysis** re-sends only `failed` rows. HTTP 402 credit/token errors leave reviews pending for retry.

If discovery pages are empty, Live Data explains the actual reason (no reviews, pending analysis, missing API key, or a stored analysis error). It never treats stored reviews as “no data collected.” If reviews were stored but analysis failed, the UI shows `Reviews collected successfully, but OpenRouter analysis failed.` plus the actual error.

On **Live Data** use **Test OpenRouter Connection** to send a real OpenRouter request. Success is reported only when that request succeeds. The API key is shown only as Configured / Missing.

## Full discovery

**🚀 Run Full Discovery Pipeline** (Overview / Live Data) runs:

1. Collect Google Play last 30 days (skipped if reviews are already stored)
2. Collect Apple App Store last 30 days (India, then US fallback; skipped if reviews are already stored)
3. Normalize, deduplicate, store
4. Analyze pending reviews with OpenRouter
5. Rebuild themes, segments, and opportunity scores
6. Refresh the dashboard

If more reviews are already in the local database than `MAX_TOTAL_REVIEWS` (default **500**), storage is pruned to the newest real Google Play and Apple reviews. AI analysis uses at most `MAX_ANALYSIS_REVIEWS` (default **150**) of those stored reviews, in batches of 10. It does not fabricate reviews to reach either cap.

## Diagnostics

```bash
python -m utils.diagnostics
```

Prints database path and counts, collector status, OpenRouter configuration (never the key), connection test result, and discovery table counts.

## Troubleshooting

| Symptom | What it actually means |
| --- | --- |
| No real reviews available for analysis. | Database has 0 stored reviews. Click Collect Last 30 Days. |
| X real reviews are awaiting AI analysis. | Reviews are stored as `pending`. Set `OPENROUTER_API_KEY`, Test OpenRouter Connection, then Analyze Pending Reviews. |
| OpenRouter analysis failed for X reviews. | Per-review `failed` rows exist. Open Live Data for `Last error`. Click Retry Failed Analysis. |
| OpenRouter API key is not configured. | Missing from `.env` locally or Streamlit Secrets in the cloud. |
| OpenRouter API quota/rate limit reached. | OpenRouter quota or rate limit. Wait and retry later. |
| Google Play collection failed | The scraper was blocked or the store request failed. The safe error is shown; Apple RSS may still succeed. |
| Streamlit Cloud empty after reboot | Local SQLite is ephemeral on Cloud. Collect again after restart. |

## Dashboard

The Streamlit dashboard is a single scrolling page: header, KPI cards, wishlist → purchase indicator, numbered discovery sections (problems through evidence explorer), the Product Manager assistant, and data limitations.

The sidebar holds compact controls only (date range, refresh, analyze). There is no Auto Refresh ON/OFF toggle. Sources are checked automatically every 5 minutes.

Period defaults to **Last 30 Days** and can be switched to **All Time**. Last-30-day metrics use review timestamps only.

If nothing has been collected yet, the UI shows:

`No real reviews available for analysis.`

plus **Collect Last 30 Days** and **Refresh Latest Reviews**.

Every insight opens original review text, source, URL, review ID, review date, rating, region, and classification. Quotes are never generated.

Default filter: **Myntra-valid evidence only**.

## Deploy to Streamlit Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Connect GitHub.
3. Select repository: `kumarsak621-max/Myntra-AI-discovery-wishlist`
4. Select branch: `main`
5. Select the Streamlit main file: `app.py`
6. Add Streamlit Secrets (App settings → Secrets) in TOML:

```toml
OPENROUTER_API_KEY = "your_openrouter_key"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
MAX_TOTAL_REVIEWS = 500
MAX_DATASET_REVIEWS = 500
MAX_ANALYSIS_REVIEWS = 150
MAX_DISCOVERY_REVIEWS = 150
AI_MAX_TOKENS = 2000
AI_BATCH_SIZE = 10
```

Use a real OpenRouter key that starts with `sk-or-v1-`. Do not paste a Google Gemini `AIza...` key. Do not wrap the key in extra quotes inside the value. After saving secrets, click **Test OpenRouter Connection**. "API KEY Configured" only means a value exists; HTTP 401 means OpenRouter rejected that credential.

7. Deploy.

Official source IDs used by the app (do not change them):

- Google Play: `com.myntra.android`
- Apple App Store: `907394059` (India first, US fallback)

Do not put a real API key in this README. Do not commit `.env` or `.streamlit/secrets.toml`.

Python version: 3.12 (`runtime.txt`).

## Tests

```bash
pytest -q
python -m utils.diagnostics
```

## Security

- `.env` and API keys must never be committed
- Author names are not shown in the UI
- Only publicly available store reviews are collected
- Original review text is never overwritten

## Known limitations

- Streamlit Cloud disk is ephemeral; SQLite does not persist across restarts or redeploys.
- Google Play collection can fail from some cloud IPs if the Play Store blocks the scraper. Apple iTunes RSS is usually more reliable from cloud hosts.
- Keep “max reviews per source” small on Streamlit Cloud to stay within request time limits.
