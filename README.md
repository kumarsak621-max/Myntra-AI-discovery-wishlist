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
```

Leave `GOOGLE_PLAY_APP_ID=com.myntra.android` and `APPLE_APP_ID=907394059`.

Never put a real API key in this README. Never commit `.env`.

Collection still works without an AI key. Analysis is skipped until `OPENROUTER_API_KEY` is set.

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

On **Data Collection**:

- **Collect Google Play Reviews** — live Play Store reviews for `com.myntra.android`
- **Collect Apple App Store Reviews** — iTunes RSS for `907394059` (India first)
- **Collect All** — both sources
- **Analyze stored Myntra-valid reviews** — OpenRouter JSON analysis for new or changed reviews only

Raw reviews are stored in local SQLite (`myntra_discovery.db`), which is gitignored. Collect after clone; do not commit the database.

On Streamlit Cloud the filesystem is **ephemeral**. SQLite is available for the running session/demo and is wiped on reboot or redeploy. This app does not provide permanent cloud persistence.

## AI Analysis

OpenRouter is the default gateway (`AI_PROVIDER=openrouter`). The model is set with `OPENROUTER_MODEL` / `AI_MODEL` (default `google/gemini-2.5-flash`).

Locally the key comes from `.env`. On Streamlit Cloud it comes from Secrets. The API key is never shown in the UI or printed in logs.

The LLM extracts structured JSON: relevance, wishlist/purchase signals, intents, barriers, uncertainties, information-seeking, behavioral signals, and observed / inferred / hypothesized root causes.

Python (not the LLM) calculates counts, percentages, clustering inputs, and opportunity scores:

```
score = reach × frequency × purchase_impact × severity × evidence_confidence
```

Each dimension is 1–5.

Analysis is cached by review content hash. Unchanged reviews are not sent to the model again.

## Dashboard

The dashboard includes Overview, Data Collection, Feedback Explorer, Wishlist Motivations, Purchase Barriers, Uncertainties, Root Causes, Themes, User Segments, Opportunity Matrix, Evidence Explorer, and Discovery Report.

If nothing has been collected yet, the UI shows:

`No data collected yet. Run the data collection pipeline.`

Every insight opens the original review text, source, URL, review ID, date, and classification. Quotes are never generated.

Default filter: **Myntra-valid evidence only**.

## Deploy to Streamlit Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Connect GitHub.
3. Select repository: `kumarsak621-max/Myntra-AI-discovery-wishlist`
4. Select branch: `main`
5. Select the Streamlit main file: `app.py`
6. Add Streamlit Secrets (App settings → Secrets) in TOML:

```toml
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
```

7. Deploy.

Official source IDs used by the app (do not change them):

- Google Play: `com.myntra.android`
- Apple App Store: `907394059` (India first, US fallback)

Do not put a real API key in this README. Do not commit `.env` or `.streamlit/secrets.toml`.

Python version: 3.12 (`runtime.txt`).

## Tests

```bash
pytest -q
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
