# Data

Collected reviews live in the local SQLite file `myntra_discovery.db` at the project root.

That database is **not** committed. It can contain public review text and should be rebuilt on each machine.

On Streamlit Cloud the filesystem is ephemeral. SQLite works for a running session but is not permanent production storage.

## How to collect

1. Copy `.env.example` to `.env`.
2. Run `python app.py`.
3. Open http://127.0.0.1:8000
4. Use **Collect Google Play Reviews**, **Collect Apple App Store Reviews**, or **Collect All**.

Official sources:

- Google Play: `com.myntra.android`
- App Store: `907394059` (India first, US fallback)

Do not commit raw dumps, cookies, credentials, or `.env`.
