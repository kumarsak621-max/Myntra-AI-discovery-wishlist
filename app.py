"""
Myntra AI Discovery Engine — main entry point.

Local FastAPI dashboard:
    python app.py
    http://127.0.0.1:8000

Streamlit (local or Streamlit Cloud):
    streamlit run app.py
    http://127.0.0.1:8501
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config.settings import (  # noqa: E402
    OFFICIAL_APPLE_APP_ID,
    OFFICIAL_APPLE_APP_URL,
    OFFICIAL_GOOGLE_PLAY_APP_ID,
    OFFICIAL_GOOGLE_PLAY_URL,
    get_settings,
    is_banned_app_id,
)
from utils.logging import configure_logging, startup_line  # noqa: E402

LOGGER = logging.getLogger("myntra.discovery")


class DiscoveryEngineApp:
    """Orchestrator. Business logic lives in collectors, analysis, and services."""

    def __init__(self) -> None:
        self.settings = None
        self.google_play = None
        self.app_store = None
        self.ai = None
        self.identities: dict = {}

    def initialize(self) -> None:
        configure_logging()
        print("=" * 60)
        print("MYNTRA AI DISCOVERY ENGINE")
        print("=" * 60)

        self.settings = get_settings()
        self._validate_configuration()
        startup_line("Configuration loaded")

        from app.database import init_db

        init_db()
        startup_line("Database initialized")

        from services.ai_service import GeminiAIService

        self.ai = GeminiAIService(self.settings)
        if self.ai.available():
            startup_line(f"AI provider initialized (Google Gemini / {self.settings.resolved_model})")
        else:
            startup_line(
                "AI provider initialized (Gemini API key is not configured. Collection still works; analysis is skipped.)",
                ok=True,
            )

        from app.collectors.app_store import AppStoreCollector
        from app.collectors.google_play import GooglePlayCollector

        self.google_play = GooglePlayCollector(self.settings)
        self.app_store = AppStoreCollector(self.settings)
        self._validate_identities()
        startup_line("Google Play collector ready")
        startup_line("Apple App Store collector ready")
        startup_line("Analysis pipeline ready")
        startup_line("Dashboard ready")
        print()
        print("Application running...")
        print(f"Open http://{self.settings.host}:{self.settings.port}")
        print()

    def _validate_configuration(self) -> None:
        s = self.settings
        errors: list[str] = []
        if is_banned_app_id(s.google_play_app_id):
            errors.append(
                f"GOOGLE_PLAY_APP_ID={s.google_play_app_id} is banned. "
                f"Use {OFFICIAL_GOOGLE_PLAY_APP_ID}"
            )
        if is_banned_app_id(s.apple_app_id):
            errors.append(
                f"APPLE_APP_ID={s.apple_app_id} is banned. Use {OFFICIAL_APPLE_APP_ID}"
            )
        if s.google_play_app_id != OFFICIAL_GOOGLE_PLAY_APP_ID:
            errors.append(
                f"GOOGLE_PLAY_APP_ID must be {OFFICIAL_GOOGLE_PLAY_APP_ID} "
                f"(got {s.google_play_app_id})"
            )
        if s.apple_app_id != OFFICIAL_APPLE_APP_ID:
            errors.append(
                f"APPLE_APP_ID must be {OFFICIAL_APPLE_APP_ID} (got {s.apple_app_id})"
            )
        if errors:
            for item in errors:
                LOGGER.error(item)
                startup_line(item, ok=False)
            LOGGER.error("Fix .env before collecting Myntra evidence. Dashboard will still start.")

    def _validate_identities(self) -> None:
        try:
            gp = self.google_play.validate_source()
        except Exception as exc:
            LOGGER.exception("Google Play identity validation failed")
            startup_line(f"Google Play identity validation error: {exc}", ok=False)
            gp = None
        try:
            apple = self.app_store.validate_source()
        except Exception as exc:
            LOGGER.exception("Apple App Store identity validation failed")
            startup_line(f"Apple App Store identity validation error: {exc}", ok=False)
            apple = None

        self.identities = {"google_play": gp, "apple_app_store": apple}

        if gp is not None:
            print()
            print("Source: Google Play")
            print(f"Configured ID: {gp.app_id}")
            print(f"Expected ID: {OFFICIAL_GOOGLE_PLAY_APP_ID}")
            print(f"URL: {OFFICIAL_GOOGLE_PLAY_URL}")
            print(f"Detected App: {gp.detected_app_name or '—'}")
            print(f"Expected App: {gp.expected_app}")
            print(f"Validation: {gp.validation_result}")
            if gp.is_valid_for_myntra:
                startup_line("Google Play: Myntra identity validated")
            else:
                startup_line(gp.warning or "Google Play identity validation FAIL", ok=False)

        if apple is not None:
            print()
            print("Source: Apple App Store")
            print(f"Configured ID: {apple.app_id}")
            print(f"Expected ID: {OFFICIAL_APPLE_APP_ID}")
            print(f"URL: {OFFICIAL_APPLE_APP_URL}")
            print(f"Detected App: {apple.detected_app_name or '—'}")
            print(f"Expected App: {apple.expected_app}")
            print(f"Validation: {apple.validation_result}")
            if apple.is_valid_for_myntra:
                startup_line("Apple App Store: Myntra identity validated")
            else:
                startup_line(apple.warning or "Apple App Store identity validation FAIL", ok=False)
        print()

    def run(self) -> None:
        import uvicorn

        uvicorn.run(
            "app.main:app",
            host=self.settings.host,
            port=self.settings.port,
            reload=False,
            log_level="info",
        )


def main() -> None:
    application = DiscoveryEngineApp()
    try:
        application.initialize()
        application.run()
    except KeyboardInterrupt:
        print("\nShutting down.")
    except Exception as exc:
        LOGGER.exception("Application failed to start")
        print(f"[FAIL] {exc}")
        sys.exit(1)


def running_in_streamlit() -> bool:
    """True when this file is executed by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if running_in_streamlit():
    from dashboard.streamlit_app import render

    render()
elif __name__ == "__main__":
    main()
