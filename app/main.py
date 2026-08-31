"""FastAPI entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Myntra Wishlist-to-Purchase Discovery Engine",
    description=(
        "Discovers why wishlisted fashion products fail to become purchases, "
        "using real public feedback. Does not propose product solutions."
    ),
    version="1.0.0",
)
app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    init_db()


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
