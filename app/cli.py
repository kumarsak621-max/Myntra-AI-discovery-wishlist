"""CLI entrypoint: python -m app.cli collect --max-reviews 50"""

from __future__ import annotations

import argparse
import json
import logging

from app.collectors.engine import CollectionEngine
from app.database import SessionLocal, init_db


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Myntra wishlist-to-purchase discovery engine")
    sub = parser.add_subparsers(dest="cmd", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--sources", default="google_play,apple_app_store")
    collect.add_argument("--max-reviews", type=int, default=None)
    collect.add_argument("--no-analyze", action="store_true")
    analyze = sub.add_parser("analyze")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.cmd == "collect":
            engine = CollectionEngine(db)
            sources = [s.strip() for s in args.sources.split(",") if s.strip()]
            stats = engine.run(
                sources,
                max_reviews=args.max_reviews,
                analyze=not args.no_analyze,
                progress=lambda e: logging.info("%s", e.get("status") or e.get("stage")),
            )
            print(json.dumps(stats.model_dump(mode="json"), indent=2, default=str))
        elif args.cmd == "analyze":
            from app.pipeline.orchestrator import run_analysis_pipeline

            n = run_analysis_pipeline(db)
            print(json.dumps({"analyzed": n}))
    finally:
        db.close()


if __name__ == "__main__":
    main()
