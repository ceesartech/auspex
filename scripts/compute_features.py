"""Compute features for matches missing them in features_cache.

Wraps services/feature-engineering's RealTimeFeatureComputer. For every
scheduled match in the next N days (default 7) that doesn't have a fresh
features_cache row, runs the full 250+ feature pipeline and writes the
result to features_cache. Idempotent — re-running only touches matches
that lack cached features.

Usage:
    python /app/scripts/compute_features.py             # next 7 days scheduled
    python /app/scripts/compute_features.py --days 14
    python /app/scripts/compute_features.py --match-ids id1,id2,id3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("compute_features")

# Make the feature-engineering package importable. orchestrator.py and
# its siblings use relative imports (from .categories.base import …),
# which requires them to load as part of a package — not as top-level
# modules. So we put the *parent* of src/ on sys.path and import the
# modules via the `src.*` namespace.
sys.path.insert(0, "/app/services/feature-engineering")


def list_target_matches(database_url: str, days: int) -> list[str]:
    """Return UUIDs of scheduled matches in the next N days without features."""
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT m.id::text AS id
                FROM matches m
                LEFT JOIN features_cache f
                  ON f.match_id = m.id AND f.expires_at > NOW()
                WHERE m.status = 'scheduled'
                  AND m.match_date BETWEEN NOW() AND NOW() + INTERVAL %s
                  AND f.id IS NULL
                ORDER BY m.match_date ASC
                """,
                (f"{days} days",),
            )
            return [r["id"] for r in cur.fetchall()]


def compute(match_ids: list[str]) -> dict[str, int]:
    """Run the orchestrator for each match id. Returns {ok,fail} counts."""
    from redis import Redis
    from src.core.config import FeatureConfig  # type: ignore
    from src.core.database import DatabaseManager  # type: ignore
    from src.orchestrator import RealTimeFeatureComputer  # type: ignore

    config = FeatureConfig()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url, decode_responses=False)

    computer = RealTimeFeatureComputer(config, db, redis)
    ok = fail = 0
    for mid in match_ids:
        try:
            features = computer.compute_features(mid, use_cache=True, validate=False)
            if features:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            logger.warning("Feature compute failed for %s: %s", mid, e)
            fail += 1
    return {"ok": ok, "fail": fail}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--match-ids", help="Comma-separated UUIDs to compute (overrides --days).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2

    if args.match_ids:
        match_ids = [s.strip() for s in args.match_ids.split(",") if s.strip()]
    else:
        match_ids = list_target_matches(args.database_url, args.days)

    if not match_ids:
        logger.info("No matches needing features in the next %d days", args.days)
        return 0

    logger.info("Computing features for %d match(es)...", len(match_ids))
    counts = compute(match_ids)
    logger.info("Done: %d ok / %d failed", counts["ok"], counts["fail"])
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
