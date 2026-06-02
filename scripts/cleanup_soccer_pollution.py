"""One-shot cleanup of soccer-derived predictions on non-soccer matches.

Before commit 895eb6f, scripts/precompute_predictions.py's
list_upcoming() didn't filter by sport — it walked EVERY scheduled
match and ran the soccer ensemble + Dixon-Coles derivation on each.
Result: NHL and NBA matches accumulated ~30 garbage prediction rows
each (one per derived soccer market: 1X2, over_under at multiple
lines, asian_handicap, double_chance, draw_no_bet, correct_score,
etc.) keyed on model_name='ensemble'.

Nothing breaks downstream because the API filters predictions by
sport via TaskSpec — those rows never get served. But they:
  - Bloat the predictions table (~30 rows × ~25 non-soccer matches
    per pipeline run = ~750 rows/week of pure junk on each retrain)
  - Confuse diagnostic queries (the "150 ensemble rows for NBA"
    finding from our prod debug)
  - Waste a few CPU-seconds per pipeline run

The fix landed in 895eb6f to STOP new pollution. This script cleans
up the rows that accumulated before that.

Idempotent: re-runs after the first cleanup are no-ops. Safe to run
on prod whenever — the rows are never served, so there's no risk of
breaking active predictions.

Usage:
    # See how many rows WOULD be deleted, without touching anything.
    python /app/scripts/cleanup_soccer_pollution.py --database-url "$DATABASE_URL"

    # Actually delete.
    python /app/scripts/cleanup_soccer_pollution.py --database-url "$DATABASE_URL" --confirm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("cleanup_soccer_pollution")

# The legacy soccer artifact name. Every soccer prediction (whether
# from the trained ensemble or the Dixon-Coles derivation) lands in
# the predictions table with this model_name — see TaskSpec.db_model_name
# in services/api/src/services/prediction_service.py. So pollution rows
# are identifiable by (model_name='ensemble', match.sport != 'soccer').
SOCCER_MODEL_NAME = "ensemble"


def count_pollution(cur) -> dict:
    """Tally rows per non-soccer sport that match the pollution
    signature. Lets the operator eyeball cleanup scope before
    consenting to delete."""
    cur.execute(
        """
        SELECT l.sport, count(*) AS rows
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        JOIN leagues l ON l.id = m.league_id
        WHERE p.model_name = %s
          AND l.sport <> 'soccer'
        GROUP BY l.sport
        ORDER BY rows DESC
        """,
        (SOCCER_MODEL_NAME,),
    )
    return {r["sport"]: r["rows"] for r in cur.fetchall()}


def delete_pollution(cur) -> int:
    """Delete the pollution rows and return the affected row count.
    Uses a sub-SELECT on the matches join to keep the DELETE plan
    indexable (vs a wide IN clause)."""
    cur.execute(
        """
        DELETE FROM predictions p
        WHERE p.model_name = %s
          AND p.match_id IN (
              SELECT m.id
              FROM matches m
              JOIN leagues l ON l.id = m.league_id
              WHERE l.sport <> 'soccer'
          )
        """,
        (SOCCER_MODEL_NAME,),
    )
    return cur.rowcount


def run(database_url: str, confirm: bool) -> dict:
    summary = {"dry_run": not confirm, "by_sport": {}, "total": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            by_sport = count_pollution(cur)
            summary["by_sport"] = by_sport
            summary["total"] = sum(by_sport.values())
            if not by_sport:
                logger.info("No soccer-on-non-soccer pollution rows found — DB is clean.")
                return summary

            logger.info("Pollution rows by sport:")
            for sport, n in by_sport.items():
                logger.info("  %s: %d rows", sport, n)
            logger.info("Total: %d rows", summary["total"])

            if not confirm:
                logger.info(
                    "DRY RUN — re-run with --confirm to actually delete these rows. "
                    "They're invisible to the API (filtered by sport via TaskSpec) "
                    "and safe to remove."
                )
                return summary

            deleted = delete_pollution(cur)
            conn.commit()
            summary["deleted"] = deleted
            logger.info("Deleted %d rows.", deleted)
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without this flag, the script reports counts and exits.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    summary = run(args.database_url, args.confirm)
    logger.info("Done. %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
