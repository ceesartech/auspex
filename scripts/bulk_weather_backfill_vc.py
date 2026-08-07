"""Bulk Visual Crossing weather backfill for the 7x soccer corpus (weather
revisit, option 1 — operator re-subscribed VC 2026-08-07).

Reuses fetch_weather_visual_crossing's fetch_timeline / match_window_summary
/ write_weather so rows are byte-identical vc_actual snapshots; this driver
just changes the SELECTION (every finished soccer match lacking vc_actual,
no 14-day window, no LIMIT) and the VENUE RESOLUTION (the 715
'team-stadium:{team_id}' venue_coords rows from the corpus expansion —
football-data matches carry no venue strings).

Cost model: one timeline call per (stadium, match day) = 1 metered record
(~$0.0001) — fetching only match days beats range queries ~10:1 because a
stadium hosts ~20 matches across a 270-day season. Full remaining corpus
~= $12-14 of records, ~6h at 0.15s pacing.

    python /app/scripts/bulk_weather_backfill_vc.py
    python /app/scripts/bulk_weather_backfill_vc.py --limit-matches 500
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("bulk_weather_backfill_vc")


def _load_vc():
    if "fetch_weather_visual_crossing" in sys.modules:
        return sys.modules["fetch_weather_visual_crossing"]
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "fetch_weather_visual_crossing", os.path.join(here, "fetch_weather_visual_crossing.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_weather_visual_crossing"] = mod
    spec.loader.exec_module(mod)
    return mod


vc = _load_vc()
DELAY_S = 0.15


def missing_matches(cur, limit: Optional[int]) -> list[dict]:
    """Finished soccer matches lacking a vc_actual snapshot, joined to
    their home team's stadium coords. Ordered newest-first so the most
    model-relevant (recent) matches land earliest under any quota cut."""
    cur.execute(
        """
        SELECT m.id::text AS match_id, m.match_date,
               vc.id::text AS venue_id, vc.latitude, vc.longitude
        FROM matches m
        JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
        JOIN venue_coords vc
          ON vc.normalized_venue_name = 'team-stadium:' || m.home_team_id::text
        WHERE m.status = 'finished'
          AND NOT EXISTS (
              SELECT 1 FROM match_weather mw
              WHERE mw.match_id = m.id AND mw.data_kind = 'vc_actual'
          )
        ORDER BY m.match_date DESC
        """
        + (" LIMIT %d" % int(limit) if limit else "")
    )
    return list(cur.fetchall())


def run(database_url: str, limit: Optional[int]) -> dict:
    counts = {"written": 0, "fetch_failed": 0, "no_window": 0}
    key = vc._api_key()
    if not key:
        logger.error("VISUAL_CROSSING_API_KEY not set")
        return counts
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            todo = missing_matches(cur, limit)
            logger.info("VC backfill: %d matches to fetch", len(todo))
            for i, m in enumerate(todo, 1):
                raw = vc.fetch_timeline(
                    float(m["latitude"]), float(m["longitude"]), m["match_date"], "UTC", api_key=key
                )
                if raw is None:
                    counts["fetch_failed"] += 1
                    # A run of consecutive failures = quota/auth problem; bail
                    # loudly rather than burn hours on 429s.
                    if counts["fetch_failed"] >= 50 and counts["written"] == 0:
                        logger.error("50 consecutive failures — aborting (check key/quota)")
                        break
                else:
                    summary = vc.match_window_summary(raw, m["match_date"])
                    if summary:
                        vc.write_weather(
                            cur,
                            match_id=m["match_id"],
                            venue_id=m["venue_id"],
                            data_kind="vc_actual",
                            summary=summary,
                            raw=None,
                        )
                        counts["written"] += 1
                    else:
                        counts["no_window"] += 1
                if i % 500 == 0:
                    conn.commit()
                    logger.info("progress %d/%d: %s", i, len(todo), counts)
                time.sleep(DELAY_S)
            conn.commit()
    logger.info("Done: %s", counts)
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-matches", type=int, help="Cap this run (default: all missing).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    counts = run(args.database_url, args.limit_matches)
    return 0 if counts["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
