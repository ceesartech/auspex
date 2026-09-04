"""Dedup-aware tennis weather backfill (one-off).

`fetch_weather.py --backfill-days N` makes one Open-Meteo archive
call per match. Tennis has 11,541 finished matches with seeded
outdoor venues but only 938 distinct (venue, date) tuples — many
matches share a city + day. This script collapses to those 938
tuples (~3 minutes at 0.2s throttle), reusing the fetch + write
helpers from fetch_weather.py.

Motivation: the v4b tennis A/B (scripts/ab_tennis_weather.py) ran
at 4.9% outdoor weather coverage and was ambiguous (non-negative
on every metric but ΔBrier < 0.005). Hypothesis is that coverage,
not the feature itself, is the bottleneck. This backfill lifts
coverage to whatever fraction of finished tennis matches have a
seeded outdoor venue (≈47%) before re-running the A/B.

Reads the same `match_weather`, `venue_coords`, `matches`, and
`leagues` tables fetch_weather.py uses. Writes one
`match_weather` row per match (so the existing
`match_weather_latest` view continues to function unchanged).

Run on prod via:
    docker compose exec api python /app/scripts/backfill_tennis_weather.py
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Import the helpers from fetch_weather rather than re-implementing
# them — keeps the API/throttle/normalization logic in one place so
# the two scripts can't drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_weather import REQUEST_DELAY_SEC, fetch_archive, match_window_summary, write_weather  # noqa: E402

LOGGER = logging.getLogger("backfill_tennis_weather")


# All (venue, date) tuples that:
#   - belong to a finished tennis match
#   - the venue is seeded in venue_coords AND outdoor
#   - no existing weather row covers any match at this tuple
# Each tuple is one API call. Per-tuple, we then write the same
# fetched summary to every match_id at (venue, date).
TUPLES_QUERY = """
    SELECT
        vc.id::text AS venue_id,
        vc.latitude,
        vc.longitude,
        vc.timezone,
        DATE(m.match_date) AS match_date,
        array_agg(DISTINCT m.id::text) AS match_ids,
        MIN(m.match_date) AS earliest_match_dt
    FROM matches m
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'tennis'
    JOIN venue_coords vc
        ON vc.normalized_venue_name = LOWER(TRIM(m.venue))
       AND vc.is_indoor = FALSE
    LEFT JOIN match_weather_latest mwl ON mwl.match_id = m.id
    WHERE m.status = 'finished'
      AND mwl.match_id IS NULL
    GROUP BY vc.id, vc.latitude, vc.longitude, vc.timezone, DATE(m.match_date)
    ORDER BY match_date ASC
"""


def run(database_url: str, limit: Optional[int]) -> dict:
    counts = {
        "tuples_total": 0,
        "tuples_processed": 0,
        "tuples_fetch_failed": 0,
        "tuples_no_window_data": 0,
        "matches_written": 0,
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(TUPLES_QUERY)
            tuples = cur.fetchall()
            counts["tuples_total"] = len(tuples)
            LOGGER.info(
                "Found %d distinct (venue, date) tuples to backfill.",
                len(tuples),
            )
            if limit is not None:
                tuples = tuples[:limit]
                LOGGER.info("Capping to first %d tuples (--limit).", len(tuples))

            for i, t in enumerate(tuples, start=1):
                # Use the earliest match_dt at this (venue, date) as
                # the window anchor. Multiple matches at the same
                # venue on the same day usually run within hours
                # of each other; the window summary covers that.
                match_dt = t["earliest_match_dt"]
                raw = fetch_archive(
                    float(t["latitude"]),
                    float(t["longitude"]),
                    match_dt,
                    t["timezone"],
                )
                time.sleep(REQUEST_DELAY_SEC)

                if not raw:
                    counts["tuples_fetch_failed"] += 1
                    LOGGER.warning(
                        "Fetch failed for venue=%s date=%s",
                        t["venue_id"],
                        t["match_date"],
                    )
                    continue

                hourly = raw.get("hourly") or {}
                summary = match_window_summary(hourly, match_dt)
                if not summary:
                    counts["tuples_no_window_data"] += 1
                    continue

                for match_id in t["match_ids"]:
                    write_weather(
                        cur,
                        match_id=match_id,
                        venue_id=t["venue_id"],
                        data_kind="actual",
                        summary=summary,
                        raw=raw,
                    )
                    counts["matches_written"] += 1

                counts["tuples_processed"] += 1

                # Commit + progress log every 50 tuples so a mid-run
                # interruption doesn't lose hours of work.
                if i % 50 == 0:
                    conn.commit()
                    LOGGER.info(
                        "%d/%d tuples done (%d matches written so far)",
                        i,
                        len(tuples),
                        counts["matches_written"],
                    )

            conn.commit()

    LOGGER.info("Done. %s", counts)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres URL (defaults to DATABASE_URL env)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: cap to the first N tuples (smoke test).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
    )

    if not args.database_url:
        LOGGER.error("DATABASE_URL not set.")
        return 1

    run(args.database_url, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
