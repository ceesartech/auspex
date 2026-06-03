"""Resolve tennis venue-cities → lat/lon via Open-Meteo geocoding.

Tennis matches.venue strings in this corpus are city names (e.g.
"paris, france", "guadalajara, mexico", "auckland, new zealand")
rather than literal stadium names, because the upstream tennis
data source records the host city, not the venue. This is GOOD
for /v1/search (which indexes populated places) — the Phase C
finding that Open-Meteo geocoder gets ~0% hit rate on stadia
(see weather-features-attempted memory) doesn't apply here.

Direct counterpart of scripts/geocode_soccer_venues.py with the
only material change being WHERE l.sport = 'tennis'. The soccer
script is hardcoded to sport='soccer' so this script reuses its
geocode / upsert helpers via import rather than parameterising
both scripts on --sport. Same source tag shape so manually-seeded
and geocoded rows interoperate.

Run on prod via:
    docker compose exec api python /app/scripts/geocode_tennis_venues.py
"""

import argparse
import logging
import os
import sys
import time
from datetime import date
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geocode_soccer_venues import (  # noqa: E402
    DEFAULT_TIMEOUT_SEC,
    geocode,
    upsert_geocoded,
)

LOGGER = logging.getLogger("geocode_tennis_venues")

# Throttle matches geocode_soccer_venues so the two scripts can't
# run side-by-side and double Open-Meteo's effective request rate.
REQUEST_DELAY_SEC = 0.20


def list_unresolved_tennis_venues(cur, limit: Optional[int]) -> list[str]:
    """Distinct tennis venue strings not already in venue_coords,
    ordered by match count desc so the top unseeded cities get
    picked up first under --limit."""
    cur.execute(
        """
        SELECT m.venue, COUNT(*) AS matches
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'tennis'
          AND m.venue IS NOT NULL
          AND m.venue <> ''
          AND NOT EXISTS (
              SELECT 1 FROM venue_coords vc
              WHERE vc.normalized_venue_name = LOWER(TRIM(m.venue))
          )
        GROUP BY m.venue
        ORDER BY matches DESC
        """
    )
    rows = cur.fetchall()
    venues = [r["venue"] for r in rows]
    if limit is not None:
        venues = venues[:limit]
    return venues


def run(database_url: str, *, limit: Optional[int], update: bool, dry_run: bool) -> dict:
    source = f"open_meteo_geocode_{date.today().isoformat()}"
    counts = {
        "venues_checked": 0,
        "hits": 0,
        "misses": 0,
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            venues = list_unresolved_tennis_venues(cur, limit)
            LOGGER.info("%d unresolved tennis venues to geocode.", len(venues))

            for i, venue in enumerate(venues, start=1):
                counts["venues_checked"] += 1
                hit = geocode(venue, timeout=DEFAULT_TIMEOUT_SEC)
                time.sleep(REQUEST_DELAY_SEC)
                if not hit:
                    counts["misses"] += 1
                    continue
                counts["hits"] += 1
                if dry_run:
                    LOGGER.info(
                        "[dry-run] would seed venue=%r lat=%.4f lon=%.4f tz=%s",
                        venue,
                        hit["latitude"],
                        hit["longitude"],
                        hit["timezone"],
                    )
                    continue
                outcome = upsert_geocoded(cur, venue, hit, source, update)
                counts[outcome] = counts.get(outcome, 0) + 1
                if i % 25 == 0:
                    conn.commit()
                    LOGGER.info("%d/%d venues done", i, len(venues))
            if not dry_run:
                conn.commit()

    LOGGER.info("Done. %s", counts)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap to the first N unresolved venues (smoke test).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Re-geocode venues already in venue_coords (default: skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print would-be inserts without touching the DB.",
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

    run(
        args.database_url,
        limit=args.limit,
        update=args.update,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
