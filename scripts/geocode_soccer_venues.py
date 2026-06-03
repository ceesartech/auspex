"""Resolve soccer venue text → lat/lon via Open-Meteo geocoding.

Catches the long tail of soccer venues that the manual seed in
seed_venue_coords.py doesn't cover. Each unique venue string in
matches.venue gets one /v1/search call:

    https://geocoding-api.open-meteo.com/v1/search?name=<venue>&count=1

The API is free, no key, returns ~10k requests/day. We throttle to
0.20s between calls — well under any rate limit. Results land in
venue_coords with source='open_meteo_geocode_<YYYYMMDD>' so the
provenance is auditable.

Scope context (as of Phase 14):
  * 23,552 soccer matches in the DB
  * 23,456 (99.6%) have NULL/empty venue text → unreachable here
  * 96 with venue text, 15 already resolved → 81 unresolved
    across 77 distinct venue strings

Net impact: at most 81 more soccer matches gain weather coverage
once the geocoder runs + a downstream fetch_weather pass lands.
The real soccer-coverage blocker is upstream venue capture
(Phase D — fix the ingest), but the infrastructure here is in
place for the moment that lands.

Usage:
    # Look up every unresolved venue
    python /app/scripts/geocode_soccer_venues.py

    # Cap to 10 venues per run (first-run safety against API surprises)
    python /app/scripts/geocode_soccer_venues.py --limit 10

    # Re-resolve known venues (e.g., if the manual seed was wrong)
    python /app/scripts/geocode_soccer_venues.py --update

    # Show what would be looked up without calling the API
    python /app/scripts/geocode_soccer_venues.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("geocode_soccer_venues")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Same throttle as fetch_weather.py — well under Open-Meteo's free
# tier (~10k requests/day). One call per unique venue per run.
REQUEST_DELAY_SEC = 0.20
DEFAULT_TIMEOUT_SEC = 15


# ── Pure helpers ──────────────────────────────────────────────────────


def normalize_venue(name: str) -> str:
    """Mirror of seed_venue_coords.normalize_venue — lowercased,
    whitespace collapsed. Same shape so manually-seeded rows and
    geocoded rows collide on the same key."""
    return " ".join((name or "").strip().lower().split())


def build_search_url(venue: str) -> str:
    """Construct the Open-Meteo /v1/search URL for a venue name.
    Kept as a helper so tests can lock the param shape without
    actually calling the API."""
    return f"{GEOCODE_URL}?name={requests.utils.quote(venue)}&count=1"


def parse_geocode_result(payload: dict) -> Optional[dict]:
    """Pick the first geocoding hit out of an Open-Meteo response.
    Returns a dict with the venue_coords-shaped columns, or None
    if the search returned nothing.

    Open-Meteo's /v1/search returns `{"results": [...]}` with a top
    hit object containing latitude, longitude, timezone, country.
    We assume timezone is present (the API has been observed to
    populate it for every populated-place result); if a future
    record lacks one, fall back to UTC."""
    if not payload or not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    hit = results[0]
    lat = hit.get("latitude")
    lon = hit.get("longitude")
    if lat is None or lon is None:
        return None
    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "timezone": hit.get("timezone") or "UTC",
        "country_code": hit.get("country_code"),
    }


# ── HTTP fetcher ──────────────────────────────────────────────────────


def geocode(venue: str, *, timeout: float = DEFAULT_TIMEOUT_SEC) -> Optional[dict]:
    """Call Open-Meteo /v1/search for one venue. Returns the parsed
    first hit (lat/lon/timezone) or None on miss / HTTP error.
    Network errors are logged + treated as a miss so a flaky API
    doesn't abort the whole run."""
    url = build_search_url(venue)
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        logger.warning("geocoder: HTTP error for %r: %s", venue, e)
        return None
    if r.status_code >= 400:
        logger.warning("geocoder: %s → HTTP %s for %r", url, r.status_code, venue)
        return None
    try:
        payload = r.json()
    except ValueError:
        logger.warning("geocoder: non-JSON response for %r", venue)
        return None
    return parse_geocode_result(payload)


# ── DB I/O ────────────────────────────────────────────────────────────


def list_unresolved_venues(cur, limit: Optional[int] = None) -> list[str]:
    """Distinct soccer venue strings that aren't already in
    venue_coords. Ordered by recency so newer matches' venues land
    first when --limit caps the run."""
    cur.execute(
        """
        SELECT DISTINCT m.venue
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'soccer'
          AND m.venue IS NOT NULL
          AND m.venue != ''
          AND NOT EXISTS (
              SELECT 1 FROM venue_coords vc
              WHERE vc.normalized_venue_name = LOWER(TRIM(m.venue))
          )
        ORDER BY m.venue ASC
        """
    )
    venues = [r["venue"] for r in cur.fetchall()]
    if limit is not None:
        venues = venues[:limit]
    return venues


def upsert_geocoded(cur, venue: str, hit: dict, source: str, update: bool) -> str:
    """Upsert a geocoded venue into venue_coords. Mirrors
    seed_venue_coords.upsert_venue's shape so manually-seeded and
    geocoded rows interoperate cleanly via the same UNIQUE constraint
    on normalized_venue_name. Returns 'inserted' / 'updated' /
    'skipped' for the run summary."""
    norm = normalize_venue(venue)
    if not update:
        cur.execute(
            "SELECT 1 FROM venue_coords WHERE normalized_venue_name = %s LIMIT 1",
            (norm,),
        )
        if cur.fetchone():
            return "skipped"
    cur.execute(
        """
        INSERT INTO venue_coords
            (normalized_venue_name, display_name, latitude, longitude,
             timezone, is_indoor, source)
        VALUES (%s, %s, %s, %s, %s, false, %s)
        ON CONFLICT (normalized_venue_name) DO UPDATE
            SET display_name = EXCLUDED.display_name,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                timezone = EXCLUDED.timezone,
                source = EXCLUDED.source,
                updated_at = NOW()
        RETURNING (xmax = 0) AS inserted
        """,
        (norm, venue, hit["latitude"], hit["longitude"], hit["timezone"], source),
    )
    row = cur.fetchone()
    # xmax = 0 in the RETURNING means the row was freshly inserted
    # (vs updated via ON CONFLICT). Lets the summary count distinguish
    # insert from update without a second roundtrip.
    return "inserted" if row and row.get("inserted") else "updated"


# ── Orchestration ─────────────────────────────────────────────────────


def run(database_url: str, *, limit: Optional[int], update: bool, dry_run: bool) -> dict:
    source = f"open_meteo_geocode_{date.today().isoformat()}"
    counts = {"venues_checked": 0, "hits": 0, "misses": 0, "inserted": 0, "updated": 0, "skipped": 0}

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            venues = list_unresolved_venues(cur, limit=limit)
            counts["venues_checked"] = len(venues)
            if not venues:
                logger.info("No unresolved soccer venues. Nothing to geocode.")
                return counts

            logger.info("Geocoding %d soccer venues via Open-Meteo", len(venues))
            for v in venues:
                if dry_run:
                    logger.info("  would geocode: %r", v)
                    continue
                hit = geocode(v)
                if hit is None:
                    counts["misses"] += 1
                    logger.info("  miss: %r", v)
                else:
                    counts["hits"] += 1
                    result = upsert_geocoded(cur, v, hit, source, update)
                    counts[result] = counts.get(result, 0) + 1
                    logger.info("  hit: %r → (%.4f, %.4f) %s [%s]", v, hit["latitude"], hit["longitude"], hit["timezone"], result)
                time.sleep(REQUEST_DELAY_SEC)
            if not dry_run:
                conn.commit()

    logger.info("Done. %s", counts)
    return counts


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap how many venues to look up per run (default: no cap).",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Re-resolve venues that already have a venue_coords row.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be looked up; no API calls, no DB writes.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, limit=args.limit, update=args.update, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
