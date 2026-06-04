"""Fetch weather for outdoor-sport matches via Visual Crossing.

Visual Crossing (https://www.visualcrossing.com/) is the paid weather
provider chosen 2026-06-04 to retest the weather-features hypothesis
after Open-Meteo's 11km grid showed no signal for NFL TOTAL or tennis
moneyline (see weather-features-attempted memory). VC's 1-4km
resolution in NA/EU (sourced from NOAA + ECMWF) is 2-3x sharper, with
50+ years of historical depth — enough to run multi-season A/Bs on
the full NFL + tennis backfill in a single endpoint call per match.

Pricing context (2026-06-04 plans):
  * Free tier: 1,000 records/day — covers development.
  * Pro $35/mo: 100,000 records/month — covers full historical
    backfill (~25k tennis + ~1k NFL = ~26k records, one-time) AND
    the ~60 records/day live forecast volume forever.
  * Standard $100/mo: 1M records/month — overkill at our scale.

Architecture mirrors scripts/fetch_weather.py (the Open-Meteo
fetcher) so the two can run side-by-side during the A/B period
without colliding on the match_weather row schema:

  1. List candidate matches (upcoming forecast OR finished backfill)
     that lack a weather row.
  2. Resolve venue → (lat, lon, timezone) via venue_coords.
  3. Fetch one VC `timeline` call per (venue, date). Single endpoint
     returns hourly + daily blocks for forecast AND historical, so
     no separate paths needed.
  4. Reduce hourly data to the match-window summary and INSERT into
     match_weather with data_kind='vc_forecast' / 'vc_actual'.

API key: VISUAL_CROSSING_API_KEY env var (required for any non-
--dry-run invocation). Free-tier keys work for testing.

Usage:
    python scripts/fetch_weather_visual_crossing.py                       # next 14 days fcst
    python scripts/fetch_weather_visual_crossing.py --days 7              # next 7 only
    python scripts/fetch_weather_visual_crossing.py --backfill-days 90    # last 90
    python scripts/fetch_weather_visual_crossing.py --sport nfl           # scope to NFL
    python scripts/fetch_weather_visual_crossing.py --dry-run --backfill-days 30  # no DB writes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_weather_visual_crossing")


BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/" "rest/services/timeline"
DEFAULT_TIMEOUT_SEC = 30
# VC docs ask for "reasonable rate" — 1/sec is well below any
# documented limit and stays polite during large backfills.
REQUEST_DELAY_SEC = 1.0


def _api_key() -> Optional[str]:
    return os.environ.get("VISUAL_CROSSING_API_KEY")


def fetch_timeline(
    lat: float,
    lon: float,
    target_date: datetime,
    tz: str,
    *,
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Optional[dict]:
    """Fetch one VC timeline page for (lat, lon, date). Returns the
    raw JSON dict, or None on failure. Historical + forecast unified
    on this endpoint — VC infers from the date which dataset to use.
    """
    iso_date = target_date.date().isoformat()
    url = f"{BASE_URL}/{lat},{lon}/{iso_date}/{iso_date}"
    params = {
        "key": api_key,
        "unitGroup": "metric",
        "include": "hours",
        "elements": ("datetime,temp,windspeed,precip,humidity,conditions"),
        "contentType": "json",
        "timezone": tz or "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        logger.warning("Visual Crossing fetch failed for %s: %s", iso_date, e)
        return None
    if r.status_code >= 400:
        body_snippet = r.text[:200] if r.text else ""
        logger.warning(
            "Visual Crossing %s → %d: %s",
            iso_date,
            r.status_code,
            body_snippet,
        )
        return None
    try:
        return r.json()
    except ValueError as e:
        logger.warning("Visual Crossing non-JSON response: %s", e)
        return None


def match_window_summary(raw: dict, match_dt: datetime) -> dict:
    """Reduce the hourly array in a VC timeline response to a single
    summary dict matching the existing match_weather column shape:
    {temperature_c, wind_kmh, precipitation_mm, humidity_pct,
     conditions}. Same averaging window the Open-Meteo fetcher uses
     (±2h around the kickoff hour) for direct comparability."""
    days = raw.get("days") or []
    if not days:
        return {}
    hours = days[0].get("hours") or []
    if not hours:
        return {}

    match_hour = match_dt.hour
    # VC hourly entries are local to the venue's timezone (matches
    # the `timezone=` request param). Pull a ±2h window around the
    # match hour so the summary smooths out single-hour noise.
    window = [
        h
        for h in hours
        if isinstance(h.get("datetime"), str) and abs(int(h["datetime"].split(":")[0]) - match_hour) <= 2
    ]
    if not window:
        return {}

    def _avg(key: str) -> Optional[float]:
        vals = [h[key] for h in window if isinstance(h.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    def _sum(key: str) -> Optional[float]:
        vals = [h[key] for h in window if isinstance(h.get(key), (int, float))]
        return sum(vals) if vals else None

    conditions = next(
        (h.get("conditions") for h in window if h.get("conditions")),
        None,
    )
    return {
        "temperature_c": _avg("temp"),
        "wind_kmh": _avg("windspeed"),
        "precipitation_mm": _sum("precip"),
        "humidity_pct": _avg("humidity"),
        "conditions": conditions,
    }


# ── DB I/O (read-only helpers shared with fetch_weather.py shape) ──


# Path to the team → home-stadium JSON generated by
# scripts/build_soccer_stadium_map.py. The VC fetcher falls back to
# this when matches.venue is NULL (~99.6% of soccer matches because
# football-data.co.uk CSVs don't ship venue text). Path is relative
# to the repo root inside the api container (/app).
DEFAULT_STADIUM_MAP_PATH = "/app/data/soccer_team_stadiums.json"


def load_stadium_map(path: Optional[str] = None) -> dict:
    """Load the team → stadium fallback dict. Returns {} when the
    file isn't present so the script keeps working (just no soccer
    fallback). Path resolves to DEFAULT_STADIUM_MAP_PATH if None."""
    p = Path(path or DEFAULT_STADIUM_MAP_PATH)
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load stadium map %s: %s", p, e)
        return {}


def lookup_venue(cur, venue_name: Optional[str]) -> Optional[dict]:
    if not venue_name:
        return None
    cur.execute(
        """
        SELECT id::text AS id, latitude, longitude, timezone, is_indoor
        FROM venue_coords
        WHERE normalized_venue_name = LOWER(TRIM(%s))
        """,
        (venue_name,),
    )
    return cur.fetchone()


def lookup_venue_via_team(
    cur,
    stadium_map: dict,
    home_team_id: str,
) -> Optional[dict]:
    """Fallback when matches.venue is NULL: look up the home team's
    home stadium from the JSON map. Returns the same shape as
    lookup_venue (with id=None because we're not pulling from
    venue_coords). Returns None on miss; caller treats it like
    'no_venue' (writes vc_unknown sentinel)."""
    if not stadium_map or not home_team_id:
        return None
    entry = stadium_map.get(home_team_id)
    if not entry:
        return None
    return {
        "id": None,  # No venue_coords row; weather row stores NULL FK.
        "latitude": float(entry["latitude"]),
        "longitude": float(entry["longitude"]),
        "timezone": entry.get("timezone") or "UTC",
        "is_indoor": bool(entry.get("is_indoor", False)),
    }


def write_weather(
    cur,
    *,
    match_id: str,
    venue_id: Optional[str],
    data_kind: str,
    summary: dict,
    raw: Optional[dict] = None,
) -> None:
    """Persist one weather snapshot — same INSERT shape as
    fetch_weather.write_weather so match_weather_latest joins cleanly
    across both data sources. data_kind in {'vc_forecast', 'vc_actual',
    'vc_unknown'}."""
    cur.execute(
        """
        INSERT INTO match_weather
            (match_id, venue_coords_id, data_kind,
             temperature_c, wind_kmh, precipitation_mm,
             humidity_pct, conditions, raw)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            match_id,
            venue_id,
            data_kind,
            summary.get("temperature_c"),
            summary.get("wind_kmh"),
            summary.get("precipitation_mm"),
            summary.get("humidity_pct"),
            summary.get("conditions"),
            Json(raw) if raw else None,
        ),
    )


# ── Match selectors (sport-agnostic) ───────────────────────────────


def list_upcoming(cur, sport_filter: Optional[str], days: int) -> list[dict]:
    """Outdoor-sport matches in the next N days that don't yet have
    a VC weather snapshot. Matches the Open-Meteo selector shape so
    the orchestration code below stays sport-agnostic."""
    q = """
        SELECT m.id::text AS match_id, m.match_date, m.venue,
               m.home_team_id::text AS home_team_id, l.sport
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE m.status = 'scheduled'
          AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
          {sport_clause}
          AND NOT EXISTS (
            SELECT 1 FROM match_weather mw
            WHERE mw.match_id = m.id AND mw.data_kind LIKE 'vc_%%'
          )
        ORDER BY m.match_date ASC
    """
    if sport_filter:
        cur.execute(
            q.format(sport_clause="AND l.sport = %s"),
            (str(days), sport_filter),
        )
    else:
        cur.execute(
            q.format(sport_clause=""),
            (str(days),),
        )
    return cur.fetchall()


def list_finished_for_backfill(
    cur,
    sport_filter: Optional[str],
    days: int,
) -> list[dict]:
    """Finished matches in the last N days that don't yet have a VC
    snapshot. Open-Meteo's archive lag is 2 days; VC's historical
    coverage is real-time, so no skew window needed."""
    q = """
        SELECT m.id::text AS match_id, m.match_date, m.venue,
               m.home_team_id::text AS home_team_id, l.sport
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE m.status = 'finished'
          AND m.match_date > NOW() - (%s || ' days')::interval
          {sport_clause}
          AND NOT EXISTS (
            SELECT 1 FROM match_weather mw
            WHERE mw.match_id = m.id AND mw.data_kind LIKE 'vc_%%'
          )
        ORDER BY m.match_date ASC
    """
    if sport_filter:
        cur.execute(
            q.format(sport_clause="AND l.sport = %s"),
            (str(days), sport_filter),
        )
    else:
        cur.execute(
            q.format(sport_clause=""),
            (str(days),),
        )
    return cur.fetchall()


# ── Orchestration ──────────────────────────────────────────────────


def fetch_and_store(
    cur,
    match: dict,
    kind: str,
    *,
    api_key: str,
    dry_run: bool = False,
    stadium_map: Optional[dict] = None,
) -> str:
    """Per-match: resolve venue → fetch → write. Returns the result
    label ('written', 'indoor', 'no_venue', 'fetch_failed',
    'no_window_data', 'bad_date').

    Venue resolution order:
      1. matches.venue → venue_coords (precise per-match path).
      2. Soccer-only fallback: home_team_id → stadium_map JSON
         (covers the ~99% of soccer matches with NULL venue text).
    """
    venue = lookup_venue(cur, match.get("venue"))
    if not venue and match.get("sport") == "soccer":
        venue = lookup_venue_via_team(
            cur,
            stadium_map or {},
            match.get("home_team_id") or "",
        )
    if not venue:
        if not dry_run:
            write_weather(
                cur,
                match_id=match["match_id"],
                venue_id=None,
                data_kind="vc_unknown",
                summary={},
            )
        return "no_venue"
    if venue["is_indoor"]:
        return "indoor"

    match_dt = match["match_date"]
    if not isinstance(match_dt, datetime):
        return "bad_date"

    raw = fetch_timeline(
        float(venue["latitude"]),
        float(venue["longitude"]),
        match_dt,
        venue["timezone"] or "UTC",
        api_key=api_key,
    )
    time.sleep(REQUEST_DELAY_SEC)
    if not raw:
        return "fetch_failed"
    summary = match_window_summary(raw, match_dt)
    if not summary:
        return "no_window_data"

    if dry_run:
        logger.info(
            "[dry-run] %s → %s",
            match["match_id"],
            {k: v for k, v in summary.items() if k != "conditions"},
        )
        return "written"

    data_kind = "vc_forecast" if kind == "forecast" else "vc_actual"
    write_weather(
        cur,
        match_id=match["match_id"],
        venue_id=venue["id"],
        data_kind=data_kind,
        summary=summary,
        raw=raw,
    )
    return "written"


def run(
    database_url: str,
    days: int,
    backfill_days: int,
    sport: Optional[str],
    *,
    dry_run: bool = False,
    stadium_map_path: Optional[str] = None,
) -> dict:
    counts = {
        "forecast_written": 0,
        "actual_written": 0,
        "indoor": 0,
        "no_venue": 0,
        "fetch_failed": 0,
        "no_window_data": 0,
        "bad_date": 0,
    }
    api_key = _api_key()
    if not api_key and not dry_run:
        raise SystemExit("VISUAL_CROSSING_API_KEY env var is required (or pass --dry-run).")
    stadium_map = load_stadium_map(stadium_map_path)
    if stadium_map:
        logger.info("Loaded soccer stadium fallback map: %d teams.", len(stadium_map))
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Pass 1: forecasts for upcoming matches.
            upcoming = list_upcoming(cur, sport, days)
            logger.info(
                "Forecast pass: %d upcoming matches needing VC weather",
                len(upcoming),
            )
            for m in upcoming:
                result = fetch_and_store(
                    cur,
                    m,
                    "forecast",
                    api_key=api_key or "",
                    dry_run=dry_run,
                    stadium_map=stadium_map,
                )
                if result == "written":
                    counts["forecast_written"] += 1
                else:
                    counts[result] = counts.get(result, 0) + 1
            if not dry_run:
                conn.commit()

            # Pass 2: archive for finished matches.
            finished = list_finished_for_backfill(cur, sport, backfill_days)
            logger.info(
                "Archive pass: %d finished matches needing VC weather",
                len(finished),
            )
            for m in finished:
                result = fetch_and_store(
                    cur,
                    m,
                    "actual",
                    api_key=api_key or "",
                    dry_run=dry_run,
                    stadium_map=stadium_map,
                )
                if result == "written":
                    counts["actual_written"] += 1
                else:
                    counts[result] = counts.get(result, 0) + 1
            if not dry_run:
                conn.commit()
    logger.info("Done. %s", counts)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Forecast lookahead window in days (default 14).",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=14,
        help="Archive-lookup window for finished matches (default 14).",
    )
    parser.add_argument(
        "--sport",
        choices=("nfl", "nhl", "soccer", "tennis", "mma"),
        help="Restrict to one sport. Defaults to ALL outdoor sports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + log but don't write to the DB (API key optional).",
    )
    parser.add_argument(
        "--stadium-map",
        default=None,
        help=(
            "Path to the soccer team→stadium fallback JSON. "
            "Defaults to /app/data/soccer_team_stadiums.json. Missing "
            "file → soccer fallback disabled (other sports unaffected)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logger.setLevel(args.log_level)
    if not args.database_url:
        logger.error("DATABASE_URL not set.")
        return 1
    run(
        args.database_url,
        args.days,
        args.backfill_days,
        args.sport,
        dry_run=args.dry_run,
        stadium_map_path=args.stadium_map,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
