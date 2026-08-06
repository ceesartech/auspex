"""Fetch weather for outdoor-sport matches via Open-Meteo.

Open-Meteo is the free + no-API-key weather provider used by Auspex
for outdoor-sport weather features. Two endpoints:

  /v1/forecast?latitude=...&longitude=...
      For upcoming matches (kickoff within 16 days).

  /archive-api/v1/archive?latitude=...&longitude=...&start_date=...&end_date=...
      For finished matches (historical archive, no API limit).

Each match needs a venue_coords row (lat/lon + timezone) before we
can fetch — see scripts/seed_venue_coords.py for the stadium seed.
Indoor venues (is_indoor=true) are skipped entirely; weather doesn't
matter in a dome.

Per-match weather snapshots land in match_weather. compute_features_*
scripts read the LATEST snapshot via the match_weather_latest view.

Two passes per run:
  1. Upcoming matches in [now, now + --days]: forecast.
  2. Finished matches without an actual-data snapshot in
     [now - --backfill-days, now]: archive lookup.

Usage:
    python /app/scripts/fetch_weather.py                       # next 14 days fcst
    python /app/scripts/fetch_weather.py --days 7              # next 7 only
    python /app/scripts/fetch_weather.py --backfill-days 30    # archive last 30
    python /app/scripts/fetch_weather.py --sport nfl           # scope to one sport
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_weather")

# Open-Meteo endpoints. No API key needed.
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Throttle between requests. Open-Meteo's free tier is generous
# (10k requests/day) but bursts can rate-limit. 0.20s between calls
# keeps us well under any threshold.
REQUEST_DELAY_SEC = 0.20

# Hourly variables we pull from Open-Meteo. The match-window
# aggregation in match_window_summary() reduces them to scalars
# the compute_features scripts consume.
HOURLY_VARS = "temperature_2m,wind_speed_10m,precipitation,relative_humidity_2m,weather_code"

# WMO weather codes → simplified condition labels.
# Reference: https://open-meteo.com/en/docs (search "WMO Weather interpretation codes")
WMO_CODE_TO_CONDITIONS = {
    0: "clear",
    1: "mainly_clear",
    2: "partly_cloudy",
    3: "overcast",
    45: "fog",
    48: "fog",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    56: "freezing_drizzle",
    57: "freezing_drizzle",
    61: "rain",
    63: "rain",
    65: "rain",
    66: "freezing_rain",
    67: "freezing_rain",
    71: "snow",
    73: "snow",
    75: "snow",
    77: "snow",
    80: "rain_showers",
    81: "rain_showers",
    82: "rain_showers",
    85: "snow_showers",
    86: "snow_showers",
    95: "thunderstorm",
    96: "thunderstorm",
    99: "thunderstorm",
}


def normalize_venue(name: str) -> str:
    """Same loose normalization shape as teams.normalized_name —
    lowercase, collapsed whitespace. Keeps the venue_coords lookup
    stable across vendor name variants."""
    return " ".join((name or "").strip().lower().split())


def lookup_venue(cur, venue_name: Optional[str]) -> Optional[dict]:
    """Resolve a venue string to a venue_coords row (or None when the
    venue is missing or unrecognised). Indoor venues return the row
    but the caller checks is_indoor before fetching weather."""
    if not venue_name:
        return None
    norm = normalize_venue(venue_name)
    cur.execute(
        """
        SELECT id::text, latitude, longitude, timezone, is_indoor, display_name
        FROM venue_coords
        WHERE normalized_venue_name = %s
        LIMIT 1
        """,
        (norm,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def match_window_summary(hourly: dict, match_dt: datetime) -> dict:
    """Reduce Open-Meteo's hourly arrays to scalar weather features
    representing the 4-hour window centered on kickoff (3h before
    kickoff to 1h after). Captures pregame conditions plus the first
    quarter / first set / first round depending on sport — the
    weather most likely to affect early-game decisions.

    Returns a dict with: temperature_c, wind_kmh, precipitation_mm
    (sum over window, not per-hour), humidity_pct, conditions (most
    common WMO code in window).
    """
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    precs = hourly.get("precipitation") or []
    hums = hourly.get("relative_humidity_2m") or []
    codes = hourly.get("weather_code") or []

    window_start = match_dt - timedelta(hours=3)
    window_end = match_dt + timedelta(hours=1)

    indices: list[int] = []
    for i, t_str in enumerate(times):
        try:
            t = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if t.tzinfo is None:
            # Open-Meteo returns local time without tz when timezone=auto
            # — assume UTC for safety; the seed pins venues to UTC by
            # default so this matches.
            t = t.replace(tzinfo=timezone.utc)
        if window_start <= t <= window_end:
            indices.append(i)

    if not indices:
        return {}

    def _avg(arr):
        vals = [arr[i] for i in indices if i < len(arr) and arr[i] is not None]
        return sum(vals) / len(vals) if vals else None

    def _sum(arr):
        vals = [arr[i] for i in indices if i < len(arr) and arr[i] is not None]
        return sum(vals) if vals else None

    # Most common WMO code in the window
    window_codes = [codes[i] for i in indices if i < len(codes) and codes[i] is not None]
    cond = None
    if window_codes:
        modal = max(set(window_codes), key=window_codes.count)
        cond = WMO_CODE_TO_CONDITIONS.get(modal, "unknown")

    return {
        "temperature_c": _avg(temps),
        "wind_kmh": _avg(winds),
        "precipitation_mm": _sum(precs),
        "humidity_pct": _avg(hums),
        "conditions": cond,
    }


def fetch_forecast(lat: float, lon: float, match_dt: datetime, tz: str) -> Optional[dict]:
    """Fetch forecast for the day containing match_dt. Returns the
    raw hourly dict (with all HOURLY_VARS) or None on network error."""
    target_date = match_dt.date().isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "timezone": tz or "UTC",
        "start_date": target_date,
        "end_date": target_date,
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Open-Meteo forecast failed for %s: %s", target_date, e)
        return None
    return r.json()


def fetch_archive(lat: float, lon: float, match_dt: datetime, tz: str) -> Optional[dict]:
    """Fetch historical weather for the day containing match_dt."""
    target_date = match_dt.date().isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "timezone": tz or "UTC",
        "start_date": target_date,
        "end_date": target_date,
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Open-Meteo archive failed for %s: %s", target_date, e)
        return None
    return r.json()


def write_weather(
    cur,
    *,
    match_id: str,
    venue_id: Optional[str],
    data_kind: str,
    summary: dict,
    raw: Optional[dict] = None,
) -> None:
    """Persist one weather snapshot. data_kind in
    {'forecast', 'actual', 'unknown'} — 'unknown' is used when the
    venue couldn't be resolved (so we don't keep re-attempting)."""
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


# ── Match selectors ────────────────────────────────────────────────


def list_upcoming(cur, sport_filter: Optional[str], days: int) -> list[dict]:
    """Scheduled outdoor matches in the next N days that lack a
    forecast snapshot. Filters indoor venues at the SQL level via
    venue_coords.is_indoor; matches without a venue_coords row are
    skipped here and surface in the 'unknown venue' branch."""
    q = """
        SELECT m.id::text AS match_id, m.match_date, m.venue, l.sport,
               m.home_team_id::text AS home_team_id
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE m.status = 'scheduled'
          AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
          {sport_clause}
          AND NOT EXISTS (
            SELECT 1 FROM match_weather mw
            WHERE mw.match_id = m.id AND mw.data_kind = 'forecast'
              AND mw.fetched_at > NOW() - INTERVAL '6 hours'
          )
        ORDER BY m.match_date ASC
    """
    if sport_filter:
        q = q.format(sport_clause="AND l.sport = %s")
        cur.execute(q, (str(days), sport_filter))
    else:
        q = q.format(sport_clause="")
        cur.execute(q, (str(days),))
    return [dict(r) for r in cur.fetchall()]


def list_finished_for_backfill(cur, sport_filter: Optional[str], days: int) -> list[dict]:
    """Finished matches in the last N days that don't yet have an
    'actual' weather snapshot. Open-Meteo archive lags ~2 days
    behind real-time so the script only attempts matches at least
    2 days old."""
    q = """
        SELECT m.id::text AS match_id, m.match_date, m.venue, l.sport,
               m.home_team_id::text AS home_team_id
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE m.status = 'finished'
          AND m.match_date < NOW() - INTERVAL '2 days'
          AND m.match_date > NOW() - (%s || ' days')::interval
          {sport_clause}
          AND NOT EXISTS (
            SELECT 1 FROM match_weather mw
            WHERE mw.match_id = m.id AND mw.data_kind = 'actual'
          )
        ORDER BY m.match_date ASC
        LIMIT 500
    """
    if sport_filter:
        q = q.format(sport_clause="AND l.sport = %s")
        cur.execute(q, (str(days), sport_filter))
    else:
        q = q.format(sport_clause="")
        cur.execute(q, (str(days),))
    return [dict(r) for r in cur.fetchall()]


# ── Orchestration ──────────────────────────────────────────────────


def fetch_and_store(cur, match: dict, kind: str) -> str:
    """Per-match: resolve venue → fetch → write. Returns the result
    label for the run summary ('written', 'indoor', 'no_venue',
    'fetch_failed', 'no_window_data')."""
    venue = lookup_venue(cur, match.get("venue"))
    if not venue and match.get("home_team_id"):
        # football-data-loaded matches carry no venue string — fall back to
        # the home team's stadium (geocode_team_stadiums.py rows).
        venue = lookup_venue(cur, f"team-stadium:{match['home_team_id']}")
    if not venue:
        # Record the attempt with conditions='unknown' so we don't
        # keep retrying the same dead-end venue every run.
        write_weather(cur, match_id=match["match_id"], venue_id=None, data_kind="unknown", summary={})
        return "no_venue"
    if venue["is_indoor"]:
        return "indoor"

    match_dt = match["match_date"]
    if not isinstance(match_dt, datetime):
        return "bad_date"

    fetcher = fetch_forecast if kind == "forecast" else fetch_archive
    raw = fetcher(venue["latitude"], venue["longitude"], match_dt, venue["timezone"])
    time.sleep(REQUEST_DELAY_SEC)
    if not raw:
        return "fetch_failed"
    hourly = raw.get("hourly") or {}
    summary = match_window_summary(hourly, match_dt)
    if not summary:
        return "no_window_data"

    write_weather(
        cur,
        match_id=match["match_id"],
        venue_id=venue["id"],
        data_kind=kind,
        summary=summary,
        raw=raw,
    )
    return "written"


def run(database_url: str, days: int, backfill_days: int, sport: Optional[str]) -> dict:
    counts = {
        "forecast_written": 0,
        "actual_written": 0,
        "indoor": 0,
        "no_venue": 0,
        "fetch_failed": 0,
        "no_window_data": 0,
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Pass 1: forecasts for upcoming matches.
            upcoming = list_upcoming(cur, sport, days)
            logger.info("Forecast pass: %d upcoming matches needing weather", len(upcoming))
            for m in upcoming:
                result = fetch_and_store(cur, m, "forecast")
                if result == "written":
                    counts["forecast_written"] += 1
                elif result == "indoor":
                    counts["indoor"] += 1
                elif result == "no_venue":
                    counts["no_venue"] += 1
                elif result == "fetch_failed":
                    counts["fetch_failed"] += 1
                elif result == "no_window_data":
                    counts["no_window_data"] += 1
            conn.commit()

            # Pass 2: archive for finished matches without actual data.
            finished = list_finished_for_backfill(cur, sport, backfill_days)
            logger.info("Archive pass: %d finished matches needing weather", len(finished))
            for m in finished:
                result = fetch_and_store(cur, m, "actual")
                if result == "written":
                    counts["actual_written"] += 1
                elif result == "indoor":
                    counts["indoor"] += 1
                elif result == "no_venue":
                    counts["no_venue"] += 1
                elif result == "fetch_failed":
                    counts["fetch_failed"] += 1
                elif result == "no_window_data":
                    counts["no_window_data"] += 1
            conn.commit()
    logger.info("Done. %s", counts)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--days",
        type=int,
        default=14,
        help="Forecast lookahead window in days (default 14 — matches the DAG lookahead).",
    )
    p.add_argument(
        "--backfill-days",
        type=int,
        default=14,
        help="Archive-lookup window in days for finished matches (default 14).",
    )
    p.add_argument(
        "--sport",
        default=None,
        help="Optional sport filter ('nfl' / 'soccer' / 'tennis'). Skips other sports entirely.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.backfill_days, args.sport)
    return 0


_ = json  # forward-compat (unused; suppress lint)

if __name__ == "__main__":
    sys.exit(main())
