"""Fetch upcoming fixtures from ESPN's free public API.

Populates the `matches` table with future (status='scheduled') fixtures
for the configured soccer leagues. Matched to existing teams by
normalized name; new teams are inserted on the fly.

ESPN's scoreboard endpoint returns the current scoring window (a few
days) plus you can pass ?dates=YYYYMMDD to walk forward up to ~2 weeks.

Usage:
    python scripts/fetch_upcoming.py                       # default 7 days
    python scripts/fetch_upcoming.py --days 14
    python scripts/fetch_upcoming.py --leagues eng.1,ger.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_upcoming")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# ESPN league slug -> our canonical (football-data code, league name, country).
# Keep these aligned with promote_raw.LEAGUE_MAP so the same `leagues` row
# is reused regardless of which loader wrote it first.
LEAGUE_MAP: dict[str, tuple[str, str, str]] = {
    "eng.1": ("E0",  "Premier League",  "England"),
    "eng.2": ("E1",  "Championship",    "England"),
    "ger.1": ("D1",  "Bundesliga",      "Germany"),
    "ita.1": ("I1",  "Serie A",         "Italy"),
    "esp.1": ("SP1", "La Liga",         "Spain"),
    "fra.1": ("F1",  "Ligue 1",         "France"),
    "ned.1": ("N1",  "Eredivisie",      "Netherlands"),
    "por.1": ("P1",  "Primeira Liga",   "Portugal"),
}


def normalize_team(name: str) -> str:
    return " ".join(name.strip().lower().split())


def fetch_day(league_slug: str, day: date) -> list[dict]:
    url = f"{ESPN_BASE}/{league_slug}/scoreboard"
    params = {"dates": day.strftime("%Y%m%d")}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("ESPN fetch failed for %s %s: %s", league_slug, day, e)
        return []
    return r.json().get("events", []) or []


def ensure_league(cur, code: str, name: str, country: str) -> str | None:
    cur.execute(
        """
        INSERT INTO leagues (name, country, sport, external_ids)
        VALUES (%s, %s, 'soccer', %s)
        ON CONFLICT (name, country, sport) DO UPDATE
            SET external_ids = leagues.external_ids || EXCLUDED.external_ids
        RETURNING id
        """,
        (name, country, Json({"football_data": code})),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def ensure_team(cur, name: str, league_id: str) -> str | None:
    norm = normalize_team(name)
    cur.execute(
        """
        INSERT INTO teams (name, normalized_name, league_id, sport)
        VALUES (%s, %s, %s, 'soccer')
        ON CONFLICT (normalized_name, sport) DO UPDATE
            SET league_id = COALESCE(teams.league_id, EXCLUDED.league_id)
        RETURNING id
        """,
        (name, norm, league_id),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def insert_scheduled_match(cur, league_id, home_id, away_id, match_dt, venue) -> int:
    cur.execute(
        """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date,
                             status, venue, season)
        VALUES (%s, %s, %s, %s, 'scheduled', %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = CASE
                  WHEN matches.status = 'finished' THEN matches.status
                  ELSE EXCLUDED.status
                END,
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                updated_at = NOW()
        """,
        (league_id, home_id, away_id, match_dt, venue, _season_for(match_dt)),
    )
    return cur.rowcount


def _season_for(dt: datetime) -> str:
    """Soccer season runs Aug→May. Aug-Dec → YYYY-YYYY+1; Jan-Jul → YYYY-1-YYYY."""
    if dt.month >= 7:
        return f"{dt.year}-{dt.year + 1}"
    return f"{dt.year - 1}-{dt.year}"


def process_event(cur, league_id: str, event: dict) -> bool:
    comp = (event.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not (home and away):
        return False

    home_name = home.get("team", {}).get("displayName") or home.get("team", {}).get("name")
    away_name = away.get("team", {}).get("displayName") or away.get("team", {}).get("name")
    if not (home_name and away_name):
        return False

    try:
        match_dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return False

    state = (event.get("status", {}).get("type", {}).get("state") or "pre").lower()
    if state != "pre":
        return False  # skip in-progress and finished events here

    venue = (comp.get("venue") or {}).get("fullName")

    home_id = ensure_team(cur, home_name, league_id)
    away_id = ensure_team(cur, away_name, league_id)
    if not (home_id and away_id):
        return False

    return bool(insert_scheduled_match(cur, league_id, home_id, away_id, match_dt, venue))


def fetch_all(database_url: str, leagues: list[str], days: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    today = date.today()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for slug in leagues:
                if slug not in LEAGUE_MAP:
                    logger.warning("Unknown league slug: %s — skipping", slug)
                    continue
                code, name, country = LEAGUE_MAP[slug]
                league_id = ensure_league(cur, code, name, country)
                if not league_id:
                    continue
                n = 0
                for offset in range(days):
                    day = today + timedelta(days=offset)
                    for ev in fetch_day(slug, day):
                        if process_event(cur, league_id, ev):
                            n += 1
                counts[slug] = n
                conn.commit()
                logger.info("Fetched %d upcoming fixtures for %s (%s)", n, name, slug)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leagues", default=",".join(LEAGUE_MAP.keys()),
                   help="Comma-separated ESPN league slugs (default: all known soccer).")
    p.add_argument("--days", type=int, default=7,
                   help="How many days forward to look (default: 7).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2
    leagues = [s.strip() for s in args.leagues.split(",") if s.strip()]
    counts = fetch_all(args.database_url, leagues, args.days)
    total = sum(counts.values())
    logger.info("Fetched %d upcoming fixtures across %d leagues", total, len(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
