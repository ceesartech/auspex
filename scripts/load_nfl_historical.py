"""Backfill historical NFL games (with final scores) from ESPN's scoreboard.

ESPN's scoreboard endpoint
(/apis/site/v2/sports/football/nfl/scoreboard?dates=YYYYMMDD)
returns every NFL game on the requested date, including finished
ones with their final scores. We walk the date range one day at a
time and upsert each finished game into the `matches` table as
status='finished' with home_score / away_score populated.

Idempotent: matches.{home_team_id, away_team_id, match_date} is the
upsert key. Re-running picks up missing games + refreshes scores for
games whose final state changed.

Why ESPN (not NFL Stats API or PFR scraper):
  * NFL has no official open Stats API. PFR (pro-football-reference)
    works but needs a scraper that respects their rate limits + ToS.
  * ESPN is open and already used for fetch_upcoming, so team-name
    normalization is consistent across paths.

NFL volume vs NBA/NHL:
  * NFL plays ~285 games/season (32 teams × 17 games / 2 + playoffs).
    Far fewer than NBA (~1230) or NHL (~1312), so 3 seasons ≈ 855
    games. Backfill takes ~5-10 min vs NBA's 15-20.
  * Each game is on a specific game-day (Sun/Mon/Thu/Sat in
    December) so most dates are dark — the script processes
    ~110-130 game-days per season, not the full 365.

Usage:
    # One season (good for sanity checks)
    python scripts/load_nfl_historical.py --start-date 2024-09-05 --end-date 2025-02-15

    # Three-season roadmap backfill (~5-10 min with throttle)
    python scripts/load_nfl_historical.py --start-date 2022-09-01 --end-date 2025-02-15
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

# Reuse the ensure_team + ensure_league + normalize_team logic from
# fetch_upcoming.py — same DB constraints, same fuzzy-matching rules.
sys.path.insert(0, os.path.dirname(__file__))

from fetch_upcoming import ESPN_BASE, NFL_LEAGUES, SPORT_CONFIGS, ensure_league, ensure_team  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_nfl_historical")

# Throttle between requests so a 3-season run (~750 days requested,
# ~330 with games) is rate-friendly. 0.25s × 750 = ~3 min plus actual
# request time = ~5-10 min total.
REQUEST_DELAY_SEC = 0.25

# ESPN status.type.state values for finished games. "post" is the
# canonical one; "final" is a defensive fallback in case the API
# changes its mind.
FINAL_STATES = {"post", "final"}

# Status values that indicate the game is still open / not yet played.
# Skipping these prevents us from writing rows that the next live-feed
# run would have to fix.
LIVE_PRE_STATES = {"pre", "in"}


# ── Pure parsing helpers (unit-tested) ──────────────────────────────


def parse_score(competitor: dict) -> Optional[int]:
    """ESPN puts score on the competitor as a string. Returns int or
    None if the field is missing / non-numeric (postponed games etc.)."""
    raw = competitor.get("score")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def extract_finished_game(event: dict) -> Optional[dict]:
    """Pluck the fields we need for a finished-game UPSERT, or None
    if this event isn't a usable finished game.

    Returns a dict with: home_name, away_name, match_dt, home_score,
    away_score, venue. The caller resolves team IDs and writes the row.
    """
    state = (event.get("status", {}).get("type", {}).get("state") or "").lower()
    if state in LIVE_PRE_STATES:
        return None
    if state not in FINAL_STATES:
        # Could be 'postponed', 'cancelled', 'suspended' — skip
        # rather than write an inconsistent row.
        return None

    comp = (event.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not (home and away):
        return None

    home_name = home.get("team", {}).get("displayName") or home.get("team", {}).get("name")
    away_name = away.get("team", {}).get("displayName") or away.get("team", {}).get("name")
    if not (home_name and away_name):
        return None

    home_score = parse_score(home)
    away_score = parse_score(away)
    if home_score is None or away_score is None:
        # Game is "final" but missing a score — data quality issue.
        # Skip rather than write NULL scores.
        return None

    try:
        match_dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None

    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_score": home_score,
        "away_score": away_score,
        "match_dt": match_dt,
        "venue": (comp.get("venue") or {}).get("fullName"),
    }


# ── ESPN fetcher ────────────────────────────────────────────────────


def fetch_day(day: date) -> list[dict]:
    """One day's NFL scoreboard from ESPN. Returns the raw events
    list (or [] on network error so the caller can press on)."""
    url = f"{ESPN_BASE}/football/nfl/scoreboard"
    params = {"dates": day.strftime("%Y%m%d")}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("ESPN fetch failed for %s: %s", day, e)
        return []
    return r.json().get("events", []) or []


# ── DB I/O ──────────────────────────────────────────────────────────


def upsert_finished_match(
    cur,
    league_id: str,
    home_id: str,
    away_id: str,
    match_dt: datetime,
    home_score: int,
    away_score: int,
    venue: Optional[str],
    season: str,
) -> int:
    """Insert OR update one finished match. The unique key
    (home_team_id, away_team_id, match_date) means re-running on a
    date that already has a row updates the score / status rather
    than duplicating."""
    cur.execute(
        """
        INSERT INTO matches
            (league_id, home_team_id, away_team_id, match_date,
             status, venue, season, home_score, away_score)
        VALUES (%s, %s, %s, %s, 'finished', %s, %s, %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = 'finished',
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                updated_at = NOW()
        """,
        (league_id, home_id, away_id, match_dt, venue, season, home_score, away_score),
    )
    return cur.rowcount


# ── Orchestration ──────────────────────────────────────────────────


def iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def run(database_url: str, start: date, end: date) -> dict:
    cfg = SPORT_CONFIGS["nfl"]
    slug = "nfl"  # only one NFL league in NFL_LEAGUES
    code, name, country = NFL_LEAGUES[slug]

    counts = {"days_processed": 0, "events_seen": 0, "matches_written": 0, "skipped": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            league_id = ensure_league(cur, cfg, code, name, country)
            if not league_id:
                logger.error("Could not ensure NFL league row")
                return counts

            for day in iter_dates(start, end):
                events = fetch_day(day)
                counts["days_processed"] += 1
                counts["events_seen"] += len(events)
                for ev in events:
                    parsed = extract_finished_game(ev)
                    if parsed is None:
                        counts["skipped"] += 1
                        continue
                    home_id = ensure_team(cur, cfg, parsed["home_name"], league_id)
                    away_id = ensure_team(cur, cfg, parsed["away_name"], league_id)
                    if not (home_id and away_id):
                        counts["skipped"] += 1
                        continue
                    season = cfg.season_func(parsed["match_dt"])
                    counts["matches_written"] += upsert_finished_match(
                        cur,
                        league_id,
                        home_id,
                        away_id,
                        parsed["match_dt"],
                        parsed["home_score"],
                        parsed["away_score"],
                        parsed["venue"],
                        season,
                    )
                conn.commit()
                time.sleep(REQUEST_DELAY_SEC)
                # NFL has fewer game-days than NBA so the progress log
                # cadence is tighter — every 60 days vs NBA's 30.
                if counts["days_processed"] % 60 == 0:
                    logger.info(
                        "Progress: %d days processed, %d matches written",
                        counts["days_processed"],
                        counts["matches_written"],
                    )

    logger.info("Done. %s", counts)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--start-date",
        required=True,
        help="Inclusive start date (YYYY-MM-DD). Typical NFL seasons start in early September; preseason in August.",
    )
    p.add_argument(
        "--end-date",
        required=True,
        help="Inclusive end date (YYYY-MM-DD). Typical NFL seasons end with Super Bowl in early February.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date)
    except ValueError as e:
        logger.error("Bad date: %s", e)
        return 2
    if end < start:
        logger.error("--end-date must be >= --start-date")
        return 2
    run(args.database_url, start, end)
    return 0


# Quiet the import-unused lint for Json (kept for forward compat).
_ = Json

if __name__ == "__main__":
    sys.exit(main())
