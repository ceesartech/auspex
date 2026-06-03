"""Backfill historical MMA fights (with final outcomes) from ESPN's scoreboard.

ESPN's MMA endpoint
  /apis/site/v2/sports/mma/ufc/scoreboard?dates=YYYYMMDD

returns active CARDS on the requested date. Each event is one fight
card (e.g., "UFC Fight Night: Covington vs. Buckley") with
competitions[] holding the individual fights (12-15 per card).
Competitions have competitors[] with a `winner` boolean and
`athlete.displayName` — no homeAway field, no per-fight score.

We use positional ordering (index 0 → home, 1 → away) since
homeAway is missing, and read the winner boolean to set
home_score / away_score = 1 / 0 (binary winner flag). MMA fights
always produce a winner (or a draw — see below).

Draws: MMA does have draws (~1% of decisions). When both winner
booleans are false (no winner declared) or both true (data quality
issue), the parser skips the row — the catchup grading path
handles draws by leaving is_correct NULL.

Volume vs other sports: UFC schedules ~40-45 cards/year × ~12
fights/card = ~500-540 fights/year. Three seasons ≈ 1500 fights.
Smaller corpus than NBA/tennis but larger than NFL.

Usage:
    python scripts/load_mma_historical.py --start-date 2024-01-01 --end-date 2024-12-31
    python scripts/load_mma_historical.py --start-date 2022-01-01 --end-date 2024-12-31
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

sys.path.insert(0, os.path.dirname(__file__))

from fetch_upcoming import (  # noqa: E402
    ESPN_BASE,
    MMA_LEAGUES,
    SPORT_CONFIGS,
    _competitor_name,
    ensure_league,
    ensure_team,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_mma_historical")

# UFC fight cards happen ~40-45 weeks/year; 3 years = ~120 active
# days. 0.25s throttle × 1095 day-requests ≈ 5 min including dark
# days (most days have zero events).
REQUEST_DELAY_SEC = 0.25

FINAL_STATES = {"post", "final"}
LIVE_PRE_STATES = {"pre", "in"}

# Slug identifies the ESPN MMA league. UFC is the only one in v1;
# Bellator / PFL / ONE could layer in via additional slugs.
DEFAULT_LEAGUE = "ufc"


# ── Pure parsing helpers (unit-tested) ──────────────────────────────


def extract_finished_fight(competition: dict) -> Optional[dict]:
    """Pluck the fields we need for a finished-fight UPSERT, or None
    if this competition isn't usable.

    Returns a dict with: home_name, away_name, match_dt, home_score,
    away_score (binary winner flag — 1 for winner, 0 for loser),
    venue. The caller resolves fighter IDs and writes the row.

    MMA-specific:
    * Competitors use .athlete.displayName.
    * Each competitor carries a `winner` boolean — exactly one True
      for a decisive finish. Both-True or both-False rows are skipped
      (draws + data quality issues stay ungraded).
    * No homeAway field — positional ordering (index 0 → home).
    """
    state = (competition.get("status", {}).get("type", {}).get("state") or "").lower()
    if state in LIVE_PRE_STATES:
        return None
    if state not in FINAL_STATES:
        # Skip 'cancelled' fights (weight miss / injury withdrawal).
        return None

    competitors = competition.get("competitors") or []
    if len(competitors) < 2:
        return None
    # No homeAway on MMA — positional ordering.
    home, away = competitors[0], competitors[1]

    home_name = _competitor_name(home, is_individual=True)
    away_name = _competitor_name(away, is_individual=True)
    if not (home_name and away_name):
        return None

    home_won = bool(home.get("winner"))
    away_won = bool(away.get("winner"))
    if home_won == away_won:
        # Draw (both false) or data corruption (both true). Skip;
        # don't write ambiguous row.
        return None

    raw_date = competition.get("date") or competition.get("startDate")
    if not raw_date:
        return None
    try:
        match_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_score": 1 if home_won else 0,
        "away_score": 1 if away_won else 0,
        "match_dt": match_dt,
        "venue": (competition.get("venue") or {}).get("fullName"),
    }


# ── ESPN fetcher ────────────────────────────────────────────────────


def fetch_day(slug: str, day: date) -> list[dict]:
    """One day's MMA cards from ESPN. Returns the raw events list
    (or [] on network error so the caller can press on)."""
    url = f"{ESPN_BASE}/mma/{slug}/scoreboard"
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


def run(database_url: str, start: date, end: date, slug: str = DEFAULT_LEAGUE) -> dict:
    cfg = SPORT_CONFIGS["mma"]
    if slug not in MMA_LEAGUES:
        logger.error("Unknown MMA league slug: %s", slug)
        return {"matches_written": 0}
    code, name, country = MMA_LEAGUES[slug]

    counts = {"days_processed": 0, "events_seen": 0, "matches_written": 0, "skipped": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            league_id = ensure_league(cur, cfg, code, name, country)
            if not league_id:
                logger.error("Could not ensure MMA league row")
                return counts

            for day in iter_dates(start, end):
                events = fetch_day(slug, day)
                counts["days_processed"] += 1
                for ev in events:
                    # Each ESPN MMA event is a CARD with
                    # competitions[] = individual fights.
                    for competition in ev.get("competitions") or []:
                        counts["events_seen"] += 1
                        parsed = extract_finished_fight(competition)
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
                # MMA cards are sparse so progress every 90 days
                # gives ~12 chunks per year of useful logging.
                if counts["days_processed"] % 90 == 0:
                    logger.info(
                        "Progress: %d days processed, %d fights written",
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
        help="Inclusive start date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        required=True,
        help="Inclusive end date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--league",
        default=DEFAULT_LEAGUE,
        choices=list(MMA_LEAGUES.keys()),
        help="Promotion slug (currently only 'ufc' is wired).",
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
    run(args.database_url, start, end, args.league)
    return 0


_ = Json  # forward-compat (unused; suppress lint)

if __name__ == "__main__":
    sys.exit(main())
