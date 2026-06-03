"""Backfill historical tennis matches (with final scores) from ESPN's scoreboard.

ESPN's tennis scoreboard endpoints
  /apis/site/v2/sports/tennis/atp/scoreboard?dates=YYYYMMDD
  /apis/site/v2/sports/tennis/wta/scoreboard?dates=YYYYMMDD

return active TOURNAMENTS (e.g., "Wimbledon") on the requested date,
NOT individual matches. The match list is nested:
    events[] (tournaments)
      groupings[] (brackets: men's singles, women's singles, doubles)
        competitions[] (individual matches)
            competitors[] (athletes with winner + linescores)

We walk that nested structure, treating each leaf competition as a
match. Each match's home_score/away_score is set to 1/0 (or 0/1)
based on the competitor.winner boolean — tennis matches always have
exactly one winner, no draws/ties. Per-set linescore parsing is a
v2 follow-up (needed for total-games modeling).

Competitors DO carry homeAway in tennis (set by ESPN's own ordering,
typically higher-seeded player as home). v2 could read from
.linescores to recover actual game counts; for now the binary
winner is sufficient for moneyline training.

Idempotent: matches.{home_team_id, away_team_id, match_date} is the
upsert key. Re-running picks up missing matches + refreshes scores
for matches whose final state changed (rare — score corrections from
data entry fixes).

Why ESPN (not ATP/WTA official APIs):
  * ATP/WTA don't publish open historical match APIs.
  * Jeff Sackmann's GitHub data is excellent but requires CSV
    parsing + a different team-name normalization pass — defer
    to v2 if we need richer per-set data.
  * ESPN is already used for fetch_upcoming, so the player-name
    normalization is consistent across the live + backfill paths.

Tennis volume vs team sports:
  * ATP + WTA combined: ~70-90 matches/day during Grand Slams,
    ~30-50/day during regular tour weeks. ~12-15k matches across
    3 seasons. Larger than NFL's ~850 but smaller than the soccer
    universe (~50k+).
  * Each year has ~200 active match-days (off-season Nov-Dec is
    sparse). Progress logged every 60 days.

Usage:
    # One year (sanity check)
    python scripts/load_tennis_historical.py --start-date 2024-01-01 --end-date 2024-12-31

    # Three-year backfill (~10-20 min with throttle)
    python scripts/load_tennis_historical.py --start-date 2022-01-01 --end-date 2024-12-31
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

# Reuse the ensure_team + ensure_league + normalize_team helpers from
# fetch_upcoming.py — same DB constraints, same fuzzy-matching rules,
# same _competitor_name helper for individual-sport competitors.
sys.path.insert(0, os.path.dirname(__file__))

from fetch_upcoming import (  # noqa: E402
    ESPN_BASE,
    SPORT_CONFIGS,
    TENNIS_LEAGUES,
    _competitor_name,
    ensure_league,
    ensure_team,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_tennis_historical")

# Throttle between requests. Tennis backfill hits the ESPN endpoint
# twice per day (once for ATP, once for WTA) so 3 seasons × 365 days
# × 2 tours = ~2200 requests. 0.20s × 2200 = ~7 min plus actual
# request time = ~10-20 min total.
REQUEST_DELAY_SEC = 0.20

FINAL_STATES = {"post", "final"}
LIVE_PRE_STATES = {"pre", "in"}

# Both ATP and WTA tours are pulled by default. Single --tour flag
# scopes a run to one of them (useful for partial reloads).
DEFAULT_TOURS = ("atp", "wta")


# ── Pure parsing helpers (unit-tested) ──────────────────────────────


def iter_match_competitions(event: dict):
    """Walk one tournament's nested structure and yield each leaf
    competition (= individual match). ESPN tennis events are
    tournaments (Wimbledon, US Open, etc.); the actual matches live
    in events[].groupings[].competitions[]."""
    for grouping in event.get("groupings") or []:
        for competition in grouping.get("competitions") or []:
            yield competition


def extract_finished_match(competition: dict, venue_fallback: Optional[str] = None) -> Optional[dict]:
    """Pluck the fields we need for a finished-match UPSERT, or None
    if this competition isn't usable.

    Returns a dict with: home_name, away_name, match_dt, home_score,
    away_score (binary winner flag — 1 for winner, 0 for loser),
    venue. The caller resolves player IDs and writes the row.

    Tennis-specific:
    * Competitors use .athlete.displayName instead of .team.displayName.
    * The competition.competitors carry a winner boolean — tennis
      always has exactly one winner, no draws. We use the boolean
      directly (per-set linescore parsing is a v2 follow-up).
    * Competitors DO carry homeAway in tennis (ESPN orders them by
      seed); we use that rather than positional fallback.
    * Match date is on the competition, not the parent event (event
      date is the tournament-start date which spans 2 weeks).
    """
    state = (competition.get("status", {}).get("type", {}).get("state") or "").lower()
    if state in LIVE_PRE_STATES:
        return None
    if state not in FINAL_STATES:
        # Skip 'postponed', 'cancelled', 'walkover', 'retired' (when
        # ESPN flags it explicitly — the score on retirement is
        # incomplete and would skew rolling-form stats).
        return None

    competitors = competition.get("competitors") or []
    if len(competitors) < 2:
        return None

    # Tennis competitors carry homeAway (set by ESPN seed ordering).
    # If for some reason it's missing, fall back to positional ordering.
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if home is None or away is None:
        home, away = competitors[0], competitors[1]

    home_name = _competitor_name(home, is_individual=True)
    away_name = _competitor_name(away, is_individual=True)
    if not (home_name and away_name):
        return None

    # winner is a boolean on each competitor — exactly one is True
    # for a completed match (no draws in tennis).
    home_won = bool(home.get("winner"))
    away_won = bool(away.get("winner"))
    if home_won == away_won:
        # Both true or both false → data quality issue (often a
        # walkover or in-progress that the state filter missed).
        # Skip rather than insert ambiguous row.
        return None
    home_score = 1 if home_won else 0
    away_score = 1 if away_won else 0

    # competition.date carries the actual match time. Fall back to
    # any explicit date field at the competition level.
    raw_date = competition.get("date") or competition.get("startDate")
    if not raw_date:
        return None
    try:
        match_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    venue = (competition.get("venue") or {}).get("fullName") or venue_fallback

    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_score": home_score,
        "away_score": away_score,
        "match_dt": match_dt,
        "venue": venue,
    }


# ── ESPN fetcher ────────────────────────────────────────────────────


def fetch_day(tour: str, day: date) -> list[dict]:
    """One day's tennis scoreboard for one tour. Returns the raw events
    list (or [] on network error so the caller can press on)."""
    url = f"{ESPN_BASE}/tennis/{tour}/scoreboard"
    params = {"dates": day.strftime("%Y%m%d")}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("ESPN fetch failed for %s/%s: %s", tour, day, e)
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
    """Insert OR update one finished tennis match. Same upsert key as
    the team-sport loaders — (home_team_id, away_team_id, match_date)
    — so re-running on a date that already has the row updates the
    score/status rather than duplicating."""
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


def run(database_url: str, start: date, end: date, tours: tuple[str, ...]) -> dict:
    cfg = SPORT_CONFIGS["tennis"]
    counts = {
        "days_processed": 0,
        "events_seen": 0,
        "matches_written": 0,
        "skipped": 0,
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Ensure both tour leagues exist up front so the per-event
            # ensure_team calls have a parent league_id ready.
            league_ids: dict[str, str] = {}
            for tour in tours:
                if tour not in TENNIS_LEAGUES:
                    logger.warning("Unknown tennis tour %r — skipping", tour)
                    continue
                code, name, country = TENNIS_LEAGUES[tour]
                lid = ensure_league(cur, cfg, code, name, country)
                if not lid:
                    logger.error("Could not ensure %s league row", tour)
                    continue
                league_ids[tour] = lid

            for day in iter_dates(start, end):
                for tour, lid in league_ids.items():
                    events = fetch_day(tour, day)
                    for ev in events:
                        # Each ESPN tennis event is a TOURNAMENT;
                        # individual matches live in its nested
                        # competitions. Walk the structure and treat
                        # each leaf competition as a match.
                        ev_venue = (ev.get("venue") or {}).get("fullName")
                        for competition in iter_match_competitions(ev):
                            counts["events_seen"] += 1
                            parsed = extract_finished_match(competition, venue_fallback=ev_venue)
                            if parsed is None:
                                counts["skipped"] += 1
                                continue
                            home_id = ensure_team(cur, cfg, parsed["home_name"], lid)
                            away_id = ensure_team(cur, cfg, parsed["away_name"], lid)
                            if not (home_id and away_id):
                                counts["skipped"] += 1
                                continue
                            season = cfg.season_func(parsed["match_dt"])
                            counts["matches_written"] += upsert_finished_match(
                                cur,
                                lid,
                                home_id,
                                away_id,
                                parsed["match_dt"],
                                parsed["home_score"],
                                parsed["away_score"],
                                parsed["venue"],
                                season,
                            )
                    time.sleep(REQUEST_DELAY_SEC)
                counts["days_processed"] += 1
                conn.commit()
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
        help="Inclusive start date (YYYY-MM-DD). Tennis tour runs Jan-Nov plus exhibitions in Dec.",
    )
    p.add_argument(
        "--end-date",
        required=True,
        help="Inclusive end date (YYYY-MM-DD). The year-end Finals wrap in late November.",
    )
    p.add_argument(
        "--tour",
        choices=("atp", "wta", "both"),
        default="both",
        help="Tour to backfill — atp (men), wta (women), or both (default).",
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
    tours = DEFAULT_TOURS if args.tour == "both" else (args.tour,)
    run(args.database_url, start, end, tours)
    return 0


# Quiet the import-unused lint for Json (kept for forward compat).
_ = Json

if __name__ == "__main__":
    sys.exit(main())
