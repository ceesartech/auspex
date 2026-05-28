"""Fetch live (pre-match) odds from the-odds-api.com.

the-odds-api.com (https://the-odds-api.com) free tier: 500 requests per
month, no card required. Each call returns all upcoming events for one
sport + one region. We default to one region ("us") and one call per
sport per run = 1 quota unit / sport / run. With the configured 9
sports the daily DAG burns 270 quota / month — well under the limit.

The script:
  1. Pulls /v4/sports/{sport_key}/odds for each configured sport.
  2. For each returned event, looks up the matching match in our
     `matches` table by (home_team_name, away_team_name, match_date).
     Team-name matching falls back through {exact, normalized, fuzzy}.
  3. Inserts one row per (bookmaker × market × selection) into the
     odds table with is_opening=false, is_live=false. Duplicate rows
     within the same window are skipped via the NOT EXISTS guard.

Sport keys (https://the-odds-api.com/sports-odds-data/sports-apis.html):
    soccer_epl                   English Premier League
    soccer_germany_bundesliga    Bundesliga
    soccer_italy_serie_a         Serie A
    soccer_spain_la_liga         La Liga
    soccer_france_ligue_one      Ligue 1
    soccer_usa_mls               MLS
    soccer_brazil_campeonato     Brasileirão
    soccer_argentina_primera_division  Primera División
    soccer_uefa_champs_league    UEFA Champions League
    soccer_uefa_europa_league    UEFA Europa League
    soccer_fifa_world_cup        FIFA World Cup
    soccer_conmebol_copa_america Copa América
    soccer_concacaf_gold_cup     CONCACAF Gold Cup

Usage:
    python scripts/fetch_live_odds.py                 # all configured sports
    python scripts/fetch_live_odds.py --sports soccer_epl,soccer_usa_mls
    python scripts/fetch_live_odds.py --regions eu     # default 'us'
    python scripts/fetch_live_odds.py --quota-only     # check remaining quota
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_live_odds")

BASE_URL = "https://api.the-odds-api.com/v4"

# the-odds-api sport keys we pull. Aligned with the in-season leagues in
# fetch_upcoming.LEAGUE_MAP. Each entry costs 1 quota / day × 30 = 30/mo
# so the full list below = 14 × 30 = 420 quota / month — within the
# 500 free-tier ceiling. The script gracefully handles 422 (sport not
# in season) so off-season EU leagues here just no-op.
DEFAULT_SPORTS = [
    # European top flights
    "soccer_epl",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_spain_la_liga",
    "soccer_france_ligue_one",
    # Summer-season leagues with historical data in our DB
    "soccer_usa_mls",
    "soccer_brazil_campeonato",
    "soccer_norway_eliteserien",
    "soccer_sweden_allsvenskan",
    "soccer_japan_j_league",
    "soccer_argentina_primera_division",
    "soccer_mexico_ligamx",
    # International competitions
    "soccer_uefa_champs_league",
    "soccer_fifa_world_cup",
]

# Map the-odds-api market_key + outcome.name → our (market_type, selection).
MARKET_TYPE_MAP = {
    "h2h": "1x2",
    "totals": "over_under",
}


def normalize_name(name: str) -> str:
    """Loose normalisation: lowercase, strip punctuation, collapse spaces."""
    return " ".join(name.lower().replace(".", "").replace("&", "and").split())


def fetch_sport_odds(sport_key: str, api_key: str, regions: str) -> list[dict]:
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("the-odds-api: invalid API key (401)")
    if r.status_code == 422:
        logger.warning("the-odds-api: %s — sport not in season or unknown", sport_key)
        return []
    r.raise_for_status()

    remaining = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")
    logger.info("Quota: used=%s remaining=%s", used, remaining)

    return r.json()


def find_match_id(cur, home_team: str, away_team: str, commence_time: datetime) -> str | None:
    """Locate the matching match row. The-odds-api uses team names that
    don't always match what's in our `teams` table; try three strategies:
      1. Exact normalized match on home + away.
      2. Fuzzy match (SequenceMatcher >= 0.8) on each side, within ±2 days.
    """
    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    window_start = commence_time - timedelta(days=2)
    window_end = commence_time + timedelta(days=2)

    cur.execute(
        """
        SELECT m.id::text AS id, ht.name AS home_name, at.name AS away_name, m.match_date
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.status = 'scheduled'
          AND m.match_date BETWEEN %s AND %s
        """,
        (window_start, window_end),
    )
    candidates = cur.fetchall()
    if not candidates:
        return None

    # 1. Exact normalised match.
    for c in candidates:
        if normalize_name(c["home_name"]) == home_norm and normalize_name(c["away_name"]) == away_norm:
            return c["id"]

    # 2. Fuzzy match.
    best = None
    best_score = 0.0
    for c in candidates:
        h_score = SequenceMatcher(None, normalize_name(c["home_name"]), home_norm).ratio()
        a_score = SequenceMatcher(None, normalize_name(c["away_name"]), away_norm).ratio()
        score = (h_score + a_score) / 2.0
        if score > best_score:
            best_score = score
            best = c
    if best is not None and best_score >= 0.8:
        return best["id"]
    return None


def insert_odds_row(
    cur, match_id: str, bookmaker: str, market_type: str, selection: str, odds_decimal: float, line: float | None
) -> bool:
    """Idempotent insert — skip if same row already exists."""
    cur.execute(
        """
        INSERT INTO odds (match_id, bookmaker, market_type, selection,
                          odds_decimal, line, is_opening, is_live)
        SELECT %s, %s, %s, %s, %s, %s, false, false
        WHERE NOT EXISTS (
            SELECT 1 FROM odds
            WHERE match_id = %s AND bookmaker = %s
              AND market_type = %s AND selection = %s
              AND COALESCE(line, -1) = COALESCE(%s, -1)
              AND is_opening = false
        )
        """,
        (
            match_id,
            bookmaker,
            market_type,
            selection,
            odds_decimal,
            line,
            match_id,
            bookmaker,
            market_type,
            selection,
            line,
        ),
    )
    return cur.rowcount > 0


def map_outcome(
    market_key: str, outcome_name: str, home_team_name: str, away_team_name: str
) -> tuple[str | None, str | None]:
    """Translate the-odds-api's market.key + outcome.name into our
    (market_type, selection) format. Returns (None, None) if unmappable.
    """
    market_type = MARKET_TYPE_MAP.get(market_key)
    if market_type is None:
        return None, None

    if market_type == "1x2":
        if outcome_name == home_team_name:
            return market_type, "home"
        if outcome_name == away_team_name:
            return market_type, "away"
        if outcome_name.lower() == "draw":
            return market_type, "draw"
        return None, None

    if market_type == "over_under":
        name = outcome_name.lower()
        if name == "over":
            return market_type, "over"
        if name == "under":
            return market_type, "under"
        return None, None

    return None, None


def process_event(cur, event: dict, unmatched_log: list | None = None) -> int:
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    commence = event.get("commence_time")
    if not (home and away and commence):
        return 0

    try:
        commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    except ValueError:
        return 0

    match_id = find_match_id(cur, home, away, commence_dt)
    if not match_id:
        if unmatched_log is not None:
            unmatched_log.append(f"{home} vs {away} @ {commence}")
        return 0

    inserted = 0
    for bm in event.get("bookmakers", []):
        bookmaker_name = bm.get("title") or bm.get("key") or "unknown"
        for market in bm.get("markets", []):
            market_key = market.get("key")
            for outcome in market.get("outcomes", []):
                market_type, selection = map_outcome(market_key, outcome.get("name", ""), home, away)
                if not market_type:
                    continue
                price = outcome.get("price")
                if price is None:
                    continue
                line = outcome.get("point")  # over/under threshold
                # We only care about the 2.5 line for over/under (matches
                # how training_data extracts implied_prob_over25).
                if market_type == "over_under" and line is not None and abs(float(line) - 2.5) > 0.01:
                    continue
                if insert_odds_row(
                    cur,
                    match_id,
                    bookmaker_name,
                    market_type,
                    selection,
                    float(price),
                    float(line) if line is not None else None,
                ):
                    inserted += 1
    return inserted


def run(database_url: str, sports: list[str], api_key: str, regions: str) -> dict[str, int]:
    results = {"events_seen": 0, "events_matched": 0, "odds_rows_inserted": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for sport_key in sports:
                events = fetch_sport_odds(sport_key, api_key, regions)
                logger.info("%s: %d events returned", sport_key, len(events))
                results["events_seen"] += len(events)
                unmatched: list[str] = []
                for event in events:
                    inserted = process_event(cur, event, unmatched_log=unmatched)
                    if inserted > 0:
                        results["events_matched"] += 1
                        results["odds_rows_inserted"] += inserted
                # Log unmatched events for visibility — usually means no
                # scheduled fixture in the matches table covering that
                # date, or a team-name spelling mismatch.
                if unmatched:
                    logger.info(
                        "%s: %d/%d events had no matching scheduled fixture. " "First 5: %s",
                        sport_key,
                        len(unmatched),
                        len(events),
                        unmatched[:5],
                    )
                conn.commit()
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sports", default=",".join(DEFAULT_SPORTS), help="Comma-separated the-odds-api sport keys.")
    p.add_argument("--regions", default="us", help="the-odds-api region: us, uk, eu, au (default 'us').")
    p.add_argument("--api-key", default=os.environ.get("THE_ODDS_API_KEY"))
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--quota-only", action="store_true", help="Make a single call to check remaining quota and exit.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.api_key:
        logger.error("THE_ODDS_API_KEY not set — sign up free at https://the-odds-api.com")
        return 2
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    if args.quota_only:
        # /v4/sports is a free endpoint that still returns quota headers.
        r = requests.get(f"{BASE_URL}/sports", params={"apiKey": args.api_key}, timeout=10)
        r.raise_for_status()
        print(
            f"Quota — used: {r.headers.get('x-requests-used')}, " f"remaining: {r.headers.get('x-requests-remaining')}"
        )
        return 0

    sports = [s.strip() for s in args.sports.split(",") if s.strip()]
    results = run(args.database_url, sports, args.api_key, args.regions)
    logger.info(
        "Done. events_seen=%d events_matched=%d odds_rows_inserted=%d",
        results["events_seen"],
        results["events_matched"],
        results["odds_rows_inserted"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
