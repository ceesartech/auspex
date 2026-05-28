"""Fetch live (pre-match) odds from the-odds-api.com.

the-odds-api.com (https://the-odds-api.com) bills 1 quota unit per
(region × market) per call. We default to one region ("us") and send
the sport's full market basket in a single call, so a soccer run costs
2 units (h2h + totals) and an NHL run costs 3 units (h2h + spreads +
totals). With the configured sport list and one daily run this is well
under both the 500-unit free tier and the 20,000-unit paid tier.

The script:
  1. Pulls /v4/sports/{sport_key}/odds for each configured sport.
  2. For each returned event, looks up the matching match in our
     `matches` table by (home_team_name, away_team_name, match_date)
     scoped to the event's sport.
  3. Inserts one row per (bookmaker × market × selection [× line]) into
     the odds table with is_opening=false, is_live=false. Duplicate rows
     within the same window are skipped via the NOT EXISTS guard.

Sport keys (https://the-odds-api.com/sports-odds-data/sports-apis.html):
    soccer_*                     27+ soccer leagues
    icehockey_nhl                NHL

Usage:
    python scripts/fetch_live_odds.py                 # all configured sports
    python scripts/fetch_live_odds.py --sports soccer_epl,icehockey_nhl
    python scripts/fetch_live_odds.py --regions eu     # default 'us'
    python scripts/fetch_live_odds.py --quota-only     # check remaining quota
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_live_odds")

BASE_URL = "https://api.the-odds-api.com/v4"

# the-odds-api sport keys we pull. Each entry costs (markets × regions)
# quota per daily run × 30 days/month. With ~28 soccer keys (2 markets)
# + 1 NHL key (3 markets) at one region, the full default list is
# (28×2 + 1×3) × 30 = ~1,770 quota/month — well within the 20,000-unit
# paid tier. The script gracefully handles 422 (sport off-season) so
# leagues outside their window no-op.
#
# To trim back: comment out sports you don't care about. To add more
# sports (NBA, NFL, MLB, tennis), see the sport-keys reference linked
# in the module docstring AND register the sport's market mapper below.
# An unregistered sport will be skipped with a warning.
DEFAULT_SPORTS = [
    # European top flights (in season Aug-May)
    "soccer_epl",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_spain_la_liga",
    "soccer_france_ligue_one",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga",
    "soccer_belgium_first_div",  # actual the-odds-api key (no "_a" suffix)
    "soccer_turkey_super_league",
    "soccer_greece_super_league",
    "soccer_scotland_premiership",
    # Americas
    "soccer_usa_mls",
    "soccer_brazil_campeonato",
    "soccer_argentina_primera_division",
    "soccer_mexico_ligamx",
    "soccer_chile_campeonato",
    "soccer_colombia_primera_a",
    # Asia-Pacific
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_china_superleague",
    "soccer_australia_aleague",
    # Nordic (summer-active filler)
    "soccer_norway_eliteserien",
    "soccer_sweden_allsvenskan",
    # International competitions
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_fifa_world_cup",
    "soccer_conmebol_copa_america",
    "soccer_concacaf_gold_cup",
    # NHL (in season Oct-Jun, including playoffs)
    "icehockey_nhl",
]


# ── Sport dispatch ────────────────────────────────────────────────────
# the-odds-api keys are prefixed by sport family (`soccer_*`, `icehockey_*`,
# `americanfootball_*`, etc.). We derive auspex's internal sport label
# from the key prefix to pick the right outcome mapper and team scope.
SPORT_KEY_PREFIXES: dict[str, str] = {
    "soccer_": "soccer",
    "icehockey_": "nhl",
}


# Per-sport market basket sent to the-odds-api. Each market costs 1 quota
# unit per region per call. Soccer uses the 1x2 + over/under 2.5 pair the
# soccer ensemble was trained on; NHL adds spreads (puck line) since the
# planned NHL ensemble covers all three markets.
SPORT_MARKETS: dict[str, str] = {
    "soccer": "h2h,totals",
    "nhl": "h2h,spreads,totals",
}


def sport_for_key(sport_key: str) -> str | None:
    """Return the auspex sport label for a the-odds-api sport key, or
    None if the key isn't from a registered sport family."""
    for prefix, sport in SPORT_KEY_PREFIXES.items():
        if sport_key.startswith(prefix):
            return sport
    return None


def normalize_name(name: str) -> str:
    """Loose normalisation: lowercase, strip punctuation, collapse spaces."""
    return " ".join(name.lower().replace(".", "").replace("&", "and").split())


def fetch_sport_odds(sport_key: str, markets: str, api_key: str, regions: str) -> list[dict]:
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("the-odds-api: invalid API key (401)")
    # 422 = sport off-season, 404 = unknown sport key — both mean "skip
    # this sport, keep going on the others" rather than crashing the run.
    if r.status_code in (404, 422):
        logger.warning("the-odds-api: %s — sport not available (HTTP %d)", sport_key, r.status_code)
        return []
    r.raise_for_status()

    remaining = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")
    logger.info("Quota: used=%s remaining=%s", used, remaining)

    return r.json()


def find_match_id(cur, sport: str, home_team: str, away_team: str, commence_time: datetime) -> str | None:
    """Locate the matching match row in `matches`, scoped to `sport`.

    The-odds-api uses team names that don't always match what's in our
    `teams` table; try three strategies:
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
          AND ht.sport = %s
          AND m.match_date BETWEEN %s AND %s
        """,
        (sport, window_start, window_end),
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


def _team_side(outcome_name: str, home_team_name: str, away_team_name: str) -> str | None:
    """Resolve a team-named outcome (e.g. 'Toronto Maple Leafs') to 'home'
    or 'away' relative to the event. Returns None if neither side matches."""
    n = outcome_name.strip().lower()
    if n == home_team_name.strip().lower():
        return "home"
    if n == away_team_name.strip().lower():
        return "away"
    return None


def map_outcome(
    sport: str,
    market_key: str,
    outcome_name: str,
    home_team_name: str,
    away_team_name: str,
    point: float | None,
) -> tuple[str | None, str | None, bool]:
    """Translate the-odds-api's market.key + outcome.name into our
    (market_type, selection) format, and report whether the row's
    `line` should be kept.

    Returns (market_type, selection, keep_line). market_type=None means
    the outcome isn't representable in our schema and should be skipped.

    Soccer: h2h → 1x2 (home/draw/away, no line); totals → over_under
    keyed on the 2.5 line only (the soccer training features expect
    implied_prob_over25).

    NHL: h2h → moneyline (home/away, no draw); spreads → spread (puck
    line, home/away with ±1.5 line); totals → total (over/under, every
    available line — the NHL model hasn't picked a canonical line yet).
    """
    if sport == "soccer":
        if market_key == "h2h":
            if outcome_name == home_team_name:
                return "1x2", "home", False
            if outcome_name == away_team_name:
                return "1x2", "away", False
            if outcome_name.lower() == "draw":
                return "1x2", "draw", False
            return None, None, False
        if market_key == "totals":
            n = outcome_name.lower()
            if n not in ("over", "under"):
                return None, None, False
            # Soccer training data only uses the 2.5 line; drop other lines
            # to avoid bloating the table with rows we never read.
            if point is None or abs(float(point) - 2.5) > 0.01:
                return None, None, False
            return "over_under", n, True
        return None, None, False

    if sport == "nhl":
        if market_key == "h2h":
            side = _team_side(outcome_name, home_team_name, away_team_name)
            if side is None:
                return None, None, False
            return "moneyline", side, False
        if market_key == "spreads":
            side = _team_side(outcome_name, home_team_name, away_team_name)
            if side is None or point is None:
                return None, None, False
            return "spread", side, True
        if market_key == "totals":
            n = outcome_name.lower()
            if n not in ("over", "under") or point is None:
                return None, None, False
            return "total", n, True
        return None, None, False

    return None, None, False


def process_event(cur, sport: str, event: dict, unmatched_log: list | None = None) -> int:
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    commence = event.get("commence_time")
    if not (home and away and commence):
        return 0

    try:
        commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    except ValueError:
        return 0

    match_id = find_match_id(cur, sport, home, away, commence_dt)
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
                point = outcome.get("point")
                point_f = float(point) if point is not None else None
                market_type, selection, keep_line = map_outcome(
                    sport, market_key, outcome.get("name", ""), home, away, point_f
                )
                if not market_type:
                    continue
                price = outcome.get("price")
                if price is None:
                    continue
                line = point_f if keep_line else None
                if insert_odds_row(
                    cur,
                    match_id,
                    bookmaker_name,
                    market_type,
                    selection,
                    float(price),
                    line,
                ):
                    inserted += 1
    return inserted


def run(database_url: str, sports: list[str], api_key: str, regions: str) -> dict[str, int]:
    results = {"events_seen": 0, "events_matched": 0, "odds_rows_inserted": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for sport_key in sports:
                sport = sport_for_key(sport_key)
                if sport is None:
                    logger.warning(
                        "%s: no registered sport family for this key — skipping. "
                        "Register the prefix in SPORT_KEY_PREFIXES and add a map_outcome branch.",
                        sport_key,
                    )
                    continue
                markets = SPORT_MARKETS[sport]
                events = fetch_sport_odds(sport_key, markets, api_key, regions)
                logger.info("%s (%s): %d events returned", sport_key, sport, len(events))
                results["events_seen"] += len(events)
                unmatched: list[str] = []
                for event in events:
                    inserted = process_event(cur, sport, event, unmatched_log=unmatched)
                    if inserted > 0:
                        results["events_matched"] += 1
                        results["odds_rows_inserted"] += inserted
                # Log unmatched events for visibility — usually means no
                # scheduled fixture in the matches table covering that
                # date, or a team-name spelling mismatch.
                if unmatched:
                    logger.info(
                        "%s: %d/%d events had no matching scheduled fixture. First 5: %s",
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
            f"Quota — used: {r.headers.get('x-requests-used')}, remaining: {r.headers.get('x-requests-remaining')}"
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
