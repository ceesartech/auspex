"""Fetch live (pre-match) odds from the-odds-api.com.

the-odds-api.com (https://the-odds-api.com) bills 1 quota unit per
(region × market) per call. We default to one region ("us") and send
the sport's full market basket in a single call, so a soccer run costs
3 units (h2h + totals + spreads) and an NHL run costs 3 units (h2h +
spreads + totals). With the configured sport list and one daily run this
is well under both the 500-unit free tier and the 20,000-unit paid tier.

Secondary soccer markets (BTTS, double chance, draw-no-bet) live on a
separate PER-EVENT endpoint and are opt-in via --additional-markets; they
cost quota per event, so they're off by default and quota-guarded.

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
    # NBA (in season Oct-Jun, including playoffs)
    "basketball_nba",
    # NFL (in season Sep-Feb, Super Bowl early Feb)
    "americanfootball_nfl",
    # Tennis — the-odds-api keys are EVENT-specific (Grand Slam +
    # major tour events). Out-of-tournament keys 422 silently so
    # the off-season cost is one round-trip per dormant key.
    # If we need broader coverage later, a /sports auto-discovery
    # pass can replace these hard-coded keys.
    "tennis_atp_australian_open",
    "tennis_atp_french_open",
    "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    "tennis_wta_australian_open",
    "tennis_wta_french_open",
    "tennis_wta_wimbledon",
    "tennis_wta_us_open",
    # MMA — UFC + Bellator. the-odds-api key is a single global
    # 'mma_mixed_martial_arts' that returns every active card across
    # promotions, so no per-promotion fanout needed.
    "mma_mixed_martial_arts",
]


# ── Sport dispatch ────────────────────────────────────────────────────
# the-odds-api keys are prefixed by sport family (`soccer_*`, `icehockey_*`,
# `americanfootball_*`, etc.). We derive auspex's internal sport label
# from the key prefix to pick the right outcome mapper and team scope.
SPORT_KEY_PREFIXES: dict[str, str] = {
    "soccer_": "soccer",
    "icehockey_": "nhl",
    "basketball_": "nba",
    "americanfootball_": "nfl",
    "tennis_": "tennis",
    "mma_": "mma",
}


# Per-sport market basket sent to the-odds-api bulk /odds endpoint. Each
# market costs 1 quota unit per region per call. Soccer pulls h2h (1x2),
# totals (over/under — ALL lines, the derivation layer offers 0.5…5.5) and
# spreads (goal handicap → asian_handicap); NHL keeps its three markets.
SPORT_MARKETS: dict[str, str] = {
    "soccer": "h2h,totals,spreads",
    "nhl": "h2h,spreads,totals",
    # NBA: moneyline + variable point spread + variable total. Unlike
    # NHL we want the FULL spread/total ladder (the-odds-api default
    # returns the book's main line per bookmaker) because NBA spreads
    # vary wildly per game and the training query uses the actual
    # closing line as a feature, not a fixed one.
    "nba": "h2h,spreads,totals",
    # NFL: same line-as-feature design as NBA. Spreads vary per game
    # (typical range -14.5 to +14.5), totals vary (typical 38-55).
    # h2h covers moneyline.
    "nfl": "h2h,spreads,totals",
    # Tennis: h2h (match-winner moneyline) + totals (total games
    # over/under — typical line ~22.5 for best-of-3, ~37.5 for
    # best-of-5). No "spreads" market on tennis at most US books —
    # set-handicap is a UK/EU specialty we can layer in later.
    "tennis": "h2h,totals",
    # MMA: h2h only in v1. Method-of-victory / round-group / fight-
    # goes-the-distance are MMA specialties available via per-event
    # additional markets — defer to v2 once we have the moneyline
    # corpus calibrated.
    "mma": "h2h",
}

# Secondary markets only available via the-odds-api PER-EVENT "additional
# markets" endpoint (not the bulk /odds call). Each costs quota per
# (region × market) PER EVENT, so fetching them is opt-in (--additional-markets)
# and quota-guarded. Coverage is region/bookmaker dependent (uk/eu books price
# these far more than us), hence a separate region knob (--additional-regions).
SPORT_ADDITIONAL_MARKETS: dict[str, str] = {
    "soccer": "btts,double_chance,draw_no_bet",
}

# Market keys we know how to map per sport. Anything else returned by the API
# is skipped and logged once per run (so new opportunities surface without
# crashing the job).
KNOWN_MARKET_KEYS: dict[str, set[str]] = {
    "soccer": {"h2h", "totals", "spreads", "btts", "double_chance", "draw_no_bet"},
    "nhl": {"h2h", "spreads", "totals"},
    "nba": {"h2h", "spreads", "totals"},
    "nfl": {"h2h", "spreads", "totals"},
    "tennis": {"h2h", "totals"},
    "mma": {"h2h"},
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

    # Status filter includes both 'scheduled' (live-odds path) and
    # 'finished' (historical backfill path). Both insert pre-match
    # odds with is_opening=false / is_live=false, so the row semantics
    # don't change based on which path matched. Excluding 'finished'
    # was the silent bug behind 0 NHL odds rows after the first
    # backfill run — historical games are 'finished' by definition.
    cur.execute(
        """
        SELECT m.id::text AS id, ht.name AS home_name, at.name AS away_name, m.match_date
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.status IN ('scheduled', 'finished')
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


def _double_chance_code(outcome_name: str, home_team_name: str, away_team_name: str) -> str | None:
    """Map a the-odds-api double_chance outcome to our 1X/12/X2 code.

    the-odds-api formats double-chance outcomes as the two covered results
    joined by '/', using the actual team names and the literal 'Draw'
    (e.g. 'Arsenal/Draw', 'Arsenal/Chelsea', 'Draw/Chelsea'). We detect which
    of {home, away, draw} the name covers. Returns None if it doesn't resolve
    to a known pair (so the caller skips it rather than guessing)."""
    name = outcome_name.strip().lower()
    has_home = home_team_name.strip().lower() in name
    has_away = away_team_name.strip().lower() in name
    has_draw = "draw" in name
    if has_home and has_draw:
        return "1X"
    if has_home and has_away:
        return "12"
    if has_draw and has_away:
        return "X2"
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
    (over/under, EVERY line — the derivation layer offers 0.5…5.5); spreads
    → asian_handicap (home/away, line = the outcome's own signed goal
    handicap). Secondary per-event markets: btts → btts (yes/no);
    double_chance → double_chance (1X/12/X2); draw_no_bet → draw_no_bet
    (home/away).

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
            if n not in ("over", "under") or point is None:
                return None, None, False
            # Keep every line — the derivation/recommendation layer prices
            # over/under at 0.5 … 5.5, so we store all lines the book offers.
            return "over_under", n, True
        if market_key == "spreads":
            side = _team_side(outcome_name, home_team_name, away_team_name)
            if side is None or point is None:
                return None, None, False
            # Goal handicap → asian_handicap. `line` is the outcome's OWN
            # signed point; home and away carry opposite signs. The
            # recommendation matcher converts the away point to the
            # home-perspective line.
            return "asian_handicap", side, True
        if market_key == "btts":
            n = outcome_name.strip().lower()
            if n not in ("yes", "no"):
                return None, None, False
            return "btts", n, False
        if market_key == "double_chance":
            code = _double_chance_code(outcome_name, home_team_name, away_team_name)
            if code is None:
                return None, None, False
            return "double_chance", code, False
        if market_key == "draw_no_bet":
            side = _team_side(outcome_name, home_team_name, away_team_name)
            if side is None:
                return None, None, False
            return "draw_no_bet", side, False
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

    if sport == "nba":
        # Structurally identical to NHL — same market_type values, same
        # selection shape. The difference is in the LINE: NHL puck_line
        # is fixed at ±1.5 and total at 5.5; NBA spread varies (-3.5 to
        # -13.5+) and total varies (200 to 245+) per game. The training
        # query for NBA uses the actual closing line as a feature, so
        # we keep every line the book offers (no canonical filter here).
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

    if sport == "nfl":
        # Same shape as NBA — h2h/spreads/totals with variable lines.
        # NFL spread range is ~ -14.5 to +14.5; totals 38-55. The
        # training query treats closing line as an input feature
        # (same line-as-feature design as NBA), so we accept every
        # line a book offers and let the model condition on it at
        # predict time.
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

    if sport == "tennis":
        # 1v1, no draw. h2h → match-winner moneyline; totals → games
        # over/under (line varies: best-of-3 ≈ 22.5, best-of-5 ≈ 37.5).
        # No spreads market — set-handicap is a UK/EU specialty not in
        # the default US odds basket.
        if market_key == "h2h":
            side = _team_side(outcome_name, home_team_name, away_team_name)
            if side is None:
                return None, None, False
            return "moneyline", side, False
        if market_key == "totals":
            n = outcome_name.lower()
            if n not in ("over", "under") or point is None:
                return None, None, False
            return "total", n, True
        return None, None, False

    if sport == "mma":
        # 1v1, no draws (draws are technically possible — split-decision
        # draws happen ~1% — but the model treats them as ungradable;
        # see grading_outcomes.mma handling). Single market in v1.
        if market_key == "h2h":
            side = _team_side(outcome_name, home_team_name, away_team_name)
            if side is None:
                return None, None, False
            return "moneyline", side, False
        return None, None, False

    return None, None, False


def _match_event(cur, sport: str, event: dict, unmatched_log: list | None = None) -> tuple[str | None, str, str]:
    """Resolve an event payload to (match_id, home, away). Returns
    (None, home, away) when the event can't be tied to a fixture in our
    `matches` table (logged to unmatched_log if provided)."""
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    commence = event.get("commence_time")
    if not (home and away and commence):
        return None, home, away

    try:
        commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    except ValueError:
        return None, home, away

    match_id = find_match_id(cur, sport, home, away, commence_dt)
    if not match_id and unmatched_log is not None:
        unmatched_log.append(f"{home} vs {away} @ {commence}")
    return match_id, home, away


def insert_event_odds(
    cur,
    match_id: str,
    event: dict,
    sport: str,
    home: str,
    away: str,
    unknown_markets: set | None = None,
) -> int:
    """Insert every representable odds row from one event payload. The bulk
    /odds response and the per-event additional-markets response share the
    same bookmakers → markets → outcomes shape, so both feed through here.
    Unhandled market keys are collected into `unknown_markets` for one-time
    logging. Returns the number of rows inserted."""
    inserted = 0
    for bm in event.get("bookmakers", []):
        bookmaker_name = bm.get("title") or bm.get("key") or "unknown"
        for market in bm.get("markets", []):
            market_key = market.get("key")
            if unknown_markets is not None and market_key and market_key not in KNOWN_MARKET_KEYS.get(sport, set()):
                unknown_markets.add(market_key)
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


def process_event(
    cur,
    sport: str,
    event: dict,
    unmatched_log: list | None = None,
    unknown_markets: set | None = None,
) -> int:
    """Resolve an event to a fixture and insert its bulk-market odds rows.
    Returns rows inserted (0 if the event can't be matched). Kept as the
    single-call entry point used by backfill_historical_odds.py."""
    match_id, home, away = _match_event(cur, sport, event, unmatched_log=unmatched_log)
    if not match_id:
        return 0
    return insert_event_odds(cur, match_id, event, sport, home, away, unknown_markets=unknown_markets)


def fetch_event_markets(
    sport_key: str, event_id: str, markets: str, api_key: str, regions: str
) -> tuple[dict, str | None] | None:
    """Fetch secondary (per-event) markets for one event from the-odds-api.

    Returns (event_payload, x_requests_remaining) on success — the payload is a
    single event object with the same shape as a bulk event — or None on any
    error / unavailable market, so the caller falls back to the bulk-only rows.
    Raises only on a 401 (bad key), which should abort the whole run."""
    if not event_id:
        return None
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        logger.warning("the-odds-api event %s: request failed (%s)", event_id, e)
        return None
    if r.status_code == 401:
        raise RuntimeError("the-odds-api: invalid API key (401)")
    # 404 = event expired/unknown, 422 = market unavailable for this event —
    # both mean "no secondary markets here", skip quietly.
    if r.status_code in (404, 422):
        return None
    try:
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("the-odds-api event %s: HTTP error (%s)", event_id, e)
        return None

    remaining = r.headers.get("x-requests-remaining")
    try:
        payload = r.json()
    except ValueError:
        return None
    return payload, remaining


def run(
    database_url: str,
    sports: list[str],
    api_key: str,
    regions: str,
    *,
    additional_markets: bool = False,
    additional_regions: str = "eu",
    max_additional_events: int | None = None,
    quota_floor: int = 0,
) -> dict[str, int]:
    results = {
        "events_seen": 0,
        "events_matched": 0,
        "odds_rows_inserted": 0,
        "additional_rows_inserted": 0,
    }
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

                # Secondary markets (per-event endpoint) — opt-in + quota-guarded.
                add_markets = SPORT_ADDITIONAL_MARKETS.get(sport) if additional_markets else None
                add_calls = 0
                stop_additional = False

                unmatched: list[str] = []
                unknown: set[str] = set()
                for event in events:
                    match_id, home, away = _match_event(cur, sport, event, unmatched_log=unmatched)
                    if not match_id:
                        continue
                    inserted = insert_event_odds(cur, match_id, event, sport, home, away, unknown_markets=unknown)
                    if inserted > 0:
                        results["events_matched"] += 1
                        results["odds_rows_inserted"] += inserted

                    # Per-event additional markets for matched fixtures.
                    if add_markets and not stop_additional:
                        if max_additional_events is not None and add_calls >= max_additional_events:
                            stop_additional = True
                        else:
                            res = fetch_event_markets(
                                sport_key, event.get("id", ""), add_markets, api_key, additional_regions
                            )
                            add_calls += 1
                            if res is not None:
                                payload, remaining = res
                                if payload:
                                    results["additional_rows_inserted"] += insert_event_odds(
                                        cur, match_id, payload, sport, home, away, unknown_markets=unknown
                                    )
                                if quota_floor and remaining is not None:
                                    try:
                                        if int(remaining) < quota_floor:
                                            logger.warning(
                                                "Quota remaining %s below floor %d — stopping "
                                                "additional-market fetches for this run.",
                                                remaining,
                                                quota_floor,
                                            )
                                            stop_additional = True
                                    except ValueError:
                                        pass

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
                if unknown:
                    logger.info("%s: skipped unhandled market keys %s", sport_key, sorted(unknown))
                conn.commit()
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sports", default=",".join(DEFAULT_SPORTS), help="Comma-separated the-odds-api sport keys.")
    p.add_argument("--regions", default="us", help="the-odds-api region: us, uk, eu, au (default 'us').")
    p.add_argument("--api-key", default=os.environ.get("THE_ODDS_API_KEY"))
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--quota-only", action="store_true", help="Make a single call to check remaining quota and exit.")
    p.add_argument(
        "--additional-markets",
        action="store_true",
        help="Also fetch secondary soccer markets (btts, double_chance, draw_no_bet) via the "
        "per-event endpoint. Costs extra quota per event — off by default.",
    )
    p.add_argument(
        "--additional-regions",
        default="eu",
        help="Region(s) for the per-event additional-markets call (default 'eu' — soccer "
        "secondary markets have better coverage on uk/eu books).",
    )
    p.add_argument(
        "--max-additional-events",
        type=int,
        default=None,
        help="Cap how many events get a per-event additional-markets call (quota guard).",
    )
    p.add_argument(
        "--quota-floor",
        type=int,
        default=0,
        help="Stop additional-market fetches once x-requests-remaining drops below this (0 = no floor).",
    )
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
        print(f"Quota — used: {r.headers.get('x-requests-used')}, remaining: {r.headers.get('x-requests-remaining')}")
        return 0

    sports = [s.strip() for s in args.sports.split(",") if s.strip()]
    results = run(
        args.database_url,
        sports,
        args.api_key,
        args.regions,
        additional_markets=args.additional_markets,
        additional_regions=args.additional_regions,
        max_additional_events=args.max_additional_events,
        quota_floor=args.quota_floor,
    )
    logger.info(
        "Done. events_seen=%d events_matched=%d odds_rows_inserted=%d additional_rows_inserted=%d",
        results["events_seen"],
        results["events_matched"],
        results["odds_rows_inserted"],
        results["additional_rows_inserted"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
