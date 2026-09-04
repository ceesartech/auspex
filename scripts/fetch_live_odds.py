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
  3. Upserts one row per (bookmaker × market × selection [× line]) into
     the `odds` table with is_opening=false, is_live=false: a key is
     INSERTed the first time it is seen, then UPDATEd in place — price and
     `timestamp` — every time the book moves it, and its `last_seen_at` is
     stamped on EVERY observation whether the price moved or not. `odds`
     therefore holds the CURRENT price, `odds_snapshots` holds the price
     HISTORY (one row per move), and the is_opening=true row (written by
     scripts/promote_raw.py) is the opening line and is never touched.
     A quote seen AFTER kickoff can only create a key that does not exist
     yet: it never overwrites a pre-match price, because `odds` is read as
     the pre-match/closing line by the training frames.
  4. Sanity-checks every bookmaker against the rest of the same payload and
     drops that book's whole block for the match when its handicap line AND
     its moneyline both put the other team on top — the side-swap canary,
     see detect_side_swapped_books.

WHY `odds` is a CURRENT-price table (2026-09 prod audit, defect 1): it used
to keep the FIRST price ever seen for a key forever, behind a NOT EXISTS
guard, so every recommendation was priced off a quote that averaged 178h
(1x2) / 89h (asian_handicap) / 149h (over_under) / 197h (mma) old. 48% of
settled soccer 1x2 recs failed their own 5% EV gate at the price actually
available when they were written, average claimed EV fell 0.132 -> 0.079,
realized ROI was +4.4% rather than the +14.1% claimed, and vw_rec_clv was
measuring first-seen staleness instead of closing-line value. Readers that
want history read odds_snapshots; readers that want "what is bettable now"
read odds.

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
from statistics import median

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
    # Tennis — the-odds-api keys are EVENT-specific. Coverage
    # expanded 2026-06-04 from 4 Grand Slams to include the
    # ATP/WTA 1000 (Masters/Premier) tier so we have year-round
    # tennis odds rather than just the 4 majors. Out-of-tournament
    # keys 422 silently so off-season cost is one round-trip per
    # dormant key — adds ~14 dormant-day round trips across the
    # whole list (negligible quota at 0 units each).
    # Grand Slams
    "tennis_atp_australian_open",
    "tennis_atp_french_open",
    "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    "tennis_wta_australian_open",
    "tennis_wta_french_open",
    "tennis_wta_wimbledon",
    "tennis_wta_us_open",
    # ATP Masters 1000 + year-end finals
    "tennis_atp_indian_wells",
    "tennis_atp_miami",
    "tennis_atp_monte_carlo",
    "tennis_atp_madrid_open",
    "tennis_atp_italian_open",
    "tennis_atp_canadian_open",
    "tennis_atp_cincinnati_open",
    "tennis_atp_shanghai_masters",
    "tennis_atp_paris_masters",
    "tennis_atp_finals",
    # WTA 1000 + year-end finals
    "tennis_wta_indian_wells",
    "tennis_wta_miami",
    "tennis_wta_madrid_open",
    "tennis_wta_italian_open",
    "tennis_wta_canadian_open",
    "tennis_wta_cincinnati_open",
    "tennis_wta_china_open",
    "tennis_wta_wuhan_open",
    "tennis_wta_finals",
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
    # Soccer secondary + halftime markets. The HT trio (h2h_h1,
    # totals_h1, btts_h1) lets the recs engine consume our existing
    # HT predictions (PR #11, migration 014); without them the HT
    # predictions just sit in the DB unused. They're included in
    # `--additional-markets` so enabling that flag costs more quota
    # than before but unlocks all soccer secondary + HT markets in
    # one pass.
    "soccer": "btts,double_chance,draw_no_bet,h2h_h1,totals_h1,btts_h1",
}

# Market keys we know how to map per sport. Anything else returned by the API
# is skipped and logged once per run (so new opportunities surface without
# crashing the job).
KNOWN_MARKET_KEYS: dict[str, set[str]] = {
    "soccer": {
        "h2h",
        "totals",
        "spreads",
        "btts",
        "double_chance",
        "draw_no_bet",
        "h2h_h1",
        "totals_h1",
        "btts_h1",
    },
    "nhl": {"h2h", "spreads", "totals"},
    "nba": {"h2h", "spreads", "totals"},
    "nfl": {"h2h", "spreads", "totals"},
    "tennis": {"h2h", "totals"},
    "mma": {"h2h"},
}

# insert_odds_row outcomes. Callers count rows WRITTEN (ODDS_WRITTEN) and
# report the split in the per-run summary. ODDS_UNCHANGED means "the book
# still shows this price": the row's last_seen_at is stamped even though
# nothing was rewritten. ODDS_POST_COMMENCE means an existing PRE-MATCH row
# was deliberately left alone because the match has already started.
ODDS_INSERTED = "inserted"
ODDS_UPDATED = "updated"
ODDS_UNCHANGED = "unchanged"
ODDS_POST_COMMENCE = "post_commence_protected"
ODDS_WRITTEN = frozenset({ODDS_INSERTED, ODDS_UPDATED})

# Markets where the two sides of one group carry OPPOSITE signed lines, so a
# bookmaker whose home/away sides are transposed in the feed shows up as a
# sign flip against the rest of the market. These are exactly the market_type
# values map_outcome produces for the-odds-api "spreads" key (soccer ->
# asian_handicap; nhl/nba/nfl -> spread). Totals (over_under / total) are
# deliberately excluded: over and under legitimately share one positive line,
# so a sign test there would fire on every book.
HANDICAP_MARKET_TYPES: frozenset[str] = frozenset({"asian_handicap", "spread"})

# The two-or-three-way match-winner markets map_outcome produces from the
# "h2h" key (soccer -> 1x2, everything else -> moneyline). A real side swap
# transposes these prices too, which is what the canary corroborates against.
MONEYLINE_MARKET_TYPES: frozenset[str] = frozenset({"1x2", "moneyline"})

# How many OTHER bookmakers must have priced the handicap before their median
# counts as a consensus worth rejecting a book against. Below this we keep the
# book: with one peer there is no way to tell which of the two is swapped.
MIN_SIDE_SWAP_PEERS = 2

# How lopsided a book's moneyline must be before its FAVOURITE counts as
# stated, measured as the de-vigged (p_home - p_away) margin — not p_home
# against 0.5, because a 3-way soccer market routinely prices a real home
# favourite below 0.5 once the draw takes its share. Corroboration needs this
# margin on both sides with OPPOSITE signs: a real transposition moves both
# markets (the 2026-08 BetUS runs had "the h2h prices swapped too"), while a
# genuine cross-book disagreement about who gives the handicap happens on
# pick'em games, where every book prices the two sides within a whisker of
# each other. 0.04 = a 4-point disagreement about who is the better team.
SIDE_SWAP_H2H_MARGIN = 0.04


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


# The row identity every statement in insert_odds_row matches on, and
# byte-for-byte the key migration 024 makes unique — so the writer can never
# permit a row the index would reject. Both flags go through
# COALESCE(..., false) rather than `= false` because both columns are NULLable
# (DEFAULT false, no NOT NULL): a legacy NULL would otherwise be invisible to
# the UPDATE *and* to the INSERT's guard and get shadow-duplicated. is_live is
# in the key because a live-odds feed (nothing writes one today) would be a
# different price series that must neither block nor clobber the pre-match one.
_ODDS_KEY_SQL = """
              match_id = %s AND bookmaker = %s
              AND market_type = %s AND selection = %s
              AND COALESCE(line, -1) = COALESCE(%s, -1)
              AND COALESCE(is_opening, false) = false
              AND COALESCE(is_live, false) = false
"""


def insert_odds_row(
    cur,
    match_id: str,
    bookmaker: str,
    market_type: str,
    selection: str,
    odds_decimal: float,
    line: float | None,
    pre_match: bool = True,
) -> str:
    """Upsert the CURRENT price for one (match, bookmaker, market, selection,
    line) key. Returns ODDS_INSERTED / ODDS_UPDATED / ODDS_UNCHANGED /
    ODDS_POST_COMMENCE — callers count what was actually written.

    The non-opening row for a key is UPDATED in place when the book's price
    moved; it is NOT skipped forever as it was before the 2026-09 audit (see
    the module docstring: recommendations were being priced off quotes up to
    8 days stale). The is_opening=true row is the OPENING line — other code
    reads it as such — so every statement here is scoped to the non-opening,
    non-live row and the opening row is never touched. Price history is not
    lost: every move is appended to odds_snapshots by insert_snapshot_row.

    TWO timestamps, because they answer different questions:
      * `timestamp`   — when this PRICE last changed. Advanced only on a move,
                        which is what the cross-book feature builders and the
                        API's price views have always read it as.
      * `last_seen_at`— when we last SAW the book quote it, stamped on every
                        observation whether or not the price moved. This is
                        the freshness signal the recommendation guard ages off
                        (scripts/generate_recommendations.MAX_ODDS_AGE_HOURS):
                        a book that pulls a market leaves its last price
                        behind forever, and only an unconditional stamp can
                        tell "still on offer" from "frozen since Tuesday".
    Requires migration 024 (adds the NULLable last_seen_at column). Readers
    COALESCE it back to `timestamp`, so rows written before the migration
    simply read as older than they are — the safe direction.

    `pre_match=False` means the fixture has already kicked off. An in-play
    quote must never REPLACE the pre-match price: `odds` is read as the
    closing/pre-match line by the training frames and feature builders
    (services/ml-models/src/utils/training_data.py, scripts/compute_features*),
    so overwriting it with a live number injects post-kickoff information into
    a model input. A post-commence observation therefore leaves an existing
    row completely alone (returning ODDS_POST_COMMENCE) and only ever INSERTs
    a key that does not exist yet — which is what keeps
    scripts/backfill_historical_odds.py, whose snapshots are deliberately
    taken at/after commence, writing exactly the rows it always did.

    Implemented as UPDATE-then-INSERT-if-no-rows rather than a real
    `ON CONFLICT` upsert because the `odds` table has NO unique constraint on
    this key (see services/data-ingestion/db/migrations/024_odds_current_price.sql:
    one can only be added once prod is verified duplicate-free, which cannot be
    checked from here). The statements are equivalent under the single serial
    writer the ingest DAG runs. A concurrent second writer racing between them
    would raise a duplicate only if that migration's unique index got created —
    which is the loud failure we want, not a silent second "current" price.

    The comparison rounds to the column's own scale (DECIMAL(10,4)); without
    that, a price like 1.9090909 would differ from the stored 1.9091 on every
    cycle and churn the timestamp forever.
    """
    key_params = (match_id, bookmaker, market_type, selection, line)

    if not pre_match:
        # The match has started. Never touch an existing pre-match row —
        # not its price, not even its last_seen_at (a live quote is not
        # evidence that the pre-match price is still on offer).
        cur.execute(
            f"""
            SELECT 1 FROM odds
            WHERE {_ODDS_KEY_SQL}
            LIMIT 1
            """,
            key_params,
        )
        if cur.fetchone() is not None:
            return ODDS_POST_COMMENCE
    else:
        cur.execute(
            f"""
            UPDATE odds
               SET odds_decimal = ROUND(%s::numeric, 4),
                   timestamp = NOW(),
                   last_seen_at = NOW()
             WHERE {_ODDS_KEY_SQL}
               AND odds_decimal IS DISTINCT FROM ROUND(%s::numeric, 4)
            """,
            (odds_decimal, *key_params, odds_decimal),
        )
        if (cur.rowcount or 0) > 0:
            return ODDS_UPDATED

        # Same price as last cycle: the row is not rewritten, but the book IS
        # still quoting it, and that is exactly what the freshness guard needs
        # to know. Only last_seen_at moves.
        cur.execute(
            f"""
            UPDATE odds
               SET last_seen_at = NOW()
             WHERE {_ODDS_KEY_SQL}
            """,
            key_params,
        )
        if (cur.rowcount or 0) > 0:
            return ODDS_UNCHANGED

    cur.execute(
        f"""
        INSERT INTO odds (match_id, bookmaker, market_type, selection,
                          odds_decimal, line, is_opening, is_live, last_seen_at)
        SELECT %s, %s, %s, %s, %s, %s, false, false, NOW()
        WHERE NOT EXISTS (
            SELECT 1 FROM odds
            WHERE {_ODDS_KEY_SQL}
        )
        """,
        (
            match_id,
            bookmaker,
            market_type,
            selection,
            odds_decimal,
            line,
            *key_params,
        ),
    )
    return ODDS_INSERTED if (cur.rowcount or 0) > 0 else ODDS_UNCHANGED


def insert_snapshot_row(
    cur, match_id: str, bookmaker: str, market_type: str, selection: str, odds_decimal: float, line: float | None
) -> bool:
    """Append to odds_snapshots when the price CHANGED vs the key's latest
    snapshot (or none exists yet). This is the CLV capture path (audit doc
    §2.2): `odds` carries ONE row per key — the current price (insert_odds_row
    overwrites it in place) — so the price HISTORY lives here instead. Bounded
    growth, one row per actual price move, read only by vw_rec_clv /
    vw_clv_summary. NOTE: this table stays append-only precisely because its
    readers need the time series; adding history rows to `odds` would silently
    turn every AVG(odds_decimal) reader from "average across BOOKS" into
    "average across books AND time" (migration 022's design note)."""
    cur.execute(
        """
        INSERT INTO odds_snapshots (match_id, bookmaker, market_type, selection, line, odds_decimal)
        SELECT %s, %s, %s, %s, %s, %s
        WHERE COALESCE((
            SELECT s.odds_decimal FROM odds_snapshots s
            WHERE s.match_id = %s AND s.bookmaker = %s
              AND s.market_type = %s AND s.selection = %s
              AND COALESCE(s.line, -1) = COALESCE(%s, -1)
            ORDER BY s.captured_at DESC
            LIMIT 1
        ), -1) <> %s
        """,
        (
            match_id,
            bookmaker,
            market_type,
            selection,
            line,
            odds_decimal,
            match_id,
            bookmaker,
            market_type,
            selection,
            line,
            odds_decimal,
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
        # ── Halftime markets (per-event additional endpoint) ──
        # h2h_h1 → match_result_ht (3-way: home/draw/away at HT)
        if market_key == "h2h_h1":
            if outcome_name == home_team_name:
                return "match_result_ht", "home", False
            if outcome_name == away_team_name:
                return "match_result_ht", "away", False
            if outcome_name.lower() == "draw":
                return "match_result_ht", "draw", False
            return None, None, False
        # totals_h1 → over_under_ht (over/under, keep every line —
        # books typically offer 0.5 and 1.5 at HT).
        if market_key == "totals_h1":
            n = outcome_name.lower()
            if n not in ("over", "under") or point is None:
                return None, None, False
            return "over_under_ht", n, True
        # btts_h1 → btts_ht (yes/no on both-teams-score-before-HT).
        if market_key == "btts_h1":
            n = outcome_name.strip().lower()
            if n not in ("yes", "no"):
                return None, None, False
            return "btts_ht", n, False
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


def _book_home_handicap_lines(event: dict, sport: str, home: str, away: str) -> dict[str, float]:
    """Each bookmaker's representative signed HOME handicap line in one payload.

    Routed through map_outcome so the set of handicap markets can never drift
    from what actually gets stored (and so totals are excluded for free). A
    book quoting several handicap lines for the same match is represented by
    their median. Books with no handicap price are absent from the result.
    """
    lines: dict[str, float] = {}
    for bm in event.get("bookmakers", []):
        bookmaker_name = bm.get("title") or bm.get("key") or "unknown"
        points: list[float] = []
        for market in bm.get("markets", []):
            market_key = market.get("key")
            for outcome in market.get("outcomes", []):
                point = outcome.get("point")
                if point is None:
                    continue
                try:
                    point_f = float(point)
                except (TypeError, ValueError):
                    continue
                market_type, selection, keep_line = map_outcome(
                    sport, market_key, outcome.get("name", ""), home, away, point_f
                )
                if keep_line and selection == "home" and market_type in HANDICAP_MARKET_TYPES:
                    points.append(point_f)
        if points:
            lines[bookmaker_name] = float(median(points))
    return lines


def _book_home_h2h_margins(event: dict, sport: str, home: str, away: str) -> dict[str, float]:
    """Each bookmaker's de-vigged (p_home - p_away) margin on the match-winner
    market in one payload — positive when that book makes HOME the better team.

    Routed through map_outcome for the same reason as the handicap lines: the
    market set can never drift from what actually gets stored. Normalised over
    whatever selections the book quotes (2-way, or 3-way with the draw), so the
    margin is comparable across books and across sports; the DIFFERENCE is used
    rather than p_home alone because a 3-way market gives a genuine home
    favourite well under 0.5. Books with no usable h2h block, or with a
    non-positive price, are absent from the result.
    """
    margins: dict[str, float] = {}
    for bm in event.get("bookmakers", []):
        bookmaker_name = bm.get("title") or bm.get("key") or "unknown"
        prices: dict[str, float] = {}
        for market in bm.get("markets", []):
            market_key = market.get("key")
            for outcome in market.get("outcomes", []):
                market_type, selection, _keep_line = map_outcome(
                    sport, market_key, outcome.get("name", ""), home, away, None
                )
                if market_type not in MONEYLINE_MARKET_TYPES or selection is None:
                    continue
                try:
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                if price > 0.0:
                    prices[selection] = price
        if "home" not in prices or "away" not in prices:
            continue
        overround = sum(1.0 / price for price in prices.values())
        if overround > 0.0:
            margins[bookmaker_name] = (1.0 / prices["home"] - 1.0 / prices["away"]) / overround
    return margins


def detect_side_swapped_books(event: dict, sport: str, home: str, away: str) -> dict[str, dict]:
    """Bookmakers in this payload whose HOME/AWAY sides look transposed.

    Returns {bookmaker: {line, line_consensus, peers, h2h, h2h_consensus}} for
    every book that fails BOTH tests against the rest of the same payload:

      1. its signed HOME handicap line has the OPPOSITE sign to the median of
         at least MIN_SIDE_SWAP_PEERS other books, both being non-zero; and
      2. its de-vigged moneyline margin (p_home - p_away) has the OPPOSITE
         sign to the peer median, both being at least SIDE_SWAP_H2H_MARGIN in
         size — i.e. the book and the market disagree about which team is
         better, not merely about the price.

    Never raises. A book with no peers, a zero line (pick'em), a zero
    consensus, or no corroborating h2h block is KEPT — and a book that fails
    (1) without (2) is logged at WARNING as "suspected but not corroborated"
    so the base rate stays observable.

    WHY corroboration (2026-09 prod audit, defect 2 + review): the sign test
    alone cannot tell a transposed feed from a legitimate cross-book
    disagreement about which side gives the handicap, and that disagreement is
    structural on the team-sport `spread` market — an NHL puck line is always
    ±1.5 and its sign is just "who is the favourite", so on a near-even game a
    minority book genuinely posting home +1.5 against peers at -1.5 is
    indistinguishable from a swap. A magnitude threshold does not separate them
    either (both a swap and a pick'em disagreement are a full 2-3 point gap).
    What DOES separate them is the moneyline: a real transposition flips both
    markets — the audit's own evidence for the 2026-08 BetUS runs is that "the
    h2h prices were swapped too" — while a pick'em disagreement comes with two
    books pricing the moneyline within a whisker of each other.

    WHY at all (2026-09 prod audit, defect 2): two whole BetUS ingest runs
    (2026-08-09 and 2026-08-22) carried soccer asian-handicap lines whose sign
    opposed the rest of the market. BetUS's home line opposed the same-minute
    median of >= 2 other books in 14/1296 snapshots while BetOnline (5,773),
    LowVig (5,749), Bovada (2,664) and MyBookie (911) did it exactly zero
    times, so this is an upstream feed defect, not a mapping bug (_team_side
    matches team names exactly and yields None — row skipped — on any
    mismatch, never a swapped side). Those bad quotes then persisted as their
    own line key forever under the old first-seen rule and were picked as
    "best price": the resulting recs won 38 of 47 near-even-money handicaps
    (+67% ROI) on lines the book never offered, which was the entire positive
    contribution to the settled book.
    """
    lines = _book_home_handicap_lines(event, sport, home, away)
    h2h = _book_home_h2h_margins(event, sport, home, away)
    flagged: dict[str, dict] = {}
    for book, own in lines.items():
        peers = [value for other, value in lines.items() if other != book]
        if len(peers) < MIN_SIDE_SWAP_PEERS:
            continue
        consensus = float(median(peers))
        if own == 0.0 or consensus == 0.0:
            continue
        if (own > 0.0) == (consensus > 0.0):
            continue

        own_p = h2h.get(book)
        peer_ps = [value for other, value in h2h.items() if other != book]
        peer_p = float(median(peer_ps)) if peer_ps else None
        corroborated = (
            own_p is not None
            and peer_p is not None
            and abs(own_p) >= SIDE_SWAP_H2H_MARGIN
            and abs(peer_p) >= SIDE_SWAP_H2H_MARGIN
            and (own_p > 0.0) != (peer_p > 0.0)
        )
        if not corroborated:
            logger.warning(
                "SIDE-SWAP SUSPECTED BUT NOT CORROBORATED: %s home line %+.2f opposes the median %+.2f "
                "of %d other book(s) on %s vs %s, but its moneyline does not disagree with the market "
                "about which team is better (book p_home-p_away=%s, peers %s) — keeping the book. A "
                "genuine cross-book disagreement about who gives the handicap looks exactly like this.",
                book,
                own,
                consensus,
                len(peers),
                home,
                away,
                "unknown" if own_p is None else f"{own_p:+.3f}",
                "unknown" if peer_p is None else f"{peer_p:+.3f}",
            )
            continue

        flagged[book] = {
            "line": own,
            "line_consensus": consensus,
            "peers": len(peers),
            "h2h_margin": own_p,
            "h2h_margin_consensus": peer_p,
        }
    return flagged


def is_pre_match(cur, match_id: str) -> bool:
    """Has this fixture NOT started yet?

    One cheap lookup per event, used to keep a post-kickoff (in-play) quote
    from overwriting the pre-match price — see insert_odds_row's `pre_match`.
    A match_id we cannot find (or a NULL match_date) answers False, the
    conservative direction: unknown means "do not overwrite".
    """
    cur.execute(
        """
        SELECT 1 FROM matches
        WHERE id = %s AND match_date > NOW()
        LIMIT 1
        """,
        (match_id,),
    )
    return cur.fetchone() is not None


def insert_event_odds(
    cur,
    match_id: str,
    event: dict,
    sport: str,
    home: str,
    away: str,
    unknown_markets: set | None = None,
    stats: dict | None = None,
) -> int:
    """Write every representable odds row from one event payload. The bulk
    /odds response and the per-event additional-markets response share the
    same bookmakers → markets → outcomes shape, so both feed through here.
    Unhandled market keys are collected into `unknown_markets` for one-time
    logging.

    A bookmaker flagged by the side-swap canary is dropped WHOLE for this
    match — every market, not just the signed-line one, and neither the odds
    rows nor the snapshots are written. The canary only fires when the book
    disagrees with the market about the favourite on BOTH the handicap and the
    moneyline, which is evidence that the book's home/away sides are
    transposed in this payload; keeping its 1x2 prices after proving that
    would move the phantom-price bug from asian_handicap into soccer 1x2, one
    of the two live team-sport streams (scripts/rec_gating.py). The drop is
    logged at ERROR level per (bookmaker, match) and counted; it never raises
    and never happens silently.

    Post-kickoff quotes never overwrite a pre-match price (in-play prices are
    what `odds` must not be polluted with — see insert_odds_row); they can only
    create keys that do not exist yet, which is the historical-backfill path.

    Returns the number of rows WRITTEN (inserted + price-updated). When
    `stats` is given, the ODDS_INSERTED / ODDS_UPDATED / ODDS_UNCHANGED /
    ODDS_POST_COMMENCE split and "side_swap_rejections" are accumulated into
    it for the run summary.
    """
    inserted = 0
    pre_match = is_pre_match(cur, match_id)
    swapped = detect_side_swapped_books(event, sport, home, away)
    for book, detail in sorted(swapped.items()):
        logger.error(
            "SIDE-SWAP CANARY: dropping ALL %s prices for match %s (%s vs %s) — its home line "
            "%+.2f opposes the median %+.2f of %d other book(s) in the same payload, AND its "
            "moneyline makes the other team better (p_home-p_away=%+.3f vs peers %+.3f). The book's "
            "home/away sides look transposed in the feed (the 2026-08 BetUS incident); nothing "
            "from it is ingested for this match.",
            book,
            match_id,
            home,
            away,
            detail["line"],
            detail["line_consensus"],
            detail["peers"],
            detail["h2h_margin"],
            detail["h2h_margin_consensus"],
        )
        if stats is not None:
            stats["side_swap_rejections"] = stats.get("side_swap_rejections", 0) + 1

    for bm in event.get("bookmakers", []):
        bookmaker_name = bm.get("title") or bm.get("key") or "unknown"
        if bookmaker_name in swapped:
            # Proven transposed for this match: not one market of it is
            # trustworthy, and a swapped 1x2 price is exactly the outlier that
            # wins a best-price pick.
            continue
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
                status = insert_odds_row(
                    cur,
                    match_id,
                    bookmaker_name,
                    market_type,
                    selection,
                    float(price),
                    line,
                    pre_match=pre_match,
                )
                if status in ODDS_WRITTEN:
                    inserted += 1
                if stats is not None:
                    stats[status] = stats.get(status, 0) + 1
                # CLV time-series capture — independent of the odds-table
                # current-price upsert, so every price MOVE keeps its own row
                # (the closing line is read from here). Post-commence captures
                # are kept: odds_snapshots is a time series and its readers
                # bound themselves by captured_at (migration 022's vw_rec_clv
                # requires captured_at < match_date).
                insert_snapshot_row(
                    cur,
                    match_id,
                    bookmaker_name,
                    market_type,
                    selection,
                    float(price),
                    line,
                )
    return inserted


def process_event(
    cur,
    sport: str,
    event: dict,
    unmatched_log: list | None = None,
    unknown_markets: set | None = None,
    stats: dict | None = None,
) -> int:
    """Resolve an event to a fixture and write its bulk-market odds rows.
    Returns rows written — inserted or price-updated — (0 if the event can't
    be matched). Kept as the single-call entry point used by
    backfill_historical_odds.py."""
    match_id, home, away = _match_event(cur, sport, event, unmatched_log=unmatched_log)
    if not match_id:
        return 0
    return insert_event_odds(cur, match_id, event, sport, home, away, unknown_markets=unknown_markets, stats=stats)


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
    # odds_rows_inserted / additional_rows_inserted count rows WRITTEN by each
    # path (a first insert or a price update — both change what the recs engine
    # will read). odds_rows_new / _updated / _unchanged are the run-wide split
    # across both paths, so a run where every price is already current shows
    # written=0 with unchanged>0 rather than looking like a dead fetch.
    # odds_rows_post_commence counts pre-match rows an in-play quote was NOT
    # allowed to overwrite; a number that climbs on the live path means the
    # feed is listing started matches, which is expected and harmless.
    results = {
        "events_seen": 0,
        "events_matched": 0,
        "odds_rows_inserted": 0,
        "additional_rows_inserted": 0,
        "odds_rows_new": 0,
        "odds_rows_updated": 0,
        "odds_rows_unchanged": 0,
        "odds_rows_post_commence": 0,
        "side_swap_rejections": 0,
    }
    stats: dict[str, int] = {
        ODDS_INSERTED: 0,
        ODDS_UPDATED: 0,
        ODDS_UNCHANGED: 0,
        ODDS_POST_COMMENCE: 0,
        "side_swap_rejections": 0,
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
                    inserted = insert_event_odds(
                        cur, match_id, event, sport, home, away, unknown_markets=unknown, stats=stats
                    )
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
                                        cur, match_id, payload, sport, home, away, unknown_markets=unknown, stats=stats
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

    results["odds_rows_new"] = stats[ODDS_INSERTED]
    results["odds_rows_updated"] = stats[ODDS_UPDATED]
    results["odds_rows_unchanged"] = stats[ODDS_UNCHANGED]
    results["odds_rows_post_commence"] = stats[ODDS_POST_COMMENCE]
    results["side_swap_rejections"] = stats["side_swap_rejections"]
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
        "Done. events_seen=%d events_matched=%d odds_rows_written=%d additional_rows_written=%d "
        "(new=%d price_updated=%d unchanged=%d post_commence_protected=%d) side_swap_rejections=%d",
        results["events_seen"],
        results["events_matched"],
        results["odds_rows_inserted"],
        results["additional_rows_inserted"],
        results["odds_rows_new"],
        results["odds_rows_updated"],
        results["odds_rows_unchanged"],
        results["odds_rows_post_commence"],
        results["side_swap_rejections"],
    )
    if results["side_swap_rejections"]:
        logger.error(
            "%d bookmaker payload(s) rejected whole by the side-swap canary this run — "
            "see the SIDE-SWAP CANARY lines above.",
            results["side_swap_rejections"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
