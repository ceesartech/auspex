"""Compute baseline NFL features and write them to features_cache.

Parallel pipeline to compute_features_nba.py — same line-as-feature
design (closing spread + closing total are INPUT features, not fixed
filters). NFL spread varies per game from -14.5 to +14.5; total
varies 38-55. One trained model per market handles the full ladder
of lines per game.

NFL-specific tuning vs NBA:
  * Rolling window of LAST 5 games, not last 10. NFL regular season
    is 17 games (vs NBA's 82), so a 10-game window covers ~60% of a
    season and ages quickly. 5 games is the standard "recent form"
    cadence NFL coverage uses.
  * Days-rest defaults to 6 (modal NFL Sun-to-Sun cadence). Short-
    week (Thu) games at 3 days rest. Bye-week games at 14 days
    rest. Schedule context is more impactful than NBA because the
    week-to-week structure is fixed.
  * Modern NFL points-per-game ≈ 22-24 per team (≈ 45-47 total).
  * Modern NFL spread skew: home teams favored ~57% of games
    (slightly higher than NBA's 55% due to crowd noise + travel).

Features written per match (all numeric, all pre-match — no leakage):

  Market / odds (devigged from the-odds-api):
    odds_home_ml, odds_away_ml
    implied_prob_home_ml, implied_prob_away_ml
    ml_bookie_margin
    closing_spread_home          # the actual line, e.g. -7.0
    closing_total_line           # the actual line, e.g. 47.5
    odds_spread_home, odds_spread_away    # typically -110 ≈ 1.91
    odds_total_over, odds_total_under

  Rolling form (last 5 games per team, home + away combined):
    {home,away}_roll_pts_scored
    {home,away}_roll_pts_allowed
    {home,away}_roll_margin       # mean (pts_scored - pts_allowed)
    {home,away}_roll_wins         # count of wins (0-5)

  Schedule context:
    {home,away}_days_rest
    {home,away}_short_week        # 1 if days_rest <= 4 (Thu game)
    {home,away}_long_week         # 1 if days_rest >= 10 (post-bye)

  Derived diffs (home minus away):
    pts_scored_diff, pts_allowed_diff, margin_diff, wins_diff,
    rest_diff

Stored in features_cache.features as JSONB keyed on
(match_id, feature_set='nfl_baseline', feature_version='v1').

Usage:
    python /app/scripts/compute_features_nfl.py                 # next 7 days
    python /app/scripts/compute_features_nfl.py --days 14
    python /app/scripts/compute_features_nfl.py --match-ids id1,id2
    python /app/scripts/compute_features_nfl.py --all-finished  # backfill
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("compute_features_nfl")

FEATURE_SET = "nfl_baseline"
FEATURE_VERSION = "v1"
# 5-game rolling window — NFL teams play 1 game/week so 5 games is
# ~5 weeks of recent form, the standard "last 5" cadence NFL coverage
# uses. Shorter than NBA's 10 because NFL has fewer games per season.
WINDOW = 5
CACHE_TTL_SECONDS = 3600
# Short-week (Thursday after Sunday) → ~3-4 days rest.
SHORT_WEEK_DAYS = 4.0
# Long-week (bye week or international slot) → ~10+ days rest.
LONG_WEEK_DAYS = 10.0

# Modern NFL league averages (2020-2025):
#   * Points/game per team ≈ 22.5 (sum ≈ 45)
#   * Margin/game ≈ 0 (league average)
#   * Days-rest: 7 modal (Sun-to-Sun), 3 short week, 13 bye
#   * Home win rate ≈ 57% (slightly higher than NBA)
#   * Moneyline modal odds ~1.50/2.50 (home -200/+150 average)
#   * Spread typical 3-7 points; total typical 42-50
NEUTRAL_DEFAULTS: dict[str, float] = {
    # Moneyline (devigged league averages)
    "odds_home_ml": 1.83,
    "odds_away_ml": 2.10,
    "implied_prob_home_ml": 0.57,  # home advantage built-in
    "implied_prob_away_ml": 0.43,
    "ml_bookie_margin": 0.05,
    # Spread + total — neutral line when missing. closing_spread_home=0
    # means no favorite; closing_total_line=45 is the modern modal.
    "closing_spread_home": 0.0,
    "closing_total_line": 45.0,
    "odds_spread_home": 1.91,
    "odds_spread_away": 1.91,
    "odds_total_over": 1.91,
    "odds_total_under": 1.91,
    # Rolling team form — league-average scoring + .500 record.
    "home_roll_pts_scored": 22.5,
    "home_roll_pts_allowed": 22.5,
    "home_roll_margin": 0.0,
    "home_roll_wins": 2.5,
    "away_roll_pts_scored": 22.5,
    "away_roll_pts_allowed": 22.5,
    "away_roll_margin": 0.0,
    "away_roll_wins": 2.5,
    # Schedule context — modal 7-day cadence.
    "home_days_rest": 7.0,
    "away_days_rest": 7.0,
    "home_short_week": 0.0,
    "away_short_week": 0.0,
    "home_long_week": 0.0,
    "away_long_week": 0.0,
    # Derived diffs default to zero (no advantage)
    "pts_scored_diff": 0.0,
    "pts_allowed_diff": 0.0,
    "margin_diff": 0.0,
    "wins_diff": 0.0,
    "rest_diff": 0.0,
    # Cross-book TOTAL features (landed 2026-06-03 after A/B in
    # scripts/ab_nfl_cross_book.py — ΔBrier -0.0110, ΔECE -0.0189
    # on 2024-2025 walk-forward). Captures market consensus +
    # book-disagreement signal that `closing_total_line` alone misses.
    # SPREAD did NOT benefit from the equivalent features and was NOT
    # landed. Neutral defaults: 1 book (degenerate market), 45 mean
    # (modal NFL total), zero disagreement, 0.5 over-prob.
    "total_book_count": 1.0,
    "total_consensus_mean": 45.0,
    "total_max_minus_min": 0.0,
    "total_std": 0.0,
    "total_consensus_implied_prob": 0.5,
    # Cross-book MONEYLINE features (landed 2026-06-04 after A/B in
    # scripts/ab_nfl_moneyline_cross_book.py — ΔBrier -0.0154 (3x
    # KEEP threshold), ΔECE -0.0366, +2.69pts accuracy on 2024-2025
    # walk-forward). Same NFL TOTAL pattern but for the 2-way ML
    # market. Neutral defaults: 1 book, 0.57 home prob (the modern
    # home-advantage modal — matches implied_prob_home_ml default),
    # zero disagreement.
    "ml_book_count": 1.0,
    "ml_consensus_home_prob": 0.57,
    "ml_max_minus_min_home_prob": 0.0,
    "ml_std_home_prob": 0.0,
}


def _with_defaults(features: dict) -> dict:
    """Replace any None / non-numeric value with the neutral default for that
    key. Guarantees every model input is finite. Mirror of the same helper
    in compute_features_nba.py / compute_features_nhl.py."""
    out: dict = {}
    for k, default in NEUTRAL_DEFAULTS.items():
        v = features.get(k)
        out[k] = v if isinstance(v, (int, float)) and v is not None else default
    for k, v in features.items():
        if k not in out:
            out[k] = v
    return out


# ── DB I/O ────────────────────────────────────────────────────────────


def list_target_matches(conn, days: int, force: bool, all_finished: bool, match_ids: Optional[list[str]]) -> list[str]:
    """Scheduled NFL matches that need feature computation (or, with
    --all-finished, every finished NFL match for training backfill).
    Mirror of the NBA version's selection logic."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if match_ids:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'nfl' AND m.id::text = ANY(%s)
                """,
                (match_ids,),
            )
            return [r["id"] for r in cur.fetchall()]

        if all_finished:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'nfl' AND m.status = 'finished'
                  AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
                ORDER BY m.match_date ASC
                """
            )
            return [r["id"] for r in cur.fetchall()]

        if force:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'nfl' AND m.status = 'scheduled'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY m.match_date ASC
                """,
                (str(days),),
            )
            return [r["id"] for r in cur.fetchall()]

        # Default: scheduled matches in window without fresh cache row.
        cur.execute(
            """
            SELECT m.id::text AS id FROM matches m
            JOIN leagues l ON l.id = m.league_id
            LEFT JOIN features_cache fc
              ON fc.match_id = m.id
             AND fc.feature_set = %s
             AND fc.feature_version = %s
             AND fc.expires_at > NOW()
            WHERE l.sport = 'nfl' AND m.status = 'scheduled'
              AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
              AND fc.match_id IS NULL
            ORDER BY m.match_date ASC
            """,
            (FEATURE_SET, FEATURE_VERSION, str(days)),
        )
        return [r["id"] for r in cur.fetchall()]


def fetch_match_meta(cur, match_id: str) -> Optional[dict]:
    cur.execute(
        """
        SELECT m.id::text AS id, m.match_date,
               m.home_team_id::text AS home_team_id,
               m.away_team_id::text AS away_team_id
        FROM matches m
        WHERE m.id = %s
        """,
        (match_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def fetch_closing_odds(cur, match_id: str) -> dict:
    """Pull the most-recent pre-match odds row per (market_type, selection,
    line) and project into the feature shape. NFL uses 'moneyline',
    'spread', 'total' with line as an INPUT FEATURE."""
    cur.execute(
        """
        SELECT DISTINCT ON (market_type, selection)
               market_type, selection, line, odds_decimal
        FROM odds
        WHERE match_id = %s AND is_live = false
          AND market_type IN ('moneyline', 'spread', 'total')
        ORDER BY market_type, selection, timestamp DESC NULLS LAST
        """,
        (match_id,),
    )
    out: dict[str, float] = {}
    for r in cur.fetchall():
        mt, sel, line, odds = r["market_type"], r["selection"], r["line"], float(r["odds_decimal"])
        if mt == "moneyline":
            if sel == "home":
                out["odds_home_ml"] = odds
            elif sel == "away":
                out["odds_away_ml"] = odds
        elif mt == "spread":
            if sel == "home":
                out["odds_spread_home"] = odds
                out["closing_spread_home"] = float(line) if line is not None else None
            elif sel == "away":
                out["odds_spread_away"] = odds
        elif mt == "total":
            if sel == "over":
                out["odds_total_over"] = odds
                out["closing_total_line"] = float(line) if line is not None else None
            elif sel == "under":
                out["odds_total_under"] = odds

    # Devigged implied probabilities for moneyline.
    if "odds_home_ml" in out and "odds_away_ml" in out:
        raw_h = 1.0 / out["odds_home_ml"]
        raw_a = 1.0 / out["odds_away_ml"]
        margin = raw_h + raw_a - 1.0
        if raw_h + raw_a > 0:
            out["implied_prob_home_ml"] = raw_h / (raw_h + raw_a)
            out["implied_prob_away_ml"] = raw_a / (raw_h + raw_a)
            out["ml_bookie_margin"] = margin

    return out


def fetch_team_rolling(cur, team_id: str, before_date) -> dict:
    """Last-5-games rolling form for one team. Reads finished NFL
    matches from the `matches` table — scores only, no per-drive
    advanced stats (those come in v2). Returns the bare numeric dict
    keyed by 'roll_pts_scored', etc., un-prefixed; caller prefixes
    with home_ / away_."""
    cur.execute(
        """
        WITH last_games AS (
            SELECT m.id, m.match_date, m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            WHERE l.sport = 'nfl'
              AND m.status = 'finished'
              AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
              AND (m.home_team_id = %s OR m.away_team_id = %s)
              AND m.match_date < %s
            ORDER BY m.match_date DESC
            LIMIT %s
        )
        SELECT
            AVG(CASE WHEN home_team_id = %s THEN home_score ELSE away_score END) AS pts_scored,
            AVG(CASE WHEN home_team_id = %s THEN away_score ELSE home_score END) AS pts_allowed,
            AVG(CASE WHEN home_team_id = %s THEN (home_score - away_score)
                     ELSE (away_score - home_score) END) AS margin,
            SUM(CASE
                    WHEN home_team_id = %s AND home_score > away_score THEN 1
                    WHEN away_team_id = %s AND away_score > home_score THEN 1
                    ELSE 0
                END) AS wins,
            COUNT(*) AS games_played
        FROM last_games
        """,
        (team_id, team_id, before_date, WINDOW, team_id, team_id, team_id, team_id, team_id),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row and row.get("games_played", 0) > 0:
        out["roll_pts_scored"] = float(row["pts_scored"]) if row["pts_scored"] is not None else None
        out["roll_pts_allowed"] = float(row["pts_allowed"]) if row["pts_allowed"] is not None else None
        out["roll_margin"] = float(row["margin"]) if row["margin"] is not None else None
        out["roll_wins"] = float(row["wins"]) if row["wins"] is not None else None
    return out


def fetch_schedule_context(cur, team_id: str, before_date) -> dict:
    """Days-rest + short-week / long-week flags for one team relative
    to the given match date. NFL's weekly cadence makes these high-
    impact features (vs NBA where back-to-back is the only schedule
    flag that consistently moves the line)."""
    cur.execute(
        """
        SELECT m.match_date
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'nfl'
          AND m.status IN ('finished', 'live')
          AND (m.home_team_id = %s OR m.away_team_id = %s)
          AND m.match_date < %s
        ORDER BY m.match_date DESC
        LIMIT 1
        """,
        (team_id, team_id, before_date),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row:
        last = row["match_date"]
        delta_days = (before_date - last).total_seconds() / 86400.0
        out["days_rest"] = float(delta_days)
        out["short_week"] = 1.0 if delta_days <= SHORT_WEEK_DAYS else 0.0
        out["long_week"] = 1.0 if delta_days >= LONG_WEEK_DAYS else 0.0
    return out


def _diff(features: dict, h_key: str, a_key: str, out_key: str) -> None:
    h, a = features.get(h_key), features.get(a_key)
    if isinstance(h, (int, float)) and isinstance(a, (int, float)):
        features[out_key] = float(h) - float(a)


# Note: weather feature integration was attempted three times for
# NFL (v3 at commit 61d1d84 with NEUTRAL_DEFAULTS, v4a at 5e699f5 /
# a72a3b4 with EXISTS-gate + OMIT-on-missing, v4b at 01d9c69 with
# always-emit + NaN-safe). All three underperformed a v1-equivalent
# control (no weather features) on a clean apples-to-apples
# walk-forward — v4b ensemble at 62.7% vs control 65.2% on the
# same 365-match 2024 test set. Consistent ~2.5pt drag across base
# learners; the signal isn't there at the corpus size we have
# (~620 pre-2024 rows + Open-Meteo 11km grid resolution). Removed
# on 2026-06-03; weather infrastructure (migration 012, fetch_weather,
# seed_venue_coords, geocode_soccer_venues) left in place for any
# future retry with a higher-resolution paid API. See
# `weather-features-attempted` memory for the full history + the
# correct ~65% NFL ML baseline (the 71.3% in earlier records does
# not reproduce on the current methodology).


def fetch_total_crossbook(cur, match_id: str) -> dict:
    """Cross-book TOTAL features. Reads every (book, over) odds row for
    this match, computes consensus + disagreement statistics.

    Validated by scripts/ab_nfl_cross_book.py on 2026-06-03: walk-forward
    on 2024-2025 showed ΔBrier -0.0110 (vs 0.005 threshold) and ΔECE
    -0.0189 — clean KEEP signal. Spread cross-book features were tested
    in the same A/B and did NOT help, so this fetcher is total-only.

    All 5 keys are emitted unconditionally (None → default at the end of
    compute_for_match) so the model input shape is stable whether or not
    the cross-book data is present for this match."""
    cur.execute(
        """
        WITH per_book AS (
            SELECT DISTINCT ON (o.bookmaker)
                o.bookmaker,
                o.line AS p_line,
                o.odds_decimal AS p_odds,
                (
                    SELECT od2.odds_decimal FROM odds od2
                    WHERE od2.match_id = o.match_id
                      AND od2.bookmaker = o.bookmaker
                      AND od2.market_type = 'total'
                      AND od2.selection = 'under'
                      AND od2.is_live = false
                    ORDER BY od2.timestamp DESC LIMIT 1
                ) AS c_odds
            FROM odds o
            WHERE o.match_id = %s
              AND o.market_type = 'total'
              AND o.selection = 'over'
              AND o.is_live = false
              AND o.line IS NOT NULL
              AND o.odds_decimal IS NOT NULL
            ORDER BY o.bookmaker, o.timestamp DESC
        )
        SELECT bookmaker, p_line, p_odds, c_odds FROM per_book
        """,
        (match_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return {}

    lines = [float(r["p_line"]) for r in rows]
    devigged = []
    for r in rows:
        p_odds = float(r["p_odds"])
        c_odds = r["c_odds"]
        if p_odds <= 0 or c_odds is None:
            continue
        c_odds = float(c_odds)
        if c_odds <= 0:
            continue
        raw_over = 1.0 / p_odds
        raw_under = 1.0 / c_odds
        denom = raw_over + raw_under
        if denom <= 0:
            continue
        devigged.append(raw_over / denom)

    out: dict = {
        "total_book_count": float(len(rows)),
        "total_consensus_mean": float(sum(lines) / len(lines)),
        "total_max_minus_min": float(max(lines) - min(lines)),
    }
    # Population std (ddof=0) for consistency with the A/B harness.
    if len(lines) > 1:
        mean = out["total_consensus_mean"]
        out["total_std"] = float((sum((x - mean) ** 2 for x in lines) / len(lines)) ** 0.5)
    else:
        out["total_std"] = 0.0
    if devigged:
        out["total_consensus_implied_prob"] = float(sum(devigged) / len(devigged))
    return out


def fetch_moneyline_crossbook(cur, match_id: str) -> dict:
    """Cross-book MONEYLINE features. Returns one row per match
    aggregating across every (book, home) odds row paired with the
    same book's (away) odds for devigged home prob.

    Validated by scripts/ab_nfl_moneyline_cross_book.py on 2026-06-04:
    ΔBrier -0.0154 (3x the 0.005 KEEP threshold), ΔECE -0.0366,
    +2.69pts accuracy on 2024-2025 walk-forward. Same shape as
    fetch_total_crossbook — emits 4 keys unconditionally so the
    model input shape is stable whether or not cross-book data is
    present."""
    cur.execute(
        """
        WITH per_book AS (
            SELECT DISTINCT ON (o.bookmaker)
                o.bookmaker,
                o.odds_decimal AS home_odds,
                (
                    SELECT od2.odds_decimal FROM odds od2
                    WHERE od2.match_id = o.match_id
                      AND od2.bookmaker = o.bookmaker
                      AND od2.market_type = 'moneyline'
                      AND od2.selection = 'away'
                      AND od2.is_live = false
                    ORDER BY od2.timestamp DESC LIMIT 1
                ) AS away_odds
            FROM odds o
            WHERE o.match_id = %s
              AND o.market_type = 'moneyline'
              AND o.selection = 'home'
              AND o.is_live = false
              AND o.odds_decimal IS NOT NULL
            ORDER BY o.bookmaker, o.timestamp DESC
        )
        SELECT bookmaker, home_odds, away_odds FROM per_book
        """,
        (match_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return {}

    home_probs: list[float] = []
    for r in rows:
        h_odds = float(r["home_odds"])
        a_odds = r["away_odds"]
        if a_odds is None or h_odds <= 0:
            continue
        a_odds = float(a_odds)
        if a_odds <= 0:
            continue
        raw_h = 1.0 / h_odds
        raw_a = 1.0 / a_odds
        denom = raw_h + raw_a
        if denom <= 0:
            continue
        home_probs.append(raw_h / denom)

    if not home_probs:
        return {}

    out: dict = {
        "ml_book_count": float(len(home_probs)),
        "ml_consensus_home_prob": float(sum(home_probs) / len(home_probs)),
        "ml_max_minus_min_home_prob": float(max(home_probs) - min(home_probs)),
    }
    if len(home_probs) > 1:
        mean = out["ml_consensus_home_prob"]
        out["ml_std_home_prob"] = float(
            (sum((p - mean) ** 2 for p in home_probs) / len(home_probs)) ** 0.5
        )
    else:
        out["ml_std_home_prob"] = 0.0
    return out


def compute_for_match(cur, match_id: str) -> Optional[dict]:
    """End-to-end feature computation for one match. Returns None
    if the match itself can't be found."""
    meta = fetch_match_meta(cur, match_id)
    if not meta:
        return None
    when = meta["match_date"]

    features: dict = {}
    features.update(fetch_closing_odds(cur, match_id))
    features.update(fetch_total_crossbook(cur, match_id))
    features.update(fetch_moneyline_crossbook(cur, match_id))
    for side, team_id in (("home", meta["home_team_id"]), ("away", meta["away_team_id"])):
        roll = fetch_team_rolling(cur, team_id, when)
        for k, v in roll.items():
            features[f"{side}_{k}"] = v
        sched = fetch_schedule_context(cur, team_id, when)
        for k, v in sched.items():
            features[f"{side}_{k}"] = v
    _diff(features, "home_roll_pts_scored", "away_roll_pts_scored", "pts_scored_diff")
    _diff(features, "home_roll_pts_allowed", "away_roll_pts_allowed", "pts_allowed_diff")
    _diff(features, "home_roll_margin", "away_roll_margin", "margin_diff")
    _diff(features, "home_roll_wins", "away_roll_wins", "wins_diff")
    _diff(features, "home_days_rest", "away_days_rest", "rest_diff")
    return _with_defaults(features)


def write_features(cur, match_id: str, features: dict) -> None:
    cur.execute(
        """
        INSERT INTO features_cache
            (match_id, feature_set, feature_version, features, expires_at, computed_at)
        VALUES (%s, %s, %s, %s::jsonb, NOW() + (%s || ' seconds')::interval, NOW())
        ON CONFLICT (match_id, feature_set, feature_version)
        DO UPDATE SET features = EXCLUDED.features,
                      expires_at = EXCLUDED.expires_at,
                      computed_at = NOW()
        """,
        (match_id, FEATURE_SET, FEATURE_VERSION, json.dumps(features), str(CACHE_TTL_SECONDS)),
    )


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7, help="Lookahead window in days (default 7).")
    p.add_argument("--force", action="store_true", help="Recompute even if cache row is fresh.")
    p.add_argument(
        "--all-finished",
        action="store_true",
        help="Backfill features for every finished NFL match (one-shot training prep).",
    )
    p.add_argument("--match-ids", help="Comma-separated UUID list to recompute specific matches.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def run(database_url: str, days: int, force: bool, all_finished: bool, match_ids: Optional[list[str]]) -> dict:
    written = 0
    skipped = 0
    with psycopg2.connect(database_url) as conn:
        targets = list_target_matches(conn, days, force, all_finished, match_ids)
        if not targets:
            logger.info("No NFL matches need feature computation")
            return {"written": 0, "skipped": 0}
        logger.info("Computing features for %d NFL matches", len(targets))
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for mid in targets:
                features = compute_for_match(cur, mid)
                if features is None:
                    skipped += 1
                    continue
                write_features(cur, mid, features)
                written += 1
            conn.commit()
    logger.info("Wrote %d NFL feature rows (%d skipped)", written, skipped)
    return {"written": written, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    match_ids = [s.strip() for s in args.match_ids.split(",") if s.strip()] if args.match_ids else None
    run(args.database_url, args.days, args.force, args.all_finished, match_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
