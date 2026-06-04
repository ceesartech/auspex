"""Compute baseline NBA features and write them to features_cache.

Parallel pipeline to compute_features.py (soccer) and
compute_features_nhl.py: same write path, separate feature_set/version,
NBA-specific source tables (matches + odds) and markets (moneyline,
spread at ANY line, total at ANY line).

Critical NBA-vs-NHL design choice: NHL puck_line is fixed at ±1.5 and
total at 5.5, so the features carry implied probabilities at those
canonical lines. NBA spread + total vary per game, so the features
carry the CLOSING LINE itself as a numeric input — the trained model
will consume it as a feature and emit a probability conditional on
that line. One trained model handles the full ladder of lines the
book offers (-3.5 to -13.5+ for spreads, 210 to 245+ for totals).

Features written per match (all numeric, all pre-match — no leakage):

  Market / odds (devigged from the-odds-api):
    odds_home_ml, odds_away_ml
    implied_prob_home_ml, implied_prob_away_ml
    ml_bookie_margin
    closing_spread_home          # the actual line, e.g. -7.5
    closing_total_line           # the actual line, e.g. 220.5
    odds_spread_home, odds_spread_away    # -110-ish, varies by book
    odds_total_over, odds_total_under

  Rolling form (last 10 finished games for the team, home/away combined):
    {home,away}_roll_pts_scored
    {home,away}_roll_pts_allowed
    {home,away}_roll_margin       # mean (pts_scored - pts_allowed)
    {home,away}_roll_wins         # count of wins (0-10)

  Schedule context:
    {home,away}_days_rest
    {home,away}_back_to_back   # 1 if previous game was within 1.5 days
    {home,away}_games_in_last_7

  Derived diffs (home minus away):
    pts_scored_diff, pts_allowed_diff, margin_diff, wins_diff, rest_diff

Stored in features_cache.features as JSONB keyed on
(match_id, feature_set='nba_baseline', feature_version='v1').

Usage:
    python /app/scripts/compute_features_nba.py                 # next 7 days
    python /app/scripts/compute_features_nba.py --days 14
    python /app/scripts/compute_features_nba.py --match-ids id1,id2
    python /app/scripts/compute_features_nba.py --all-finished  # backfill
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
logger = logging.getLogger("compute_features_nba")

FEATURE_SET = "nba_baseline"
# v1: scores + closing odds (moneyline / spread line / total line).
# A v2 will add advanced rate stats (pace, off/def rating, eFG%) once
# we ingest a per-game NBA stats table similar to nhl_match_stats.
FEATURE_VERSION = "v1"
# 10-game rolling window — NBA teams play ~3.5 games/week so 10 games
# is ~3 weeks of recent form, the standard "last 10" cadence NBA
# coverage uses.
WINDOW = 10
CACHE_TTL_SECONDS = 3600
# Back-to-back flag threshold in days. NBA back-to-backs are
# typically ~24h apart but UTC + timezone variation can push them to
# ~26h. 1.5 days catches both without bleeding into 2-days-rest games.
BACK_TO_BACK_DAYS = 1.5

# Modern NBA league averages (2020-2025):
#   * Points/game per team ≈ 113-115
#   * Margin/game ≈ 0 (league averages out)
#   * Pace ≈ 100 possessions
#   * .500 record over last 10 → 5 wins
#   * Days-rest: most NBA games are on 1-2 days rest; modal is 1
#   * Moneyline: average game is ~1.91 each side (close to pickem);
#     a typical favorite/dog game is ~1.55 / 2.50
#   * Spread: average line is 0 (no favorite); typical fav is -4 to -7
#   * Total: average is ~225, typical 215-235
NEUTRAL_DEFAULTS: dict[str, float] = {
    # Moneyline (devigged league averages)
    "odds_home_ml": 1.91,
    "odds_away_ml": 1.91,
    "implied_prob_home_ml": 0.52,  # home court advantage built-in
    "implied_prob_away_ml": 0.48,
    "ml_bookie_margin": 0.04,
    # Spread + total — set the LINE itself to neutral when missing.
    # closing_spread_home=0 means no line edge; closing_total_line=225
    # is the league-modal total.
    "closing_spread_home": 0.0,
    "closing_total_line": 225.0,
    "odds_spread_home": 1.91,
    "odds_spread_away": 1.91,
    "odds_total_over": 1.91,
    "odds_total_under": 1.91,
    # Rolling team form — league-average scoring + .500 record.
    "home_roll_pts_scored": 114.0,
    "home_roll_pts_allowed": 114.0,
    "home_roll_margin": 0.0,
    "home_roll_wins": 5.0,
    "away_roll_pts_scored": 114.0,
    "away_roll_pts_allowed": 114.0,
    "away_roll_margin": 0.0,
    "away_roll_wins": 5.0,
    # Schedule context — modal NBA days-rest is 1, ~12% of starts are
    # back-to-back, modal games-in-last-7 is 3.
    "home_days_rest": 1.0,
    "away_days_rest": 1.0,
    "home_back_to_back": 0.0,
    "away_back_to_back": 0.0,
    "home_games_in_last_7": 3.0,
    "away_games_in_last_7": 3.0,
    # Derived diffs default to zero (no advantage)
    "pts_scored_diff": 0.0,
    "pts_allowed_diff": 0.0,
    "margin_diff": 0.0,
    "wins_diff": 0.0,
    "rest_diff": 0.0,
    # Cross-book SPREAD features (landed 2026-06-04 after A/B in
    # scripts/ab_nba_cross_book.py — ΔBrier -0.0061 on 2024-2025
    # walk-forward). NBA moneyline + total did NOT benefit and
    # were NOT landed. Neutral defaults: 1 book, 0.0 mean (no
    # favourite), zero disagreement, 0.5 home-cover prob.
    "spread_book_count": 1.0,
    "spread_consensus_mean": 0.0,
    "spread_max_minus_min": 0.0,
    "spread_std": 0.0,
    "spread_consensus_implied_prob": 0.5,
}


def _with_defaults(features: dict) -> dict:
    """Replace any None / non-numeric value with the neutral default for that
    key. Guarantees every model input is finite. Mirror of the same helper
    in compute_features_nhl.py."""
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
    """Scheduled NBA matches that need feature computation (or, with
    --all-finished, every finished NBA match for training backfill).
    Mirror of the NHL version's selection logic."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if match_ids:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'nba' AND m.id::text = ANY(%s)
                """,
                (match_ids,),
            )
            return [r["id"] for r in cur.fetchall()]

        if all_finished:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'nba' AND m.status = 'finished'
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
                WHERE l.sport = 'nba' AND m.status = 'scheduled'
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
            WHERE l.sport = 'nba' AND m.status = 'scheduled'
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
    line) and project into the feature shape. NBA uses 'moneyline',
    'spread', 'total' with line as an INPUT FEATURE (not a fixed filter
    like NHL)."""
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

    # Devigged implied probabilities for moneyline. The raw 1/odds sums
    # to ~1.04-1.06 because of the book's vig; we strip that proportionally
    # so the two implied probs sum to exactly 1.0.
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
    """Last-10 games rolling form for one team. Reads finished NBA
    matches from the `matches` table — scores only, no per-possession
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
            WHERE l.sport = 'nba'
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
    """Days-rest, back-to-back flag, and games-in-last-7 for one team
    relative to the given match date."""
    cur.execute(
        """
        SELECT m.match_date
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'nba'
          AND m.status IN ('finished', 'live')
          AND (m.home_team_id = %s OR m.away_team_id = %s)
          AND m.match_date < %s
        ORDER BY m.match_date DESC
        LIMIT 7
        """,
        (team_id, team_id, before_date),
    )
    rows = cur.fetchall()
    out: dict[str, float] = {}
    if rows:
        last = rows[0]["match_date"]
        delta_days = (before_date - last).total_seconds() / 86400.0
        out["days_rest"] = float(delta_days)
        out["back_to_back"] = 1.0 if delta_days <= BACK_TO_BACK_DAYS else 0.0
        # Games in the 7 days prior to this match.
        seven_ago = before_date.timestamp() - 7 * 86400
        out["games_in_last_7"] = float(sum(1 for r in rows if r["match_date"].timestamp() >= seven_ago))
    return out


def _diff(features: dict, h_key: str, a_key: str, out_key: str) -> None:
    h, a = features.get(h_key), features.get(a_key)
    if isinstance(h, (int, float)) and isinstance(a, (int, float)):
        features[out_key] = float(h) - float(a)


def fetch_spread_crossbook(cur, match_id: str) -> dict:
    """Cross-book SPREAD features. Mirrors fetch_total_crossbook in
    compute_features_nfl.py — same SQL shape, same 5 keys (just
    with the 'spread' prefix).

    Validated by scripts/ab_nba_cross_book.py on 2026-06-04:
    ΔBrier -0.0061 on 2024-2025 walk-forward, clears the 0.005 KEEP
    threshold. NBA moneyline + total were tested in the same A/B
    and didn't help, so this fetcher is spread-only."""
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
                      AND od2.market_type = 'spread'
                      AND od2.selection = 'away'
                      AND od2.is_live = false
                    ORDER BY od2.timestamp DESC LIMIT 1
                ) AS c_odds
            FROM odds o
            WHERE o.match_id = %s
              AND o.market_type = 'spread'
              AND o.selection = 'home'
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
        raw_home = 1.0 / p_odds
        raw_away = 1.0 / c_odds
        denom = raw_home + raw_away
        if denom <= 0:
            continue
        devigged.append(raw_home / denom)

    out: dict = {
        "spread_book_count": float(len(rows)),
        "spread_consensus_mean": float(sum(lines) / len(lines)),
        "spread_max_minus_min": float(max(lines) - min(lines)),
    }
    if len(lines) > 1:
        mean = out["spread_consensus_mean"]
        out["spread_std"] = float(
            (sum((x - mean) ** 2 for x in lines) / len(lines)) ** 0.5
        )
    else:
        out["spread_std"] = 0.0
    if devigged:
        out["spread_consensus_implied_prob"] = float(sum(devigged) / len(devigged))
    return out


def compute_for_match(cur, match_id: str) -> Optional[dict]:
    """End-to-end feature computation for one match. Returns None
    if the match itself can't be found."""
    meta = fetch_match_meta(cur, match_id)
    if not meta:
        return None
    when = meta["match_date"]

    features: dict = {}
    # Odds
    features.update(fetch_closing_odds(cur, match_id))
    features.update(fetch_spread_crossbook(cur, match_id))
    # Per-team rolling form
    for side, team_id in (("home", meta["home_team_id"]), ("away", meta["away_team_id"])):
        roll = fetch_team_rolling(cur, team_id, when)
        for k, v in roll.items():
            features[f"{side}_{k}"] = v
        sched = fetch_schedule_context(cur, team_id, when)
        for k, v in sched.items():
            features[f"{side}_{k}"] = v
    # Diffs
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
        help="Backfill features for every finished NBA match (one-shot training prep).",
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
            logger.info("No NBA matches need feature computation")
            return {"written": 0, "skipped": 0}
        logger.info("Computing features for %d NBA matches", len(targets))
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for mid in targets:
                features = compute_for_match(cur, mid)
                if features is None:
                    skipped += 1
                    continue
                write_features(cur, mid, features)
                written += 1
            conn.commit()
    logger.info("Wrote %d NBA feature rows (%d skipped)", written, skipped)
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
