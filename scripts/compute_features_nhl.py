"""Compute baseline NHL features and write them to features_cache.

Parallel pipeline to compute_features.py: same write path, separate
feature_set/version, NHL-specific source tables (nhl_match_stats) and
markets (moneyline, puck-line spread, totals at 5.5).

Features written per match (all numeric, all pre-match — no leakage):

  Market / odds (devigged from the-odds-api):
    odds_home_ml, odds_away_ml
    implied_prob_home_ml, implied_prob_away_ml
    ml_bookie_margin
    odds_home_pl15, odds_away_pl15
    implied_prob_home_cover_pl15, implied_prob_away_cover_pl15
    odds_over55, odds_under55, implied_prob_over55

  Rolling form (last 10 finished games for the team, home/away combined):
    {home,away}_roll_goals_for
    {home,away}_roll_goals_against
    {home,away}_roll_shots_for
    {home,away}_roll_shots_against
    {home,away}_roll_pp_pct        # SUM(PPG) / SUM(PPO)
    {home,away}_roll_pk_pct        # 1 - opp PP rate
    {home,away}_roll_save_pct
    {home,away}_roll_standings_pts # 2 win / 1 OT-SO loss / 0 reg loss

  Schedule context:
    {home,away}_days_rest
    {home,away}_back_to_back   # 1 if previous game was within ~1.5 days
    {home,away}_games_in_last_7

  Derived diffs (home minus away — model often learns these faster
  than the raw pair):
    form_diff_goals_for, form_diff_goals_against
    form_diff_shots_for, form_diff_shots_against
    form_diff_pp_pct, form_diff_pk_pct, form_diff_save_pct
    form_diff_standings_pts
    rest_diff

Stored in features_cache.features as JSONB keyed on
(match_id, feature_set='nhl_baseline', feature_version='v1').

Usage:
    python /app/scripts/compute_features_nhl.py                 # next 7 days
    python /app/scripts/compute_features_nhl.py --days 14
    python /app/scripts/compute_features_nhl.py --match-ids id1,id2,id3
    python /app/scripts/compute_features_nhl.py --all-finished  # backfill training-set features
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("compute_features_nhl")

FEATURE_SET = "nhl_baseline"
# v2 adds starting-goalie rolling stats (save%, GAA, days-rest, B2B
# flag) plus team pace stats (shots/goals per 60 minutes). Phase 3c
# training showed the v1 puck-line + total models couldn't beat naive
# baselines — odds correlate with WINNER but tell you almost nothing
# about MARGIN or TOTAL GOALS. Goalie identity + pace are the two
# biggest missing signals.
#
# Inference-time gap: for UPCOMING games we don't yet know who's
# starting in goal, so the goalie features fall to NEUTRAL_DEFAULTS
# at predict time. The benefit is currently training-only until we
# add projected-goalie ingestion (DailyFaceoff scrape or NHL roster
# API). Pace features work at both train and inference time because
# they're rolled from each team's prior games.
FEATURE_VERSION = "v2"
# Rolling window — NHL teams play ~3.5 games/week so 10 games is ~3 weeks
# of recent form, the standard "last 10" cadence NHL coverage uses.
WINDOW = 10
# How long a computed feature row stays valid before recompute.
CACHE_TTL_SECONDS = 3600
# Threshold for the back-to-back flag, in days. Games are often 24h
# apart but the API sends UTC timestamps and timezone variation pushes
# some pairs to ~26h — 1.5 days covers both without bleeding into
# 2-days-rest games.
BACK_TO_BACK_DAYS = 1.5
# Approximate game length in minutes used to normalize per-60 pace
# stats. The right-rail boxscore stores period_scores but not exact
# OT minutes; we use 60/65/65 as a reasonable approximation (real OT
# adds 0-5 min, SO adds ~0-2 min). Error is <2% on per-60 rates which
# is well within the noise of a 10-game rolling window.
GAME_LENGTH_MIN = {"REG": 60.0, "OT": 65.0, "SO": 65.0}

# NHL-tuned neutral defaults, rough league averages 2018-2024:
#   * Goals/game per team ≈ 3.05
#   * SOG/game per team ≈ 30
#   * PP% ≈ 0.20, PK% ≈ 0.80, save% ≈ 0.905
#   * Standings pts/game ≈ 1.10 (2 pts × win-rate ~0.50 + 1 pt × OT-loss ~0.11)
#   * Moneyline: home ~−170 (1.59 decimal), away ~+155 (2.55 decimal),
#     but devigging across the books recent average is closer to
#     1.82 / 2.10 — using those to keep implied probs reasonable.
#   * Totals at 5.5: roughly even-money each way → 1.95 / 1.95.
NEUTRAL_DEFAULTS: dict[str, float] = {
    # Moneyline
    "odds_home_ml": 1.82,
    "odds_away_ml": 2.10,
    "implied_prob_home_ml": 0.54,
    "implied_prob_away_ml": 0.46,
    "ml_bookie_margin": 0.05,
    # Puck line (always ±1.5)
    "odds_home_pl15": 2.30,
    "odds_away_pl15": 1.65,
    "implied_prob_home_cover_pl15": 0.42,
    "implied_prob_away_cover_pl15": 0.58,
    # Totals at 5.5 (canonical NHL line)
    "odds_over55": 1.95,
    "odds_under55": 1.95,
    "implied_prob_over55": 0.50,
    # Rolling team form
    "home_roll_goals_for": 3.05,
    "home_roll_goals_against": 3.05,
    "home_roll_shots_for": 30.0,
    "home_roll_shots_against": 30.0,
    "home_roll_pp_pct": 0.20,
    "home_roll_pk_pct": 0.80,
    "home_roll_save_pct": 0.905,
    "home_roll_standings_pts": 1.10,
    "away_roll_goals_for": 3.05,
    "away_roll_goals_against": 3.05,
    "away_roll_shots_for": 30.0,
    "away_roll_shots_against": 30.0,
    "away_roll_pp_pct": 0.20,
    "away_roll_pk_pct": 0.80,
    "away_roll_save_pct": 0.905,
    "away_roll_standings_pts": 1.10,
    # Schedule context — most NHL teams play every 2 days during the
    # regular season, so 2 is the modal days-rest value.
    "home_days_rest": 2.0,
    "away_days_rest": 2.0,
    "home_back_to_back": 0.0,
    "away_back_to_back": 0.0,
    "home_games_in_last_7": 3.0,
    "away_games_in_last_7": 3.0,
    # Derived diffs default to zero (no advantage on either side)
    "form_diff_goals_for": 0.0,
    "form_diff_goals_against": 0.0,
    "form_diff_shots_for": 0.0,
    "form_diff_shots_against": 0.0,
    "form_diff_pp_pct": 0.0,
    "form_diff_pk_pct": 0.0,
    "form_diff_save_pct": 0.0,
    "form_diff_standings_pts": 0.0,
    "rest_diff": 0.0,
    # ── Starting goalie rolling stats (v2) ────────────────────────
    # Save% league average ~0.905 (same as team-level). GAA ~2.75/60
    # for a league-average starter. Days-rest 3 is the modal start-
    # to-start gap for an NHL starter who plays ~60-65% of games. B2B
    # flag defaults to 0 (most starts aren't on no-rest).
    "home_goalie_roll_save_pct": 0.905,
    "away_goalie_roll_save_pct": 0.905,
    "home_goalie_roll_gaa": 2.75,
    "away_goalie_roll_gaa": 2.75,
    "home_goalie_days_rest": 3.0,
    "away_goalie_days_rest": 3.0,
    "home_goalie_back_to_back": 0.0,
    "away_goalie_back_to_back": 0.0,
    "goalie_save_pct_diff": 0.0,
    "goalie_gaa_diff": 0.0,
    # ── Pace stats (v2) ────────────────────────────────────────────
    # Per-60 normalization removes the OT-game inflation that biased
    # raw rolling averages. League averages: 30 shots/60 for, 30
    # against; 3.05 goals/60 each way.
    "home_shots_for_per_60": 30.0,
    "home_shots_against_per_60": 30.0,
    "home_goals_for_per_60": 3.05,
    "home_goals_against_per_60": 3.05,
    "away_shots_for_per_60": 30.0,
    "away_shots_against_per_60": 30.0,
    "away_goals_for_per_60": 3.05,
    "away_goals_against_per_60": 3.05,
    "pace_shots_diff": 0.0,
    "pace_goals_diff": 0.0,
}


def _with_defaults(features: dict) -> dict:
    """Replace any None / non-numeric value with the neutral default for that
    key. Guarantees every model input is finite (mirrors compute_features.py)."""
    out: dict = {}
    for k, default in NEUTRAL_DEFAULTS.items():
        v = features.get(k)
        out[k] = v if isinstance(v, (int, float)) and v is not None else default
    for k, v in features.items():
        if k not in out:
            out[k] = v
    return out


def list_target_matches(conn, days: int, force: bool = False) -> list[str]:
    """Scheduled NHL matches in the next N days that lack a fresh
    nhl_baseline features_cache row. With force=True, drops the cache-
    freshness filter so every match in the window is recomputed."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if force:
            cur.execute(
                """
                SELECT m.id::text AS id
                FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE m.status = 'scheduled'
                  AND l.sport = 'nhl'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY m.match_date ASC
                """,
                (str(days),),
            )
        else:
            cur.execute(
                """
                SELECT m.id::text AS id
                FROM matches m
                JOIN leagues l ON l.id = m.league_id
                LEFT JOIN features_cache f
                  ON f.match_id = m.id
                 AND f.feature_set = %s
                 AND f.feature_version = %s
                 AND f.expires_at > NOW()
                WHERE m.status = 'scheduled'
                  AND l.sport = 'nhl'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                  AND f.id IS NULL
                ORDER BY m.match_date ASC
                """,
                (FEATURE_SET, FEATURE_VERSION, str(days)),
            )
        return [r["id"] for r in cur.fetchall()]


def list_all_finished_matches(conn, force: bool = False) -> list[str]:
    """Every finished NHL match. Without force, skips matches that
    already have a fresh nhl_baseline features_cache row (1-hour TTL).
    With force=True, drops the cache-freshness filter — needed after a
    historical odds backfill, where the existing features_cache rows
    are still within their TTL but reflect the pre-backfill defaults
    rather than the new market lines. The ON CONFLICT clause in
    write_features upserts cleanly, so re-running with --force is safe.
    Ordered ASC so a crash mid-run leaves an early-seasons-first
    partial cache that's still useful for early training experiments.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if force:
            cur.execute(
                """
                SELECT m.id::text AS id
                FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE m.status = 'finished'
                  AND l.sport = 'nhl'
                  AND m.home_score IS NOT NULL
                  AND m.away_score IS NOT NULL
                ORDER BY m.match_date ASC
                """,
            )
            return [r["id"] for r in cur.fetchall()]
        cur.execute(
            """
            SELECT m.id::text AS id
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            LEFT JOIN features_cache f
              ON f.match_id = m.id
             AND f.feature_set = %s
             AND f.feature_version = %s
             AND f.expires_at > NOW()
            WHERE m.status = 'finished'
              AND l.sport = 'nhl'
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
              AND f.id IS NULL
            ORDER BY m.match_date ASC
            """,
            (FEATURE_SET, FEATURE_VERSION),
        )
        return [r["id"] for r in cur.fetchall()]


# ── Pure derivations: implied probabilities ──────────────────────────


def _devig_two_way(odds_a: float | None, odds_b: float | None) -> tuple[float | None, float | None, float | None]:
    """Devig a 2-outcome market (moneyline, puck-line cover, over/under).
    Returns (prob_a, prob_b, margin) or all-None on missing inputs."""
    if not (odds_a and odds_b):
        return None, None, None
    inv_a, inv_b = 1.0 / odds_a, 1.0 / odds_b
    total = inv_a + inv_b
    return inv_a / total, inv_b / total, total - 1.0


# ── DB lookups ───────────────────────────────────────────────────────


def _odds_avg(cur, match_id: str, market_type: str, selection: str, line=None):
    if line is None:
        cur.execute(
            """
            SELECT AVG(odds_decimal)::float AS o
            FROM odds
            WHERE match_id = %s AND market_type = %s AND selection = %s
              AND NOT is_live
            """,
            (match_id, market_type, selection),
        )
    else:
        cur.execute(
            """
            SELECT AVG(odds_decimal)::float AS o
            FROM odds
            WHERE match_id = %s AND market_type = %s AND selection = %s
              AND line = %s AND NOT is_live
            """,
            (match_id, market_type, selection, line),
        )
    row = cur.fetchone()
    return row["o"] if row else None


def _rolling_nhl_form(cur, team_id: str, before_date, side: str) -> dict:
    """Mean per-game stats over the team's last WINDOW finished games,
    joining nhl_match_stats for shots/PP/PK/saves. Special-teams rates
    are computed as SUM(num)/SUM(den) over the window, not AVG(per-game
    rate) — a 1-of-1 game shouldn't carry the same weight as 1-of-5.

    `side` ∈ {'home', 'away'} prefixes the output keys; the underlying
    rolling stats pull from both home and away games for the team."""
    cur.execute(
        """
        WITH team_games AS (
            SELECT
                m.id,
                m.match_date,
                CASE WHEN m.home_team_id = %(tid)s THEN m.home_score
                     ELSE m.away_score END AS goals_for,
                CASE WHEN m.home_team_id = %(tid)s THEN m.away_score
                     ELSE m.home_score END AS goals_against,
                s.shots_on_goal      AS shots_for,
                opp.shots_on_goal    AS shots_against,
                s.power_play_goals,
                s.power_play_opportunities,
                opp.power_play_goals AS opp_pp_goals,
                opp.power_play_opportunities AS opp_pp_opps,
                s.save_pct,
                COALESCE(m.metadata->>'period_type_final', 'REG') AS final_period
            FROM matches m
            JOIN nhl_match_stats s
              ON s.match_id = m.id AND s.team_id = %(tid)s
            JOIN nhl_match_stats opp
              ON opp.match_id = m.id AND opp.team_id <> %(tid)s
            WHERE (m.home_team_id = %(tid)s OR m.away_team_id = %(tid)s)
              AND m.status = 'finished'
              AND m.match_date < %(before)s
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT %(n)s
        )
        SELECT
            AVG(goals_for)::float                AS gf,
            AVG(goals_against)::float            AS ga,
            AVG(shots_for)::float                AS sf,
            AVG(shots_against)::float            AS sa,
            CASE WHEN SUM(power_play_opportunities) > 0
                 THEN SUM(power_play_goals)::float / SUM(power_play_opportunities)
                 ELSE NULL
            END                                  AS pp_pct,
            CASE WHEN SUM(opp_pp_opps) > 0
                 THEN 1.0 - (SUM(opp_pp_goals)::float / SUM(opp_pp_opps))
                 ELSE NULL
            END                                  AS pk_pct,
            AVG(save_pct)::float                 AS save_pct,
            AVG(
                CASE
                    WHEN goals_for > goals_against THEN 2.0
                    WHEN goals_for < goals_against
                         AND final_period IN ('OT', 'SO') THEN 1.0
                    ELSE 0.0
                END
            )::float                             AS standings_pts,
            COUNT(*)::int                        AS n
        FROM team_games
        """,
        {"tid": team_id, "before": before_date, "n": WINDOW},
    )
    row = cur.fetchone() or {}
    return {
        f"{side}_roll_goals_for": row.get("gf"),
        f"{side}_roll_goals_against": row.get("ga"),
        f"{side}_roll_shots_for": row.get("sf"),
        f"{side}_roll_shots_against": row.get("sa"),
        f"{side}_roll_pp_pct": row.get("pp_pct"),
        f"{side}_roll_pk_pct": row.get("pk_pct"),
        f"{side}_roll_save_pct": row.get("save_pct"),
        f"{side}_roll_standings_pts": row.get("standings_pts"),
    }


def _team_schedule_context(cur, team_id: str, before_date, side: str) -> dict:
    """Days-rest, back-to-back flag, and games-played-in-last-7-days
    against `before_date`. Returns Nones (filled by neutral defaults
    downstream) only if the team has no prior finished games."""
    cur.execute(
        """
        SELECT
            EXTRACT(EPOCH FROM (%(before)s - MAX(match_date))) / 86400.0 AS days_rest,
            COUNT(*) FILTER (
                WHERE match_date >= %(before)s - INTERVAL '7 days'
            )::int AS games_last_7
        FROM matches
        WHERE (home_team_id = %(tid)s OR away_team_id = %(tid)s)
          AND status = 'finished'
          AND match_date < %(before)s
        """,
        {"tid": team_id, "before": before_date},
    )
    row = cur.fetchone() or {}
    days_rest = row.get("days_rest")
    return {
        f"{side}_days_rest": float(days_rest) if days_rest is not None else None,
        f"{side}_back_to_back": (1.0 if days_rest is not None and float(days_rest) <= BACK_TO_BACK_DAYS else 0.0),
        f"{side}_games_in_last_7": int(row.get("games_last_7") or 0),
    }


def _rolling_pace_stats(cur, team_id: str, before_date, side: str) -> dict:
    """Per-60-minute rate stats over the team's last WINDOW finished
    games. Normalizes for OT/SO game-length inflation so totals models
    see clean rate signals instead of OT-inflated raw averages."""
    cur.execute(
        f"""
        WITH team_games AS (
            SELECT
                CASE WHEN m.home_team_id = %(tid)s THEN m.home_score
                     ELSE m.away_score END AS goals_for,
                CASE WHEN m.home_team_id = %(tid)s THEN m.away_score
                     ELSE m.home_score END AS goals_against,
                s.shots_on_goal AS shots_for,
                opp.shots_on_goal AS shots_against,
                CASE COALESCE(m.metadata->>'period_type_final', 'REG')
                    WHEN 'REG' THEN {GAME_LENGTH_MIN['REG']}
                    WHEN 'OT'  THEN {GAME_LENGTH_MIN['OT']}
                    WHEN 'SO'  THEN {GAME_LENGTH_MIN['SO']}
                    ELSE {GAME_LENGTH_MIN['REG']}
                END AS game_minutes
            FROM matches m
            JOIN nhl_match_stats s
              ON s.match_id = m.id AND s.team_id = %(tid)s
            JOIN nhl_match_stats opp
              ON opp.match_id = m.id AND opp.team_id <> %(tid)s
            WHERE (m.home_team_id = %(tid)s OR m.away_team_id = %(tid)s)
              AND m.status = 'finished'
              AND m.match_date < %(before)s
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT %(n)s
        )
        SELECT
            CASE WHEN SUM(game_minutes) > 0
                 THEN 60.0 * SUM(shots_for)::float / SUM(game_minutes)
                 ELSE NULL
            END AS shots_for_per_60,
            CASE WHEN SUM(game_minutes) > 0
                 THEN 60.0 * SUM(shots_against)::float / SUM(game_minutes)
                 ELSE NULL
            END AS shots_against_per_60,
            CASE WHEN SUM(game_minutes) > 0
                 THEN 60.0 * SUM(goals_for)::float / SUM(game_minutes)
                 ELSE NULL
            END AS goals_for_per_60,
            CASE WHEN SUM(game_minutes) > 0
                 THEN 60.0 * SUM(goals_against)::float / SUM(game_minutes)
                 ELSE NULL
            END AS goals_against_per_60
        FROM team_games
        """,
        {"tid": team_id, "before": before_date, "n": WINDOW},
    )
    row = cur.fetchone() or {}
    return {
        f"{side}_shots_for_per_60": row.get("shots_for_per_60"),
        f"{side}_shots_against_per_60": row.get("shots_against_per_60"),
        f"{side}_goals_for_per_60": row.get("goals_for_per_60"),
        f"{side}_goals_against_per_60": row.get("goals_against_per_60"),
    }


def _find_starting_goalie(cur, team_id: str, before_date) -> str | None:
    """Look up the team's starting goalie for the upcoming match. For
    HISTORICAL games (where this feature compute runs during the
    --all-finished sweep) the goalie is already set on the matching
    nhl_match_stats row, so we read it from there. For UPCOMING games
    there's no such row yet — returns None and the caller falls back
    to NEUTRAL_DEFAULTS for all goalie features. Closing that gap
    needs a projected-goalie ingestion path (e.g., DailyFaceoff scrape)."""
    cur.execute(
        """
        SELECT s.starting_goalie_id::text AS gid
        FROM nhl_match_stats s
        JOIN matches m ON m.id = s.match_id
        WHERE s.team_id = %s
          AND m.match_date >= %s::timestamptz - INTERVAL '6 hours'
          AND m.match_date <= %s::timestamptz + INTERVAL '6 hours'
        LIMIT 1
        """,
        (team_id, before_date, before_date),
    )
    row = cur.fetchone()
    return row["gid"] if row and row.get("gid") else None


def _goalie_rolling_form(cur, goalie_id: str | None, before_date, side: str) -> dict:
    """Per-goalie rolling stats over their last WINDOW starts: save%,
    GAA (per 60), days since last start, back-to-back flag. Returns
    Nones (filled with NEUTRAL_DEFAULTS downstream) when goalie_id is
    None — i.e., we don't know who'll start for the upcoming game."""
    if goalie_id is None:
        return {
            f"{side}_goalie_roll_save_pct": None,
            f"{side}_goalie_roll_gaa": None,
            f"{side}_goalie_days_rest": None,
            f"{side}_goalie_back_to_back": None,
        }

    cur.execute(
        f"""
        WITH goalie_starts AS (
            SELECT
                m.match_date,
                s.saves,
                opp.shots_on_goal AS shots_faced,
                s.goals_against,
                CASE COALESCE(m.metadata->>'period_type_final', 'REG')
                    WHEN 'REG' THEN {GAME_LENGTH_MIN['REG']}
                    WHEN 'OT'  THEN {GAME_LENGTH_MIN['OT']}
                    WHEN 'SO'  THEN {GAME_LENGTH_MIN['SO']}
                    ELSE {GAME_LENGTH_MIN['REG']}
                END AS game_minutes
            FROM nhl_match_stats s
            JOIN matches m ON m.id = s.match_id
            JOIN nhl_match_stats opp
              ON opp.match_id = s.match_id AND opp.team_id <> s.team_id
            WHERE s.starting_goalie_id = %(gid)s
              AND m.status = 'finished'
              AND m.match_date < %(before)s
            ORDER BY m.match_date DESC
            LIMIT %(n)s
        )
        SELECT
            CASE WHEN SUM(shots_faced) > 0
                 THEN SUM(saves)::float / SUM(shots_faced)
                 ELSE NULL
            END AS save_pct,
            CASE WHEN SUM(game_minutes) > 0
                 THEN 60.0 * SUM(goals_against)::float / SUM(game_minutes)
                 ELSE NULL
            END AS gaa,
            EXTRACT(EPOCH FROM (%(before)s - MAX(match_date))) / 86400.0 AS days_rest
        FROM goalie_starts
        """,
        {"gid": goalie_id, "before": before_date, "n": WINDOW},
    )
    row = cur.fetchone() or {}
    days_rest = row.get("days_rest")
    return {
        f"{side}_goalie_roll_save_pct": row.get("save_pct"),
        f"{side}_goalie_roll_gaa": row.get("gaa"),
        f"{side}_goalie_days_rest": float(days_rest) if days_rest is not None else None,
        f"{side}_goalie_back_to_back": (
            1.0 if days_rest is not None and float(days_rest) <= BACK_TO_BACK_DAYS else 0.0
        ),
    }


# ── Per-match feature assembly ───────────────────────────────────────


def compute_match_features(cur, match_id: str):
    """Return the feature dict for one NHL match, or None if the match
    has no team ids."""
    cur.execute(
        "SELECT home_team_id::text AS h, away_team_id::text AS a, match_date AS d " "FROM matches WHERE id = %s",
        (match_id,),
    )
    m = cur.fetchone()
    if not m or not (m["h"] and m["a"]):
        return None

    features: dict = {}

    # Moneyline (h2h)
    odds_h_ml = _odds_avg(cur, match_id, "moneyline", "home")
    odds_a_ml = _odds_avg(cur, match_id, "moneyline", "away")
    features["odds_home_ml"] = odds_h_ml
    features["odds_away_ml"] = odds_a_ml
    p_h, p_a, margin = _devig_two_way(odds_h_ml, odds_a_ml)
    features["implied_prob_home_ml"] = p_h
    features["implied_prob_away_ml"] = p_a
    features["ml_bookie_margin"] = margin

    # Puck line (±1.5)
    odds_h_pl = _odds_avg(cur, match_id, "spread", "home", -1.5)
    odds_a_pl = _odds_avg(cur, match_id, "spread", "away", 1.5)
    features["odds_home_pl15"] = odds_h_pl
    features["odds_away_pl15"] = odds_a_pl
    p_h_pl, p_a_pl, _ = _devig_two_way(odds_h_pl, odds_a_pl)
    features["implied_prob_home_cover_pl15"] = p_h_pl
    features["implied_prob_away_cover_pl15"] = p_a_pl

    # Totals at 5.5 (canonical NHL line)
    odds_over = _odds_avg(cur, match_id, "total", "over", 5.5)
    odds_under = _odds_avg(cur, match_id, "total", "under", 5.5)
    features["odds_over55"] = odds_over
    features["odds_under55"] = odds_under
    p_over, _, _ = _devig_two_way(odds_over, odds_under)
    features["implied_prob_over55"] = p_over

    # Rolling form per team
    home_form = _rolling_nhl_form(cur, m["h"], m["d"], "home")
    away_form = _rolling_nhl_form(cur, m["a"], m["d"], "away")
    features.update(home_form)
    features.update(away_form)

    # Schedule context per team
    features.update(_team_schedule_context(cur, m["h"], m["d"], "home"))
    features.update(_team_schedule_context(cur, m["a"], m["d"], "away"))

    # Per-60 pace stats per team (controls for OT/SO game-length inflation)
    home_pace = _rolling_pace_stats(cur, m["h"], m["d"], "home")
    away_pace = _rolling_pace_stats(cur, m["a"], m["d"], "away")
    features.update(home_pace)
    features.update(away_pace)

    # Starting goalie rolling stats. For historical games (current_match
    # is finished), the goalie row exists; for upcoming games it's
    # absent and goalie features default. Document a v2-of-v2 path:
    # add projected-goalie ingestion so prediction-time benefits too.
    home_goalie_id = _find_starting_goalie(cur, m["h"], m["d"])
    away_goalie_id = _find_starting_goalie(cur, m["a"], m["d"])
    home_goalie = _goalie_rolling_form(cur, home_goalie_id, m["d"], "home")
    away_goalie = _goalie_rolling_form(cur, away_goalie_id, m["d"], "away")
    features.update(home_goalie)
    features.update(away_goalie)

    # Derived diffs — pull from the raw rolling values BEFORE neutral
    # defaults are applied so a missing-on-both-sides stat doesn't get
    # a misleading zero diff. _safe_diff returns None on either-side
    # missing, which then gets the neutral 0.0 default downstream.
    def _safe_diff(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    features["form_diff_goals_for"] = _safe_diff(
        home_form.get("home_roll_goals_for"), away_form.get("away_roll_goals_for")
    )
    features["form_diff_goals_against"] = _safe_diff(
        home_form.get("home_roll_goals_against"), away_form.get("away_roll_goals_against")
    )
    features["form_diff_shots_for"] = _safe_diff(
        home_form.get("home_roll_shots_for"), away_form.get("away_roll_shots_for")
    )
    features["form_diff_shots_against"] = _safe_diff(
        home_form.get("home_roll_shots_against"), away_form.get("away_roll_shots_against")
    )
    features["form_diff_pp_pct"] = _safe_diff(home_form.get("home_roll_pp_pct"), away_form.get("away_roll_pp_pct"))
    features["form_diff_pk_pct"] = _safe_diff(home_form.get("home_roll_pk_pct"), away_form.get("away_roll_pk_pct"))
    features["form_diff_save_pct"] = _safe_diff(
        home_form.get("home_roll_save_pct"), away_form.get("away_roll_save_pct")
    )
    features["form_diff_standings_pts"] = _safe_diff(
        home_form.get("home_roll_standings_pts"), away_form.get("away_roll_standings_pts")
    )
    features["rest_diff"] = _safe_diff(features.get("home_days_rest"), features.get("away_days_rest"))

    # Goalie diffs (home minus away). Defaults fire when goalie unknown.
    features["goalie_save_pct_diff"] = _safe_diff(
        home_goalie.get("home_goalie_roll_save_pct"), away_goalie.get("away_goalie_roll_save_pct")
    )
    features["goalie_gaa_diff"] = _safe_diff(
        home_goalie.get("home_goalie_roll_gaa"), away_goalie.get("away_goalie_roll_gaa")
    )

    # Pace diffs — shots/60 and goals/60 (home minus away).
    features["pace_shots_diff"] = _safe_diff(
        home_pace.get("home_shots_for_per_60"), away_pace.get("away_shots_for_per_60")
    )
    features["pace_goals_diff"] = _safe_diff(
        home_pace.get("home_goals_for_per_60"), away_pace.get("away_goals_for_per_60")
    )

    return features


def write_features(cur, match_id: str, features: dict) -> None:
    cur.execute(
        """
        INSERT INTO features_cache
            (match_id, feature_set, features, feature_version,
             computed_at, expires_at)
        VALUES
            (%s, %s, %s::jsonb, %s, NOW(), NOW() + (%s || ' seconds')::interval)
        ON CONFLICT (match_id, feature_set, feature_version) DO UPDATE
            SET features = EXCLUDED.features,
                computed_at = NOW(),
                expires_at = EXCLUDED.expires_at
        """,
        (match_id, FEATURE_SET, json.dumps(features), FEATURE_VERSION, str(CACHE_TTL_SECONDS)),
    )


def compute_all(database_url: str, match_ids: list[str]) -> dict:
    ok = fail = 0
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for mid in match_ids:
                try:
                    features = compute_match_features(cur, mid)
                    if features is None:
                        fail += 1
                        continue
                    features = _with_defaults(features)
                    write_features(cur, mid, features)
                    ok += 1
                except Exception as e:
                    logger.warning("Failed for %s: %s", mid, e)
                    fail += 1
            conn.commit()
    return {"ok": ok, "fail": fail}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--match-ids", help="Comma-separated UUIDs (overrides --days / --all-finished).")
    p.add_argument(
        "--all-finished",
        action="store_true",
        help="Backfill features for every finished NHL match that doesn't already have a "
        "fresh nhl_baseline row. Use this once after a historical load to populate the "
        "training set; the daily DAG handles upcoming matches via --days.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore the cache-freshness filter and recompute features for every match in "
        "scope. Use this after a backfill_historical_odds run: the existing features_cache "
        "rows are still within their 1-hour TTL but reflect the pre-backfill default odds, "
        "so the default --all-finished would skip them and the model would re-train on the "
        "same stale features. The write path uses ON CONFLICT upsert so repeats are safe.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    force_note = " (force=on, ignoring cache TTL)" if args.force else ""

    if args.match_ids:
        match_ids = [s.strip() for s in args.match_ids.split(",") if s.strip()]
        scope_desc = f"{len(match_ids)} explicit match id(s){force_note}"
    elif args.all_finished:
        with psycopg2.connect(args.database_url) as conn:
            match_ids = list_all_finished_matches(conn, force=args.force)
        scope_desc = (
            f"{len(match_ids)} finished NHL match(es)" + force_note
            if args.force
            else f"{len(match_ids)} finished NHL match(es) without fresh features"
        )
    else:
        with psycopg2.connect(args.database_url) as conn:
            match_ids = list_target_matches(conn, args.days, force=args.force)
        scope_desc = f"{len(match_ids)} upcoming NHL match(es) in next {args.days} day(s){force_note}"

    if not match_ids:
        logger.info("No NHL matches to compute features for (%s)", scope_desc)
        return 0

    logger.info("Computing NHL features for %s...", scope_desc)
    counts = compute_all(args.database_url, match_ids)
    logger.info("Done: %d ok / %d failed", counts["ok"], counts["fail"])
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
