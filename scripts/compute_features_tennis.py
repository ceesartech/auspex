"""Compute baseline tennis features and write them to features_cache.

Tennis is the first 1v1 sport in the system — features differ from the
team sports in two key ways:

  * No "points scored" rolling form — tennis matches are binary win/
    lose at the match level. The rolling stat is win-rate over the
    last N matches, not points/game.
  * Head-to-head between this specific PAIR of players is a high-
    signal feature (tennis matchups are stylistically dependent —
    Nadal vs Djokovic vs Federer career H2H matters a lot more than
    a soccer H2H would). NFL/NBA/NHL have it in passing; tennis
    treats it as first-class.

Features written per match (all numeric, all pre-match — no leakage):

  Market / odds (devigged from the-odds-api):
    odds_home_ml, odds_away_ml
    implied_prob_home_ml, implied_prob_away_ml
    ml_bookie_margin

  Rolling form (last 10 matches per player, ATP + WTA combined):
    {home,away}_roll_wins      # count 0-10
    {home,away}_roll_matches   # count, < 10 for new players
    {home,away}_roll_win_pct   # wins / matches, 0.0-1.0

  Head-to-head (career, this exact pair):
    h2h_home_wins, h2h_away_wins, h2h_matches

  Schedule context:
    {home,away}_days_rest     # days since their last match
    {home,away}_active        # 1.0 if played in last 30 days, else 0.0

  Derived diffs (home minus away):
    roll_win_pct_diff
    h2h_balance               # home_wins - away_wins
    days_rest_diff
    odds_implied_diff         # implied_prob_home_ml - implied_prob_away_ml

Stored in features_cache.features as JSONB keyed on
(match_id, feature_set='tennis_baseline', feature_version='v1').

Note v1 scope: no surface (hard/clay/grass), no ranking, no service
stats. Those can layer in as v2 features once we have a data source.
The odds-implied probability already encodes most ranking signal
since the books factor surface and form into their lines.

Usage:
    python /app/scripts/compute_features_tennis.py                 # next 7 days
    python /app/scripts/compute_features_tennis.py --days 14
    python /app/scripts/compute_features_tennis.py --match-ids id1,id2
    python /app/scripts/compute_features_tennis.py --all-finished  # backfill
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("compute_features_tennis")

FEATURE_SET = "tennis_baseline"
FEATURE_VERSION = "v1"

# 10-match rolling window. ATP/WTA top players play ~70-90 matches/year
# so 10 matches covers ~6-8 weeks. Long enough to smooth out a single
# bad day; short enough to track recent form.
WINDOW = 10
CACHE_TTL_SECONDS = 3600

# Days since last match defining an "active" player. 30+ days idle is
# rare for tour players and often signals injury return — useful binary
# flag for the model.
ACTIVE_DAYS = 30.0


# Tennis tour-level averages (ATP + WTA combined, 2020-2025):
#   * Best-of-3 modal total games ≈ 22.5 (Grand Slam best-of-5 ≈ 37.5,
#     but most events are best-of-3).
#   * Average match implied probability for the favorite ≈ 0.65 (the
#     -200 / +160 favorite/dog split is the modal price).
#   * Most tour players play once every 4-7 days during tournaments,
#     once every 7-14 days between tournaments.
NEUTRAL_DEFAULTS: dict[str, float] = {
    # Moneyline (devigged, 65/35 modal favorite split)
    "odds_home_ml": 1.54,
    "odds_away_ml": 2.60,
    "implied_prob_home_ml": 0.65,
    "implied_prob_away_ml": 0.35,
    "ml_bookie_margin": 0.05,
    # Rolling form — neutral .500 win rate over partial window. Tour
    # players average ~50% win rate (top-100 averages a bit higher,
    # ranking floor at .500).
    "home_roll_wins": 5.0,
    "home_roll_matches": 10.0,
    "home_roll_win_pct": 0.50,
    "away_roll_wins": 5.0,
    "away_roll_matches": 10.0,
    "away_roll_win_pct": 0.50,
    # Head-to-head — defaults to "never played" (a common case
    # since the tour has 3000+ players). 0/0/0 is the explicit
    # "no prior data" signal — the model learns to weight this
    # alongside the no-prior-data H2H.
    "h2h_home_wins": 0.0,
    "h2h_away_wins": 0.0,
    "h2h_matches": 0.0,
    # Schedule context — assume both players coming off a typical
    # 7-day inter-match rest. active=1 means "played in last 30 days"
    # which is true for any tour regular.
    "home_days_rest": 7.0,
    "away_days_rest": 7.0,
    "home_active": 1.0,
    "away_active": 1.0,
    # Derived diffs default to zero (no advantage)
    "roll_win_pct_diff": 0.0,
    "h2h_balance": 0.0,
    "days_rest_diff": 0.0,
    "odds_implied_diff": 0.30,  # matches the 65/35 favorite/dog default
}


def _with_defaults(features: dict) -> dict:
    """Replace any None / non-numeric value with the neutral default for that
    key. Guarantees every model input is finite. Mirror of the same helper
    in compute_features_{nba,nfl,nhl}.py."""
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
    """Scheduled tennis matches that need feature computation (or, with
    --all-finished, every finished tennis match for training backfill).
    Mirror of the NFL version's selection logic."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if match_ids:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'tennis' AND m.id::text = ANY(%s)
                """,
                (match_ids,),
            )
            return [r["id"] for r in cur.fetchall()]

        if all_finished:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'tennis' AND m.status = 'finished'
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
                WHERE l.sport = 'tennis' AND m.status = 'scheduled'
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
            WHERE l.sport = 'tennis' AND m.status = 'scheduled'
              AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
              AND fc.id IS NULL
            ORDER BY m.match_date ASC
            """,
            (FEATURE_SET, FEATURE_VERSION, str(days)),
        )
        return [r["id"] for r in cur.fetchall()]


def fetch_match_meta(cur, match_id: str) -> Optional[dict]:
    """Pull the match's core columns needed for feature computation."""
    cur.execute(
        """
        SELECT m.id::text AS id, m.match_date, m.home_team_id, m.away_team_id
        FROM matches m
        WHERE m.id = %s
        """,
        (match_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def fetch_closing_odds(cur, match_id: str) -> dict:
    """Pull the most-recent pre-match odds row per (market_type, selection)
    and project into the feature shape. Tennis registers 'moneyline'
    and 'total' markets — we only need moneyline here since the v1
    feature set doesn't condition on total line (model handles ML only)."""
    cur.execute(
        """
        SELECT DISTINCT ON (market_type, selection)
               market_type, selection, line, odds_decimal
        FROM odds
        WHERE match_id = %s AND is_live = false
          AND market_type IN ('moneyline', 'total')
        ORDER BY market_type, selection, timestamp DESC NULLS LAST
        """,
        (match_id,),
    )
    out: dict[str, float] = {}
    for r in cur.fetchall():
        mt, sel, odds = r["market_type"], r["selection"], float(r["odds_decimal"])
        if mt == "moneyline":
            if sel == "home":
                out["odds_home_ml"] = odds
            elif sel == "away":
                out["odds_away_ml"] = odds
        # Total odds are captured by the precompute path; we don't
        # need them as features for the v1 moneyline-only model.

    # Devigged implied probabilities for moneyline. Same math as the
    # team-sport feature scripts.
    if "odds_home_ml" in out and "odds_away_ml" in out:
        raw_h = 1.0 / out["odds_home_ml"]
        raw_a = 1.0 / out["odds_away_ml"]
        margin = raw_h + raw_a - 1.0
        if raw_h + raw_a > 0:
            out["implied_prob_home_ml"] = raw_h / (raw_h + raw_a)
            out["implied_prob_away_ml"] = raw_a / (raw_h + raw_a)
            out["ml_bookie_margin"] = margin

    return out


def fetch_player_rolling(cur, player_id: str, before_date) -> dict:
    """Last-N-matches rolling form for one player. Reads finished
    tennis matches from the `matches` table — match-level win/lose only
    (no per-set or per-game stats in v1). Returns un-prefixed dict;
    caller prefixes with home_ / away_."""
    cur.execute(
        """
        WITH last_matches AS (
            SELECT m.id, m.match_date, m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            WHERE l.sport = 'tennis'
              AND m.status = 'finished'
              AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
              AND (m.home_team_id = %s OR m.away_team_id = %s)
              AND m.match_date < %s
            ORDER BY m.match_date DESC
            LIMIT %s
        )
        SELECT
            SUM(CASE
                    WHEN home_team_id = %s AND home_score > away_score THEN 1
                    WHEN away_team_id = %s AND away_score > home_score THEN 1
                    ELSE 0
                END) AS wins,
            COUNT(*) AS matches_played
        FROM last_matches
        """,
        (player_id, player_id, before_date, WINDOW, player_id, player_id),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row and row.get("matches_played", 0) > 0:
        wins = float(row["wins"]) if row["wins"] is not None else 0.0
        matches = float(row["matches_played"])
        out["roll_wins"] = wins
        out["roll_matches"] = matches
        out["roll_win_pct"] = wins / matches if matches > 0 else 0.5
    return out


def fetch_head_to_head(cur, home_id: str, away_id: str, before_date) -> dict:
    """Career head-to-head between this specific pair. Bidirectional —
    pulls matches where (home_id is home AND away_id is away) OR vice
    versa, counts wins from the perspective of the CURRENT match's
    home player. Tennis H2H is high-signal because matchups are
    stylistically dependent."""
    cur.execute(
        """
        SELECT
            SUM(CASE
                    WHEN home_team_id = %s AND home_score > away_score THEN 1
                    WHEN away_team_id = %s AND away_score > home_score THEN 1
                    ELSE 0
                END) AS home_wins,
            SUM(CASE
                    WHEN home_team_id = %s AND home_score > away_score THEN 1
                    WHEN away_team_id = %s AND away_score > home_score THEN 1
                    ELSE 0
                END) AS away_wins,
            COUNT(*) AS total_matches
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'tennis'
          AND m.status = 'finished'
          AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
          AND (
            (m.home_team_id = %s AND m.away_team_id = %s)
            OR (m.home_team_id = %s AND m.away_team_id = %s)
          )
          AND m.match_date < %s
        """,
        (
            home_id,
            home_id,  # home_wins counts
            away_id,
            away_id,  # away_wins counts
            home_id,
            away_id,  # match pair (home-away)
            away_id,
            home_id,  # match pair (away-home)
            before_date,
        ),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row:
        out["h2h_home_wins"] = float(row["home_wins"] or 0)
        out["h2h_away_wins"] = float(row["away_wins"] or 0)
        out["h2h_matches"] = float(row["total_matches"] or 0)
    return out


def fetch_player_schedule(cur, player_id: str, before_date) -> dict:
    """Days since this player's last match + active flag (played in
    last 30 days). Tennis matchplay is much more frequent than NFL
    week-to-week so the cadence matters less than the
    extended-layoff (injury-return) signal."""
    cur.execute(
        """
        SELECT m.match_date
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'tennis'
          AND m.status IN ('finished', 'live')
          AND (m.home_team_id = %s OR m.away_team_id = %s)
          AND m.match_date < %s
        ORDER BY m.match_date DESC
        LIMIT 1
        """,
        (player_id, player_id, before_date),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row:
        last = row["match_date"]
        delta_days = (before_date - last).total_seconds() / 86400.0
        out["days_rest"] = float(delta_days)
        out["active"] = 1.0 if delta_days <= ACTIVE_DAYS else 0.0
    return out


def _diff(features: dict, h_key: str, a_key: str, out_key: str) -> None:
    h, a = features.get(h_key), features.get(a_key)
    if isinstance(h, (int, float)) and isinstance(a, (int, float)):
        features[out_key] = float(h) - float(a)


def compute_for_match(cur, match_id: str) -> Optional[dict]:
    """End-to-end feature computation for one match. Returns None
    if the match itself can't be found."""
    meta = fetch_match_meta(cur, match_id)
    if not meta:
        return None
    when = meta["match_date"]

    features: dict = {}
    features.update(fetch_closing_odds(cur, match_id))
    for side, player_id in (("home", meta["home_team_id"]), ("away", meta["away_team_id"])):
        roll = fetch_player_rolling(cur, player_id, when)
        for k, v in roll.items():
            features[f"{side}_{k}"] = v
        sched = fetch_player_schedule(cur, player_id, when)
        for k, v in sched.items():
            features[f"{side}_{k}"] = v
    features.update(fetch_head_to_head(cur, meta["home_team_id"], meta["away_team_id"], when))

    _diff(features, "home_roll_win_pct", "away_roll_win_pct", "roll_win_pct_diff")
    _diff(features, "home_days_rest", "away_days_rest", "days_rest_diff")
    _diff(features, "implied_prob_home_ml", "implied_prob_away_ml", "odds_implied_diff")
    # H2H balance = home wins minus away wins (career, this pair).
    h, a = features.get("h2h_home_wins"), features.get("h2h_away_wins")
    if isinstance(h, (int, float)) and isinstance(a, (int, float)):
        features["h2h_balance"] = float(h) - float(a)

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
        help="Backfill features for every finished tennis match (one-shot training prep).",
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
            logger.info("No tennis matches need feature computation")
            return {"written": 0, "skipped": 0}
        logger.info("Computing features for %d tennis matches", len(targets))
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for mid in targets:
                features = compute_for_match(cur, mid)
                if features is None:
                    skipped += 1
                    continue
                write_features(cur, mid, features)
                written += 1
            conn.commit()
    logger.info("Wrote %d tennis feature rows (%d skipped)", written, skipped)
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
    raise SystemExit(main())
