"""Compute baseline MMA features and write them to features_cache.

MMA shares the 1v1 / winner-boolean / single-moneyline-market design
with tennis. Feature shape is essentially the same with MMA-specific
defaults:

  * Rolling window of 5 fights (UFC fighters fight ~3-4 times/year,
    so 5 fights covers ~14-18 months of recent form). Shorter than
    tennis (10 matches) because MMA fight frequency is much lower.
  * Active flag at 365 days — MMA fighters routinely go a full year
    between fights (injury, weight cuts, contract disputes). Tennis
    uses 30 days because tour cadence is much tighter.
  * Modal favorite implied probability ≈ 0.62 (slightly less heavy
    than tennis since MMA upsets are more frequent — striker vs
    grappler matchups can flip the script).

Features written per fight (all numeric, all pre-fight — no leakage):

  Market / odds (devigged from the-odds-api):
    odds_home_ml, odds_away_ml
    implied_prob_home_ml, implied_prob_away_ml
    ml_bookie_margin

  Rolling form (last 5 fights per fighter):
    {home,away}_roll_wins
    {home,away}_roll_matches
    {home,away}_roll_win_pct

  Head-to-head (career, this exact pair):
    h2h_home_wins, h2h_away_wins, h2h_matches

  Schedule context:
    {home,away}_days_rest
    {home,away}_active        # 1.0 if fought in last 365 days, else 0.0

  Derived diffs (home minus away):
    roll_win_pct_diff
    h2h_balance
    days_rest_diff
    odds_implied_diff

Stored in features_cache.features as JSONB keyed on
(match_id, feature_set='mma_baseline', feature_version='v1').

v1 scope: no weight class, striker vs grappler classification, reach,
age, southpaw stance. Those layer in as v2 features once we have
fighter-profile data (UFCStats.com scrape is the obvious source).

Usage:
    python /app/scripts/compute_features_mma.py                 # next 7 days
    python /app/scripts/compute_features_mma.py --days 14
    python /app/scripts/compute_features_mma.py --match-ids id1,id2
    python /app/scripts/compute_features_mma.py --all-finished  # backfill
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
logger = logging.getLogger("compute_features_mma")

FEATURE_SET = "mma_baseline"
FEATURE_VERSION = "v1"

# 5-fight rolling window. UFC fighters fight ~3-4 times/year so 5
# fights covers ~14-18 months — long enough to track recent form,
# short enough that ancient career stats don't dilute.
WINDOW = 5
CACHE_TTL_SECONDS = 3600

# 365 days = "active fighter". MMA layoffs of 6-12 months are
# routine (injury rehab, contract disputes); >365 days idle often
# signals borderline retirement / cage rust.
ACTIVE_DAYS = 365.0


NEUTRAL_DEFAULTS: dict[str, float] = {
    # Moneyline — modal MMA favorite is ~ -160 / +135 = 62/38 split.
    # Slightly less heavy than tennis (-200 / +160 modal) since MMA
    # upsets are more frequent.
    "odds_home_ml": 1.62,
    "odds_away_ml": 2.45,
    "implied_prob_home_ml": 0.62,
    "implied_prob_away_ml": 0.38,
    "ml_bookie_margin": 0.05,
    # Rolling form — neutral .500 win rate over partial window.
    "home_roll_wins": 2.5,
    "home_roll_matches": 5.0,
    "home_roll_win_pct": 0.50,
    "away_roll_wins": 2.5,
    "away_roll_matches": 5.0,
    "away_roll_win_pct": 0.50,
    # H2H — defaults to never-fought (the common case in MMA where
    # most matchups are first meetings).
    "h2h_home_wins": 0.0,
    "h2h_away_wins": 0.0,
    "h2h_matches": 0.0,
    # Schedule context — modal inter-fight gap is ~120 days (4 months).
    # active=1 means "fought in last 365 days" → true for any roster
    # regular.
    "home_days_rest": 120.0,
    "away_days_rest": 120.0,
    "home_active": 1.0,
    "away_active": 1.0,
    # Derived diffs default to zero (no advantage).
    "roll_win_pct_diff": 0.0,
    "h2h_balance": 0.0,
    "days_rest_diff": 0.0,
    "odds_implied_diff": 0.24,  # matches the 62/38 favorite/dog default
}


def _with_defaults(features: dict) -> dict:
    """Replace any None / non-numeric value with the neutral default for that
    key. Guarantees every model input is finite."""
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
    """Scheduled MMA fights that need feature computation (or, with
    --all-finished, every finished MMA fight for training backfill)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if match_ids:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'mma' AND m.id::text = ANY(%s)
                """,
                (match_ids,),
            )
            return [r["id"] for r in cur.fetchall()]

        if all_finished:
            cur.execute(
                """
                SELECT m.id::text AS id FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = 'mma' AND m.status = 'finished'
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
                WHERE l.sport = 'mma' AND m.status = 'scheduled'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY m.match_date ASC
                """,
                (str(days),),
            )
            return [r["id"] for r in cur.fetchall()]

        cur.execute(
            """
            SELECT m.id::text AS id FROM matches m
            JOIN leagues l ON l.id = m.league_id
            LEFT JOIN features_cache fc
              ON fc.match_id = m.id
             AND fc.feature_set = %s
             AND fc.feature_version = %s
             AND fc.expires_at > NOW()
            WHERE l.sport = 'mma' AND m.status = 'scheduled'
              AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
              AND fc.id IS NULL
            ORDER BY m.match_date ASC
            """,
            (FEATURE_SET, FEATURE_VERSION, str(days)),
        )
        return [r["id"] for r in cur.fetchall()]


def fetch_match_meta(cur, match_id: str) -> Optional[dict]:
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
    """Pull the most-recent pre-fight odds row per (market_type,
    selection). MMA registers 'moneyline' only in v1."""
    cur.execute(
        """
        SELECT DISTINCT ON (market_type, selection)
               market_type, selection, line, odds_decimal
        FROM odds
        WHERE match_id = %s AND is_live = false
          AND market_type = 'moneyline'
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

    if "odds_home_ml" in out and "odds_away_ml" in out:
        raw_h = 1.0 / out["odds_home_ml"]
        raw_a = 1.0 / out["odds_away_ml"]
        margin = raw_h + raw_a - 1.0
        if raw_h + raw_a > 0:
            out["implied_prob_home_ml"] = raw_h / (raw_h + raw_a)
            out["implied_prob_away_ml"] = raw_a / (raw_h + raw_a)
            out["ml_bookie_margin"] = margin

    return out


def fetch_fighter_rolling(cur, fighter_id: str, before_date) -> dict:
    """Last-N-fights rolling form for one fighter. Same shape as
    tennis but with the MMA WINDOW (5)."""
    cur.execute(
        """
        WITH last_fights AS (
            SELECT m.id, m.match_date, m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            WHERE l.sport = 'mma'
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
            COUNT(*) AS fights_played
        FROM last_fights
        """,
        (fighter_id, fighter_id, before_date, WINDOW, fighter_id, fighter_id),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row and row.get("fights_played", 0) > 0:
        wins = float(row["wins"]) if row["wins"] is not None else 0.0
        fights = float(row["fights_played"])
        out["roll_wins"] = wins
        out["roll_matches"] = fights
        out["roll_win_pct"] = wins / fights if fights > 0 else 0.5
    return out


def fetch_head_to_head(cur, home_id: str, away_id: str, before_date) -> dict:
    """Career head-to-head between this specific fighter pair. Rare in
    MMA — most matchups are first meetings."""
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
        WHERE l.sport = 'mma'
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
            home_id,
            away_id,
            away_id,
            home_id,
            away_id,
            away_id,
            home_id,
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


def fetch_fighter_schedule(cur, fighter_id: str, before_date) -> dict:
    """Days since this fighter's last fight + active flag (fought in
    last 365 days). MMA fight cadence is way slower than tennis so
    the active threshold is much higher."""
    cur.execute(
        """
        SELECT m.match_date
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma'
          AND m.status IN ('finished', 'live')
          AND (m.home_team_id = %s OR m.away_team_id = %s)
          AND m.match_date < %s
        ORDER BY m.match_date DESC
        LIMIT 1
        """,
        (fighter_id, fighter_id, before_date),
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
    meta = fetch_match_meta(cur, match_id)
    if not meta:
        return None
    when = meta["match_date"]

    features: dict = {}
    features.update(fetch_closing_odds(cur, match_id))
    for side, fighter_id in (("home", meta["home_team_id"]), ("away", meta["away_team_id"])):
        roll = fetch_fighter_rolling(cur, fighter_id, when)
        for k, v in roll.items():
            features[f"{side}_{k}"] = v
        sched = fetch_fighter_schedule(cur, fighter_id, when)
        for k, v in sched.items():
            features[f"{side}_{k}"] = v
    features.update(fetch_head_to_head(cur, meta["home_team_id"], meta["away_team_id"], when))

    _diff(features, "home_roll_win_pct", "away_roll_win_pct", "roll_win_pct_diff")
    _diff(features, "home_days_rest", "away_days_rest", "days_rest_diff")
    _diff(features, "implied_prob_home_ml", "implied_prob_away_ml", "odds_implied_diff")
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
        help="Backfill features for every finished MMA fight (one-shot training prep).",
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
            logger.info("No MMA fights need feature computation")
            return {"written": 0, "skipped": 0}
        logger.info("Computing features for %d MMA fights", len(targets))
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for mid in targets:
                features = compute_for_match(cur, mid)
                if features is None:
                    skipped += 1
                    continue
                write_features(cur, mid, features)
                written += 1
            conn.commit()
    logger.info("Wrote %d MMA feature rows (%d skipped)", written, skipped)
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
