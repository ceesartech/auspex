"""Compute baseline features for matches and write them to features_cache.

The full 250+ feature orchestrator (services/feature-engineering) was
written against a different schema than what's actually in the DB, so
its SQL queries fail. Rather than rewriting all of that, this script
computes the **same baseline features** the training pipeline uses
(closing-odds implied probabilities + rolling team form) directly from
the canonical leagues/teams/matches/odds tables. Same shape as what
training_data.prepare_training_frame produces, just for individual
upcoming matches.

Features written per match (all numeric, all pre-match — no leakage):

    implied_prob_home, implied_prob_draw, implied_prob_away
    bookie_margin
    implied_prob_over25
    home_roll_goals_for, home_roll_goals_against, home_roll_points
    away_roll_goals_for, away_roll_goals_against, away_roll_points

These are written into features_cache.features as a JSONB blob, keyed
on (match_id, feature_set='baseline', feature_version='v1').
precompute_predictions.py reads them back via the same key.

Usage:
    python /app/scripts/compute_features.py             # next 7 days scheduled
    python /app/scripts/compute_features.py --days 14
    python /app/scripts/compute_features.py --match-ids id1,id2,id3
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
logger = logging.getLogger("compute_features")

FEATURE_SET = "baseline"
FEATURE_VERSION = "v1"
# Rolling window (last N team matches counted toward form features).
WINDOW = 5
# How long a computed feature row stays valid before recompute.
CACHE_TTL_SECONDS = 3600


def list_target_matches(conn, days: int) -> list[str]:
    """Scheduled matches in the next N days lacking a fresh features_cache row."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT m.id::text AS id
            FROM matches m
            LEFT JOIN features_cache f
              ON f.match_id = m.id
             AND f.feature_set = %s
             AND f.feature_version = %s
             AND f.expires_at > NOW()
            WHERE m.status = 'scheduled'
              AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
              AND f.id IS NULL
            ORDER BY m.match_date ASC
            """,
            (FEATURE_SET, FEATURE_VERSION, str(days)),
        )
        return [r["id"] for r in cur.fetchall()]


def _compute_implied_probs(odds_h: float | None, odds_d: float | None, odds_a: float | None) -> dict[str, float | None]:
    if not (odds_h and odds_d and odds_a):
        return {
            "implied_prob_home": None,
            "implied_prob_draw": None,
            "implied_prob_away": None,
            "bookie_margin": None,
        }
    inv_h, inv_d, inv_a = 1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a
    margin = inv_h + inv_d + inv_a
    return {
        "implied_prob_home": inv_h / margin,
        "implied_prob_draw": inv_d / margin,
        "implied_prob_away": inv_a / margin,
        "bookie_margin": margin - 1.0,
    }


def _compute_over_under(odds_over: float | None, odds_under: float | None) -> float | None:
    if not (odds_over and odds_under):
        return None
    inv_o, inv_u = 1.0 / odds_over, 1.0 / odds_under
    return inv_o / (inv_o + inv_u)


def _rolling_team_form(cur, team_id: str, before_date, side: str) -> dict[str, float | None]:
    """Mean goals-for/against and points over the team's last `WINDOW` matches
    before `before_date`. `side` is 'home' or 'away' for label prefixing only —
    the rolling stats include BOTH home and away matches for the team.
    """
    cur.execute(
        """
        WITH team_matches AS (
            SELECT
                CASE WHEN m.home_team_id = %(tid)s THEN m.home_score ELSE m.away_score END AS goals_for,
                CASE WHEN m.home_team_id = %(tid)s THEN m.away_score ELSE m.home_score END AS goals_against
            FROM matches m
            WHERE (m.home_team_id = %(tid)s OR m.away_team_id = %(tid)s)
              AND m.status = 'finished'
              AND m.match_date < %(before)s
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT %(n)s
        )
        SELECT
            AVG(goals_for)::float        AS gf,
            AVG(goals_against)::float    AS ga,
            AVG(CASE
                WHEN goals_for > goals_against THEN 3.0
                WHEN goals_for = goals_against THEN 1.0
                ELSE 0.0
            END)::float                  AS pts,
            COUNT(*)::int                AS n
        FROM team_matches
        """,
        {"tid": team_id, "before": before_date, "n": WINDOW},
    )
    row = cur.fetchone()
    return {
        f"{side}_roll_goals_for": row["gf"] if row else None,
        f"{side}_roll_goals_against": row["ga"] if row else None,
        f"{side}_roll_points": row["pts"] if row else None,
    }


def _odds_avg(cur, match_id: str, market_type: str, selection: str, line: float | None = None) -> float | None:
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


def compute_match_features(cur, match_id: str) -> dict | None:
    """Build the feature dict for one match. Returns None if the match
    has no team ids — caller should skip."""
    cur.execute(
        "SELECT home_team_id::text AS h, away_team_id::text AS a, match_date AS d " "FROM matches WHERE id = %s",
        (match_id,),
    )
    m = cur.fetchone()
    if not m or not (m["h"] and m["a"]):
        return None

    features: dict[str, float | None] = {}

    # ── Closing 1x2 odds + over/under 2.5 (averaged across books) ───
    odds_h = _odds_avg(cur, match_id, "1x2", "home")
    odds_d = _odds_avg(cur, match_id, "1x2", "draw")
    odds_a = _odds_avg(cur, match_id, "1x2", "away")
    odds_over = _odds_avg(cur, match_id, "over_under", "over", 2.5)
    odds_under = _odds_avg(cur, match_id, "over_under", "under", 2.5)

    features.update(_compute_implied_probs(odds_h, odds_d, odds_a))
    features["implied_prob_over25"] = _compute_over_under(odds_over, odds_under)

    # ── Rolling team form ───────────────────────────────────────────
    features.update(_rolling_team_form(cur, m["h"], m["d"], "home"))
    features.update(_rolling_team_form(cur, m["a"], m["d"], "away"))

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


def compute_all(database_url: str, match_ids: list[str]) -> dict[str, int]:
    ok = fail = 0
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for mid in match_ids:
                try:
                    features = compute_match_features(cur, mid)
                    if features is None:
                        fail += 1
                        continue
                    write_features(cur, mid, features)
                    ok += 1
                except Exception as e:
                    logger.warning("Failed for %s: %s", mid, e)
                    fail += 1
            conn.commit()
    return {"ok": ok, "fail": fail}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--match-ids", help="Comma-separated UUIDs (overrides --days).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    if args.match_ids:
        match_ids = [s.strip() for s in args.match_ids.split(",") if s.strip()]
    else:
        with psycopg2.connect(args.database_url) as conn:
            match_ids = list_target_matches(conn, args.days)

    if not match_ids:
        logger.info("No matches needing features in the next %d days", args.days)
        return 0

    logger.info("Computing features for %d match(es)...", len(match_ids))
    counts = compute_all(args.database_url, match_ids)
    logger.info("Done: %d ok / %d failed", counts["ok"], counts["fail"])
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
