"""Compute baseline features for matches and write them to features_cache.

These are the features the training pipeline actually uses
(training_data.prepare_training_frame): closing-odds implied
probabilities + 5-game rolling team form. Same names, same scale.

The richer 250+ feature orchestrator (services/feature-engineering)
now works correctly against the actual schema, but the *trained
model* only saw the baseline 18-feature set during training. Feeding
it the orchestrator's superset would produce predictions on column
names the model has never heard of. Until the training pipeline is
extended to consume the orchestrator features, this script keeps
features_cache aligned with the model's expected feature space.

Features written per match (all numeric, all pre-match — no leakage):

    implied_prob_home, implied_prob_draw, implied_prob_away
    bookie_margin
    implied_prob_over25
    home_roll_goals_for, home_roll_goals_against, home_roll_points
    away_roll_goals_for, away_roll_goals_against, away_roll_points

Written into features_cache.features as a JSONB blob, keyed on
(match_id, feature_set='baseline', feature_version='v1').

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

# Rolling match_stats form (validated: scripts/ab_soccer_matchstats.py,
# held-out ΔBrier -0.0029, +0.91pts acc). xG is intentionally excluded —
# expected_goals is NULL for 100% of match_stats rows. Multi-window
# (3/5/10) stacking is what made the signal clear in the A/B.
MS_STATS = ["shots", "shots_on_target", "corners", "possession", "fouls", "shots_against", "sot_against"]
MS_WINDOWS = [3, 5, 10]
# How long a computed feature row stays valid before recompute.
CACHE_TTL_SECONDS = 3600

# Neutral defaults for each feature, used when the source data is missing
# (no live odds for this match, no rolling form for the team). Without
# these the value would be NULL; downstream predict_proba then runs
# fillna(X.median()) on a single-row DataFrame, gets median=NULL, and
# emits NaN probabilities — the prediction gets skipped and we never
# learn anything about the match. With neutral defaults the model still
# predicts but converges toward the league prior, which is the right
# behaviour for sparse-data fixtures. As real data comes in via
# fetch_live_odds and the historical loaders, predictions diverge from
# the prior with actual signal.
NEUTRAL_DEFAULTS: dict[str, float] = {
    # Closing 1x2 odds — roughly the league-average implied probs.
    "odds_home": 2.40,
    "odds_draw": 3.30,
    "odds_away": 3.10,
    "odds_over25": 1.95,
    "odds_under25": 1.95,
    # Derived implied probabilities.
    "implied_prob_home": 0.43,
    "implied_prob_draw": 0.27,
    "implied_prob_away": 0.30,
    "implied_prob_over25": 0.52,
    "bookie_margin": 0.05,
    # Rolling team form — mid-table averages (~1.4 pts/game ≈ ~5 pts/5).
    "home_roll_goals_for": 1.40,
    "home_roll_goals_against": 1.40,
    "home_roll_points": 1.40,
    "away_roll_goals_for": 1.40,
    "away_roll_goals_against": 1.40,
    "away_roll_points": 1.40,
    "form_diff_points": 0.0,
    "form_diff_goals": 0.0,
}
# Rolling match_stats diffs (home minus away) default to 0 = teams equal,
# the neutral prior — same rationale as form_diff_* above.
NEUTRAL_DEFAULTS.update({f"ms_diff_{s}_w{w}": 0.0 for s in MS_STATS for w in MS_WINDOWS})


def _with_defaults(features: dict) -> dict:
    """Replace any None / non-numeric value with the neutral default for that
    key. Guarantees every model input is finite."""
    out: dict = {}
    for k, default in NEUTRAL_DEFAULTS.items():
        v = features.get(k)
        out[k] = v if isinstance(v, (int, float)) and v is not None else default
    # Pass through any extra keys (model can't use them but features_cache
    # is just JSONB — keeping them lets us add features without losing data).
    for k, v in features.items():
        if k not in out:
            out[k] = v
    return out


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


def _compute_implied_probs(odds_h, odds_d, odds_a):
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


def _compute_over_under(odds_over, odds_under):
    if not (odds_over and odds_under):
        return None
    inv_o, inv_u = 1.0 / odds_over, 1.0 / odds_under
    return inv_o / (inv_o + inv_u)


def _rolling_team_form(cur, team_id: str, before_date, side: str) -> dict:
    """Mean goals-for/against and points over the team's last `WINDOW`
    finished matches before `before_date`. `side` ∈ {'home', 'away'}
    is for label prefixing; the underlying stats include BOTH home and
    away matches for the team to maximise the available sample."""
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
            AVG(goals_for)::float     AS gf,
            AVG(goals_against)::float AS ga,
            AVG(CASE
                WHEN goals_for > goals_against THEN 3.0
                WHEN goals_for = goals_against THEN 1.0
                ELSE 0.0
            END)::float               AS pts,
            COUNT(*)::int             AS n
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


def _rolling_team_stats(cur, team_id: str, before_date) -> dict:
    """Per-window means of the team's own match_stats (and the opponent's
    shots/SoT, for the 'against' metrics) over its last max(MS_WINDOWS)
    FINISHED matches before `before_date`. Leakage-safe: the current
    match is excluded by `match_date < before_date`. Returns
    {f'{stat}_w{w}': mean} (None when the team has no prior history)."""
    cur.execute(
        """
        SELECT
            ms.shots, ms.shots_on_target, ms.corners, ms.possession, ms.fouls,
            opp.shots            AS shots_against,
            opp.shots_on_target  AS sot_against
        FROM matches m
        JOIN match_stats ms ON ms.match_id = m.id AND ms.team_id = %(tid)s
        LEFT JOIN match_stats opp ON opp.match_id = m.id AND opp.team_id <> %(tid)s
        WHERE (m.home_team_id = %(tid)s OR m.away_team_id = %(tid)s)
          AND m.status = 'finished'
          AND m.match_date < %(before)s
        ORDER BY m.match_date DESC
        LIMIT %(n)s
        """,
        {"tid": team_id, "before": before_date, "n": max(MS_WINDOWS)},
    )
    rows = cur.fetchall()
    out: dict = {}
    for w in MS_WINDOWS:
        sub = rows[:w]
        for c in MS_STATS:
            vals = [float(r[c]) for r in sub if r.get(c) is not None]
            out[f"{c}_w{w}"] = (sum(vals) / len(vals)) if vals else None
    return out


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


def compute_match_features(cur, match_id: str):
    """Return the feature dict for one match, or None if the match has
    no team ids."""
    cur.execute(
        "SELECT home_team_id::text AS h, away_team_id::text AS a, match_date AS d " "FROM matches WHERE id = %s",
        (match_id,),
    )
    m = cur.fetchone()
    if not m or not (m["h"] and m["a"]):
        return None

    features: dict = {}

    odds_h = _odds_avg(cur, match_id, "1x2", "home")
    odds_d = _odds_avg(cur, match_id, "1x2", "draw")
    odds_a = _odds_avg(cur, match_id, "1x2", "away")
    odds_over = _odds_avg(cur, match_id, "over_under", "over", 2.5)
    odds_under = _odds_avg(cur, match_id, "over_under", "under", 2.5)

    # Raw odds — what the currently-trained model references directly.
    features["odds_home"] = odds_h
    features["odds_draw"] = odds_d
    features["odds_away"] = odds_a
    features["odds_over25"] = odds_over
    features["odds_under25"] = odds_under

    features.update(_compute_implied_probs(odds_h, odds_d, odds_a))
    features["implied_prob_over25"] = _compute_over_under(odds_over, odds_under)

    home_form = _rolling_team_form(cur, m["h"], m["d"], "home")
    away_form = _rolling_team_form(cur, m["a"], m["d"], "away")
    features.update(home_form)
    features.update(away_form)

    # form_diff_* — what the currently-trained model references.
    def _safe_diff(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    features["form_diff_points"] = _safe_diff(home_form.get("home_roll_points"), away_form.get("away_roll_points"))
    features["form_diff_goals"] = _safe_diff(home_form.get("home_roll_goals_for"), away_form.get("away_roll_goals_for"))

    # Rolling match_stats diffs (home minus away) per stat per window —
    # the validated signal. Raw per-side values are computed internally
    # but only the diffs are written (matches the A/B feature set).
    home_ms = _rolling_team_stats(cur, m["h"], m["d"])
    away_ms = _rolling_team_stats(cur, m["a"], m["d"])
    for c in MS_STATS:
        for w in MS_WINDOWS:
            features[f"ms_diff_{c}_w{w}"] = _safe_diff(home_ms.get(f"{c}_w{w}"), away_ms.get(f"{c}_w{w}"))

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
                    # Fill any None values with neutral defaults before
                    # persisting — guarantees downstream predict_proba
                    # gets finite inputs and never NaN-skips a match.
                    features = _with_defaults(features)
                    write_features(cur, mid, features)
                    ok += 1
                except Exception as e:
                    logger.warning("Failed for %s: %s", mid, e)
                    fail += 1
            conn.commit()
    return {"ok": ok, "fail": fail}


def list_finished_matches(conn, sport: str) -> list[str]:
    """All FINISHED matches (with a score) for a sport — for the
    historical features_cache backfill so the model TRAINS on the same
    feature set it serves with."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT m.id::text AS id
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            WHERE l.sport = %s
              AND m.status = 'finished'
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            ORDER BY m.match_date ASC
            """,
            (sport,),
        )
        return [r["id"] for r in cur.fetchall()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--match-ids", help="Comma-separated UUIDs (overrides --days).")
    p.add_argument(
        "--backfill-finished",
        metavar="SPORT",
        help="Recompute features_cache for ALL finished matches of SPORT " "(historical backfill; overrides --days).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    if args.match_ids:
        match_ids = [s.strip() for s in args.match_ids.split(",") if s.strip()]
    elif args.backfill_finished:
        with psycopg2.connect(args.database_url) as conn:
            match_ids = list_finished_matches(conn, args.backfill_finished)
        logger.info("Backfill: %d finished %s matches", len(match_ids), args.backfill_finished)
    else:
        with psycopg2.connect(args.database_url) as conn:
            match_ids = list_target_matches(conn, args.days)

    if not match_ids:
        logger.info("No matches to compute features for")
        return 0

    logger.info("Computing features for %d match(es)...", len(match_ids))
    counts = compute_all(args.database_url, match_ids)
    logger.info("Done: %d ok / %d failed", counts["ok"], counts["fail"])
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
