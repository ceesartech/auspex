"""Precompute NBA predictions for upcoming scheduled matches.

Parallel to scripts/precompute_predictions_nhl.py — iterates Phase 6
TASKS registry for NBA, runs every NBA task (moneyline, spread,
total) per upcoming match, writes one prediction row per (match,
task) to the predictions table. The moneyline result is also pushed
into Redis under the legacy single-prediction cache key so the API's
/predictions/ endpoint serves a cached response without re-running
the ensemble.

Key difference from NHL: NBA spread + total use LINE-AS-FEATURE
(closing_spread_home / closing_total_line are inputs the model
conditions on). The features_cache row written by
scripts/compute_features_nba.py already carries these — we just run
the trained ensembles against them, no per-line variation needed at
predict time.

Telegram digest: high-confidence picks are enqueued for the
combined pipeline digest via scripts/telegram_notify.py.
Per-market thresholds in MARKET_NOTIFY_THRESHOLDS keep the
noise-floor markets (spread, total) honest — moneyline gets the
lowest bar because NBA favorites routinely clear it.

Usage:
    python /app/scripts/precompute_predictions_nba.py                 # next 7 days
    python /app/scripts/precompute_predictions_nba.py --days 14
    python /app/scripts/precompute_predictions_nba.py --no-notify     # skip Telegram
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

# Add the scripts dir so the shared telegram_notify helper imports.
sys.path.insert(0, os.path.dirname(__file__))

from telegram_notify import Alert, enqueue_alerts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_predictions_nba")

# features_cache key for NBA. Mirrors NHL's
# (nhl_baseline, v1/v2/v3) pin — same compute-script-versioning
# contract.
FEATURE_SET = "nba_baseline"
FEATURE_VERSION = "v1"

# Per-market confidence thresholds. Tuned to surface alerts on most
# game days without spamming below-noise-floor picks:
#   - moneyline: NBA favorites clear 0.60 routinely; the model also
#     hits 75% val accuracy so high-conf picks are common
#   - spread: line-as-feature gives the model real signal; 0.58 is
#     the conservative "I have a measurable edge" floor
#   - total: same line-as-feature dynamic; 0.58 baseline
MARKET_NOTIFY_THRESHOLDS: dict[str, float] = {
    "moneyline": 0.60,
    "spread": 0.58,
    "total": 0.58,
}

# Friendly market labels for the Telegram digest. Aligns with
# MARKET_DISPLAY_LABELS in scripts/precompute_predictions_nhl.py.
MARKET_DISPLAY_LABELS: dict[str, str] = {
    "moneyline": "Moneyline",
    "spread": "Spread",
    "total": "Total Points",
}


def nba_alert(
    *,
    league_name: str,
    home_team: str,
    away_team: str,
    match_date,
    market: str,
    predicted_outcome: str,
    confidence: float,
    probabilities: dict,
) -> Alert:
    """Translate one NBA task prediction into the shared Alert shape.
    Looks up the friendly market label here so the digest never
    carries raw snake_case strings to users."""
    return Alert(
        sport="nba",
        league_name=league_name,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        market_label=MARKET_DISPLAY_LABELS.get(market, market),
        predicted_outcome=predicted_outcome,
        confidence=float(confidence),
        probabilities={k: float(v) for k, v in (probabilities or {}).items()},
    )


def list_upcoming_nba(database_url: str, days: int) -> list[dict]:
    """Scheduled NBA matches in the next N days that have a fresh
    nba_baseline features_cache row. Ordered by match_date ASC so a
    crash mid-run leaves the soonest-to-start matches predicted first."""
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    m.id::text         AS match_id,
                    m.match_date,
                    ht.name            AS home_team,
                    at.name            AS away_team,
                    l.name             AS league_name
                FROM matches m
                JOIN leagues l ON l.id = m.league_id
                JOIN teams ht  ON ht.id = m.home_team_id
                JOIN teams at  ON at.id = m.away_team_id
                JOIN features_cache fc
                  ON fc.match_id = m.id
                 AND fc.feature_set = %s
                 AND fc.feature_version = %s
                 AND fc.expires_at > NOW()
                WHERE m.status = 'scheduled'
                  AND l.sport = 'nba'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY m.match_date ASC
                """,
                (FEATURE_SET, FEATURE_VERSION, str(days)),
            )
            return [dict(r) for r in cur.fetchall()]


def fetch_features(cur, match_id: str) -> dict | None:
    """Read the most-recent nba_baseline features JSONB for a match."""
    cur.execute(
        """
        SELECT features FROM features_cache
        WHERE match_id = %s
          AND feature_set = %s
          AND feature_version = %s
          AND expires_at > NOW()
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        (match_id, FEATURE_SET, FEATURE_VERSION),
    )
    row = cur.fetchone()
    return row["features"] if row else None


def store_prediction(
    cur,
    match_id: str,
    model_name: str,
    model_version: str,
    prediction_type: str,
    predicted_outcome: str,
    confidence: float,
    probabilities: dict,
) -> None:
    """Idempotent upsert keyed on (match_id, model_name, model_version,
    prediction_type). Same 4-column constraint as the NHL path."""
    import json

    cur.execute(
        """
        INSERT INTO predictions
            (match_id, model_name, model_version, prediction_type,
             predicted_outcome, confidence, probabilities)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (match_id, model_name, model_version, prediction_type)
        DO UPDATE SET
            predicted_outcome = EXCLUDED.predicted_outcome,
            confidence = EXCLUDED.confidence,
            probabilities = EXCLUDED.probabilities,
            updated_at = NOW()
        """,
        (
            match_id,
            model_name,
            model_version,
            prediction_type,
            predicted_outcome,
            confidence,
            json.dumps(probabilities),
        ),
    )


def run(database_url: str, days: int, notify: bool = True) -> dict:
    """Walk upcoming NBA matches and write predictions for every
    registered NBA task. Returns a count summary for log / DAG return.

    With notify=True (default), high-confidence picks are queued for
    the combined pipeline digest via enqueue_alerts.
    """
    # Late imports so module load doesn't pull in numpy / sqlalchemy /
    # the predictor package unless the script is actually being run.
    from services.cache_service import CacheService  # type: ignore
    from services.prediction_service import get_model_version, load_models_into_process, tasks_for_sport  # type: ignore

    models = load_models_into_process()
    cache = CacheService()

    nba_tasks = tasks_for_sport("nba")
    if not nba_tasks:
        logger.error("No NBA tasks registered in TASKS dict — nothing to do")
        return {"predicted": 0, "match_count": 0}
    logger.info(
        "NBA tasks to predict per match: %s",
        [t.market for t in nba_tasks],
    )

    # Quick gate: if any NBA ensemble isn't loaded, log up front so
    # the operator notices missing artifacts before we burn time
    # iterating over hundreds of matches. Same gate as the NHL path.
    missing = [f"{t.sport}:{t.market}" for t in nba_tasks if f"{t.sport}:{t.market}" not in models]
    if missing:
        logger.warning(
            "NBA tasks missing from loaded models: %s — those rows will be skipped",
            missing,
        )

    upcoming = list_upcoming_nba(database_url, days)
    if not upcoming:
        logger.info("No upcoming NBA matches with fresh features in the next %d days", days)
        return {"predicted": 0, "match_count": 0}
    logger.info("Found %d upcoming NBA matches to score", len(upcoming))

    predicted = 0
    skipped = 0
    alerts: list[Alert] = []
    # Per-market diagnostic tally — surfaces in the summary log so a
    # 0-alert run shows which market gates were the blocker (mirrors
    # the NHL precompute pattern).
    market_predicted: dict[str, int] = {t.market: 0 for t in nba_tasks}
    market_max_conf: dict[str, float] = {t.market: 0.0 for t in nba_tasks}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for m in upcoming:
                features = fetch_features(cur, m["match_id"])
                if not features:
                    # Belt-and-braces: list_upcoming already filtered
                    # to matches WITH features; this catches a TTL race
                    # between query and fetch.
                    skipped += 1
                    continue

                # Mirror unprefixed keys to their feature__ form
                # (same bridge as the NHL path — trained models'
                # feature_names list contains BOTH raw and prefixed
                # names per the training-data flattening).
                X_dict: dict = {}
                for k, v in features.items():
                    X_dict[k] = v
                    prefixed = k if k.startswith("feature__") else f"feature__{k}"
                    if prefixed not in X_dict:
                        X_dict[prefixed] = v

                headline_cached = False
                for task in nba_tasks:
                    task_key = f"{task.sport}:{task.market}"
                    model = models.get(task_key)
                    if model is None:
                        continue
                    try:
                        import numpy as np
                        import pandas as pd

                        proba = model.predict_proba(pd.DataFrame([X_dict]))[0]
                        if not np.all(np.isfinite(proba)):
                            logger.warning(
                                "Non-finite proba for %s/%s — skipping",
                                m["match_id"],
                                task.market,
                            )
                            continue
                    except Exception as e:
                        logger.warning(
                            "Predict failed for %s task=%s: %s",
                            m["match_id"],
                            task.market,
                            e,
                        )
                        continue

                    if len(proba) != len(task.labels):
                        logger.warning(
                            "Label count mismatch for %s task=%s: %d proba vs %d labels",
                            m["match_id"],
                            task.market,
                            len(proba),
                            len(task.labels),
                        )
                        continue

                    idx = int(proba.argmax())
                    probabilities = {task.labels[i]: float(proba[i]) for i in range(len(task.labels))}
                    confidence = float(proba[idx])
                    predicted_outcome = task.labels[idx]
                    market_predicted[task.market] += 1
                    if confidence > market_max_conf[task.market]:
                        market_max_conf[task.market] = confidence

                    version_str = get_model_version(task_key)
                    store_prediction(
                        cur,
                        m["match_id"],
                        task.ensemble_name,
                        version_str,
                        task.prediction_type,
                        predicted_outcome,
                        confidence,
                        probabilities,
                    )
                    predicted += 1

                    # Cache the moneyline as the headline so the
                    # /api/v1/predictions/ endpoint can serve it
                    # without re-running the ensemble. Other markets
                    # are exposed via /predictions/match/{match_id}.
                    if task.market == "moneyline" and not headline_cached:
                        cache.set_prediction(
                            m["match_id"],
                            {
                                "predicted_outcome": predicted_outcome,
                                "confidence": confidence,
                                "probabilities": probabilities,
                                "model_version": version_str,
                            },
                            model_version=version_str,
                        )
                        headline_cached = True

                    # Accumulate for the end-of-run digest. Per-market
                    # thresholds gate which picks surface; unknown
                    # markets fall back to 0.65 (mid-range floor).
                    if notify:
                        threshold = MARKET_NOTIFY_THRESHOLDS.get(task.market, 0.65)
                        if confidence >= threshold:
                            alerts.append(
                                nba_alert(
                                    league_name=m["league_name"],
                                    home_team=m["home_team"],
                                    away_team=m["away_team"],
                                    match_date=m["match_date"],
                                    market=task.market,
                                    predicted_outcome=predicted_outcome,
                                    confidence=confidence,
                                    probabilities=probabilities,
                                )
                            )
            conn.commit()

    # Push NBA picks onto the same shared Redis queue the soccer +
    # NHL scripts use. The DAG's downstream send_pipeline_digest task
    # drains all three sports into ONE combined Telegram message.
    queue_depth = enqueue_alerts(alerts) if alerts else 0
    # Per-market breakdown for diagnostics.
    market_breakdown = ", ".join(
        f"{market}: n={market_predicted[market]}, "
        f"max={market_max_conf[market]:.2f}, "
        f"gate={MARKET_NOTIFY_THRESHOLDS.get(market, 0.65):.2f}"
        for market in market_predicted
    )
    logger.info(
        "Wrote %d NBA prediction rows across %d matches (%d skipped); "
        "queued %d picks (queue depth now %d). Per-market: %s",
        predicted,
        len(upcoming),
        skipped,
        len(alerts),
        queue_depth,
        market_breakdown,
    )
    return {
        "predicted": predicted,
        "match_count": len(upcoming),
        "skipped": skipped,
        "alerts_queued": len(alerts),
        "queue_depth": queue_depth,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7, help="How many days forward to score (default 7).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip Telegram dispatch for high-confidence predictions.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    # Make services.* importable. Same path manipulation as the NHL
    # precompute — Dockerfile.api puts api src on PYTHONPATH for the
    # running api container, but `docker compose exec` for a one-off
    # script invocation still needs the bootstrap.
    for p in ("/app/services/api/src", "/app/services/ml-models/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    counts = run(args.database_url, args.days, notify=not args.no_notify)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
