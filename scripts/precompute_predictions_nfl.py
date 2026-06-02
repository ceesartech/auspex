"""Precompute NFL predictions for upcoming scheduled matches.

Parallel to scripts/precompute_predictions_nba.py — iterates the
TASKS registry for NFL, runs every NFL task (moneyline, spread,
total) per upcoming match, writes one prediction row per (match,
task) to the predictions table. The moneyline result is also pushed
into Redis under the legacy single-prediction cache key so the API's
/predictions/ endpoint serves a cached response without re-running
the ensemble.

Same line-as-feature design as NBA: closing_spread_home /
closing_total_line are baked into the features_cache row written by
scripts/compute_features_nfl.py — we just run the trained ensembles
against them.

Telegram digest: high-confidence picks are enqueued for the
combined pipeline digest via scripts/telegram_notify.py. Per-market
thresholds match NBA's (0.60 / 0.58 / 0.58) — NFL favorites and
covers are roughly as common as NBA, and we re-tune from real
calibration data later if needed.

Usage:
    python /app/scripts/precompute_predictions_nfl.py                 # next 7 days
    python /app/scripts/precompute_predictions_nfl.py --days 14
    python /app/scripts/precompute_predictions_nfl.py --no-notify     # skip Telegram
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
logger = logging.getLogger("precompute_predictions_nfl")

# features_cache key for NFL. Same compute-script-versioning contract
# as NBA / NHL.
FEATURE_SET = "nfl_baseline"
FEATURE_VERSION = "v1"

# Per-market confidence thresholds. NFL favorites clear 0.60 less
# often than NBA (more parity, more variance per game) but the
# 0.58 line-as-feature floor still expresses a "measurable edge"
# — re-tune from real calibration metrics after we have a few
# weeks of OOS picks.
MARKET_NOTIFY_THRESHOLDS: dict[str, float] = {
    "moneyline": 0.60,
    "spread": 0.58,
    "total": 0.58,
}

MARKET_DISPLAY_LABELS: dict[str, str] = {
    "moneyline": "Moneyline",
    "spread": "Spread",
    "total": "Total Points",
}


def nfl_alert(
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
    """Translate one NFL task prediction into the shared Alert shape."""
    return Alert(
        sport="nfl",
        league_name=league_name,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        market_label=MARKET_DISPLAY_LABELS.get(market, market),
        predicted_outcome=predicted_outcome,
        confidence=float(confidence),
        probabilities={k: float(v) for k, v in (probabilities or {}).items()},
    )


def list_upcoming_nfl(database_url: str, days: int) -> list[dict]:
    """Scheduled NFL matches in the next N days that have a fresh
    nfl_baseline features_cache row. Ordered by match_date ASC."""
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
                  AND l.sport = 'nfl'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY m.match_date ASC
                """,
                (FEATURE_SET, FEATURE_VERSION, str(days)),
            )
            return [dict(r) for r in cur.fetchall()]


def fetch_features(cur, match_id: str) -> dict | None:
    """Read the most-recent nfl_baseline features JSONB for a match."""
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
    prediction_type)."""
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
    """Walk upcoming NFL matches and write predictions for every
    registered NFL task. Returns a count summary for log / DAG return.
    """
    from services.cache_service import CacheService  # type: ignore
    from services.prediction_service import get_model_version, load_models_into_process, tasks_for_sport  # type: ignore

    models = load_models_into_process()
    cache = CacheService()

    nfl_tasks = tasks_for_sport("nfl")
    if not nfl_tasks:
        logger.error("No NFL tasks registered in TASKS dict — nothing to do")
        return {"predicted": 0, "match_count": 0}
    logger.info(
        "NFL tasks to predict per match: %s",
        [t.market for t in nfl_tasks],
    )

    missing = [f"{t.sport}:{t.market}" for t in nfl_tasks if f"{t.sport}:{t.market}" not in models]
    if missing:
        logger.warning(
            "NFL tasks missing from loaded models: %s — those rows will be skipped",
            missing,
        )

    upcoming = list_upcoming_nfl(database_url, days)
    if not upcoming:
        logger.info("No upcoming NFL matches with fresh features in the next %d days", days)
        return {"predicted": 0, "match_count": 0}
    logger.info("Found %d upcoming NFL matches to score", len(upcoming))

    predicted = 0
    skipped = 0
    alerts: list[Alert] = []
    market_predicted: dict[str, int] = {t.market: 0 for t in nfl_tasks}
    market_max_conf: dict[str, float] = {t.market: 0.0 for t in nfl_tasks}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for m in upcoming:
                features = fetch_features(cur, m["match_id"])
                if not features:
                    skipped += 1
                    continue

                X_dict: dict = {}
                for k, v in features.items():
                    X_dict[k] = v
                    prefixed = k if k.startswith("feature__") else f"feature__{k}"
                    if prefixed not in X_dict:
                        X_dict[prefixed] = v

                headline_cached = False
                for task in nfl_tasks:
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

                    if notify:
                        threshold = MARKET_NOTIFY_THRESHOLDS.get(task.market, 0.65)
                        if confidence >= threshold:
                            alerts.append(
                                nfl_alert(
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

    queue_depth = enqueue_alerts(alerts) if alerts else 0
    market_breakdown = ", ".join(
        f"{market}: n={market_predicted[market]}, "
        f"max={market_max_conf[market]:.2f}, "
        f"gate={MARKET_NOTIFY_THRESHOLDS.get(market, 0.65):.2f}"
        for market in market_predicted
    )
    logger.info(
        "Wrote %d NFL prediction rows across %d matches (%d skipped); "
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
    for p in ("/app/services/api/src", "/app/services/ml-models/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    counts = run(args.database_url, args.days, notify=not args.no_notify)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
