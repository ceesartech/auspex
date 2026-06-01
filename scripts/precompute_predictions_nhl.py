"""Precompute NHL predictions for upcoming scheduled matches.

Parallel to scripts/precompute_predictions.py but iterates Phase 4a's
TASKS registry — for each upcoming NHL match it runs every NHL task
(moneyline, regulation, puck-line, total) and writes one prediction
row per (match, task) to the predictions table. The moneyline result
is also pushed into Redis under the legacy single-prediction cache key
so the API's /predictions/ endpoint can serve a cached response
without re-running the ensemble.

Soccer's market_result + Dixon-Coles market derivation is not used
here — NHL tasks are direct classifications and don't share that
pipeline.

Telegram notifications (Phase 4d) fire one alert per (match, market)
when the prediction's confidence clears that market's threshold. The
thresholds are per-market because the markets sit at different
points in the noise floor: puck-line and total only marginally beat
naive baselines so they need a stricter bar than moneyline. The
template uses 🏒 and prints the friendly market label so a single
chat can receive both soccer and NHL alerts without confusion.

Usage:
    python /app/scripts/precompute_predictions_nhl.py                 # next 7 days
    python /app/scripts/precompute_predictions_nhl.py --days 14
    python /app/scripts/precompute_predictions_nhl.py --no-notify     # skip Telegram
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_predictions_nhl")

# Pin features_cache lookups to v3 (the latest NHL feature version after
# Phase 3e's 5v5 advanced stats). If a future version lands, bump here
# and the lookup automatically prefers the newer rows.
FEATURE_SET = "nhl_baseline"
FEATURE_VERSION = "v3"

# Per-market confidence thresholds (Phase 4d). Moneyline gets the lowest
# bar because favorites routinely clear 0.60 and the model is
# well-calibrated there. Regulation is 3-class so the achievable max
# probability is lower (~0.55 is "strong"). Puck-line and total sit
# close to a naive marginal baseline — only the model's most confident
# picks are worth surfacing, so we set 0.70.
MARKET_NOTIFY_THRESHOLDS: dict[str, float] = {
    "moneyline": 0.60,
    "regulation": 0.55,
    "puck_line": 0.70,
    "total": 0.70,
}

# Friendly market label for the Telegram message. Mirrors the
# MARKET_LABELS map in services/frontend/src/components/predictions/
# prediction-card.tsx — keep both in sync if a new market lands.
MARKET_DISPLAY_LABELS: dict[str, str] = {
    "moneyline": "Moneyline",
    "regulation": "Regulation (60 min)",
    "puck_line": "Puck Line",
    "total": "Total Goals O/U 5.5",
}


def send_telegram_nhl(
    *,
    league_name: str,
    home_team: str,
    away_team: str,
    match_date,
    market: str,
    predicted_outcome: str,
    confidence: float,
    probabilities: dict,
    threshold: float,
) -> bool:
    """Sport-aware Telegram dispatch for NHL predictions.

    Parallel to the soccer send_telegram in scripts/precompute_predictions.py
    but uses the 🏒 emoji and a 2-class-friendly probability line. Returns
    True iff a message was actually sent (config present + threshold met +
    network success).

    The gate ordering is intentional: confidence check first so we can
    early-exit without paying the env-lookup cost on every prediction,
    then the env-flag check, then the credential check.
    """
    if confidence < threshold:
        return False
    if os.environ.get("ENABLE_TELEGRAM_NOTIFICATIONS", "false").lower() != "true":
        return False
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram not configured — skipping NHL notification")
        return False

    market_label = MARKET_DISPLAY_LABELS.get(market, market)
    text = (
        f"🏒 <b>{league_name}</b> · {market_label}\n"
        f"{home_team} vs {away_team}\n"
        f"\U0001f551 {match_date.strftime('%Y-%m-%d %H:%M %Z')}\n\n"
        f"Predicted: <b>{predicted_outcome}</b>\n"
        f"Confidence: <b>{confidence:.1%}</b>\n"
        f"Probabilities: " + ", ".join(f"{k} {v:.1%}" for k, v in (probabilities or {}).items())
    )
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.warning("Telegram send failed: %s", e)
        return False


def list_upcoming_nhl(database_url: str, days: int) -> list[dict]:
    """Scheduled NHL matches in the next N days that have a fresh
    nhl_baseline features_cache row. Ordered by match_date ASC so a
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
                  AND l.sport = 'nhl'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY m.match_date ASC
                """,
                (FEATURE_SET, FEATURE_VERSION, str(days)),
            )
            return [dict(r) for r in cur.fetchall()]


def fetch_features(cur, match_id: str) -> dict | None:
    """Read the most-recent nhl_baseline features JSONB for a match."""
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
    prediction_type). The 4-column constraint was added by migration
    007; predict_match's _store_prediction uses the same shape."""
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
    """Walk upcoming NHL matches and write predictions for every
    registered NHL task. Returns a count summary suitable for log /
    DAG return.

    With notify=True (default), high-confidence predictions also
    trigger a Telegram alert per market. The per-market thresholds
    in MARKET_NOTIFY_THRESHOLDS keep low-signal markets (puck-line,
    total) from spamming the channel.
    """
    # Late imports so module load doesn't pull in numpy / sqlalchemy /
    # the predictor package unless the script is actually being run.
    from services.cache_service import CacheService  # type: ignore
    from services.prediction_service import get_model_version, load_models_into_process, tasks_for_sport  # type: ignore

    models = load_models_into_process()
    cache = CacheService()

    nhl_tasks = tasks_for_sport("nhl")
    if not nhl_tasks:
        logger.error("No NHL tasks registered in TASKS dict — nothing to do")
        return {"predicted": 0, "match_count": 0}
    logger.info(
        "NHL tasks to predict per match: %s",
        [t.market for t in nhl_tasks],
    )

    # Quick gate: if any NHL ensemble isn't loaded, log up front so the
    # operator notices missing artifacts before we burn time iterating
    # over hundreds of matches.
    missing = [f"{t.sport}:{t.market}" for t in nhl_tasks if f"{t.sport}:{t.market}" not in models]
    if missing:
        logger.warning(
            "NHL tasks missing from loaded models: %s — those rows will be skipped",
            missing,
        )

    upcoming = list_upcoming_nhl(database_url, days)
    if not upcoming:
        logger.info("No upcoming NHL matches with fresh features in the next %d days", days)
        return {"predicted": 0, "match_count": 0}
    logger.info("Found %d upcoming NHL matches to score", len(upcoming))

    predicted = 0
    skipped = 0
    notified = 0
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for m in upcoming:
                features = fetch_features(cur, m["match_id"])
                if not features:
                    # Belt-and-braces: list_upcoming filtered to matches
                    # WITH features so this should never fire, but the
                    # TTL race between query and fetch is theoretically
                    # possible.
                    skipped += 1
                    continue

                # Mirror unprefixed keys to their feature__-prefixed form
                # (mirrors _predict_one's bridge — trained models' feature
                # lists contain both raw and prefixed names).
                X_dict: dict = {}
                for k, v in features.items():
                    X_dict[k] = v
                    prefixed = k if k.startswith("feature__") else f"feature__{k}"
                    if prefixed not in X_dict:
                        X_dict[prefixed] = v
                # home_team / away_team are needed by hockey-Poisson and
                # don't live in the features JSON.
                X_dict["home_team"] = m["home_team"]
                X_dict["away_team"] = m["away_team"]

                headline_cached = False
                for task in nhl_tasks:
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

                    # Cache the moneyline as the headline so the /api/v1/predictions/
                    # endpoint can serve it without re-running the ensemble.
                    # Other markets are now exposed via Phase 4b's
                    # /predictions/match/{match_id} endpoint; this cache stays
                    # moneyline-only because it backs the legacy single-
                    # prediction code path.
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

                    # Telegram alert (Phase 4d). One message per (match,
                    # market) when confidence clears the per-market gate.
                    # Unknown markets fall back to 0.70 — strict enough
                    # to avoid noise if a new market lands before we
                    # update MARKET_NOTIFY_THRESHOLDS.
                    if notify:
                        threshold = MARKET_NOTIFY_THRESHOLDS.get(task.market, 0.70)
                        if send_telegram_nhl(
                            league_name=m["league_name"],
                            home_team=m["home_team"],
                            away_team=m["away_team"],
                            match_date=m["match_date"],
                            market=task.market,
                            predicted_outcome=predicted_outcome,
                            confidence=confidence,
                            probabilities=probabilities,
                            threshold=threshold,
                        ):
                            notified += 1
            conn.commit()

    logger.info(
        "Wrote %d NHL prediction rows across %d matches (%d skipped, %d notified)",
        predicted,
        len(upcoming),
        skipped,
        notified,
    )
    return {
        "predicted": predicted,
        "match_count": len(upcoming),
        "skipped": skipped,
        "notified": notified,
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
    # Make services.* importable. The same path manipulation is done
    # by the soccer precompute_predictions; Dockerfile.api puts the
    # api src on PYTHONPATH for the running api container but a one-off
    # docker compose exec call still needs the bootstrap.
    for p in ("/app/services/api/src", "/app/services/ml-models/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    counts = run(args.database_url, args.days, notify=not args.no_notify)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
