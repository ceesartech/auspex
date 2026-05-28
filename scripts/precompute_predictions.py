"""Precompute predictions for upcoming matches and warm Redis.

For every scheduled match in the next N days that has features_cache
populated, run the ensemble model, store the prediction in:
  - the `predictions` table (history + auditability)
  - the `prediction:<match_id>:...` Redis key (low-latency serve)

Optionally fire a Telegram message for any prediction whose confidence
exceeds --notify-threshold (default 0.65).

Usage (inside the api container):
    python /app/scripts/precompute_predictions.py
    python /app/scripts/precompute_predictions.py --notify-threshold 0.7
    python /app/scripts/precompute_predictions.py --no-notify
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_predictions")

# Make the api src importable so we reuse PredictionService + caching.
sys.path.insert(0, "/app/services/api/src")


def list_upcoming(database_url: str, days: int) -> list[dict]:
    """Matches in the next N days with features available, no fresh
    prediction. The query also pulls human-readable team names so the
    Telegram payload is useful."""
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT m.id::text AS match_id,
                       m.match_date,
                       ht.name AS home_team,
                       at.name AS away_team,
                       l.name AS league
                FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                JOIN leagues l ON l.id = m.league_id
                JOIN features_cache f
                  ON f.match_id = m.id AND f.expires_at > NOW()
                WHERE m.status = 'scheduled'
                  AND m.match_date BETWEEN NOW() AND NOW() + INTERVAL %s
                ORDER BY m.match_date ASC
                """,
                (f"{days} days",),
            )
            return list(cur.fetchall())


def send_telegram(prediction: dict, match: dict, threshold: float) -> bool:
    """Send a Telegram message if confidence exceeds threshold.

    Reads bot token + chat id from env. Returns True on send, False
    otherwise (missing config / network error / below threshold).
    """
    if prediction.get("confidence", 0.0) < threshold:
        return False
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if os.environ.get("ENABLE_TELEGRAM_NOTIFICATIONS", "false").lower() != "true":
        return False
    if not token or not chat_id:
        logger.warning("Telegram not configured — skipping notification")
        return False

    text = (
        f"⚽ <b>{match['league']}</b>\n"
        f"{match['home_team']} vs {match['away_team']}\n"
        f"\U0001F551 {match['match_date'].strftime('%Y-%m-%d %H:%M %Z')}\n\n"
        f"Predicted: <b>{prediction['predicted_label']}</b>\n"
        f"Confidence: <b>{prediction['confidence']:.1%}</b>\n"
        f"Probabilities: " + ", ".join(f"{k} {v:.1%}" for k, v in (prediction.get("probabilities") or {}).items())
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


def store_prediction(cur, match_id: str, prediction: dict, model_version: str) -> None:
    cur.execute(
        """
        INSERT INTO predictions
        (match_id, model_name, model_version, prediction_type,
         predicted_outcome, confidence, probabilities)
        VALUES (%(match_id)s, 'ensemble', %(model_version)s, 'match_result',
                %(predicted)s, %(confidence)s, %(probs)s::jsonb)
        ON CONFLICT (match_id, model_name, model_version) DO UPDATE
            SET predicted_outcome = EXCLUDED.predicted_outcome,
                confidence = EXCLUDED.confidence,
                probabilities = EXCLUDED.probabilities,
                updated_at = NOW()
        """,
        {
            "match_id": match_id,
            "model_version": model_version,
            "predicted": prediction["predicted_label"],
            "confidence": prediction["confidence"],
            # allow_nan=False raises a ValueError locally instead of
            # silently producing "NaN" tokens that Postgres rejects with
            # "invalid input syntax for type json". By the time we reach
            # here we've already validated the prediction is finite.
            "probs": json.dumps(prediction.get("probabilities") or {}, allow_nan=False),
        },
    )


def run(database_url: str, days: int, notify_threshold: float, notify: bool) -> dict:
    from services.cache_service import CacheService  # type: ignore
    from services.prediction_service import (  # type: ignore
        PredictionService,
        load_models_into_process,
        get_model_version,
    )

    load_models_into_process()
    model_version = get_model_version()
    cache = CacheService()

    upcoming = list_upcoming(database_url, days)
    if not upcoming:
        logger.info("No upcoming matches with features in the next %d days", days)
        return {"predicted": 0, "notified": 0}

    predicted = 0
    notified = 0
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for m in upcoming:
                # Use the request-path service so behavior is identical to
                # what /api/v1/predictions/ produces.
                svc = PredictionService(db=None)
                # PredictionService needs a SQLAlchemy session for DB ops;
                # we sidestep that by calling lower-level helpers directly.
                # For simplicity, hit features_cache + run the model ourselves:
                cur.execute(
                    "SELECT features FROM features_cache "
                    "WHERE match_id = %s AND expires_at > NOW() "
                    "ORDER BY computed_at DESC LIMIT 1",
                    (m["match_id"],),
                )
                row = cur.fetchone()
                if not row:
                    continue
                features = row["features"] or {}

                models = svc.models
                ensemble = models.get("ensemble")
                if not ensemble:
                    logger.error("No ensemble model loaded; aborting")
                    return {"predicted": predicted, "notified": notified}

                try:
                    import numpy as np
                    import pandas as pd

                    proba = ensemble.predict_proba(pd.DataFrame([features]))[0]
                    # When every feature is None/NaN (e.g. no live odds for
                    # this match yet, no historical rolling form for the
                    # team), the median-fill in each base model produces
                    # NaN medians and the ensemble emits NaN probabilities.
                    # Skip cleanly — predicting nonsense and storing it
                    # would corrupt downstream analytics.
                    if not np.all(np.isfinite(proba)):
                        logger.info(
                            "Skipping %s: features insufficient (predict produced NaN)",
                            m["match_id"],
                        )
                        continue
                    labels = ["home", "draw", "away"]
                    idx = int(proba.argmax())
                    pred = {
                        "predicted_label": labels[idx],
                        "confidence": float(proba[idx]),
                        "probabilities": {labels[i]: float(proba[i]) for i in range(3)},
                    }
                except Exception as e:
                    logger.warning("Predict failed for %s: %s", m["match_id"], e)
                    continue

                store_prediction(cur, m["match_id"], pred, model_version)
                cache.set_prediction(
                    m["match_id"],
                    {
                        "predicted_outcome": pred["predicted_label"],
                        "confidence": pred["confidence"],
                        "probabilities": pred.get("probabilities", {}),
                        "model_version": model_version,
                    },
                    model_version=model_version,
                )
                predicted += 1

                if notify and send_telegram(pred, m, notify_threshold):
                    notified += 1

            conn.commit()
    logger.info("Stored %d predictions, sent %d Telegram alerts", predicted, notified)
    return {"predicted": predicted, "notified": notified}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7)
    p.add_argument(
        "--notify-threshold",
        type=float,
        default=0.65,
        help="Min confidence to trigger a Telegram alert (default: 0.65).",
    )
    p.add_argument("--no-notify", action="store_true")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.notify_threshold, not args.no_notify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
