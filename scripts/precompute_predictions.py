"""Precompute predictions for upcoming matches and warm Redis.

For every scheduled match in the next N days that has features_cache
populated, run the ensemble model, store the prediction in:
  - the `predictions` table (history + auditability)
  - the `prediction:<match_id>:...` Redis key (low-latency serve)

Picks that clear --notify-threshold are bundled into a single Telegram
digest message at the end of the run. Previously this fired one
message per pick — for a 7-day window that meant a flurry of alerts
the user had to scroll through individually.

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

import psycopg2
from psycopg2.extras import RealDictCursor

# Make the api src + shared scripts package importable so we reuse
# PredictionService + the digest helper.
sys.path.insert(0, "/app/services/api/src")
sys.path.insert(0, os.path.dirname(__file__))

from telegram_notify import Alert, send_telegram_digest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_predictions")


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


def soccer_alert(prediction: dict, match: dict) -> Alert:
    """Translate a soccer prediction dict into the shared Alert shape.
    The market label is hardcoded to '1X2' because precompute_predictions
    produces match_result picks only; market-derivation rows are written
    DB-only and don't get bundled into the digest."""
    return Alert(
        sport="soccer",
        league_name=match["league"],
        home_team=match["home_team"],
        away_team=match["away_team"],
        match_date=match["match_date"],
        market_label="1X2",
        predicted_outcome=prediction["predicted_label"],
        confidence=float(prediction.get("confidence", 0.0)),
        probabilities={k: float(v) for k, v in (prediction.get("probabilities") or {}).items()},
    )


def store_prediction(cur, match_id: str, prediction: dict, model_version: str) -> None:
    cur.execute(
        """
        INSERT INTO predictions
        (match_id, model_name, model_version, prediction_type,
         predicted_outcome, confidence, probabilities)
        VALUES (%(match_id)s, 'ensemble', %(model_version)s, 'match_result',
                %(predicted)s, %(confidence)s, %(probs)s::jsonb)
        ON CONFLICT (match_id, model_name, model_version, prediction_type) DO UPDATE
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


# Engine market_type -> DB predictions.prediction_type. The engine's "1x2"
# market is intentionally absent: the ensemble `match_result` row already
# carries the (reconciled-against) home/draw/away prediction, so re-storing it
# would be redundant.
MARKET_PREDICTION_TYPE: dict[str, str] = {
    "over_under": "over_under",
    "btts": "btts",
    "correct_score": "correct_score",
    "double_chance": "double_chance",
    "draw_no_bet": "draw_no_bet",
    "asian_handicap": "asian_handicap",
    "team_total": "team_total",
    "clean_sheet": "clean_sheet",
    "win_to_nil": "win_to_nil",
    "odd_even": "odd_even",
    "winning_margin": "winning_margin",
    "total_goals": "total_goals",
    "result_btts": "result_btts",
    "result_over_under": "result_over_under",
}


def store_market_predictions(cur, match_id: str, markets: dict, model_version: str) -> int:
    """Upsert one `predictions` row per derived market_type.

    `markets` is the {market_type: {selection: prob}} dict from the derivation
    engine. Each row stores the full selection dict in `probabilities`
    (JSONB); `predicted_outcome`/`confidence` are the argmax selection and its
    probability. Returns the number of rows written.

    Multi-line markets (over_under, asian_handicap, team_total) keep every
    line's selections in the single JSONB blob — consumers (the recommendation
    generator) read the full dict, so the row-level argmax is just a headline.
    asian_handicap `*_push` keys are kept in `probabilities` (the recommender
    needs them for push-aware EV) but excluded from the argmax since a push
    isn't a backable outcome.
    """
    written = 0
    for engine_mt, probs in markets.items():
        ptype = MARKET_PREDICTION_TYPE.get(engine_mt)
        if ptype is None or not probs:
            continue
        selectable = {k: v for k, v in probs.items() if not k.endswith("_push")}
        if not selectable:
            continue
        top_sel = max(selectable, key=selectable.get)
        cur.execute(
            """
            INSERT INTO predictions
            (match_id, model_name, model_version, prediction_type,
             predicted_outcome, confidence, probabilities)
            VALUES (%(match_id)s, 'ensemble', %(model_version)s, %(ptype)s,
                    %(predicted)s, %(confidence)s, %(probs)s::jsonb)
            ON CONFLICT (match_id, model_name, model_version, prediction_type) DO UPDATE
                SET predicted_outcome = EXCLUDED.predicted_outcome,
                    confidence = EXCLUDED.confidence,
                    probabilities = EXCLUDED.probabilities,
                    updated_at = NOW()
            """,
            {
                "match_id": match_id,
                "model_version": model_version,
                "ptype": ptype,
                "predicted": top_sel,
                "confidence": float(selectable[top_sel]),
                "probs": json.dumps(probs, allow_nan=False),
            },
        )
        written += 1
    return written


def compute_feature_medians(database_url: str) -> dict[str, float]:
    """Compute per-column median across every features_cache row.

    Used as a fallback to fill missing/None values before a model
    predicts on a single row. The base models do fillna(X.median()) at
    predict time, which yields NaN when X is a single row and the
    column is None — single-row median of None is None, NaN
    propagates, the ensemble emits NaN probabilities, and the match
    gets skipped. By pre-filling None values with a corpus-wide
    median, we avoid the trap and still produce a sensible prediction
    (just a more conservative one when some features are missing).
    """
    medians: dict[str, float] = {}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT features FROM features_cache WHERE features IS NOT NULL")
            rows = cur.fetchall()
    if not rows:
        return medians
    # Aggregate values per key across all rows.
    by_key: dict[str, list[float]] = {}
    for r in rows:
        for k, v in (r["features"] or {}).items():
            if isinstance(v, (int, float)) and v is not None:
                # JSONB numbers come back as int/float; ignore strings/None
                by_key.setdefault(k, []).append(float(v))
    import statistics

    for k, vals in by_key.items():
        if vals:
            medians[k] = float(statistics.median(vals))
    return medians


def run(database_url: str, days: int, notify_threshold: float, notify: bool) -> dict:
    from services.cache_service import CacheService  # type: ignore
    from services.prediction_service import (  # type: ignore
        PredictionService,
        get_model_version,
        load_models_into_process,
    )

    load_models_into_process()
    model_version = get_model_version()
    cache = CacheService()

    # Market-derivation engine lives in ml-models/src, whose path is added to
    # sys.path by load_models_into_process(). If it's unavailable we still
    # produce the 1x2 match_result prediction below — derivation just no-ops.
    try:
        from predictors.market_derivation import derive_from_lambdas  # type: ignore
    except Exception as e:  # pragma: no cover - defensive import guard
        derive_from_lambdas = None
        logger.warning("market_derivation unavailable (%s); storing match_result only", e)

    feature_medians = compute_feature_medians(database_url)
    logger.info("Loaded %d feature-medians for NaN fallback", len(feature_medians))

    upcoming = list_upcoming(database_url, days)
    if not upcoming:
        logger.info("No upcoming matches with features in the next %d days", days)
        return {"predicted": 0, "market_rows": 0, "alerts": 0, "telegram_messages": 0}

    predicted = 0
    market_rows = 0
    alerts: list[Alert] = []
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
                    return {
                        "predicted": predicted,
                        "market_rows": market_rows,
                        "alerts": len(alerts),
                        "telegram_messages": 0,
                    }

                try:
                    import numpy as np
                    import pandas as pd

                    # Fill missing/None values with the corpus-wide median
                    # for that feature. Without this, a single None in the
                    # row poisons the base models' per-row median-fill and
                    # produces NaN predictions. With it, we degrade
                    # gracefully — the model gets a sensible neutral value
                    # for the missing column and still emits a finite prob.
                    filled = {
                        k: (v if isinstance(v, (int, float)) and v is not None else feature_medians.get(k))
                        for k, v in features.items()
                    }

                    proba = ensemble.predict_proba(pd.DataFrame([filled]))[0]
                    if not np.all(np.isfinite(proba)):
                        logger.info(
                            "Skipping %s: features insufficient even after median fill",
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

                # Derive the wider market catalog from the Dixon-Coles
                # scoreline, reconciled to the ensemble 1x2 so every market's
                # home/draw/away split matches `match_result`. Best-effort: any
                # failure leaves the match_result prediction intact.
                dc = ensemble.models.get("dixon_coles") if derive_from_lambdas else None
                if dc is not None and getattr(dc, "is_fitted", False):
                    try:
                        h_lam, a_lam = dc.lambdas_for_match(m["home_team"], m["away_team"])
                        markets = derive_from_lambdas(
                            h_lam,
                            a_lam,
                            float(getattr(dc, "rho", 0.0) or 0.0),
                            target_1x2=(float(proba[0]), float(proba[1]), float(proba[2])),
                        )
                        market_rows += store_market_predictions(cur, m["match_id"], markets, model_version)
                    except Exception as e:
                        logger.warning("Market derivation failed for %s: %s", m["match_id"], e)

                # Accumulate for the end-of-run digest instead of
                # firing one message per pick.
                if notify and pred["confidence"] >= notify_threshold:
                    alerts.append(soccer_alert(pred, m))

            conn.commit()

    # One bundled Telegram message instead of N per-pick alerts. The
    # digest helper handles the disabled / unconfigured / network-fail
    # gates and chunks if the digest exceeds Telegram's 4096-char limit.
    messages_sent = send_telegram_digest(
        alerts,
        header=f"Auspex soccer picks · {len(alerts)} high-confidence",
    )
    logger.info(
        "Stored %d match-result predictions (+%d market rows), %d picks digested into %d Telegram message(s)",
        predicted,
        market_rows,
        len(alerts),
        messages_sent,
    )
    return {
        "predicted": predicted,
        "market_rows": market_rows,
        "alerts": len(alerts),
        "telegram_messages": messages_sent,
    }


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
