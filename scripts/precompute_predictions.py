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

from telegram_notify import Alert, enqueue_alerts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_predictions")


def list_upcoming(database_url: str, days: int) -> list[dict]:
    """Soccer matches in the next N days with features available.

    The l.sport='soccer' filter is load-bearing: without it, this
    script picks up NHL + NBA scheduled games and runs the soccer
    ensemble on them, producing nonsense match_result predictions
    and ~15 Dixon-Coles-derived market rows per non-soccer match.
    None of those rows ever get served (the API filters by sport
    via TaskSpec) but they bloat the predictions table and waste
    compute cycles. The features_cache join already pinned to
    feature_set is the second guard."""
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
                  ON f.match_id = m.id
                 AND f.feature_set = 'baseline'
                 AND f.expires_at > NOW()
                WHERE m.status = 'scheduled'
                  AND l.sport = 'soccer'
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
    # Halftime markets (migration 014, model artifact
    # dixon_coles_ht_soccer). Three markets derived directly from
    # the HT scoreline matrix — no reconcile step (the FT 1x2 doesn't
    # constrain HT 1x2 outcomes).
    "match_result_ht": "match_result_ht",
    "over_under_ht": "over_under_ht",
    "btts_ht": "btts_ht",
    # Halftime/fulltime joint double-result market (migration 015,
    # second-half model artifact dixon_coles_h2_soccer). Single
    # market with 9 selections.
    "ht_ft_double_result": "ht_ft_double_result",
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
        # 'other' (correct_score's aggregated tail bucket) is excluded for
        # the same reason as '_push': it isn't a backable single outcome.
        # Left in the argmax it wins on essentially every row (the tail sum
        # beats any one scoreline), which made predicted_outcome 'other'
        # permanently ungradable — 0/1124 correct_score grades ever matched.
        selectable = {k: v for k, v in probs.items() if not k.endswith("_push") and k != "other"}
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


def _load_dixon_coles_artifact(path_str: str, label: str):
    """Shared helper used by HT + 2H loaders. Returns the fitted
    DixonColesPredictor, or None when the artifact is missing.
    Logging matches the previous _load_halftime_dixon_coles shape
    so behaviour around missing artifacts is unchanged.
    """
    from pathlib import Path

    path = Path(path_str)
    if not path.exists():
        logger.info("%s Dixon-Coles artifact not found at %s; %s markets disabled.", label, path, label)
        return None
    try:
        from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore
        from predictors.poisson_models import DixonColesPredictor  # type: ignore
    except Exception as e:
        logger.warning("%s model imports failed; %s markets disabled: %s", label, label, e)
        return None
    model = DixonColesPredictor(DIXON_COLES_CONFIG)
    try:
        model.load(str(path))
    except Exception as e:
        logger.warning("%s model load failed at %s: %s", label, path, e)
        return None
    return model


def _load_halftime_dixon_coles():
    """Load the halftime Dixon-Coles model artifact if present."""
    return _load_dixon_coles_artifact(
        "/app/models/production/dixon_coles_ht_soccer/1.0.0/model.bin",
        "HT",
    )


def _load_second_half_dixon_coles():
    """Load the second-half Dixon-Coles model artifact if present.
    Together with the HT model this enables ht_ft_double_result
    derivation."""
    return _load_dixon_coles_artifact(
        "/app/models/production/dixon_coles_h2_soccer/1.0.0/model.bin",
        "2H",
    )


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

    # Halftime Dixon-Coles is an OPTIONAL standalone artifact. The
    # FT pipeline keeps working when it's missing; HT derivation just
    # no-ops. Loaded once at startup so the per-match loop only does
    # arithmetic.
    ht_dc_model = _load_halftime_dixon_coles()
    second_half_dc_model = _load_second_half_dixon_coles()
    if ht_dc_model is not None:
        try:
            from predictors.market_derivation import (  # type: ignore
                MAX_GOALS_DERIVE,
                build_dc_matrix,
                derive_markets,
                derive_soccer_htft_markets,
            )

            ht_derive_ready = True
        except Exception as ht_e:
            logger.warning("HT derivation imports failed; skipping HT markets: %s", ht_e)
            ht_derive_ready = False
            derive_soccer_htft_markets = None
    else:
        ht_derive_ready = False
        derive_soccer_htft_markets = None
    # HT/FT joint requires BOTH models. Either missing → no-op.
    htft_derive_ready = ht_derive_ready and second_half_dc_model is not None and derive_soccer_htft_markets is not None

    feature_medians = compute_feature_medians(database_url)
    logger.info("Loaded %d feature-medians for NaN fallback", len(feature_medians))

    upcoming = list_upcoming(database_url, days)
    if not upcoming:
        logger.info("No upcoming matches with features in the next %d days", days)
        return {"predicted": 0, "market_rows": 0, "alerts_queued": 0, "queue_depth": 0}

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
                ensemble = models.get("soccer:match_result")
                if not ensemble:
                    logger.error("No soccer:match_result ensemble loaded; aborting")
                    return {
                        "predicted": predicted,
                        "market_rows": market_rows,
                        "alerts_queued": len(alerts),
                        "queue_depth": 0,
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

                    # Mirror plain keys into feature__-prefixed duplicates —
                    # the bridge every other sport's precompute already has
                    # (see precompute_predictions_nfl.py). Training flattens
                    # the features_cache JSONB into feature__* columns, so
                    # the trained models' feature_names use that prefix;
                    # without the mirror, XGB/LGBM/NN KeyError and only
                    # Poisson/DC's global prior survives (the July-2026
                    # audit §1.1 constant-prior incident).
                    for k in list(filled.keys()):
                        prefixed = k if k.startswith("feature__") else f"feature__{k}"
                        if prefixed not in filled:
                            filled[prefixed] = filled[k]

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
                # scoreline, reconciled to the ensemble 1x2 so every
                # market's home/draw/away split matches `match_result`.
                # Best-effort: any failure here leaves the
                # match_result prediction intact.
                dc = ensemble.models.get("dixon_coles_soccer_match_result") if derive_from_lambdas else None
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

                # Halftime markets — separate model, separate scoreline
                # matrix. Best-effort: failure leaves FT predictions intact.
                ht_P = None  # captured for the HT/FT joint block below
                if ht_derive_ready and ht_dc_model is not None:
                    try:
                        ht_h_lam, ht_a_lam = ht_dc_model.lambdas_for_match(
                            m["home_team"],
                            m["away_team"],
                        )
                        ht_P = build_dc_matrix(
                            ht_h_lam,
                            ht_a_lam,
                            float(getattr(ht_dc_model, "rho", 0.0) or 0.0),
                            max_goals=MAX_GOALS_DERIVE,
                        )
                        ht_markets = derive_markets("soccer_halftime", ht_P)
                        market_rows += store_market_predictions(
                            cur,
                            m["match_id"],
                            ht_markets,
                            model_version,
                        )
                    except Exception as e:
                        logger.warning("HT market derivation failed for %s: %s", m["match_id"], e)
                        ht_P = None

                # HT/FT joint double-result market — convolves the HT
                # scoreline matrix with a SECOND-HALF (FT - HT) Dixon-
                # Coles matrix and aggregates joint mass into the 9
                # (HT outcome × FT outcome) buckets. Best-effort: any
                # failure (incl. ht_P unavailable from the block above
                # or 2H model missing) leaves the other markets intact.
                if htft_derive_ready and ht_P is not None:
                    try:
                        h2_h_lam, h2_a_lam = second_half_dc_model.lambdas_for_match(
                            m["home_team"],
                            m["away_team"],
                        )
                        h2_P = build_dc_matrix(
                            h2_h_lam,
                            h2_a_lam,
                            float(getattr(second_half_dc_model, "rho", 0.0) or 0.0),
                            max_goals=MAX_GOALS_DERIVE,
                        )
                        htft_markets = derive_soccer_htft_markets(ht_P, h2_P)
                        market_rows += store_market_predictions(
                            cur,
                            m["match_id"],
                            htft_markets,
                            model_version,
                        )
                    except Exception as e:
                        logger.warning(
                            "HT/FT joint derivation failed for %s: %s",
                            m["match_id"],
                            e,
                        )

                # Accumulate for the end-of-run digest instead of
                # firing one message per pick.
                if notify and pred["confidence"] >= notify_threshold:
                    alerts.append(soccer_alert(pred, m))

            conn.commit()

    # Push picks onto the shared Redis queue. The DAG's downstream
    # send_pipeline_digest task drains both sports' queues and sends
    # ONE combined Telegram message — this script does NOT send
    # directly anymore, because doing so would split soccer + NHL
    # across two messages.
    queue_depth = enqueue_alerts(alerts) if alerts else 0
    logger.info(
        "Stored %d match-result predictions (+%d market rows); queued %d picks (queue depth now %d)",
        predicted,
        market_rows,
        len(alerts),
        queue_depth,
    )
    return {
        "predicted": predicted,
        "market_rows": market_rows,
        "alerts_queued": len(alerts),
        "queue_depth": queue_depth,
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
