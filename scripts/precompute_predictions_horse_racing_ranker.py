"""Score upcoming races with the trained LambdaMART ranker.

Parallel to scripts/precompute_predictions_horse_racing.py (the
market-consensus baseline). Writes to the SAME race_predictions
table but under a distinct model_name so:

  - the consensus baseline keeps producing predictions for every
    race (including races without features_cache rows or with too
    little data for the ranker),
  - the ranker's predictions sit alongside the consensus for races
    where features are available,
  - the recommendation engine can pick the ranker's view when
    present and fall back to consensus when not (see
    generate_recommendations_horse_racing.MODEL_PRECEDENCE).

Architecture (changed at the 13k-race corpus mark): the precompute
LOADS a saved model artefact from --model-dir (default
/app/models/horse_racing_ranker_v1/). Training itself runs
out-of-band via scripts/train_horse_racing_win.py — fits + tunes
temperature + saves model.bin + feature_names.json + metadata.json.
Per-tick re-training OOMed the api container at 152k-row scale;
the load-only path is bounded by the scoring window (~few hundred
races) regardless of corpus size.

Model: lightgbm_ranker_v1 / 1.0.0. Re-run the training script when
the corpus grows materially or when you change the feature set;
otherwise the same artefact serves indefinitely.

Usage (inside the api container):
    python /app/scripts/precompute_predictions_horse_racing_ranker.py
    python /app/scripts/precompute_predictions_horse_racing_ranker.py --days 3
    python /app/scripts/precompute_predictions_horse_racing_ranker.py --race-ids id1,id2
    python /app/scripts/precompute_predictions_horse_racing_ranker.py \
        --model-dir /tmp/hr_ranker_v2   # smoke a fresh training run

Bootstrap:
    python /app/scripts/train_horse_racing_win.py \
        --split-date 2026-05-15 \
        --output-dir /app/models/horse_racing_ranker_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ml-models" / "src"))
sys.path.insert(0, "/app/services/ml-models/src")

from predictors.horse_racing_ranker import HorseRacingRanker  # noqa: E402
from utils.horse_racing_data import group_array  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_predictions_horse_racing_ranker")

MODEL_NAME = "lightgbm_ranker_v1"
MODEL_VERSION = "1.0.0"
DEFAULT_MODEL_DIR = Path("/app/models/horse_racing_ranker_v1")


# ── Scoring SQL ────────────────────────────────────────────────────


SCORING_QUERY = """
    SELECT
        r.id::text          AS race_id,
        r.race_date,
        r.track_name,
        r.race_number,
        e.id::text          AS entrant_id,
        e.program_number,
        e.morning_line_odds,
        e.starting_price,
        e.finish_position,
        e.scratched,
        e.disqualified,
        rp.confidence       AS consensus_prob,
        rfc.features        AS features_blob
    FROM races r
    JOIN race_entrants e ON e.race_id = r.id
    LEFT JOIN race_predictions rp
        ON rp.entrant_id = e.id
       AND rp.model_name = 'market_consensus_v1'
       AND rp.prediction_type = 'win'
    LEFT JOIN race_features_cache rfc
        ON rfc.race_id = r.id
       AND rfc.feature_set = 'horse_racing_baseline'
    WHERE r.id::text = ANY(%s)
      AND NOT e.scratched
    ORDER BY r.race_date ASC, r.id::text, e.program_number ASC NULLS LAST
"""


# ── DB I/O ────────────────────────────────────────────────────────


def list_target_races(cur, days: int, race_ids: Optional[list[str]]) -> list[str]:
    """Scheduled races in the next N days. --race-ids overrides
    the lookahead window. Skips races without a features_cache row
    — the ranker requires features to be available."""
    if race_ids:
        cur.execute(
            """
            SELECT id::text AS id FROM races
            WHERE id::text = ANY(%s)
              AND EXISTS (
                  SELECT 1 FROM race_features_cache fc
                  WHERE fc.race_id = races.id
                    AND fc.feature_set = 'horse_racing_baseline'
              )
            """,
            (race_ids,),
        )
        return [r["id"] for r in cur.fetchall()]

    cur.execute(
        """
        SELECT r.id::text AS id FROM races r
        JOIN race_features_cache fc
          ON fc.race_id = r.id
         AND fc.feature_set = 'horse_racing_baseline'
        WHERE r.status = 'scheduled'
          AND r.race_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
        ORDER BY r.race_date ASC
        """,
        (str(days),),
    )
    return [r["id"] for r in cur.fetchall()]


def store_prediction(
    cur,
    *,
    race_id: str,
    entrant_id: str,
    confidence: float,
    field_probs: dict[str, float],
    temperature: float,
) -> None:
    """Idempotent upsert on (race_id, entrant_id, model_name,
    model_version, prediction_type). Metadata records the calibration
    temperature so we know which softmax shaped the probs."""
    cur.execute(
        """
        INSERT INTO race_predictions
            (race_id, entrant_id, model_name, model_version,
             prediction_type, confidence, probabilities, metadata)
        VALUES (%s, %s, %s, %s, 'win', %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (race_id, entrant_id, model_name, model_version, prediction_type)
        DO UPDATE SET
            confidence = EXCLUDED.confidence,
            probabilities = EXCLUDED.probabilities,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """,
        (
            race_id,
            entrant_id,
            MODEL_NAME,
            MODEL_VERSION,
            confidence,
            json.dumps(field_probs),
            json.dumps({"method": "lambdamart_softmax", "temperature": temperature}),
        ),
    )


# ── Prediction frame helpers ───────────────────────────────────────


def load_scoring_frame(database_url: str, race_ids: list[str]) -> pd.DataFrame:
    """Load + prepare per-(race, entrant) rows for the target races.
    Reuses prepare_training_frame so the JSONB-flatten + consensus
    implied prob columns line up with the training shape — load-
    bearing for feature alignment between fit and predict."""
    from sqlalchemy import create_engine
    from utils.horse_racing_data import prepare_training_frame

    engine = create_engine(database_url)
    try:
        raw = pd.read_sql(SCORING_QUERY, engine, params=(race_ids,))
    finally:
        engine.dispose()
    if raw.empty:
        return raw.copy()
    return prepare_training_frame(raw)


def predict_for_races(
    model: HorseRacingRanker,
    scoring_frame: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Run the ranker over each target race; return a dict of
    {race_id: {entrant_id: prob}} where each race's probs sum to 1.0
    after the per-race softmax."""
    if scoring_frame.empty:
        return {}
    # Reuse the training-shape feature_cols so columns line up. Any
    # column the model expects but the scoring frame lacks gets a
    # column of NaN (the model handles NaN natively via LightGBM's
    # missing-direction inference).
    X = scoring_frame.reindex(columns=feature_cols)
    groups = group_array(scoring_frame)

    race_probs = model.predict_probabilities(X, groups)

    out: dict[str, dict[str, float]] = {}
    cursor = 0
    for race_probs_arr, size in zip(race_probs, groups):
        chunk = scoring_frame.iloc[cursor : cursor + int(size)]
        cursor += int(size)
        race_id = str(chunk["race_id"].iloc[0])
        out[race_id] = {str(entrant_id): float(prob) for entrant_id, prob in zip(chunk["entrant_id"], race_probs_arr)}
    return out


# ── Orchestration ─────────────────────────────────────────────────


def run(
    database_url: str,
    days: int,
    race_ids: Optional[list[str]],
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict:
    counts = {"races": 0, "predictions": 0, "skipped_races": 0}

    # 1. Load the saved model. The training script (run out-of-band)
    #    produces these artefacts; we never re-train here because the
    #    full-corpus train OOM'd the api container at 152k-row scale.
    try:
        model = HorseRacingRanker.load(model_dir)
    except FileNotFoundError as e:
        logger.error(
            "No saved ranker at %s: %s. Bootstrap via " "scripts/train_horse_racing_win.py --output-dir %s",
            model_dir,
            e,
            model_dir,
        )
        return counts

    feature_cols = list(model.feature_names)
    logger.info(
        "Loaded ranker from %s (temperature %.3f, %d features)",
        model_dir,
        model.temperature,
        len(feature_cols),
    )

    # 2. Find target races + score them.
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            targets = list_target_races(cur, days, race_ids)
            if not targets:
                logger.info("No upcoming races needing ranker predictions")
                return counts
            logger.info("Scoring %d upcoming races", len(targets))

            scoring_frame = load_scoring_frame(database_url, targets)
            if scoring_frame.empty:
                logger.info("No scoring rows returned for %d targets", len(targets))
                return counts

            preds = predict_for_races(model, scoring_frame, feature_cols)

            # 3. Persist.
            for race_id, race_probs in preds.items():
                if not race_probs:
                    counts["skipped_races"] += 1
                    continue
                counts["races"] += 1
                for entrant_id, prob in race_probs.items():
                    store_prediction(
                        cur,
                        race_id=race_id,
                        entrant_id=entrant_id,
                        confidence=prob,
                        field_probs=race_probs,
                        temperature=model.temperature,
                    )
                    counts["predictions"] += 1
            conn.commit()

    logger.info(
        "Wrote %d ranker predictions across %d races (%d skipped)",
        counts["predictions"],
        counts["races"],
        counts["skipped_races"],
    )
    return counts


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=2, help="Lookahead window in days (default 2).")
    p.add_argument("--race-ids", help="Comma-separated UUID list to score specific races.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=("Directory containing model.bin + feature_names.json + metadata.json. " f"Default: {DEFAULT_MODEL_DIR}."),
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    race_ids = [s.strip() for s in args.race_ids.split(",") if s.strip()] if args.race_ids else None
    run(args.database_url, args.days, race_ids, model_dir=args.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
