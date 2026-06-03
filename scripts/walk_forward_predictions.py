"""Walk-forward retrospective predictions for honest backtesting.

The backtest engine (scripts/backtest_recommendations.py) needs
out-of-sample predictions to chew on. The production predictions
table only has rows for matches that were SCHEDULED during the
system's operational lifetime — historical matches loaded via the
load_*_historical scripts have status='finished' and no associated
prediction.

This script generates those missing predictions in an honest,
non-overfit way:

  1. For each --bundle (one of soccer_match_result, nhl_moneyline,
     nhl_regulation, nhl_puck_line, nhl_total, nba_moneyline,
     nba_spread, nba_total, nfl_moneyline, nfl_spread, nfl_total,
     tennis_moneyline):
       a. Load the bundle's training frame from the live DB.
       b. Filter to matches with match_date < --split-date.
       c. Persist the filtered frame to CSV.
       d. Subprocess into training.train_all_models --input-csv
          --output-dir <snapshots-dir>/<bundle> so the trained
          artifacts live in a sandbox, not /app/models/production.
  2. For each finished match with match_date >= --split-date:
       a. Load the bundle's trained ensemble from the sandbox.
       b. Read features_cache row for the match (must exist; the
          compute_features_*.py scripts backfill these as part of
          the historical-ingest flow).
       c. Run predict_proba.
       d. Insert into the predictions table with
          model_version='wf_<split_date>' so live + walk-forward
          predictions don't collide on the unique constraint.

What this script is NOT:
  * Not a ROLLING walk-forward. Single split, single model snapshot
    per bundle. The trained model doesn't update over the test
    period, so its accuracy may drift over a multi-year test window.
    True rolling walk-forward (quarterly retrains) is a future v2.
  * Not idempotent at the prediction level — re-running with the
    same --split-date will hit the predictions unique constraint
    on (match_id, model_name, model_version, prediction_type) and
    UPSERT, which is fine but means the run isn't a no-op.

Usage (inside the api container):

    # Train + predict for ONE bundle (fastest, ~30 min)
    python /app/scripts/walk_forward_predictions.py \\
        --split-date 2024-01-01 \\
        --bundle nba_moneyline \\
        --snapshots-dir /tmp/wf_snapshots

    # All bundles (multi-hour). Run overnight on prod.
    python /app/scripts/walk_forward_predictions.py \\
        --split-date 2024-01-01 \\
        --bundle all \\
        --snapshots-dir /tmp/wf_snapshots

    # Then backtest against the resulting predictions:
    python /app/scripts/backtest_recommendations.py \\
        --start 2024-01-01 --end 2025-06-30 \\
        --prediction-version wf_2024-01-01

The --prediction-version filter on the backtest is how we isolate
walk-forward predictions from production rows — without it the
backtest would mix WF + prod and the numbers wouldn't mean anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Reuse the training framework. Late-imported per-bundle to avoid a
# hard module-load cost when --bundle filters to one sport.
sys.path.insert(0, "/app/services/ml-models/src")
sys.path.insert(0, os.path.dirname(__file__))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("walk_forward_predictions")


# ── Bundle resolution ────────────────────────────────────────────────


# Bundle keys mirror train_all_models.SPORT_BUNDLES. Kept here as a
# tuple instead of imported so the script remains importable in test
# environments without ml-models on the path.
ALL_BUNDLES = (
    "soccer_match_result",
    "nhl_moneyline",
    "nhl_regulation",
    "nhl_puck_line",
    "nhl_total",
    "nba_moneyline",
    "nba_spread",
    "nba_total",
    "nfl_moneyline",
    "nfl_spread",
    "nfl_total",
    "tennis_moneyline",
)

# Each bundle's ensemble registry name — needed for prediction-side
# model loading. Mirrors what train_all_models writes to
# <output-dir>/<ensemble_name>/<version>/model.bin.
BUNDLE_TO_ENSEMBLE: dict[str, str] = {
    "soccer_match_result": "ensemble_soccer_match_result",
    "nhl_moneyline": "ensemble_nhl_ml",
    "nhl_regulation": "ensemble_nhl_reg",
    "nhl_puck_line": "ensemble_nhl_pl",
    "nhl_total": "ensemble_nhl_tot",
    "nba_moneyline": "ensemble_nba_ml",
    "nba_spread": "ensemble_nba_sp",
    "nba_total": "ensemble_nba_tot",
    "nfl_moneyline": "ensemble_nfl_ml",
    "nfl_spread": "ensemble_nfl_sp",
    "nfl_total": "ensemble_nfl_tot",
    "tennis_moneyline": "ensemble_tennis_ml",
}

# Each bundle's prediction_type value (what gets written to
# predictions.prediction_type). Multiple bundles can share a value
# (NHL regulation and soccer match_result both use 'match_result'),
# disambiguated by model_name on the read side.
BUNDLE_TO_PREDICTION_TYPE: dict[str, str] = {
    "soccer_match_result": "match_result",
    "nhl_moneyline": "moneyline",
    "nhl_regulation": "match_result",
    "nhl_puck_line": "spread",
    "nhl_total": "total",
    "nba_moneyline": "moneyline",
    "nba_spread": "spread",
    "nba_total": "total",
    "nfl_moneyline": "moneyline",
    "nfl_spread": "spread",
    "nfl_total": "total",
    "tennis_moneyline": "moneyline",
}

# Each bundle's feature_set (matches compute_features_*.py).
BUNDLE_TO_FEATURE_SET: dict[str, str] = {
    "soccer_match_result": "baseline",
    "nhl_moneyline": "nhl_baseline",
    "nhl_regulation": "nhl_baseline",
    "nhl_puck_line": "nhl_baseline",
    "nhl_total": "nhl_baseline",
    "nba_moneyline": "nba_baseline",
    "nba_spread": "nba_baseline",
    "nba_total": "nba_baseline",
    "nfl_moneyline": "nfl_baseline",
    "nfl_spread": "nfl_baseline",
    "nfl_total": "nfl_baseline",
    "tennis_moneyline": "tennis_baseline",
}


def resolve_bundles(arg: str) -> list[str]:
    """'all' → every bundle. 'soccer' / 'nhl' / 'nba' / 'nfl' / 'tennis'
    → all bundles for that sport. Otherwise treated as a single bundle
    key."""
    if arg == "all":
        return list(ALL_BUNDLES)
    if arg in {"soccer", "nhl", "nba", "nfl", "tennis"}:
        return [b for b in ALL_BUNDLES if b.startswith(arg + "_") or b == f"{arg}_match_result"]
    if arg in ALL_BUNDLES:
        return [arg]
    raise ValueError(
        f"Unknown bundle {arg!r}. Use 'all', a sport " f"('soccer'/'nhl'/'nba'/'nfl'/'tennis'), or one of {ALL_BUNDLES}"
    )


# ── Training step ────────────────────────────────────────────────────


def filter_frame_before(frame, split_date: str):
    """Drop rows with match_date >= split_date. Walk-forward training
    must NEVER see post-split data."""
    import pandas as pd

    cutoff = pd.Timestamp(split_date, tz="UTC")
    if "match_date" not in frame.columns:
        return frame
    md = pd.to_datetime(frame["match_date"], errors="coerce", utc=True)
    return frame[md < cutoff].copy()


def train_snapshot(
    bundle: str,
    split_date: str,
    snapshots_dir: Path,
    database_url: str,
    skip_models: str = "",
) -> Path:
    """Train one bundle on data BEFORE split_date. Returns the path
    where artifacts landed.

    skip_models is a comma-separated list passed through to
    train_all_models --skip-models. Use it to drop a base model that's
    known to misbehave in the snapshot context (e.g. neural_network for
    tennis where the small per-snapshot frame can't fit a stable NN).

    Pipeline:
      1. Load the bundle's frame in-process via training.train_all_models
         SPORT_BUNDLES[bundle].load_frame(database_url=db).
      2. Filter to match_date < split_date.
      3. Persist the filtered frame to CSV.
      4. Subprocess train_all_models.py --sport <bundle> --input-csv
         <csv> --output-dir <snapshots-dir>/<bundle>. This keeps the
         training entrypoint unchanged — we're just driving it from
         outside with a pre-filtered CSV.
    """
    from training.train_all_models import SPORT_BUNDLES

    if bundle not in SPORT_BUNDLES:
        raise ValueError(f"Unknown bundle: {bundle}")
    spec = SPORT_BUNDLES[bundle]

    bundle_dir = snapshots_dir / bundle
    bundle_dir.mkdir(parents=True, exist_ok=True)
    csv_path = bundle_dir / "training.csv"
    output_dir = bundle_dir / "models"

    logger.info("Loading training frame for bundle=%s", bundle)
    frame = spec.load_frame(database_url=database_url)
    n_before = len(frame)
    frame = filter_frame_before(frame, split_date)
    n_after = len(frame)
    logger.info("Filtered %d → %d rows (split_date=%s)", n_before, n_after, split_date)

    if n_after < 100:
        raise RuntimeError(
            f"Too few training rows for bundle={bundle} before {split_date}: {n_after}. "
            "Pick an earlier split_date or load more historical data."
        )

    frame.to_csv(csv_path, index=False)
    logger.info("Wrote training CSV: %s", csv_path)

    logger.info("Training bundle=%s → %s", bundle, output_dir)
    cmd = [
        sys.executable,
        "-m",
        "training.train_all_models",
        "--sport",
        bundle,
        "--input-csv",
        str(csv_path),
        "--output-dir",
        str(output_dir),
    ]
    if skip_models:
        cmd.extend(["--skip-models", skip_models])
    result = subprocess.run(
        cmd,
        cwd="/app/services/ml-models",
        env={**os.environ, "PYTHONPATH": "/app/services/ml-models/src"},
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training subprocess failed (rc={result.returncode}) for bundle={bundle}")

    return output_dir


# ── Prediction step ──────────────────────────────────────────────────


def list_test_matches(conn, bundle: str, split_date: str) -> list[dict]:
    """Finished matches with match_date >= split_date for this bundle's
    sport. Each row carries the features_cache JSON needed for
    prediction — INNER JOIN so we skip matches without features."""
    sport = bundle.split("_")[0]  # soccer / nhl / nba / nfl
    feature_set = BUNDLE_TO_FEATURE_SET[bundle]
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                m.id::text AS match_id,
                m.match_date,
                ht.name AS home_team,
                at.name AS away_team,
                fc.features
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            JOIN LATERAL (
                SELECT features
                FROM features_cache
                WHERE match_id = m.id AND feature_set = %s
                ORDER BY computed_at DESC
                LIMIT 1
            ) fc ON true
            WHERE l.sport = %s
              AND m.status = 'finished'
              AND m.match_date >= %s::date
            ORDER BY m.match_date ASC
            """,
            (feature_set, sport, split_date),
        )
        return [dict(r) for r in cur.fetchall()]


def load_snapshot_ensemble(bundle: str, snapshot_dir: Path):
    """Reconstitute the ensemble from a walk-forward snapshot. Reuses
    services.api's _build_ensemble_for_task with the snapshot path
    as the model_path, so the live registry under
    /app/models/production isn't touched."""
    from services.prediction_service import TASKS, _build_ensemble_for_task, _klass_registry  # type: ignore

    ensemble_name = BUNDLE_TO_ENSEMBLE[bundle]
    # _klass_registry returns the (class, config) lookup that
    # _build_ensemble_for_task uses to instantiate each base model
    # from its on-disk artifact. Build once, reuse for every bundle.
    klass_for = _klass_registry()

    # Find the prod TaskSpec that uses this ensemble_name; we reuse
    # its label/feature config and only swap the model_path.
    for task in TASKS.values():
        if task.ensemble_name == ensemble_name:
            ensemble, meta = _build_ensemble_for_task(snapshot_dir, task, klass_for)
            if ensemble is None:
                raise RuntimeError(f"No ensemble found at {snapshot_dir}/{ensemble_name}/. Did training succeed?")
            return ensemble, task, meta
    raise ValueError(f"No TaskSpec found for ensemble_name={ensemble_name}")


def predict_one(model, features: dict, labels: list[str]) -> tuple[str, float, dict]:
    """Single predict_proba call → (predicted_outcome, confidence,
    probabilities dict). Mirrors the live precompute scripts'
    prediction step."""
    import numpy as np
    import pandas as pd

    # Mirror the feature__-prefix bridge the live predict path uses
    # (trained models' feature_names contain BOTH raw and prefixed
    # column names).
    X_dict: dict = {}
    for k, v in features.items():
        X_dict[k] = v
        prefixed = k if k.startswith("feature__") else f"feature__{k}"
        if prefixed not in X_dict:
            X_dict[prefixed] = v

    proba = model.predict_proba(pd.DataFrame([X_dict]))[0]
    if not np.all(np.isfinite(proba)):
        raise RuntimeError("predict_proba returned non-finite values")
    if len(proba) != len(labels):
        raise RuntimeError(f"Label count mismatch: {len(proba)} proba vs {len(labels)} labels")
    idx = int(proba.argmax())
    probabilities = {labels[i]: float(proba[i]) for i in range(len(labels))}
    return labels[idx], float(proba[idx]), probabilities


def store_prediction(
    cur, match_id: str, bundle: str, split_date: str, predicted_outcome: str, confidence: float, probabilities: dict
) -> None:
    """UPSERT into predictions with the walk-forward versioning key
    so live + WF predictions can coexist."""
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
            BUNDLE_TO_ENSEMBLE[bundle],
            f"wf_{split_date}",
            BUNDLE_TO_PREDICTION_TYPE[bundle],
            predicted_outcome,
            confidence,
            json.dumps(probabilities),
        ),
    )


def predict_test_period(bundle: str, split_date: str, snapshot_dir: Path, database_url: str) -> dict:
    """Walk test-period matches, predict each with the bundle's
    snapshot model, INSERT into predictions table. Returns counters."""
    ensemble, task, _ = load_snapshot_ensemble(bundle, snapshot_dir)

    counts = {"predicted": 0, "skipped": 0, "failed": 0}
    with psycopg2.connect(database_url) as conn:
        matches = list_test_matches(conn, bundle, split_date)
        logger.info("Bundle=%s → %d test-period matches", bundle, len(matches))
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for m in matches:
                features = m.get("features") or {}
                if not features:
                    counts["skipped"] += 1
                    continue
                try:
                    # hockey-Poisson + Dixon-Coles need home_team /
                    # away_team in the input — pass them through
                    # alongside the features dict.
                    feature_payload = dict(features)
                    feature_payload["home_team"] = m["home_team"]
                    feature_payload["away_team"] = m["away_team"]
                    predicted, confidence, probabilities = predict_one(ensemble, feature_payload, list(task.labels))
                except Exception as e:
                    logger.warning("Predict failed for %s: %s", m["match_id"], e)
                    counts["failed"] += 1
                    continue
                store_prediction(
                    cur,
                    m["match_id"],
                    bundle,
                    split_date,
                    predicted,
                    confidence,
                    probabilities,
                )
                counts["predicted"] += 1
            conn.commit()
    logger.info("Bundle=%s done: %s", bundle, counts)
    return counts


# ── Top-level orchestration ──────────────────────────────────────────


def run(args: argparse.Namespace) -> dict:
    bundles = resolve_bundles(args.bundle)
    snapshots_dir = Path(args.snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Walk-forward: %d bundle(s) at split_date=%s", len(bundles), args.split_date)
    if args.dry_run:
        logger.info("DRY RUN — would process: %s", bundles)
        return {"bundles": bundles, "split_date": args.split_date, "dry_run": True}

    total: dict[str, dict] = {}
    for bundle in bundles:
        logger.info("=" * 60)
        logger.info("BUNDLE %s", bundle)
        logger.info("=" * 60)
        snapshot_dir = train_snapshot(
            bundle,
            args.split_date,
            snapshots_dir,
            args.database_url,
            skip_models=args.skip_models,
        )
        counts = predict_test_period(bundle, args.split_date, snapshot_dir, args.database_url)
        total[bundle] = counts

    logger.info("All bundles done.")
    for bundle, counts in total.items():
        logger.info("  %s: %s", bundle, counts)
    return total


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--split-date",
        required=True,
        help="Train on data BEFORE this date; predict matches on/after.",
    )
    p.add_argument(
        "--bundle",
        default="all",
        help="'all' / sport ('soccer'/'nhl'/'nba'/'nfl') / specific bundle key.",
    )
    p.add_argument(
        "--snapshots-dir",
        default="/tmp/wf_snapshots",
        help="Where to drop training CSVs + trained-model artifacts.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--skip-models",
        default="",
        help=(
            "Comma-separated list of base-model types to skip when training the "
            "snapshot (passed through to train_all_models --skip-models). E.g. "
            "'neural_network' for tennis where the small snapshot frame can't "
            "fit a stable NN."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="List bundles + exit; no training, no DB writes.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    # Basic date format guard.
    try:
        date.fromisoformat(args.split_date)
    except ValueError as e:
        logger.error("Bad --split-date: %s", e)
        return 2
    try:
        run(args)
    except Exception as e:
        logger.error("Walk-forward failed: %s", e, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
