"""Train the horse-racing win-market LambdaMART ranker.

Standalone training script — separate from
services/ml-models/src/training/train_all_models.py because horse
racing's data shape (race-grouped, variable-N, ranker loss) doesn't
fit the team-sport SportBundle abstraction.

Usage (inside the api container, against prod data):

    python /app/scripts/train_horse_racing_win.py \\
        --split-date 2026-05-01 \\
        --output-dir /tmp/hr_ranker_v1

    # With an input CSV (e.g., for re-runs without DB round-trip):
    python /app/scripts/train_horse_racing_win.py \\
        --input-csv /tmp/hr_corpus.csv \\
        --split-date 2026-05-01

Walk-forward methodology: every race with race_date < --split-date
goes to training; >= goes to test. We also reserve a slice of the
training tail as the LightGBM Ranker's validation set (driving
early-stopping + temperature calibration); the test set stays
untouched until evaluation.

Output: a directory containing:
  model.bin              — LightGBM booster (loadable via lgb.Booster)
  feature_names.json     — ordered feature list (must match at predict)
  metadata.json          — config, dates, validation metrics, top1, MRR
  feature_importance.json — gain-based importance per feature

Compares to the consensus baseline at the end so the operator can
see whether the trained model BEAT 28.4% / Brier 0.0831 on the
same test-period races.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "ml-models" / "src"))
sys.path.insert(0, "/app/services/ml-models/src")

from predictors.horse_racing_ranker import HorseRacingRanker, HorseRacingRankerConfig  # noqa: E402
from utils.horse_racing_data import (  # noqa: E402
    HORSE_RACING_TRAINING_QUERY,
    NON_FEATURE_COLUMNS,
    get_feature_columns,
    group_array,
    load_training_frame,
    split_by_date,
    validate_training_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("train_horse_racing_win")


# ── Helpers ────────────────────────────────────────────────────────


def _split_train_val(
    train_frame: pd.DataFrame, val_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the LAST val_fraction of train_frame's races as the
    early-stopping / temperature-calibration validation set. Tail
    split (not random) preserves the temporal causality LambdaMART
    expects — feeding the model future races to predict past races
    would over-estimate generalisation."""
    if val_fraction <= 0:
        return train_frame, train_frame.iloc[0:0].copy()
    race_ids = list(dict.fromkeys(train_frame["race_id"].tolist()))
    if len(race_ids) < 2:
        return train_frame, train_frame.iloc[0:0].copy()
    cutoff = int(len(race_ids) * (1.0 - val_fraction))
    train_race_set = set(race_ids[:cutoff])
    val_race_set = set(race_ids[cutoff:])
    train = train_frame[train_frame["race_id"].isin(train_race_set)].reset_index(drop=True)
    val = train_frame[train_frame["race_id"].isin(val_race_set)].reset_index(drop=True)
    return train, val


def _consensus_baseline_topk(test_frame: pd.DataFrame) -> dict:
    """Compute the consensus baseline's top-1 accuracy on the test
    set so we have an apples-to-apples comparison. The baseline's
    'pick' is the entrant with the highest market_consensus_v1
    confidence in each race; correct = that entrant won."""
    if test_frame.empty or "consensus_prob" not in test_frame.columns:
        return {"top1_accuracy": 0.0, "races": 0}
    races_with_pred = test_frame.dropna(subset=["consensus_prob"])
    if races_with_pred.empty:
        return {"top1_accuracy": 0.0, "races": 0}
    # Per-race argmax of consensus_prob → did THAT entrant win?
    grouped = races_with_pred.groupby("race_id", sort=False)
    n_races = 0
    n_correct = 0
    for race_id, race_df in grouped:
        if race_df["target"].sum() == 0:
            continue
        n_races += 1
        idx = race_df["consensus_prob"].idxmax()
        if int(race_df.loc[idx, "target"]) == 1:
            n_correct += 1
    return {
        "top1_accuracy": n_correct / n_races if n_races else 0.0,
        "races": n_races,
    }


def _save_artefacts(
    output_dir: Path,
    model: HorseRacingRanker,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # LightGBM booster saves to text by default — fine for our v1.
    if model.model is not None and getattr(model.model, "booster_", None) is not None:
        model.model.booster_.save_model(str(output_dir / "model.bin"))
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(model.feature_names, f)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(model.feature_importance, f, indent=2)


# ── Orchestration ──────────────────────────────────────────────────


def run(
    *,
    database_url: Optional[str],
    input_csv: Optional[str],
    split_date: str,
    output_dir: Path,
    val_fraction: float = 0.15,
    config: Optional[HorseRacingRankerConfig] = None,
) -> dict:
    logger.info("Loading training frame...")
    frame = load_training_frame(database_url=database_url, input_csv=input_csv)
    if frame.empty:
        logger.error("Empty training frame; nothing to do")
        return {}
    quality = validate_training_frame(frame)
    logger.info(
        "Corpus: %d rows across %d races (%s..%s); win rate %.3f; features %d; missing %.1f%%",
        quality.rows,
        quality.races,
        quality.date_min,
        quality.date_max,
        quality.win_rate,
        quality.feature_count,
        quality.missing_feature_rate * 100,
    )

    train_frame, test_frame = split_by_date(frame, split_date)
    if train_frame.empty or test_frame.empty:
        raise ValueError(
            f"Empty train or test split (train rows={len(train_frame)}, "
            f"test rows={len(test_frame)}); pick a different --split-date"
        )

    train_inner, val_inner = _split_train_val(train_frame, val_fraction)
    feature_cols = get_feature_columns(train_inner)

    X_train = train_inner[feature_cols]
    y_train = train_inner["target"].to_numpy(dtype=np.int64)
    g_train = group_array(train_inner)

    X_val = val_inner[feature_cols] if not val_inner.empty else None
    y_val = val_inner["target"].to_numpy(dtype=np.int64) if not val_inner.empty else None
    g_val = group_array(val_inner) if not val_inner.empty else None

    X_test = test_frame[feature_cols]
    y_test = test_frame["target"].to_numpy(dtype=np.int64)
    g_test = group_array(test_frame)

    model = HorseRacingRanker(config=config)
    fit_result = model.fit(
        X_train=X_train,
        y_train=y_train,
        groups_train=g_train,
        X_val=X_val,
        y_val=y_val,
        groups_val=g_val,
    )

    # Evaluate on the TRUE test set (not the inner val slice).
    test_metrics = model._evaluate(X_test, y_test, g_test)  # noqa: SLF001 — internal eval is the intended public surface
    consensus_baseline = _consensus_baseline_topk(test_frame)

    logger.info(
        "Test metrics — top1 acc %.3f, MRR %.3f, NLL %.4f, races %d",
        test_metrics.get("top1_accuracy", 0.0),
        test_metrics.get("mrr", 0.0),
        test_metrics.get("nll", 0.0),
        test_metrics.get("races", 0),
    )
    logger.info(
        "Consensus baseline on same test set — top1 acc %.3f, races %d",
        consensus_baseline["top1_accuracy"],
        consensus_baseline["races"],
    )
    delta = test_metrics.get("top1_accuracy", 0.0) - consensus_baseline["top1_accuracy"]
    logger.info("Ranker vs consensus top1 delta: %+.3f pts", delta * 100)

    metadata = {
        "model_name": "lightgbm_ranker_v1",
        "model_version": "1.0.0",
        "split_date": split_date,
        "val_fraction": val_fraction,
        "corpus": {
            "rows": quality.rows,
            "races": quality.races,
            "win_rate": quality.win_rate,
            "date_min": quality.date_min,
            "date_max": quality.date_max,
            "feature_count": quality.feature_count,
            "missing_feature_rate": quality.missing_feature_rate,
        },
        "split_sizes": {
            "train_rows": int(len(X_train)),
            "train_races": int(len(g_train)),
            "val_rows": int(len(X_val)) if X_val is not None else 0,
            "val_races": int(len(g_val)) if g_val is not None else 0,
            "test_rows": int(len(X_test)),
            "test_races": int(len(g_test)),
        },
        "fit_result": {
            "temperature": model.temperature,
            "best_iteration": fit_result.get("best_iteration"),
            "validation_metrics": fit_result.get("validation_metrics", {}),
        },
        "test_metrics": test_metrics,
        "consensus_baseline_on_test": consensus_baseline,
        "ranker_minus_baseline_top1": delta,
    }
    _save_artefacts(output_dir, model, metadata)
    logger.info("Saved artefacts to %s", output_dir)
    return metadata


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--input-csv", help="Skip the DB and load from a saved CSV.")
    p.add_argument(
        "--split-date",
        required=True,
        help="Walk-forward boundary (YYYY-MM-DD). Races on/after go to test.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/hr_ranker_v1"),
        help="Where to write model.bin + metadata.json + feature_names.json",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of train tail reserved for early stopping + temperature tuning.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input_csv and not args.database_url:
        logger.error("Provide --input-csv or --database-url")
        return 2
    run(
        database_url=args.database_url,
        input_csv=args.input_csv,
        split_date=args.split_date,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
