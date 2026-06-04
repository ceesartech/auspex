"""One-off A/B: does CatBoostRanker beat LGBMRanker on horse racing?

Memory `horse-racing-ml-ranker-v1` option 4: try CatBoost (or a
Plackett-Luce alternative) — LambdaMART isn't the only ranker
family. CatBoost is the cheaper test (drop-in replacement, same
learn-to-rank API). Hypothesis: oblivious-trees + ordered boosting
could capture different signal than LightGBM's leaf-wise growth,
specifically the calibration gap that's resisted all four prior
push attempts (race-relative features, +29% corpus, place/show,
isotonic post-cal).

A/B at split_date=2026-05-15 on the 18k-race expanded corpus:
  v_lgbm     : current production HorseRacingRanker (LGBMRanker)
  v_catboost : CatBoostRanker with comparable hyperparameters,
               same train/val split, same feature set

Both wrap their raw scores in the same softmax-temperature
calibration pipeline used by HorseRacingRanker so the Brier
comparison is on equivalent probability distributions.

Decision rule:
  ΔBrier ≤ -0.005 → KEEP v_catboost (add catboost to requirements,
  retrain the production ranker with the new class)
  Otherwise → DROP, document as the final negative result for the
  ranker calibration push.

Run on prod via (after pip install catboost --user with PYTHONUSERBASE):
    PYTHONPATH=/tmp/pip-user/lib/python3.11/site-packages:$PYTHONPATH \\
    python /app/scripts/ab_horse_racing_catboost.py
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_horse_racing_catboost")


def _per_race_softmax_with_temperature(
    scores: np.ndarray, groups: np.ndarray, temperature: float,
) -> list:
    """Same per-race softmax HorseRacingRanker.predict_probabilities
    uses — apply temperature and softmax within each race group,
    return a list of per-race np arrays (matching the existing
    Brier helper's expected shape)."""
    out = []
    cursor = 0
    for size in groups:
        size = int(size)
        race_scores = scores[cursor:cursor + size]
        cursor += size
        scaled = race_scores / max(temperature, 1e-6)
        scaled = scaled - scaled.max()  # numerical stability
        exp = np.exp(scaled)
        out.append(exp / exp.sum())
    return out


def _per_entrant_brier(race_probs: list, groups: np.ndarray,
                       y_true: np.ndarray) -> float:
    total = 0.0
    n = 0
    cursor = 0
    for race_arr, size in zip(race_probs, groups):
        size = int(size)
        actuals = y_true[cursor:cursor + size].astype(np.float64)
        cursor += size
        if actuals.sum() == 0:
            continue
        total += float(np.mean((race_arr - actuals) ** 2))
        n += 1
    return total / n if n else float("inf")


def _tune_temperature(
    scores: np.ndarray, groups: np.ndarray, y_true: np.ndarray,
) -> Tuple[float, float]:
    """Pick the temperature ∈ [0.01, 2.0] that minimises per-race
    Brier on the val set. Mirrors the prod
    HorseRacingRanker._fit_temperature loop so the A/B compares
    on the same calibration objective."""
    best_t = 1.0
    best_brier = float("inf")
    for t in (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15,
              0.18, 0.20, 0.24, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50, 2.00):
        probs = _per_race_softmax_with_temperature(scores, groups, t)
        b = _per_entrant_brier(probs, groups, y_true)
        if b < best_brier:
            best_brier = b
            best_t = t
    return best_t, best_brier


def _top1_accuracy(scores: np.ndarray, groups: np.ndarray,
                   y_true: np.ndarray) -> float:
    cursor = 0
    hits = 0
    n = 0
    for size in groups:
        size = int(size)
        race_scores = scores[cursor:cursor + size]
        race_actual = y_true[cursor:cursor + size]
        cursor += size
        if race_actual.sum() == 0:
            continue
        n += 1
        top = int(np.argmax(race_scores))
        if int(race_actual[top]) == 1:
            hits += 1
    return hits / n if n else 0.0


def train_catboost(
    X_train: pd.DataFrame, y_train: np.ndarray, g_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray, g_val: np.ndarray,
):
    """Train a CatBoostRanker with hyperparameters comparable to the
    production LGBMRanker config (HorseRacingRankerConfig defaults)."""
    from catboost import CatBoostRanker, Pool

    # CatBoost expects a group_id PER ROW (not a per-group count).
    # Convert g_train from sizes to per-row group ids.
    def _row_group_ids(groups: np.ndarray) -> np.ndarray:
        return np.repeat(np.arange(len(groups)), groups.astype(int))

    train_pool = Pool(
        data=X_train.values, label=y_train,
        group_id=_row_group_ids(g_train),
    )
    val_pool = Pool(
        data=X_val.values, label=y_val,
        group_id=_row_group_ids(g_val),
    )

    model = CatBoostRanker(
        loss_function="YetiRank",  # listwise, comparable to LambdaRank
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    return model


def evaluate(name: str, scores: np.ndarray, groups: np.ndarray,
             y_true: np.ndarray, temperature: float) -> dict:
    probs = _per_race_softmax_with_temperature(scores, groups, temperature)
    brier = _per_entrant_brier(probs, groups, y_true)
    top1 = _top1_accuracy(scores, groups, y_true)
    LOGGER.info(
        "%s :: top1=%.4f  brier=%.4f  (T=%.3f)",
        name, top1, brier, temperature,
    )
    return {"top1": top1, "brier": brier, "temperature": temperature}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--split-date", default="2026-05-15")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    if not args.database_url:
        LOGGER.error("DATABASE_URL not set.")
        return 1

    sys.path.insert(0, "/app/services/ml-models/src")
    from predictors.horse_racing_ranker import HorseRacingRanker
    from utils.horse_racing_data import (
        get_feature_columns, group_array, load_training_frame, split_by_date,
    )
    sys.path.insert(0, "/app/scripts")
    from train_horse_racing_win import _per_entrant_brier as _trainer_brier
    from train_horse_racing_win import _split_train_val

    LOGGER.info("Loading ranker training frame...")
    frame = load_training_frame(database_url=args.database_url)
    LOGGER.info("Loaded %d rows / %d races.",
                len(frame), frame["race_id"].nunique())

    train_frame, test_frame = split_by_date(frame, args.split_date)
    train_inner, val_inner = _split_train_val(train_frame, args.val_fraction)
    feature_cols = get_feature_columns(train_inner)
    LOGGER.info(
        "Train=%d Val=%d Test=%d  features=%d",
        len(train_inner), len(val_inner), len(test_frame), len(feature_cols),
    )

    median = train_inner[feature_cols].median()
    X_train = train_inner[feature_cols].fillna(median)
    y_train = train_inner["target"].to_numpy(dtype=np.int64)
    g_train = group_array(train_inner)

    X_val = val_inner[feature_cols].fillna(median)
    y_val = val_inner["target"].to_numpy(dtype=np.int64)
    g_val = group_array(val_inner)

    X_test = test_frame[feature_cols].fillna(median)
    y_test = test_frame["target"].to_numpy(dtype=np.int64)
    g_test = group_array(test_frame)

    # ── v_lgbm: current production class ──────────────────────────
    LOGGER.info("══════════════ v_lgbm (LGBMRanker) ══════════════")
    lgbm_model = HorseRacingRanker()
    lgbm_model.fit(
        X_train=train_inner[feature_cols], y_train=y_train, groups_train=g_train,
        X_val=val_inner[feature_cols], y_val=y_val, groups_val=g_val,
    )
    lgbm_scores_test = lgbm_model.model.predict(X_test)
    lgbm_test_probs = lgbm_model.predict_probabilities(
        test_frame[feature_cols], g_test,
    )
    lgbm_test_brier = _trainer_brier(lgbm_test_probs, g_test, y_test)
    lgbm_top1 = _top1_accuracy(lgbm_scores_test, g_test, y_test)
    LOGGER.info(
        "v_lgbm :: top1=%.4f brier=%.4f T=%.3f",
        lgbm_top1, lgbm_test_brier, lgbm_model.temperature,
    )

    # ── v_catboost: option 4 candidate ────────────────────────────
    LOGGER.info("══════════════ v_catboost (CatBoostRanker) ══════════════")
    cat_model = train_catboost(X_train, y_train, g_train, X_val, y_val, g_val)
    cat_val_scores = cat_model.predict(X_val.values)
    cat_temp, cat_val_brier = _tune_temperature(cat_val_scores, g_val, y_val)
    LOGGER.info(
        "v_catboost val temperature tuned: T=%.3f val_brier=%.4f",
        cat_temp, cat_val_brier,
    )
    cat_scores_test = cat_model.predict(X_test.values)
    cat_metrics = evaluate("v_catboost", cat_scores_test, g_test, y_test, cat_temp)

    # ── Final comparison ─────────────────────────────────────────
    LOGGER.info("")
    LOGGER.info("───────────── DELTAS (v_catboost - v_lgbm) ─────────────")
    d_top1 = cat_metrics["top1"] - lgbm_top1
    d_brier = cat_metrics["brier"] - lgbm_test_brier
    LOGGER.info("Δ top1:  %+.4f  (positive favours catboost)", d_top1)
    LOGGER.info("Δ brier: %+.4f  (NEGATIVE favours catboost)", d_brier)
    if d_brier <= -0.005:
        verdict = "KEEP v_catboost (clears threshold)"
    elif d_brier >= 0.005:
        verdict = "DROP v_catboost (worse Brier)"
    else:
        verdict = "DROP v_catboost (|ΔBrier|<0.005, model class isn't the bottleneck)"
    LOGGER.info("VERDICT: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
