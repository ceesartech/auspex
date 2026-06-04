"""One-off A/B: cross-book features on NHL REGULATION (3-class).

PR #13 shipped cross-book features for NHL puck_line + total but
SKIPPED nhl_regulation because the binary XGBoost harness can't
handle the 3-class target (0=home reg win, 1=tie at regulation
→ OT/SO, 2=away reg win). This script fills that gap with a
multi-class XGBoost harness so we can decide on REGULATION too.

Hypothesis:
  * Cross-book moneyline disagreement (high std of devigged home
    prob across the 24 NHL books) correlates with "this game is
    close → likely to go to OT". The existing implied_prob_home_ml
    captures the MEAN devigged home prob; std + max-min capture
    book disagreement that the mean alone doesn't.
  * If the signal exists, ΔBrier should be negative on a clean
    walk-forward.

Reuses the same cross-book feature recipe as PR #12 (NFL ML) +
PR #13 (NHL puck/total). 4 features:
  ml_book_count
  ml_consensus_home_prob
  ml_max_minus_min_home_prob
  ml_std_home_prob

Multi-class Brier: mean per-row of sum_c (p_c - one_hot_c)^2.
Standard generalisation of binary Brier to K classes.
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_nhl_regulation_cross_book")


ML_QUERY = """
    SELECT DISTINCT ON (o.match_id, o.selection, o.bookmaker)
        o.match_id::text AS match_id, o.bookmaker, o.selection,
        o.odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'nhl'
    WHERE m.status = 'finished' AND o.is_live = false
      AND o.market_type = 'moneyline'
      AND o.odds_decimal IS NOT NULL
      AND o.selection IN ('home', 'away')
    ORDER BY o.match_id, o.selection, o.bookmaker, o.timestamp DESC
"""


def _devig_pair(odds_a: float, odds_b: float) -> Tuple[float, float]:
    if odds_a <= 0 or odds_b <= 0:
        return (np.nan, np.nan)
    raw_a = 1.0 / odds_a
    raw_b = 1.0 / odds_b
    total = raw_a + raw_b
    if total <= 0:
        return (np.nan, np.nan)
    return (raw_a / total, raw_b / total)


def compute_ml_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    wide = snapshots.pivot_table(
        index=["match_id", "bookmaker"],
        columns="selection",
        values="odds_decimal",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    for col in ("home", "away"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.dropna(subset=["home", "away"])
    wide["p_home"] = [_devig_pair(h, a)[0] for h, a in zip(wide["home"], wide["away"])]
    wide = wide.dropna(subset=["p_home"])

    rows = []
    for match_id, g in wide.groupby("match_id"):
        probs = g["p_home"].astype(float).values
        rows.append(
            {
                "match_id": match_id,
                "ml_book_count": float(len(probs)),
                "ml_consensus_home_prob": float(probs.mean()),
                "ml_max_minus_min_home_prob": float(probs.max() - probs.min()),
                "ml_std_home_prob": (float(np.std(probs, ddof=0)) if len(probs) > 1 else 0.0),
            }
        )
    return pd.DataFrame(rows)


ML_KEYS = (
    "ml_book_count",
    "ml_consensus_home_prob",
    "ml_max_minus_min_home_prob",
    "ml_std_home_prob",
)


def season_walk_forward_split(
    frame: pd.DataFrame,
    test_season: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["season"] < test_season].copy()
    test = frame[frame["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Walk-forward for season={test_season!r}: train={len(train)} "
            f"test={len(test)}. Seasons present: "
            f"{sorted(frame['season'].dropna().unique())}"
        )
    return train, test


def multi_class_brier(y_true: np.ndarray, proba: np.ndarray, num_class: int) -> float:
    """Multi-class Brier — mean over samples of sum_c (p_c - one_hot_c)^2.
    Standard generalisation of binary Brier."""
    y_onehot = np.zeros((len(y_true), num_class), dtype=float)
    y_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def multi_class_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """ECE on the predicted CLASS's confidence (max prob)."""
    preds = np.argmax(proba, axis=1)
    confidences = proba[np.arange(len(proba)), preds]
    correct = (preds == y_true).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def train_and_eval_multiclass(train, test, feature_cols, target, num_class=3):
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    X_train = train[feature_cols].values
    y_train = train[target].astype(int).values
    X_test = test[feature_cols].values
    y_test = test[target].astype(int).values

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    preds = np.argmax(proba, axis=1)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "brier": multi_class_brier(y_test, proba, num_class),
        "ece": multi_class_ece(y_test, proba),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--test-season", default="2024-2025")
    parser.add_argument(
        "--log-level",
        default="INFO",
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
    from utils import training_data as td

    LOGGER.info("Loading NHL moneyline snapshots...")
    from sqlalchemy import create_engine

    engine = create_engine(args.database_url)
    try:
        ml_snapshots = pd.read_sql(ML_QUERY, engine)
    finally:
        engine.dispose()
    LOGGER.info("ML snapshots: %d", len(ml_snapshots))
    ml_features = compute_ml_features(ml_snapshots)

    LOGGER.info("Loading NHL REGULATION frame...")
    base = td.load_nhl_regulation_frame(database_url=args.database_url)
    base["match_id"] = base["match_id"].astype(str)
    ml_features["match_id"] = ml_features["match_id"].astype(str)
    merged = base.merge(ml_features, on="match_id", how="left")
    cov = merged["ml_book_count"].notna().sum()
    LOGGER.info(
        "Loaded %d REGULATION matches. Cross-book coverage: %d (%.1f%%).",
        len(merged),
        cov,
        100.0 * cov / max(1, len(merged)),
    )

    train, test = season_walk_forward_split(merged, args.test_season)
    LOGGER.info(
        "Walk-forward: train=%d test=%d (test_season=%s).",
        len(train),
        len(test),
        args.test_season,
    )

    target = td.NHL_REGULATION_TARGET
    excluded = set(td.NHL_REGULATION_NON_FEATURE_COLUMNS) | {target}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1 = [c for c in numeric_cols if c not in excluded and c not in ML_KEYS]
    vcb = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1 features: %d  |  v_crossbook features: %d (+%d keys)",
        len(v1),
        len(vcb),
        len(vcb) - len(v1),
    )

    m_v1 = train_and_eval_multiclass(train, test, v1, target)
    LOGGER.info(
        "v1-control:   acc=%.4f brier=%.4f ece=%.4f",
        m_v1["accuracy"],
        m_v1["brier"],
        m_v1["ece"],
    )
    m_cb = train_and_eval_multiclass(train, test, vcb, target)
    LOGGER.info(
        "v_crossbook:  acc=%.4f brier=%.4f ece=%.4f",
        m_cb["accuracy"],
        m_cb["brier"],
        m_cb["ece"],
    )

    d_acc = m_cb["accuracy"] - m_v1["accuracy"]
    d_brier = m_cb["brier"] - m_v1["brier"]
    d_ece = m_cb["ece"] - m_v1["ece"]
    LOGGER.info(
        "Δ acc: %+.4f  Δ brier: %+.4f  Δ ece: %+.4f",
        d_acc,
        d_brier,
        d_ece,
    )
    if d_brier <= -0.005:
        verdict = "KEEP cross-book for REGULATION"
    elif d_brier >= 0.005:
        verdict = "DROP cross-book for REGULATION (worse Brier)"
    else:
        verdict = "DROP cross-book for REGULATION (|ΔBrier|<0.005)"
    LOGGER.info("VERDICT: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
