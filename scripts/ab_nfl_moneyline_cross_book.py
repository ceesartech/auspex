"""One-off A/B: do cross-book features help NFL moneyline?

PR #8 shipped cross-book features for NFL TOTAL (ΔBrier -0.0110)
but the A/B only covered spread + total. Moneyline (which already
ships at ~71% accuracy per memory) was never tested with the same
feature class. Since the TOTAL signal came from
public-book disagreement + consensus deltas, and NFL ML has the
same 22-book structure, this script applies the same pattern.

Features per match (moneyline is 2-way: home/away):
  ml_book_count
  ml_consensus_home_prob (mean devigged home prob across books)
  ml_max_minus_min_home_prob
  ml_std_home_prob
  ml_consensus_implied_prob_disagreement (max - min - std combined)

Methodology mirrors scripts/ab_nfl_cross_book.py (the TOTAL one
that shipped). XGBoost trained twice on identical walk-forward
split (last 2024-2025 season as test).

Decision rule (per market):
  ΔBrier ≤ -0.005 → KEEP. Land features in compute_features_nfl.py
  for moneyline, retrain ensemble.

Run on prod via:
    docker compose exec api python /app/scripts/ab_nfl_moneyline_cross_book.py
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_nfl_moneyline_cross_book")


SNAPSHOTS_QUERY = """
    SELECT DISTINCT ON (o.match_id, o.selection, o.bookmaker)
        o.match_id::text AS match_id,
        o.bookmaker,
        o.selection,
        o.odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'nfl'
    WHERE m.status = 'finished'
      AND o.is_live = false
      AND o.market_type = 'moneyline'
      AND o.odds_decimal IS NOT NULL
      AND o.selection IN ('home', 'away')
    ORDER BY o.match_id, o.selection, o.bookmaker, o.timestamp DESC
"""


def _devig_pair(odds_home: float, odds_away: float) -> float:
    """Returns devigged P(home) only — for the 2-way ML market."""
    if odds_home <= 0 or odds_away <= 0:
        return np.nan
    raw_h = 1.0 / odds_home
    raw_a = 1.0 / odds_away
    total = raw_h + raw_a
    return raw_h / total if total > 0 else np.nan


def compute_crossbook_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Per-match: per-book devigged home prob, then cross-book stats.

    Returns one row per match. Books that have only home OR only
    away (not both) get dropped — can't devig without the pair.
    """
    # Wide pivot: one row per (match, book) with home + away odds.
    wide = (
        snapshots
        .pivot_table(
            index=["match_id", "bookmaker"],
            columns="selection",
            values="odds_decimal",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    for col in ("home", "away"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.dropna(subset=["home", "away"])

    wide["p_home"] = [
        _devig_pair(h, a) for h, a in zip(wide["home"], wide["away"])
    ]
    wide = wide.dropna(subset=["p_home"])

    rows = []
    for match_id, g in wide.groupby("match_id"):
        probs = g["p_home"].astype(float).values
        rows.append({
            "match_id": match_id,
            "ml_book_count": float(len(probs)),
            "ml_consensus_home_prob": float(probs.mean()),
            "ml_max_minus_min_home_prob": float(probs.max() - probs.min()),
            "ml_std_home_prob": (
                float(np.std(probs, ddof=0)) if len(probs) > 1 else 0.0
            ),
        })
    return pd.DataFrame(rows)


CROSSBOOK_KEYS = (
    "ml_book_count",
    "ml_consensus_home_prob",
    "ml_max_minus_min_home_prob",
    "ml_std_home_prob",
)


def season_walk_forward_split(
    frame: pd.DataFrame, test_season: str,
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


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob > lo) & (y_prob <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        ece += (in_bin.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def train_and_eval(train, test, feature_cols, target):
    from sklearn.metrics import accuracy_score, brier_score_loss
    from xgboost import XGBClassifier

    X_train = train[feature_cols].values
    y_train = train[target].astype(int).values
    X_test = test[feature_cols].values
    y_test = test[target].astype(int).values

    clf = XGBClassifier(
        objective="binary:logistic",
        max_depth=6, learning_rate=0.05, n_estimators=400,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.01, reg_lambda=1.0,
        tree_method="hist", random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "brier": float(brier_score_loss(y_test, proba)),
        "ece": expected_calibration_error(y_test, proba),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--test-season", default="2024-2025")
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
    from utils import training_data as td

    LOGGER.info("Loading NFL moneyline frame...")
    base = td.load_nfl_moneyline_frame(database_url=args.database_url)
    LOGGER.info("Loaded %d matches.", len(base))

    LOGGER.info("Loading cross-book snapshots...")
    from sqlalchemy import create_engine
    engine = create_engine(args.database_url)
    try:
        snapshots = pd.read_sql(SNAPSHOTS_QUERY, engine)
    finally:
        engine.dispose()
    LOGGER.info(
        "Loaded %d (match, book, selection) snapshot rows.", len(snapshots),
    )

    crossbook = compute_crossbook_features(snapshots)
    LOGGER.info(
        "Cross-book features computed for %d matches.", len(crossbook),
    )

    base["match_id"] = base["match_id"].astype(str)
    crossbook["match_id"] = crossbook["match_id"].astype(str)
    merged = base.merge(crossbook, on="match_id", how="left")
    cov = merged["ml_book_count"].notna().sum()
    LOGGER.info(
        "Test corpus: %d matches, cross-book coverage %d (%.1f%%).",
        len(merged), cov, 100.0 * cov / max(1, len(merged)),
    )

    train, test = season_walk_forward_split(merged, args.test_season)
    LOGGER.info(
        "Walk-forward: train=%d test=%d (test_season=%s)",
        len(train), len(test), args.test_season,
    )

    excluded = set(td.NFL_MONEYLINE_NON_FEATURE_COLUMNS) | {td.NFL_MONEYLINE_TARGET}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1 = [c for c in numeric_cols if c not in excluded and c not in CROSSBOOK_KEYS]
    vcb = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1 features: %d  |  v_crossbook features: %d (+%d cross-book keys)",
        len(v1), len(vcb), len(vcb) - len(v1),
    )

    m_v1 = train_and_eval(train, test, v1, td.NFL_MONEYLINE_TARGET)
    LOGGER.info(
        "v1-control:   acc=%.4f brier=%.4f ece=%.4f",
        m_v1["accuracy"], m_v1["brier"], m_v1["ece"],
    )
    m_cb = train_and_eval(train, test, vcb, td.NFL_MONEYLINE_TARGET)
    LOGGER.info(
        "v_crossbook:  acc=%.4f brier=%.4f ece=%.4f",
        m_cb["accuracy"], m_cb["brier"], m_cb["ece"],
    )

    d_acc = m_cb["accuracy"] - m_v1["accuracy"]
    d_brier = m_cb["brier"] - m_v1["brier"]
    d_ece = m_cb["ece"] - m_v1["ece"]
    LOGGER.info(
        "Δ acc: %+.4f  Δ brier: %+.4f  Δ ece: %+.4f", d_acc, d_brier, d_ece,
    )
    if d_brier <= -0.005:
        verdict = "KEEP cross-book features for NFL moneyline"
    elif d_brier >= 0.005:
        verdict = "DROP cross-book features for NFL moneyline (worse Brier)"
    else:
        verdict = "DROP cross-book features for NFL moneyline (|ΔBrier|<0.005)"
    LOGGER.info("VERDICT: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
