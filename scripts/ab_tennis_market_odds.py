"""A/B: does a REAL market price (tennis-data.co.uk closing odds, loaded by
scripts/load_tennis_odds.py) make the tennis moneyline model better — and
does the model then beat the market or merely match it?

Arms, all on the same temporal holdout of the REAL training frame
(training_data.load_tennis_moneyline_frame):
  no_market   production feature set minus every market column — what the
              current model effectively is (its odds features were the
              hard-coded default in ~99% of training rows)
  with_market production feature set, market columns now real
  market      the de-vigged closing price itself (feature__implied_prob_*),
              scored only where a price exists

Brier is the repo convention for a 2-class model: sum over both classes of
(p - y)^2 = 2 (p_away - y_away)^2, so a coin flip scores 0.5 (matches
training.calibration.brier_multiclass and held_out_metrics.json). Every
delta is PAIRED on the same rows and quoted with its standard error
(sqrt(Var(per-row delta) / n)), per audit §4.3. Ship gate for a model-side
change: dBrier <= -0.005 and clear of the noise.

Windows: the last --test-fraction of matches by date, plus per-year
walk-forward (train on everything before the year, test on the year) for
every year with enough priced test rows.

    python /app/scripts/ab_tennis_market_odds.py
    python scripts/ab_tennis_market_odds.py --input-csv tennis_frame.csv   # offline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_tennis_market_odds")

# Columns that carry the market price, in both spellings the frame holds:
# the training query's AVG(odds_decimal) (NaN when unpriced) and the
# flattened features_cache keys (NEUTRAL_DEFAULT when unpriced).
MARKET_COLUMNS = (
    "odds_home_ml",
    "odds_away_ml",
    "feature__odds_home_ml",
    "feature__odds_away_ml",
    "feature__implied_prob_home_ml",
    "feature__implied_prob_away_ml",
    "feature__ml_bookie_margin",
    "feature__odds_implied_diff",
)
MIN_PRICED_TEST_ROWS = 300

XGB_PARAMS = dict(
    objective="binary:logistic",
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
    eval_metric="logloss",
)


def brier2(p_away: np.ndarray, y_away: np.ndarray) -> np.ndarray:
    """Per-row 2-class sum-of-squares Brier (repo convention)."""
    return 2.0 * (np.asarray(p_away, float) - np.asarray(y_away, float)) ** 2


def logloss(p_away: np.ndarray, y_away: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_away, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_away, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """b - a, mean and SE (negative = b better)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "delta": round(float(d.mean()), 5),
        "se": round(se, 5),
        "z": round(float(d.mean()) / se, 2) if se else None,
    }


def has_market(frame: pd.DataFrame) -> pd.Series:
    """A real price exists: the training query's AVG(odds) is not NaN."""
    if "odds_home_ml" in frame.columns and "odds_away_ml" in frame.columns:
        return frame["odds_home_ml"].notna() & frame["odds_away_ml"].notna()
    return pd.Series(False, index=frame.index)


def market_prob_away(frame: pd.DataFrame) -> np.ndarray:
    """De-vigged P(away) from the training query's AVG(odds) — the price that
    actually exists for the match. (The cached feature can be stale or absent:
    a features_cache row computed before the odds landed still holds the
    NEUTRAL_DEFAULT, and some finished matches have no cache row at all.)"""
    inv_h = 1.0 / frame["odds_home_ml"].astype(float)
    inv_a = 1.0 / frame["odds_away_ml"].astype(float)
    return (inv_a / (inv_h + inv_a)).to_numpy()


def feature_sets(frame: pd.DataFrame, excluded: set[str]) -> tuple[list[str], list[str]]:
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    with_market = [c for c in numeric if c not in excluded]
    no_market = [c for c in with_market if c not in MARKET_COLUMNS]
    return no_market, with_market


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: str) -> np.ndarray:
    from xgboost import XGBClassifier

    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(train[cols].to_numpy(dtype=float), train[target].astype(int).to_numpy())
    return clf.predict_proba(test[cols].to_numpy(dtype=float))[:, 1]


def evaluate_window(
    train: pd.DataFrame, test: pd.DataFrame, no_mkt: list[str], with_mkt: list[str], target: str
) -> dict:
    y = test[target].astype(int).to_numpy()
    p_no = fit_predict(train, test, no_mkt, target)
    p_with = fit_predict(train, test, with_mkt, target)
    priced = has_market(test).to_numpy()
    out: dict = {
        "n_train": int(len(train)),
        "n_train_priced": int(has_market(train).sum()),
        "n_test": int(len(test)),
        "n_test_priced": int(priced.sum()),
        "all_rows": {
            "brier_no_market": round(float(brier2(p_no, y).mean()), 5),
            "brier_with_market": round(float(brier2(p_with, y).mean()), 5),
            "with_minus_no": paired(brier2(p_no, y), brier2(p_with, y)),
        },
    }
    if priced.sum() >= 30:
        yp = y[priced]
        pm = market_prob_away(test)[priced]
        b_no, b_with, b_mkt = brier2(p_no[priced], yp), brier2(p_with[priced], yp), brier2(pm, yp)
        out["priced_rows"] = {
            "brier_no_market": round(float(b_no.mean()), 5),
            "brier_with_market": round(float(b_with.mean()), 5),
            "brier_market": round(float(b_mkt.mean()), 5),
            "logloss_with_market": round(float(logloss(p_with[priced], yp).mean()), 5),
            "logloss_market": round(float(logloss(pm, yp).mean()), 5),
            "acc_with_market": round(float(((p_with[priced] >= 0.5) == (yp == 1)).mean()), 4),
            "acc_market": round(float(((pm >= 0.5) == (yp == 1)).mean()), 4),
            "with_minus_no": paired(b_no, b_with),
            "with_minus_market": paired(b_mkt, b_with),
            "no_minus_market": paired(b_mkt, b_no),
        }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--input-csv", help="raw training-query export instead of the DB")
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--ml-src", default="/app/services/ml-models/src")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

    sys.path.insert(0, args.ml_src)
    from utils import training_data as td

    frame = td.load_tennis_moneyline_frame(database_url=args.database_url, input_csv=args.input_csv)
    frame = frame.sort_values("match_date").reset_index(drop=True)
    target = td.TENNIS_MONEYLINE_TARGET
    excluded = set(td.TENNIS_MONEYLINE_NON_FEATURE_COLUMNS) | {target}
    no_mkt, with_mkt = feature_sets(frame, excluded)
    priced = has_market(frame)
    LOGGER.info("frame %d matches, %d priced (%.1f%%)", len(frame), int(priced.sum()), 100 * priced.mean())

    report: dict = {
        "frame_rows": int(len(frame)),
        "frame_priced": int(priced.sum()),
        "priced_by_year": frame.assign(p=priced)
        .groupby(frame["match_date"].dt.year)["p"]
        .agg(["size", "sum"])
        .to_dict("index"),
        "rows_without_features_by_year": frame.assign(nf=frame.get("feature__home_roll_matches").isna())
        .groupby(frame["match_date"].dt.year)["nf"]
        .sum()
        .to_dict(),
        "features_no_market": len(no_mkt),
        "features_with_market": len(with_mkt),
        "windows": {},
    }
    cut = frame["match_date"].iloc[int(len(frame) * (1 - args.test_fraction))]
    report["windows"][f"last_{args.test_fraction:.0%}_from_{cut.date()}"] = evaluate_window(
        frame[frame["match_date"] < cut], frame[frame["match_date"] >= cut], no_mkt, with_mkt, target
    )
    for year in sorted(frame["match_date"].dt.year.unique()):
        test = frame[frame["match_date"].dt.year == year]
        train = frame[frame["match_date"].dt.year < year]
        if has_market(test).sum() < MIN_PRICED_TEST_ROWS or len(train) < 2000:
            continue
        report["windows"][f"year_{year}"] = evaluate_window(train, test, no_mkt, with_mkt, target)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
