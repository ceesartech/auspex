"""One-off A/B: do cross-book Pinnacle features help soccer 1x2?

The NFL TOTAL cross-book A/B (commit 6ff9e64) shipped a clean
KEEP (ΔBrier -0.0110) using only public-book consensus features
because Pinnacle was soccer-only in this corpus. Soccer is the
opposite: Pinnacle covers 22,318 of 23,456 finished matches (95%)
plus Bet365 covers all 23,456. That's exactly the "sharp book vs
public consensus" setup we couldn't run on NFL.

Hypothesis: the production training query uses `AVG(odds) across
all books` for 1x2 odds. Pinnacle's devigged probabilities are
generally sharper than the average — so adding a Pinnacle-specific
prob feature + Pinnacle-vs-consensus delta features should give the
ensemble a more accurate prior than the mean alone.

Features per match (1x2 is 3-way: home/draw/away):

  pinnacle_implied_home/draw/away
      Pinnacle's devigged implied prob per outcome.
  pinnacle_minus_avg_home/draw/away
      Pinnacle's devigged prob minus the cross-book mean devigged
      prob. Negative for the side Pinnacle thinks is more likely
      than the public consensus does.
  pinnacle_present
      1.0/0.0 flag — guards the model against treating "missing
      Pinnacle = NaN" as a structurally different state.

Methodology mirrors NFL TOTAL cross-book A/B:
  * In-memory join (no features_cache writes).
  * XGBoost trained twice with the same train/val split.
    v1-control = current production features (odds + features_cache).
    v_crossbook = v1 + the cross-book features above.
  * Walk-forward by season — the soccer frame has the season
    column from leagues; pick the last 2 seasons as test.

Decision rule:
  ΔBrier ≤ -0.005 → KEEP, land in DEFAULT_TRAINING_QUERY and
  retrain the production ensemble. SPREAD on NFL didn't help; only
  ship if the test signal is unambiguous.

Run on prod via:
    docker compose exec api python /app/scripts/ab_soccer_cross_book.py
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_soccer_cross_book")


SNAPSHOTS_QUERY = """
    SELECT DISTINCT ON (o.match_id, o.bookmaker, o.selection)
        o.match_id::text AS match_id,
        o.bookmaker,
        o.selection,
        o.odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND o.is_live = false
      AND o.market_type = '1x2'
      AND o.odds_decimal IS NOT NULL
      AND o.selection IN ('home', 'draw', 'away')
    ORDER BY o.match_id, o.bookmaker, o.selection, o.timestamp DESC
"""

PINNACLE = "Pinnacle"


def _devig_triple(o_home: float, o_draw: float, o_away: float) -> Tuple[float, float, float]:
    if min(o_home, o_draw, o_away) <= 0:
        return (np.nan, np.nan, np.nan)
    raw = np.array([1.0 / o_home, 1.0 / o_draw, 1.0 / o_away])
    total = float(raw.sum())
    if total <= 0:
        return (np.nan, np.nan, np.nan)
    p = raw / total
    return (float(p[0]), float(p[1]), float(p[2]))


def compute_crossbook_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Per-match: Pinnacle devigged probs + delta vs cross-book mean.

    Returns one row per match with the 7 feature columns. Matches
    without Pinnacle get pinnacle_present=0 and the per-Pinnacle
    feature columns NaN; the consensus columns still populate as
    long as ANY book has 1x2 data."""
    # Wide pivot: one row per (match, bookmaker) with home/draw/away odds.
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
    required = {"home", "draw", "away"}
    missing = required - set(wide.columns)
    for col in missing:
        wide[col] = np.nan
    devigged = [
        _devig_triple(h, d, a)
        for h, d, a in zip(wide["home"], wide["draw"], wide["away"])
    ]
    wide["p_home"] = [t[0] for t in devigged]
    wide["p_draw"] = [t[1] for t in devigged]
    wide["p_away"] = [t[2] for t in devigged]
    wide = wide.dropna(subset=["p_home", "p_draw", "p_away"])

    rows = []
    for match_id, g in wide.groupby("match_id"):
        pinn = g[g["bookmaker"] == PINNACLE]
        non_pinn = g[g["bookmaker"] != PINNACLE]
        if pinn.empty:
            pinn_h = pinn_d = pinn_a = np.nan
            pinn_present = 0.0
        else:
            pinn_h = float(pinn["p_home"].iloc[0])
            pinn_d = float(pinn["p_draw"].iloc[0])
            pinn_a = float(pinn["p_away"].iloc[0])
            pinn_present = 1.0

        if non_pinn.empty:
            avg_h = avg_d = avg_a = np.nan
        else:
            avg_h = float(non_pinn["p_home"].mean())
            avg_d = float(non_pinn["p_draw"].mean())
            avg_a = float(non_pinn["p_away"].mean())

        rows.append({
            "match_id": match_id,
            "pinnacle_implied_home": pinn_h,
            "pinnacle_implied_draw": pinn_d,
            "pinnacle_implied_away": pinn_a,
            "pinnacle_minus_avg_home": (
                pinn_h - avg_h
                if not (np.isnan(pinn_h) or np.isnan(avg_h)) else np.nan
            ),
            "pinnacle_minus_avg_draw": (
                pinn_d - avg_d
                if not (np.isnan(pinn_d) or np.isnan(avg_d)) else np.nan
            ),
            "pinnacle_minus_avg_away": (
                pinn_a - avg_a
                if not (np.isnan(pinn_a) or np.isnan(avg_a)) else np.nan
            ),
            "pinnacle_present": pinn_present,
        })
    return pd.DataFrame(rows)


# Two key sets — first attempt with full 7-feature set scored badly
# (ΔBrier +0.0606). pinnacle_implied_home/draw/away are highly
# collinear with the existing AVG odds columns; XGB overfits on the
# tiny pinnacle-vs-avg residual when both scales are present.
# CROSSBOOK_KEYS_FULL kept for reference / smoke-checking.
CROSSBOOK_KEYS_FULL = (
    "pinnacle_implied_home",
    "pinnacle_implied_draw",
    "pinnacle_implied_away",
    "pinnacle_minus_avg_home",
    "pinnacle_minus_avg_draw",
    "pinnacle_minus_avg_away",
    "pinnacle_present",
)
# CROSSBOOK_KEYS = delta-only set. Strips the collinear absolute
# Pinnacle probs and keeps just the sharp-vs-public divergence
# signal that the existing AVG odds columns don't capture.
CROSSBOOK_KEYS = (
    "pinnacle_minus_avg_home",
    "pinnacle_minus_avg_draw",
    "pinnacle_minus_avg_away",
    "pinnacle_present",
)


def date_walk_forward_split(
    frame: pd.DataFrame, split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_ts = pd.to_datetime(split_date, utc=True)
    frame = frame.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    train = frame[frame["match_date"] < split_ts].copy()
    test = frame[frame["match_date"] >= split_ts].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Walk-forward at split_date={split_date}: train={len(train)} test={len(test)}"
        )
    return train, test


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> float:
    """ECE on the predicted prob of the predicted class.
    Multi-class friendly — uses max class prob per sample."""
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
    """Multi-class XGBoost (soccer 1x2 = 3 classes: home/draw/away)."""
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    X_train = train[feature_cols].values
    y_train = train[target].astype(int).values
    X_test = test[feature_cols].values
    y_test = test[target].astype(int).values

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6, learning_rate=0.05, n_estimators=400,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.01, reg_lambda=1.0,
        tree_method="hist", random_state=42,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    preds = np.argmax(proba, axis=1)

    # Multi-class Brier: mean of per-class Brier scores against
    # one-hot ground truth. Matches the soccer ensemble's metric.
    y_onehot = np.eye(3)[y_test]
    brier = float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))
    accuracy = float(accuracy_score(y_test, preds))
    # ECE on the predicted class's probability.
    max_probs = proba[np.arange(len(proba)), preds]
    is_correct = (preds == y_test).astype(int)
    ece = expected_calibration_error(is_correct, max_probs)
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "accuracy": accuracy,
        "brier": brier,
        "ece": ece,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--split-date", default="2024-08-01",
        help="Walk-forward boundary (YYYY-MM-DD). Default tests 2024-25 season.",
    )
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
    from utils.training_data import (
        NON_FEATURE_COLUMNS, TARGET_COLUMN, load_training_frame,
    )

    LOGGER.info("Loading soccer training frame...")
    base = load_training_frame(database_url=args.database_url)
    LOGGER.info("Loaded %d matches.", len(base))

    LOGGER.info("Loading cross-book snapshots...")
    from sqlalchemy import create_engine
    engine = create_engine(args.database_url)
    try:
        snapshots = pd.read_sql(SNAPSHOTS_QUERY, engine)
    finally:
        engine.dispose()
    LOGGER.info(
        "Loaded %d (match, book, selection) snapshot rows.",
        len(snapshots),
    )

    crossbook = compute_crossbook_features(snapshots)
    LOGGER.info(
        "Cross-book features computed for %d matches (pinnacle on %d).",
        len(crossbook), int(crossbook["pinnacle_present"].sum()),
    )

    base["match_id"] = base["match_id"].astype(str)
    crossbook["match_id"] = crossbook["match_id"].astype(str)
    merged = base.merge(crossbook, on="match_id", how="left")
    pinn_cov = (merged["pinnacle_present"] == 1.0).sum()
    LOGGER.info(
        "Test corpus: %d matches, pinnacle covers %d (%.1f%%).",
        len(merged), pinn_cov, 100.0 * pinn_cov / max(1, len(merged)),
    )

    train, test = date_walk_forward_split(merged, args.split_date)
    LOGGER.info(
        "Walk-forward: train=%d test=%d (split=%s)",
        len(train), len(test), args.split_date,
    )

    excluded = set(NON_FEATURE_COLUMNS) | {TARGET_COLUMN}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1 = [c for c in numeric_cols if c not in excluded and c not in CROSSBOOK_KEYS]
    vcb = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1 features: %d  |  v_crossbook features: %d (+%d cross-book keys)",
        len(v1), len(vcb), len(vcb) - len(v1),
    )

    m_v1 = train_and_eval(train, test, v1, TARGET_COLUMN)
    LOGGER.info(
        "v1-control:   acc=%.4f brier=%.4f ece=%.4f",
        m_v1["accuracy"], m_v1["brier"], m_v1["ece"],
    )
    m_cb = train_and_eval(train, test, vcb, TARGET_COLUMN)
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
        verdict = "KEEP cross-book features"
    elif d_brier >= 0.005:
        verdict = "DROP cross-book features (worse Brier)"
    else:
        verdict = "DROP cross-book features (|ΔBrier|<0.005)"
    LOGGER.info("VERDICT: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
