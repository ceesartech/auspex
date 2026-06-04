"""One-off A/B: do cross-book features help NBA moneyline + spread + total?

PR #8 (NFL TOTAL) + PR #12 (NFL ML) both shipped cross-book wins
(ΔBrier -0.0110 / -0.0154). NBA has the same 22-24 book multi-book
structure (24 ML, 23 spread, 23 total) across 3,679 finished
matches. If the cross-book signal is fundamental (book-disagreement
+ consensus features that the existing AVG odds don't capture),
the same pattern should work for NBA.

Tests all 3 markets in one run, mirroring scripts/ab_nfl_cross_book.py
+ scripts/ab_nfl_moneyline_cross_book.py.

Features per market:
  ML     : ml_book_count, ml_consensus_home_prob,
           ml_max_minus_min_home_prob, ml_std_home_prob
  SPREAD : spread_book_count, spread_consensus_mean,
           spread_max_minus_min, spread_std,
           spread_consensus_implied_prob
  TOTAL  : total_book_count, total_consensus_mean,
           total_max_minus_min, total_std,
           total_consensus_implied_prob

(Same naming as NFL so the eventual production landing reuses
fetch_total_crossbook / fetch_moneyline_crossbook helpers.)

Run on prod via:
    docker compose exec api python /app/scripts/ab_nba_cross_book.py
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_nba_cross_book")


# 3 markets in 3 separate queries — DISTINCT ON dedups when the same
# book has multiple rows from re-ingestion.
ML_QUERY = """
    SELECT DISTINCT ON (o.match_id, o.selection, o.bookmaker)
        o.match_id::text AS match_id, o.bookmaker, o.selection,
        o.odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'nba'
    WHERE m.status = 'finished' AND o.is_live = false
      AND o.market_type = 'moneyline'
      AND o.odds_decimal IS NOT NULL
      AND o.selection IN ('home', 'away')
    ORDER BY o.match_id, o.selection, o.bookmaker, o.timestamp DESC
"""

SPREAD_TOTAL_QUERY = """
    SELECT DISTINCT ON (o.match_id, o.market_type, o.selection, o.bookmaker)
        o.match_id::text AS match_id, o.bookmaker, o.market_type,
        o.selection, o.line, o.odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'nba'
    WHERE m.status = 'finished' AND o.is_live = false
      AND o.market_type IN ('spread', 'total')
      AND o.line IS NOT NULL
      AND o.odds_decimal IS NOT NULL
    ORDER BY o.match_id, o.market_type, o.selection, o.bookmaker,
             o.timestamp DESC
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
    """ML cross-book features per match (mirrors NFL ml fetcher)."""
    wide = (
        snapshots
        .pivot_table(
            index=["match_id", "bookmaker"], columns="selection",
            values="odds_decimal", aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    for col in ("home", "away"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.dropna(subset=["home", "away"])
    wide["p_home"] = [
        _devig_pair(h, a)[0] for h, a in zip(wide["home"], wide["away"])
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


def compute_spread_total_features(
    snapshots: pd.DataFrame, market: str, prefix: str,
    primary_selection: str, counter_selection: str,
) -> pd.DataFrame:
    """Spread/total cross-book features (mirrors NFL total fetcher).
    primary_selection drives the line; counter drives the devig."""
    market_rows = snapshots[snapshots["market_type"] == market]
    primary = market_rows[market_rows["selection"] == primary_selection][
        ["match_id", "bookmaker", "line", "odds_decimal"]
    ].rename(columns={"line": "p_line", "odds_decimal": "p_odds"})
    counter = market_rows[market_rows["selection"] == counter_selection][
        ["match_id", "bookmaker", "odds_decimal"]
    ].rename(columns={"odds_decimal": "c_odds"})
    wide = primary.merge(counter, on=["match_id", "bookmaker"], how="inner")
    if wide.empty:
        return pd.DataFrame(columns=["match_id"])

    devigs = [_devig_pair(p, c) for p, c in zip(wide["p_odds"], wide["c_odds"])]
    wide["p_impl"] = [d[0] for d in devigs]

    rows = []
    for match_id, g in wide.groupby("match_id"):
        lines = g["p_line"].astype(float).values
        impls = g["p_impl"].astype(float).dropna().values
        rows.append({
            "match_id": match_id,
            f"{prefix}_book_count": float(len(g)),
            f"{prefix}_consensus_mean": float(lines.mean()),
            f"{prefix}_max_minus_min": float(lines.max() - lines.min()),
            f"{prefix}_std": (
                float(np.std(lines, ddof=0)) if len(lines) > 1 else 0.0
            ),
            f"{prefix}_consensus_implied_prob": (
                float(impls.mean()) if len(impls) > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


ML_KEYS = (
    "ml_book_count", "ml_consensus_home_prob",
    "ml_max_minus_min_home_prob", "ml_std_home_prob",
)
SPREAD_KEYS = (
    "spread_book_count", "spread_consensus_mean", "spread_max_minus_min",
    "spread_std", "spread_consensus_implied_prob",
)
TOTAL_KEYS = (
    "total_book_count", "total_consensus_mean", "total_max_minus_min",
    "total_std", "total_consensus_implied_prob",
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
        "accuracy": float(accuracy_score(y_test, preds)),
        "brier": float(brier_score_loss(y_test, proba)),
        "ece": expected_calibration_error(y_test, proba),
    }


def run_market(
    label, loader_name, non_feature_set_name, target_name,
    market_keys, database_url, market_features, test_season,
):
    sys.path.insert(0, "/app/services/ml-models/src")
    from utils import training_data as td

    LOGGER.info("══════════════════ %s ══════════════════", label)
    base = getattr(td, loader_name)(database_url=database_url)
    LOGGER.info("Loaded %d %s matches.", len(base), label)

    base["match_id"] = base["match_id"].astype(str)
    market_features["match_id"] = market_features["match_id"].astype(str)
    merged = base.merge(market_features, on="match_id", how="left")
    cov = merged[market_keys[0]].notna().sum()
    LOGGER.info(
        "%s cross-book coverage: %d / %d (%.1f%%).",
        label, cov, len(merged), 100.0 * cov / max(1, len(merged)),
    )

    train, test = season_walk_forward_split(merged, test_season)
    LOGGER.info(
        "Walk-forward: train=%d test=%d (test_season=%s).",
        len(train), len(test), test_season,
    )

    target = getattr(td, target_name)
    excluded = set(getattr(td, non_feature_set_name)) | {target}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1 = [c for c in numeric_cols if c not in excluded and c not in market_keys]
    vcb = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1 features: %d  |  v_crossbook features: %d (+%d keys)",
        len(v1), len(vcb), len(vcb) - len(v1),
    )

    m_v1 = train_and_eval(train, test, v1, target)
    LOGGER.info(
        "v1-control:   acc=%.4f brier=%.4f ece=%.4f",
        m_v1["accuracy"], m_v1["brier"], m_v1["ece"],
    )
    m_cb = train_and_eval(train, test, vcb, target)
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
        verdict = f"KEEP cross-book for {label}"
    elif d_brier >= 0.005:
        verdict = f"DROP cross-book for {label} (worse Brier)"
    else:
        verdict = f"DROP cross-book for {label} (|ΔBrier|<0.005)"
    LOGGER.info("VERDICT: %s", verdict)


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

    LOGGER.info("Loading NBA snapshot streams...")
    from sqlalchemy import create_engine
    engine = create_engine(args.database_url)
    try:
        ml_snapshots = pd.read_sql(ML_QUERY, engine)
        st_snapshots = pd.read_sql(SPREAD_TOTAL_QUERY, engine)
    finally:
        engine.dispose()
    LOGGER.info("ML snapshots: %d / Spread+Total snapshots: %d.",
                len(ml_snapshots), len(st_snapshots))

    ml_features = compute_ml_features(ml_snapshots)
    spread_features = compute_spread_total_features(
        st_snapshots, "spread", "spread", "home", "away",
    )
    total_features = compute_spread_total_features(
        st_snapshots, "total", "total", "over", "under",
    )

    run_market(
        "MONEYLINE",
        loader_name="load_nba_moneyline_frame",
        non_feature_set_name="NBA_MONEYLINE_NON_FEATURE_COLUMNS",
        target_name="NBA_MONEYLINE_TARGET",
        market_keys=ML_KEYS,
        database_url=args.database_url,
        market_features=ml_features,
        test_season=args.test_season,
    )
    run_market(
        "SPREAD",
        loader_name="load_nba_spread_frame",
        non_feature_set_name="NBA_SPREAD_NON_FEATURE_COLUMNS",
        target_name="NBA_SPREAD_TARGET",
        market_keys=SPREAD_KEYS,
        database_url=args.database_url,
        market_features=spread_features,
        test_season=args.test_season,
    )
    run_market(
        "TOTAL",
        loader_name="load_nba_total_frame",
        non_feature_set_name="NBA_TOTAL_NON_FEATURE_COLUMNS",
        target_name="NBA_TOTAL_TARGET",
        market_keys=TOTAL_KEYS,
        database_url=args.database_url,
        market_features=total_features,
        test_season=args.test_season,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
