"""One-off A/B: do cross-book disagreement features help NFL spread + total?

History:
  * Commit 9dc706a added spread_movement / total_movement (close - open
    line) + bookie-margin features. Commit cbd8fa2 reverted them after
    walk-forward on the 2024 season showed mixed-to-negative results.
  * Memory `nfl-spread-total-efficient` hinted "line-movement TIMING
    (early vs late steam), not just delta" as the next try.
  * Audit on 2026-06-03: NFL historical odds have NO TEMPORAL DATA —
    every snapshot ingested at 2026-06-02 21:10:30, zero is_opening
    flags. The prior reverted "close - open" was effectively
    "same_value - same_value" for most matches. Line-movement timing
    is untestable on this corpus.

Pivot: the data DOES carry cross-book signal — 22 distinct books per
NFL match including Pinnacle (66,954 rows total; recognised industry
sharp book). Pinnacle moves first on sharp action, so its divergence
from public-book consensus is a plausible "extra-market information"
signal that the prior `closing_spread_home` baseline doesn't capture.

Cross-book features per market (spread, total):
  _book_count                books offering this market
  _pinnacle                  Pinnacle's line
  _consensus_mean            mean line across non-Pinnacle books
  _pinnacle_minus_consensus  Pinnacle - consensus_mean (sharp signal)
  _max_minus_min             best - worst line across books
  _std                       std deviation across books
  _pinnacle_implied_prob     Pinnacle's devigged prob (home for spread,
                             over for total)
  _consensus_implied_prob    devigged prob averaged across non-Pinnacle
  _pinnacle_minus_consensus_implied_prob

Methodology mirrors scripts/ab_tennis_weather.py:
  * In-memory cross-book join (no features_cache writes).
  * XGBoost trained twice per market with identical walk-forward split.
  * v1-control = current production features.
  * v_crossbook = v1 + the cross-book features above.
  * Test split = the 2024-2025 season (last in corpus).

Decision rule (per market, independently):
  ΔBrier ≤ -0.005 → KEEP cross-book features for that market.

Run on prod via:
    docker compose exec api python /app/scripts/ab_nfl_cross_book.py
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_nfl_cross_book")

# Latest pre-match (line, odds) per (match, book, market, selection).
# DISTINCT ON dedups when the same book has duplicate rows from
# multiple ingestion runs.
SNAPSHOTS_QUERY = """
    SELECT DISTINCT ON (o.match_id, o.market_type, o.selection, o.bookmaker)
        o.match_id::text AS match_id,
        o.bookmaker,
        o.market_type,
        o.selection,
        o.line,
        o.odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'nfl'
    WHERE m.status = 'finished'
      AND o.is_live = false
      AND o.market_type IN ('spread', 'total')
      AND o.line IS NOT NULL
      AND o.odds_decimal IS NOT NULL
    ORDER BY o.match_id, o.market_type, o.selection, o.bookmaker, o.timestamp DESC
"""

PINNACLE = "Pinnacle"


def _devig_pair(odds_a: float, odds_b: float) -> Tuple[float, float]:
    """Devigged implied prob for a two-outcome market."""
    if odds_a <= 0 or odds_b <= 0:
        return (np.nan, np.nan)
    raw_a = 1.0 / odds_a
    raw_b = 1.0 / odds_b
    total = raw_a + raw_b
    if total <= 0:
        return (np.nan, np.nan)
    return (raw_a / total, raw_b / total)


def _per_market_features(
    snapshots: pd.DataFrame,
    market: str,
    primary_selection: str,
    counter_selection: str,
    prefix: str,
) -> pd.DataFrame:
    """Per-match cross-book features for one market. primary_selection
    is the side whose LINE we extract ('home' for spread, 'over' for
    total). counter_selection is the other side (used for devigging
    implied probs)."""
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
        pinn = g[g["bookmaker"] == PINNACLE]
        non_pinn = g[g["bookmaker"] != PINNACLE]

        pinn_line = float(pinn["p_line"].iloc[0]) if not pinn.empty else np.nan
        pinn_impl = float(pinn["p_impl"].iloc[0]) if not pinn.empty else np.nan
        consensus_line = float(non_pinn["p_line"].mean()) if not non_pinn.empty else np.nan
        consensus_impl = float(non_pinn["p_impl"].mean()) if not non_pinn.empty else np.nan

        rows.append(
            {
                "match_id": match_id,
                f"{prefix}_book_count": float(len(g)),
                f"{prefix}_pinnacle": pinn_line,
                f"{prefix}_consensus_mean": consensus_line,
                f"{prefix}_pinnacle_minus_consensus": (
                    pinn_line - consensus_line if not (np.isnan(pinn_line) or np.isnan(consensus_line)) else np.nan
                ),
                f"{prefix}_max_minus_min": float(lines.max() - lines.min()),
                f"{prefix}_std": (float(np.std(lines, ddof=0)) if len(lines) > 1 else 0.0),
                f"{prefix}_pinnacle_implied_prob": pinn_impl,
                f"{prefix}_consensus_implied_prob": consensus_impl,
                f"{prefix}_pinnacle_minus_consensus_implied_prob": (
                    pinn_impl - consensus_impl if not (np.isnan(pinn_impl) or np.isnan(consensus_impl)) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_crossbook_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    spread = _per_market_features(snapshots, "spread", "home", "away", "spread")
    total = _per_market_features(snapshots, "total", "over", "under", "total")
    if spread.empty and total.empty:
        return pd.DataFrame(columns=["match_id"])
    if spread.empty:
        return total
    if total.empty:
        return spread
    return spread.merge(total, on="match_id", how="outer")


# Pinnacle has 0 NFL rows in this corpus (66,954 Pinnacle rows are all
# soccer). The 4 _pinnacle* features end up all-NaN for NFL, which is
# legal but pollutes the feature search. Strip them.
SPREAD_CROSSBOOK_KEYS = (
    "spread_book_count",
    "spread_consensus_mean",
    "spread_max_minus_min",
    "spread_std",
    "spread_consensus_implied_prob",
)
TOTAL_CROSSBOOK_KEYS = (
    "total_book_count",
    "total_consensus_mean",
    "total_max_minus_min",
    "total_std",
    "total_consensus_implied_prob",
)


def season_walk_forward_split(frame: pd.DataFrame, test_season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["season"] < test_season].copy()
    test = frame[frame["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Walk-forward for season={test_season!r}: train={len(train)} "
            f"test={len(test)}. Seasons present: "
            f"{sorted(frame['season'].dropna().unique())}"
        )
    return train, test


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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


def run_market(
    label,
    loader_name,
    non_feature_set_name,
    target_name,
    crossbook_keys,
    database_url,
    crossbook_features,
    test_season,
):
    sys.path.insert(0, "/app/services/ml-models/src")
    from utils import training_data as td

    LOGGER.info("══════════════════ %s ══════════════════", label)
    base = getattr(td, loader_name)(database_url=database_url)
    LOGGER.info("Loaded %d %s matches.", len(base), label)

    base["match_id"] = base["match_id"].astype(str)
    crossbook_features["match_id"] = crossbook_features["match_id"].astype(str)
    merged = base.merge(crossbook_features, on="match_id", how="left")

    coverage = merged[crossbook_keys[0]].notna().sum()
    LOGGER.info(
        "%s cross-book coverage: %d / %d matches (%.1f%%).",
        label,
        coverage,
        len(merged),
        100.0 * coverage / max(1, len(merged)),
    )
    pinn_key = "spread_pinnacle" if "spread" in label.lower() else "total_pinnacle"
    pinn_cov = merged[pinn_key].notna().sum()
    LOGGER.info(
        "Pinnacle present on %d / %d matches (%.1f%%).",
        pinn_cov,
        len(merged),
        100.0 * pinn_cov / max(1, len(merged)),
    )

    train, test = season_walk_forward_split(merged, test_season)
    LOGGER.info(
        "Walk-forward: %d train, %d test (test_season=%s).",
        len(train),
        len(test),
        test_season,
    )

    target = getattr(td, target_name)
    excluded = set(getattr(td, non_feature_set_name)) | {target}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1 = [c for c in numeric_cols if c not in excluded and c not in crossbook_keys]
    vcb = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1 features: %d  |  v_crossbook features: %d (+%d cross-book keys)",
        len(v1),
        len(vcb),
        len(vcb) - len(v1),
    )

    m_v1 = train_and_eval(train, test, v1, target)
    LOGGER.info(
        "v1-control:   acc=%.4f brier=%.4f ece=%.4f",
        m_v1["accuracy"],
        m_v1["brier"],
        m_v1["ece"],
    )
    m_cb = train_and_eval(train, test, vcb, target)
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
        verdict = f"KEEP cross-book features for {label}"
    elif d_brier >= 0.005:
        verdict = f"DROP cross-book features for {label} (worse Brier)"
    else:
        verdict = f"DROP cross-book features for {label} (|ΔBrier|<0.005)"
    LOGGER.info("VERDICT: %s", verdict)


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

    LOGGER.info("Loading NFL cross-book snapshot stream...")
    from sqlalchemy import create_engine

    engine = create_engine(args.database_url)
    try:
        snapshots = pd.read_sql(SNAPSHOTS_QUERY, engine)
    finally:
        engine.dispose()
    LOGGER.info(
        "Loaded %d (match, book, market, selection) rows across %d matches.",
        len(snapshots),
        snapshots["match_id"].nunique(),
    )
    crossbook = compute_crossbook_features(snapshots)
    LOGGER.info("Cross-book features computed for %d matches.", len(crossbook))

    run_market(
        "SPREAD",
        loader_name="load_nfl_spread_frame",
        non_feature_set_name="NFL_SPREAD_NON_FEATURE_COLUMNS",
        target_name="NFL_SPREAD_TARGET",
        crossbook_keys=SPREAD_CROSSBOOK_KEYS,
        database_url=args.database_url,
        crossbook_features=crossbook,
        test_season=args.test_season,
    )
    run_market(
        "TOTAL",
        loader_name="load_nfl_total_frame",
        non_feature_set_name="NFL_TOTAL_NON_FEATURE_COLUMNS",
        target_name="NFL_TOTAL_TARGET",
        crossbook_keys=TOTAL_CROSSBOOK_KEYS,
        database_url=args.database_url,
        crossbook_features=crossbook,
        test_season=args.test_season,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
