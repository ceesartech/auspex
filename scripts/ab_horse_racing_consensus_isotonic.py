"""One-off A/B: does isotonic post-calibration fix the consensus's
known 40+ overconfidence?

Memory `horse-racing-baseline` documents a real, well-characterised
bias in market_consensus_v1:
  bucket  predicted  actual  gap
  40-50%  0.443      0.338   -10.6pts   ← overconfident
  50+%    0.574      0.500   -7.4pts    ← overconfident
  below 40%: calibrated within ±2.4pts

EVERY rec the consensus generates for a horse at 40%+ implied prob
is built on inflated probability → inflated EV → false-positive
rec. Isotonic regression fit on (raw_prob → actual_outcome) pairs
from the graded history would learn to pull down the 40+ buckets
while leaving the lower buckets alone (since isotonic is
monotonic).

Methodology:
  * Pull all graded race_predictions for model_name=
    market_consensus_v1 (129k rows / 15.7k races / 12 months).
  * Walk-forward split by race_date: train on the first 10 months
    (2025-06-03 → 2026-04-30), test on the last month
    (2026-05-01 → 2026-06-02). Race-level integrity preserved —
    every entrant of a race shares the race_date.
  * Fit sklearn IsotonicRegression on train (raw_confidence,
    actual_outcome) pairs.
  * Apply to test; renormalize per race so probs sum to 1 again
    (isotonic breaks the per-race normalisation by mapping each
    raw prob through an independent monotonic function).
  * Compare overall Brier, per-bucket calibration, and top-1
    accuracy. Top-1 accuracy MUST be unchanged because isotonic
    is monotone and the per-race renormalisation preserves order.

Decision rule:
  ΔBrier ≤ -0.001 (tighter than feature A/Bs since calibration
  changes work in tenths-of-a-thousandth of Brier and the prior is
  that the 40+ bucket fix MUST help if it works at all).

Run on prod via:
    docker compose exec api python /app/scripts/ab_horse_racing_consensus_isotonic.py
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_horse_racing_consensus_isotonic")


GRADED_QUERY = """
    SELECT
        rp.race_id::text AS race_id,
        rp.entrant_id::text AS entrant_id,
        r.race_date,
        rp.confidence::float AS raw_prob,
        rp.actual_outcome::float AS actual
    FROM race_predictions rp
    JOIN races r ON r.id = rp.race_id
    WHERE rp.model_name = 'market_consensus_v1'
      AND rp.prediction_type = 'win'
      AND rp.actual_outcome IS NOT NULL
      AND rp.confidence IS NOT NULL
    ORDER BY r.race_date, rp.race_id, rp.entrant_id
"""


# Memory's canonical bucket structure so the A/B output is directly
# comparable to the recorded baseline numbers.
BUCKET_EDGES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]
BUCKET_LABELS = [
    "00-05", "05-10", "10-15", "15-20", "20-25",
    "25-30", "30-40", "40-50", "50+ ",
]


def per_race_renormalize(df: pd.DataFrame, prob_col: str) -> np.ndarray:
    """Per-race renormalize so probs sum to 1 within each race.
    Without this isotonic outputs can sum to >1 or <1 per race,
    which would break downstream EV math."""
    sums = df.groupby("race_id")[prob_col].transform("sum")
    out = df[prob_col].astype(float).to_numpy() / sums.replace(0, 1).to_numpy()
    return out


def per_race_brier(df: pd.DataFrame, prob_col: str) -> float:
    """Mean per-race Brier on the entrant-level prob/outcome pairs.
    Mirrors the trainer's _per_entrant_brier so the number compares
    directly to the consensus baseline 0.0831 quoted in memory."""
    total = 0.0
    n = 0
    for _race, grp in df.groupby("race_id", sort=False):
        if grp["actual"].sum() == 0:
            continue
        diff = grp[prob_col].astype(float).to_numpy() - grp["actual"].astype(float).to_numpy()
        total += float(np.mean(diff ** 2))
        n += 1
    return total / n if n else float("inf")


def per_bucket_table(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """Per-bucket calibration table matching the memory format."""
    rows = []
    bucket_idx = pd.cut(
        df[prob_col].astype(float),
        bins=BUCKET_EDGES,
        labels=BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )
    for label, grp in df.groupby(bucket_idx, observed=True):
        if len(grp) == 0:
            continue
        rows.append({
            "bucket": str(label),
            "predicted": float(grp[prob_col].mean()),
            "actual": float(grp["actual"].mean()),
            "n": int(len(grp)),
        })
    table = pd.DataFrame(rows)
    table["gap"] = table["actual"] - table["predicted"]
    return table


def top1_accuracy(df: pd.DataFrame, prob_col: str) -> float:
    """Per-race: was the model's argmax-prob entrant the actual
    winner? Isotonic + renormalize is monotonic so this should
    be identical to raw."""
    hits = 0
    n = 0
    for _race, grp in df.groupby("race_id", sort=False):
        if grp["actual"].sum() == 0:
            continue
        n += 1
        idx = grp[prob_col].astype(float).idxmax()
        if int(grp.loc[idx, "actual"]) == 1:
            hits += 1
    return hits / n if n else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--split-date", default="2026-05-01",
        help="Walk-forward boundary (YYYY-MM-DD). Races on/after go to test.",
    )
    parser.add_argument(
        "--train-window-days", type=int, default=90,
        help="Days back from split-date to use for the 'recent' isotonic.",
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

    from sqlalchemy import create_engine
    from sklearn.isotonic import IsotonicRegression

    LOGGER.info("Loading graded consensus predictions...")
    engine = create_engine(args.database_url)
    try:
        df = pd.read_sql(GRADED_QUERY, engine)
    finally:
        engine.dispose()
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    LOGGER.info(
        "Loaded %d graded predictions across %d races (%s..%s).",
        len(df), df["race_id"].nunique(),
        df["race_date"].min().date(), df["race_date"].max().date(),
    )

    split_ts = pd.to_datetime(args.split_date, utc=True)
    train_full = df[df["race_date"] < split_ts].copy()
    test = df[df["race_date"] >= split_ts].copy()
    # Time-aware calibration: also try fitting on just the last
    # `args.train_window_days` of train data. If the overconfidence
    # drifts over time (memory's -10.6 vs test's -21.5 at 40-50%
    # suggests it does), recent-only fit may generalise better than
    # all-of-history.
    recent_start = split_ts - pd.Timedelta(days=args.train_window_days)
    train_recent = train_full[train_full["race_date"] >= recent_start].copy()
    LOGGER.info(
        "Walk-forward: train_full=%d (%d races) train_recent=%d (%d races, last %d days) test=%d (%d races) split=%s",
        len(train_full), train_full["race_id"].nunique(),
        len(train_recent), train_recent["race_id"].nunique(),
        args.train_window_days,
        len(test), test["race_id"].nunique(), args.split_date,
    )

    # Fit isotonic on full + recent train pairs.
    iso_full = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True,
    )
    iso_full.fit(train_full["raw_prob"].to_numpy(), train_full["actual"].to_numpy())
    iso_recent = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True,
    )
    iso_recent.fit(train_recent["raw_prob"].to_numpy(), train_recent["actual"].to_numpy())
    iso = iso_full  # keep `iso` name for the rest of the script (full-history default)
    LOGGER.info(
        "Isotonic fits: full=%d knots, recent=%d knots.",
        len(iso_full.X_thresholds_), len(iso_recent.X_thresholds_),
    )

    # Apply to test. Compare 4 variants:
    #   raw                — unchanged consensus
    #   calibrated_full    — full-history isotonic, no renorm
    #   calibrated_recent  — recent-window isotonic, no renorm
    #   calibrated         — full-history isotonic + per-race renorm (prod shape)
    test["calibrated_raw"] = iso_full.predict(test["raw_prob"].to_numpy())
    test["calibrated_recent_raw"] = iso_recent.predict(test["raw_prob"].to_numpy())
    test["calibrated"] = per_race_renormalize(test, "calibrated_raw")
    test["calibrated_recent"] = per_race_renormalize(test, "calibrated_recent_raw")
    test["raw"] = test["raw_prob"]

    LOGGER.info("")
    LOGGER.info("═════════ Overall test metrics ═════════")
    raw_brier = per_race_brier(test, "raw")
    full_brier = per_race_brier(test, "calibrated_raw")
    recent_brier = per_race_brier(test, "calibrated_recent_raw")
    full_renorm_brier = per_race_brier(test, "calibrated")
    recent_renorm_brier = per_race_brier(test, "calibrated_recent")
    raw_top1 = top1_accuracy(test, "raw")
    full_top1 = top1_accuracy(test, "calibrated_raw")
    LOGGER.info("raw                          :: brier=%.4f top1=%.4f", raw_brier, raw_top1)
    LOGGER.info("calibrated_full   (norenorm) :: brier=%.4f", full_brier)
    LOGGER.info("calibrated_recent (norenorm) :: brier=%.4f", recent_brier)
    LOGGER.info("calibrated_full   (renorm)   :: brier=%.4f", full_renorm_brier)
    LOGGER.info("calibrated_recent (renorm)   :: brier=%.4f", recent_renorm_brier)
    LOGGER.info("")
    LOGGER.info("Δ brier vs raw (lower = better):")
    LOGGER.info("  full   norenorm: %+.4f", full_brier - raw_brier)
    LOGGER.info("  recent norenorm: %+.4f", recent_brier - raw_brier)
    LOGGER.info("  full   renorm:   %+.4f", full_renorm_brier - raw_brier)
    LOGGER.info("  recent renorm:   %+.4f", recent_renorm_brier - raw_brier)
    LOGGER.info("Δ top1: %+.4f  (should be ~0)", full_top1 - raw_top1)
    # Keep cal_brier name pointing to the production-shape variant
    # (full-history + renorm) for the verdict line below.
    cal_brier = full_renorm_brier

    LOGGER.info("")
    LOGGER.info("═════════ Per-bucket calibration (test set) ═════════")
    LOGGER.info("Bucket  | raw_pred  raw_actual  raw_gap | cal_pred  cal_actual  cal_gap | n")
    raw_table = per_bucket_table(test, "raw").set_index("bucket")
    cal_table = per_bucket_table(test, "calibrated").set_index("bucket")
    for label in BUCKET_LABELS:
        if label not in raw_table.index:
            continue
        rr = raw_table.loc[label]
        cc = cal_table.loc[label] if label in cal_table.index else None
        if cc is None:
            LOGGER.info(
                "%-7s | %.3f     %.3f       %+.3f | (bucket empty in calibrated)",
                label, rr["predicted"], rr["actual"], rr["gap"],
            )
            continue
        LOGGER.info(
            "%-7s | %.3f     %.3f       %+.3f | %.3f     %.3f       %+.3f | %d",
            label, rr["predicted"], rr["actual"], rr["gap"],
            cc["predicted"], cc["actual"], cc["gap"], int(rr["n"]),
        )

    LOGGER.info("")
    if cal_brier - raw_brier <= -0.001:
        verdict = "KEEP isotonic calibration"
    elif cal_brier - raw_brier >= 0.001:
        verdict = "DROP isotonic calibration (Brier worse)"
    else:
        verdict = "neutral (|ΔBrier|<0.001) — calibration doesn't move the needle"
    LOGGER.info("VERDICT: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
