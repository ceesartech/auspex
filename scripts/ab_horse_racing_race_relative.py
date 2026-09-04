"""One-off A/B: do race-relative features help the LambdaMART ranker?

Memory `horse-racing-ml-ranker-v1` option 2: the ranker uses
absolute features (consensus_implied_prob, form, etc.) but doesn't
see the per-race RELATIVE consensus signal. A horse at 5% implied
prob in a 20-runner field is different from a horse at 5% implied
prob in a 5-runner field; the model doesn't currently get that
field context.

Features tested (computed in-memory per race_id group, no
features_cache writes):

  consensus_prob_normalized
      entrant's consensus_implied_prob / sum across race. Each
      race's values sum to 1.0 so the model sees pure within-race
      ranking, scrubbed of field-size correlation.

  consensus_rank
      1 = favorite, 2 = second favorite, ... ; ties broken
      stably by entrant index. Captures ordinal position which
      raw prob doesn't directly encode.

  consensus_gap_to_favorite
      max(consensus_implied_prob in race) - this entrant's. 0 for
      the favorite. Tells the model how far back from chalk this
      runner is in implied-prob terms.

Methodology mirrors NFL/tennis A/B scripts:
  * Load training frame from horse_racing_data.load_training_frame.
  * Build race-relative features over the existing
    consensus_implied_prob column.
  * Walk-forward split at --split-date (default 2026-05-15, matches
    the production ranker's split per memory).
  * Train HorseRacingRanker twice (same config, same val slice).
  * Compare top1, MRR, NLL, Brier — Brier is the load-bearing
    metric because the ranker's calibration is the gap we're
    trying to close (test Brier 0.1050 vs consensus 0.0831).

Decision rule (Brier-first, matches the ranker's existing
KEEP/DROP framing in memory):
  ΔBrier ≤ -0.005 → KEEP race-relative features.

Run on prod via:
    docker compose exec api python /app/scripts/ab_horse_racing_race_relative.py
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_horse_racing_race_relative")


RACE_RELATIVE_KEYS = (
    "consensus_prob_normalized",
    "consensus_rank",
    "consensus_gap_to_favorite",
)


def add_race_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-race relative features over consensus_implied_prob.

    All three keys are populated for every row. Race-groups with all
    NaN consensus_implied_prob (no consensus prediction for any
    entrant) get the neutral fallback: normalized = 1/field_size,
    rank = field_size middle, gap = 0. Mirrors the
    NEUTRAL_DEFAULTS pattern in compute_features_horse_racing for
    rows the ranker would otherwise see as 3-way-NaN."""
    out = frame.copy()
    out[list(RACE_RELATIVE_KEYS)] = np.nan

    if "race_id" not in out.columns or "consensus_implied_prob" not in out.columns:
        return out

    for race_id, idx in out.groupby("race_id", sort=False).groups.items():
        slice_idx = list(idx)
        probs = out.loc[slice_idx, "consensus_implied_prob"].astype(float)
        field_size = len(slice_idx)

        valid = probs.notna()
        if valid.sum() == 0:
            # No consensus for any runner — neutral fallback.
            uniform = 1.0 / field_size if field_size > 0 else np.nan
            out.loc[slice_idx, "consensus_prob_normalized"] = uniform
            out.loc[slice_idx, "consensus_rank"] = (field_size + 1) / 2.0
            out.loc[slice_idx, "consensus_gap_to_favorite"] = 0.0
            continue

        total = float(probs[valid].sum())
        if total > 0:
            normalized = probs / total
        else:
            normalized = pd.Series([1.0 / field_size] * len(probs), index=probs.index)
        out.loc[slice_idx, "consensus_prob_normalized"] = normalized.values

        # Rank: 1 = highest implied prob. NaN-aware: NaN rows get
        # the worst rank (field_size) so the model can tell them
        # apart from "ranked last with a real number."
        ranks = probs.rank(method="min", ascending=False, na_option="bottom")
        out.loc[slice_idx, "consensus_rank"] = ranks.values

        favorite = float(probs[valid].max())
        out.loc[slice_idx, "consensus_gap_to_favorite"] = (favorite - probs.fillna(favorite)).values

    return out


def evaluate_variant(
    label: str,
    frame: pd.DataFrame,
    split_date: str,
    drop_relative: bool,
    val_fraction: float,
) -> dict:
    """Train + evaluate one variant. drop_relative=True gives the
    v1-control, =False gives v_relative."""
    sys.path.insert(0, "/app/services/ml-models/src")
    from predictors.horse_racing_ranker import HorseRacingRanker
    from utils.horse_racing_data import get_feature_columns, group_array, split_by_date

    # Import helpers from the production trainer so the A/B's
    # train/val split + Brier math match what gets shipped.
    sys.path.insert(0, "/app/scripts")
    from train_horse_racing_win import _per_entrant_brier, _split_train_val

    LOGGER.info("══════════════ %s ══════════════", label)
    work = frame.copy()
    if drop_relative:
        work = work.drop(columns=list(RACE_RELATIVE_KEYS), errors="ignore")
    train_frame, test_frame = split_by_date(work, split_date)

    train_inner, val_inner = _split_train_val(train_frame, val_fraction)
    feature_cols = get_feature_columns(train_inner)
    LOGGER.info(
        "%s: features=%d train_rows=%d val_rows=%d test_rows=%d",
        label,
        len(feature_cols),
        len(train_inner),
        len(val_inner),
        len(test_frame),
    )

    X_train = train_inner[feature_cols]
    y_train = train_inner["target"].to_numpy(dtype=np.int64)
    g_train = group_array(train_inner)

    X_val = val_inner[feature_cols] if not val_inner.empty else None
    y_val = val_inner["target"].to_numpy(dtype=np.int64) if not val_inner.empty else None
    g_val = group_array(val_inner) if not val_inner.empty else None

    X_test = test_frame[feature_cols]
    y_test = test_frame["target"].to_numpy(dtype=np.int64)
    g_test = group_array(test_frame)

    model = HorseRacingRanker()
    model.fit(
        X_train=X_train,
        y_train=y_train,
        groups_train=g_train,
        X_val=X_val,
        y_val=y_val,
        groups_val=g_val,
    )
    test_metrics = model._evaluate(X_test, y_test, g_test)
    test_probs = model.predict_probabilities(X_test, g_test)
    brier = _per_entrant_brier(test_probs, g_test, y_test)
    out = {
        "label": label,
        "n_features": len(feature_cols),
        "top1": test_metrics.get("top1_accuracy", 0.0),
        "mrr": test_metrics.get("mrr", 0.0),
        "nll": test_metrics.get("nll", 0.0),
        "brier": brier,
        "races": test_metrics.get("races", 0),
    }
    LOGGER.info(
        "%s: top1=%.4f mrr=%.4f nll=%.4f brier=%.4f races=%d",
        out["label"],
        out["top1"],
        out["mrr"],
        out["nll"],
        out["brier"],
        out["races"],
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--split-date",
        default="2026-05-15",
        help="Walk-forward split date (default matches production ranker).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
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
    from utils.horse_racing_data import load_training_frame

    LOGGER.info("Loading ranker training frame...")
    frame = load_training_frame(database_url=args.database_url)
    LOGGER.info(
        "Loaded %d entrant rows across %d races.",
        len(frame),
        frame["race_id"].nunique() if "race_id" in frame else 0,
    )

    LOGGER.info("Computing race-relative features...")
    frame_with_relative = add_race_relative_features(frame)
    coverage = frame_with_relative["consensus_prob_normalized"].notna().sum()
    LOGGER.info(
        "Race-relative coverage: %d / %d rows (%.1f%%).",
        coverage,
        len(frame_with_relative),
        100.0 * coverage / max(1, len(frame_with_relative)),
    )

    v1 = evaluate_variant(
        "v1-control",
        frame_with_relative,
        split_date=args.split_date,
        drop_relative=True,
        val_fraction=args.val_fraction,
    )
    vrel = evaluate_variant(
        "v_relative",
        frame_with_relative,
        split_date=args.split_date,
        drop_relative=False,
        val_fraction=args.val_fraction,
    )

    d_top1 = vrel["top1"] - v1["top1"]
    d_mrr = vrel["mrr"] - v1["mrr"]
    d_brier = vrel["brier"] - v1["brier"]
    d_nll = vrel["nll"] - v1["nll"]

    LOGGER.info("")
    LOGGER.info("───────────── DELTAS (v_relative - v1) ─────────────")
    LOGGER.info("Δ top1:  %+.4f (positive favours v_relative)", d_top1)
    LOGGER.info("Δ mrr:   %+.4f (positive favours v_relative)", d_mrr)
    LOGGER.info("Δ nll:   %+.4f (NEGATIVE favours v_relative)", d_nll)
    LOGGER.info("Δ brier: %+.4f (NEGATIVE favours v_relative)", d_brier)
    if d_brier <= -0.005:
        verdict = "KEEP race-relative features"
    elif d_brier >= 0.005:
        verdict = "DROP race-relative features (worse Brier)"
    else:
        verdict = "DROP race-relative features (|ΔBrier|<0.005)"
    LOGGER.info("VERDICT: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
