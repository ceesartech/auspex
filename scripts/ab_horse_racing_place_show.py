"""One-off A/B: does a multi-grade ranker (label_gain=[0,1,2,3]) help
horse-racing place + show markets without giving up win-market quality?

Memory `horse-racing-ml-ranker-v1` option 5: lift LambdaRank
truncation to top-3, train so the model sees ordinal structure
(1st > 2nd > 3rd > rest) instead of winner-take-all. Hypothesis:
the extra signal lets the ranker learn finishing-position
ordering without losing top-1 quality, AND produces useful
place/show probabilities for free.

A/B at split_date=2026-05-15 on the 18k-race corpus:
  v_win   : label_gain=[0,1], eval_at_n=1 (current production shape)
  v_multi : label_gain=[0,1,2,3], eval_at_n=3 (memory's option 5)

Both models evaluated on three markets:
  WIN   : top1 hit rate, Brier on (1=winner, 0=loser)
  PLACE : top2 hit rate, Brier on (1=top2, 0=not top2)
  SHOW  : top3 hit rate, Brier on (1=top3, 0=not top3)

Top-K hit rate per race: are the model's K highest-score entrants
exactly the K actual top finishers (set intersection >= K)?

Decision rules (per market, independently):
  KEEP v_multi for that market if its Brier is at least 0.005
  better than v_win's Brier in that market.

If v_multi wins place AND show without giving up >0.005 on win
Brier, it's strictly better than v_win and should replace the
production ranker plus unlock the two new markets.

Run on prod via:
    docker compose exec api python /app/scripts/ab_horse_racing_place_show.py
"""

import argparse
import logging
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_horse_racing_place_show")


def derive_multigrade_target(finish_position: pd.Series) -> np.ndarray:
    """Map finish_position → label_gain index.
    1 → 3 (best), 2 → 2, 3 → 1, 4+ or NaN → 0."""
    out = np.zeros(len(finish_position), dtype=np.int64)
    pos = pd.to_numeric(finish_position, errors="coerce")
    out[pos == 1] = 3
    out[pos == 2] = 2
    out[pos == 3] = 1
    # 4+ and NaN stay 0.
    return out


def top_k_hit_rate(
    scores: np.ndarray, groups: np.ndarray, y_topk: np.ndarray, k: int,
) -> float:
    """Per-race: are the model's k highest-score entrants exactly
    the k actual top-finishers? Returns mean over races of
    (intersection size / k)."""
    cursor = 0
    hits = 0
    n = 0
    for size in groups:
        size = int(size)
        race_scores = scores[cursor:cursor + size]
        race_actual = y_topk[cursor:cursor + size]
        cursor += size
        if race_actual.sum() == 0:
            # Race with fewer than k actual top finishers (rare
            # truncation case) — skip rather than penalize.
            continue
        # Take indices of top-k scores (ties broken by argsort
        # which is stable). The set intersection with actual
        # top-k positions is the hit count.
        topk_idx = np.argsort(-race_scores, kind="stable")[:k]
        hit_count = int(np.sum(race_actual[topk_idx]))
        hits += hit_count / k
        n += 1
    return hits / n if n else 0.0


def brier_on_target(
    race_probs: list, groups: np.ndarray, y_target: np.ndarray,
) -> float:
    """Per-entrant Brier with race_probs being a list of np arrays
    (one per race, from predict_probabilities) and y_target a flat
    array aligned with the original frame."""
    total = 0.0
    n = 0
    cursor = 0
    for race_arr, size in zip(race_probs, groups):
        size = int(size)
        actuals = y_target[cursor:cursor + size].astype(np.float64)
        cursor += size
        if actuals.sum() == 0:
            continue
        total += float(np.mean((race_arr - actuals) ** 2))
        n += 1
    return total / n if n else float("inf")


def train_variant(label: str, frame: pd.DataFrame, split_date: str,
                  variant: str, val_fraction: float) -> dict:
    """Train + score one variant. Returns the trained model + test
    arrays needed for downstream market evaluation.

    variant ∈ {"win_binary", "multi_grade", "place_binary",
    "show_binary"}. The first two are general models scored on all
    three markets; place_binary / show_binary are dedicated binary
    rankers trained against their market's truth (top-2 / top-3
    finishers) for a market-appropriate Brier comparison."""
    sys.path.insert(0, "/app/services/ml-models/src")
    from predictors.horse_racing_ranker import (
        HorseRacingRanker, HorseRacingRankerConfig,
    )
    from utils.horse_racing_data import (
        get_feature_columns, group_array, split_by_date,
    )

    sys.path.insert(0, "/app/scripts")
    from train_horse_racing_win import _split_train_val

    LOGGER.info("══════════════ %s ══════════════", label)
    work = frame.copy()
    config = HorseRacingRankerConfig()
    if variant == "multi_grade":
        work["target"] = derive_multigrade_target(work["finish_position"])
        config = replace(config, label_gain=[0, 1, 2, 3], eval_at_n=3)
    elif variant == "place_binary":
        # Dedicated PLACE ranker: target = 1 if finish_position ≤ 2.
        # Stays binary so the softmax → Brier comparison is calibrated
        # against the same outcome the model was optimized for.
        work["target"] = work["_is_place"].astype(np.int64)
        config = replace(config, eval_at_n=2)
    elif variant == "show_binary":
        # Dedicated SHOW ranker — same shape, target = top-3.
        work["target"] = work["_is_show"].astype(np.int64)
        config = replace(config, eval_at_n=3)
    elif variant == "win_binary":
        # Production-shape control: target stays as the existing
        # binary (finish_position == 1).
        pass
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    train_frame, test_frame = split_by_date(work, split_date)
    train_inner, val_inner = _split_train_val(train_frame, val_fraction)
    feature_cols = get_feature_columns(train_inner)
    # Strip the per-market truth columns we added to `frame` for
    # downstream evaluation — get_feature_columns is generic and
    # would otherwise let them leak into X_train as features
    # (gives NDCG=1.0 at iteration 1, fake-perfect metrics).
    feature_cols = [
        c for c in feature_cols
        if c not in {"_is_win", "_is_place", "_is_show"}
    ]
    LOGGER.info(
        "%s: train_rows=%d val_rows=%d test_rows=%d features=%d label_gain=%s",
        label, len(train_inner), len(val_inner), len(test_frame),
        len(feature_cols), config.label_gain,
    )

    X_train = train_inner[feature_cols]
    y_train = train_inner["target"].to_numpy(dtype=np.int64)
    g_train = group_array(train_inner)

    X_val = val_inner[feature_cols] if not val_inner.empty else None
    y_val = (
        val_inner["target"].to_numpy(dtype=np.int64)
        if not val_inner.empty else None
    )
    g_val = group_array(val_inner) if not val_inner.empty else None

    X_test = test_frame[feature_cols]
    g_test = group_array(test_frame)

    model = HorseRacingRanker(config=config)
    model.fit(
        X_train=X_train, y_train=y_train, groups_train=g_train,
        X_val=X_val, y_val=y_val, groups_val=g_val,
    )

    scores = model.model.predict(X_test[model.feature_names].fillna(X_train.median()))
    probs = model.predict_probabilities(X_test, g_test)
    return {
        "label": label,
        "model": model,
        "test_frame": test_frame,
        "test_scores": scores,
        "test_probs": probs,
        "groups": g_test,
    }


def evaluate_market(
    result: dict, target_col: str, label_for_log: str,
) -> dict:
    """Evaluate one model on one market (win/place/show)."""
    test_frame = result["test_frame"]
    y_market = test_frame[target_col].to_numpy(dtype=np.int64)
    scores = result["test_scores"]
    probs = result["test_probs"]
    groups = result["groups"]

    k = {"_is_win": 1, "_is_place": 2, "_is_show": 3}[target_col]
    hit_rate = top_k_hit_rate(scores, groups, y_market, k)
    brier = brier_on_target(probs, groups, y_market)
    LOGGER.info(
        "%s :: market=%s  topK_hit=%.4f  brier=%.4f",
        result["label"], label_for_log, hit_rate, brier,
    )
    return {"hit_rate": hit_rate, "brier": brier}


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
    from utils.horse_racing_data import load_training_frame

    LOGGER.info("Loading ranker training frame...")
    frame = load_training_frame(database_url=args.database_url)
    LOGGER.info(
        "Loaded %d rows across %d races.",
        len(frame), frame["race_id"].nunique() if "race_id" in frame else 0,
    )

    # Pre-compute per-market binary targets on the FULL frame so both
    # variants score against the same truth in train + test slices.
    pos = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame["_is_win"] = (pos == 1).astype(int)
    frame["_is_place"] = (pos <= 2).astype(int)
    frame["_is_show"] = (pos <= 3).astype(int)

    results = {
        name: train_variant(
            name, frame, args.split_date, variant=name,
            val_fraction=args.val_fraction,
        )
        for name in ("win_binary", "multi_grade", "place_binary", "show_binary")
    }

    LOGGER.info("")
    LOGGER.info("═════════════ MARKET-BY-MARKET COMPARISON ═════════════")
    # Each market evaluates the dedicated binary model PLUS the
    # win-binary baseline (the current production shape) PLUS the
    # multi-grade attempt. Dedicated models are the only ones with
    # native softmax calibration on their market.
    win_market = {
        name: evaluate_market(results[name], "_is_win", "WIN")
        for name in ("win_binary", "multi_grade")
    }
    place_market = {
        name: evaluate_market(results[name], "_is_place", "PLACE")
        for name in ("win_binary", "multi_grade", "place_binary")
    }
    show_market = {
        name: evaluate_market(results[name], "_is_show", "SHOW")
        for name in ("win_binary", "multi_grade", "show_binary")
    }

    LOGGER.info("")
    LOGGER.info("───────────── PER-MARKET BRIER LEADERBOARDS ─────────────")
    for market_label, lookup in (
        ("WIN  ", win_market),
        ("PLACE", place_market),
        ("SHOW ", show_market),
    ):
        leaderboard = sorted(lookup.items(), key=lambda kv: kv[1]["brier"])
        winner_name, winner = leaderboard[0]
        LOGGER.info(
            "%s leader: %-13s brier=%.4f hitK=%.4f",
            market_label, winner_name, winner["brier"], winner["hit_rate"],
        )
        for name, m in leaderboard[1:]:
            LOGGER.info(
                "         %-13s brier=%.4f (Δ%+.4f vs leader) hitK=%.4f",
                name, m["brier"], m["brier"] - winner["brier"], m["hit_rate"],
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
