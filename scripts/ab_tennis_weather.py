"""One-off A/B: does adding weather features improve tennis moneyline?

Tennis weather integration was reverted on 2026-06-03 (commit 7f299c4)
PRECAUTIONARILY — never actually validated against control on the
24k-match tennis corpus. NFL ran the full v1-vs-v4b walk-forward and
weather features were a clean -2.5pt drag at 65.2 → 64.4 with
Open-Meteo's 11km grid + 585 pre-2024 train rows.

This script answers: would weather features help tennis (24,446
finished matches, 6,116 cached weather rows, 140 seeded venues)?

Methodology:
  1. Load the existing tennis training frame from features_cache
     (current state, no weather features).
  2. In-memory join of weather snapshot per match — no DB writes,
     no features_cache pollution. Identical to the v4b schema:
     8 canonical TENNIS_WEATHER_KEYS, sport-tuned thresholds
     (HIGH_WIND=25kmh, HOT=32C, WET=5mm).
  3. Walk-forward split at the 90th percentile of match_date —
     ~2,400 test matches, ~22,000 train.
  4. Train two XGBoost models with IDENTICAL hyperparameters
     (XGBOOST_TENNIS_MONEYLINE from model_config.py):
       v1-control: existing features only
       v4b      : existing + 8 weather features
  5. Report test accuracy, Brier, ECE for both. Compute deltas.

Decision rule (mirrors NFL v4b):
  v4b strictly better on Brier by >=0.005 -> KEEP weather, re-land
  the revert-revert on main.
  Otherwise -> DROP, update memory, weather chapter closed for tennis.

Run on prod via:
    docker compose exec api python /app/scripts/ab_tennis_weather.py

See `weather-features-attempted.md` for the NFL v4b precedent
this experiment is modelled on.
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_tennis_weather")

# Sport-tuned thresholds from the reverted compute_features_tennis.py.
# Kept here verbatim so the A/B replicates the v4b schema exactly.
HIGH_WIND_KMH = 25.0
WET_PRECIP_MM = 5.0
HOT_TEMP_C = 32.0

# The 8 canonical keys the v4b code emitted. Tennis omits the freezing
# flag and adds a hot flag vs NFL. None-for-missing keeps train/predict
# shapes aligned (the always-emit invariant).
TENNIS_WEATHER_KEYS = (
    "weather_indoor",
    "weather_temp_c",
    "weather_wind_kmh",
    "weather_precip_mm",
    "weather_humidity_pct",
    "weather_high_wind",
    "weather_wet",
    "weather_hot",
)

# Bulk weather query — one round trip for the entire corpus. Falls
# back to venue_coords lookup by normalized name when no match_weather
# row exists (mirrors the per-match fallback path in the reverted
# fetch_weather function).
WEATHER_QUERY = """
    SELECT
        m.id::text AS match_id,
        mwl.temperature_c,
        mwl.wind_kmh,
        mwl.precipitation_mm,
        mwl.humidity_pct,
        vc_w.is_indoor   AS weather_venue_indoor,
        vc_name.is_indoor AS name_lookup_indoor
    FROM matches m
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'tennis'
    LEFT JOIN match_weather_latest mwl ON mwl.match_id = m.id
    LEFT JOIN venue_coords vc_w ON vc_w.id = mwl.venue_coords_id
    LEFT JOIN venue_coords vc_name
        ON vc_name.normalized_venue_name = LOWER(TRIM(m.venue))
    WHERE m.status = 'finished'
"""


def fetch_weather_frame(database_url: str) -> pd.DataFrame:
    """Pull per-match weather snapshot + indoor flag into a DataFrame
    keyed by match_id. Each match contributes exactly one row; columns
    are NaN when neither match_weather nor a venue lookup has an
    is_indoor / numeric value."""
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        raw = pd.read_sql(WEATHER_QUERY, engine)
    finally:
        engine.dispose()

    out = pd.DataFrame({"match_id": raw["match_id"]})

    # Indoor source-of-truth: prefer the weather row's venue link,
    # fall back to the by-name venue lookup. Both NaN -> NaN.
    indoor = raw["weather_venue_indoor"].combine_first(
        raw["name_lookup_indoor"]
    )
    out["weather_indoor"] = indoor.map(
        lambda v: 1.0 if v is True or v == 1 else (0.0 if v is False or v == 0 else np.nan)
    )

    # Indoor matches: keep weather_indoor=1, leave outdoor numerics
    # NaN. The reverted fetch_weather() short-circuited indoor venues
    # to avoid leaking irrelevant outdoor readings into indoor matches.
    is_indoor = out["weather_indoor"] == 1.0
    temp = raw["temperature_c"].astype(float)
    wind = raw["wind_kmh"].astype(float)
    precip = raw["precipitation_mm"].astype(float)
    humidity = raw["humidity_pct"].astype(float)

    out["weather_temp_c"] = temp.where(~is_indoor, np.nan)
    out["weather_wind_kmh"] = wind.where(~is_indoor, np.nan)
    out["weather_precip_mm"] = precip.where(~is_indoor, np.nan)
    out["weather_humidity_pct"] = humidity.where(~is_indoor, np.nan)

    # Threshold flags. NaN-safe: flag stays NaN when underlying
    # numeric is NaN; sklearn / XGBoost both handle NaN natively.
    out["weather_high_wind"] = np.where(
        out["weather_wind_kmh"].notna(),
        (out["weather_wind_kmh"] > HIGH_WIND_KMH).astype(float),
        np.nan,
    )
    out["weather_wet"] = np.where(
        out["weather_precip_mm"].notna(),
        (out["weather_precip_mm"] > WET_PRECIP_MM).astype(float),
        np.nan,
    )
    out["weather_hot"] = np.where(
        out["weather_temp_c"].notna(),
        (out["weather_temp_c"] > HOT_TEMP_C).astype(float),
        np.nan,
    )

    return out


def walk_forward_split(frame: pd.DataFrame, test_fraction: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Last `test_fraction` of rows by match_date go to test. Anything
    on the cutoff date or earlier goes to train (no leakage from
    same-day matches because we split on date, not row index)."""
    sorted_frame = frame.sort_values("match_date").reset_index(drop=True)
    cutoff_idx = int(len(sorted_frame) * (1 - test_fraction))
    cutoff_date = sorted_frame.iloc[cutoff_idx]["match_date"]
    train = sorted_frame[sorted_frame["match_date"] < cutoff_date].copy()
    test = sorted_frame[sorted_frame["match_date"] >= cutoff_date].copy()
    return train, test


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard 10-bin ECE on the predicted probability of the
    predicted class. Mirrors what the project's metrics layer uses
    so numbers compare to the 0.006 ECE quoted in tennis memory."""
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


def train_and_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    target: str,
) -> dict:
    """Train XGBoost on train, predict on test, return metrics."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres URL (defaults to DATABASE_URL env)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of matches (by date) reserved for the test set",
    )
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
        LOGGER.error("DATABASE_URL not set. Pass --database-url or set env.")
        return 1

    # Import here so the script's --help doesn't require pulling in
    # the full training_data + sqlalchemy graph just to print usage.
    sys.path.insert(0, "/app/services/ml-models/src")
    from utils.training_data import (
        TENNIS_MONEYLINE_NON_FEATURE_COLUMNS,
        TENNIS_MONEYLINE_TARGET,
        load_tennis_moneyline_frame,
    )

    LOGGER.info("Loading tennis moneyline frame...")
    base = load_tennis_moneyline_frame(database_url=args.database_url)
    LOGGER.info("Loaded %d matches.", len(base))

    LOGGER.info("Fetching weather snapshot for the corpus...")
    weather = fetch_weather_frame(args.database_url)
    LOGGER.info(
        "Weather rows: %d (indoor=%d, outdoor=%d, unknown=%d).",
        len(weather),
        int((weather["weather_indoor"] == 1.0).sum()),
        int((weather["weather_indoor"] == 0.0).sum()),
        int(weather["weather_indoor"].isna().sum()),
    )

    base["match_id"] = base["match_id"].astype(str)
    weather["match_id"] = weather["match_id"].astype(str)
    merged = base.merge(weather, on="match_id", how="left")

    # Coverage stats — show how many test-fold matches actually have
    # outdoor weather numerics (the only kind where v4b can possibly
    # differ from v1-control).
    has_outdoor = merged["weather_temp_c"].notna().sum()
    LOGGER.info(
        "Outdoor weather coverage: %d / %d matches (%.1f%%).",
        has_outdoor,
        len(merged),
        100.0 * has_outdoor / max(1, len(merged)),
    )

    train, test = walk_forward_split(merged, args.test_fraction)
    LOGGER.info(
        "Walk-forward split: %d train, %d test (cutoff date: %s).",
        len(train),
        len(test),
        test["match_date"].min(),
    )

    # v1-control feature set = current production set. v4b = same +
    # the 8 weather keys. Build both column lists from the merged
    # frame so any column the trainer would normally pick up is
    # treated identically.
    excluded = set(TENNIS_MONEYLINE_NON_FEATURE_COLUMNS) | {TENNIS_MONEYLINE_TARGET}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1_features = [
        c for c in numeric_cols
        if c not in excluded and c not in TENNIS_WEATHER_KEYS
    ]
    v4b_features = [c for c in numeric_cols if c not in excluded]

    LOGGER.info("v1-control features: %d", len(v1_features))
    LOGGER.info("v4b features:        %d (+%d weather keys)", len(v4b_features), len(v4b_features) - len(v1_features))

    LOGGER.info("Training v1-control (no weather)...")
    v1_metrics = train_and_eval(train, test, v1_features, TENNIS_MONEYLINE_TARGET)
    LOGGER.info(
        "v1-control: acc=%.4f brier=%.4f ece=%.4f (n_train=%d, n_test=%d, n_features=%d)",
        v1_metrics["accuracy"], v1_metrics["brier"], v1_metrics["ece"],
        v1_metrics["n_train"], v1_metrics["n_test"], v1_metrics["n_features"],
    )

    LOGGER.info("Training v4b (with weather)...")
    v4b_metrics = train_and_eval(train, test, v4b_features, TENNIS_MONEYLINE_TARGET)
    LOGGER.info(
        "v4b:        acc=%.4f brier=%.4f ece=%.4f (n_train=%d, n_test=%d, n_features=%d)",
        v4b_metrics["accuracy"], v4b_metrics["brier"], v4b_metrics["ece"],
        v4b_metrics["n_train"], v4b_metrics["n_test"], v4b_metrics["n_features"],
    )

    delta_acc = v4b_metrics["accuracy"] - v1_metrics["accuracy"]
    delta_brier = v4b_metrics["brier"] - v1_metrics["brier"]  # NEGATIVE delta = better
    delta_ece = v4b_metrics["ece"] - v1_metrics["ece"]

    LOGGER.info("")
    LOGGER.info("──────────────────────── DELTAS (v4b - v1) ────────────────────────")
    LOGGER.info("Δ accuracy: %+.4f (positive favours v4b)", delta_acc)
    LOGGER.info("Δ brier:    %+.4f (NEGATIVE favours v4b)", delta_brier)
    LOGGER.info("Δ ece:      %+.4f (NEGATIVE favours v4b)", delta_ece)

    if delta_brier <= -0.005:
        verdict = "KEEP weather (v4b strictly better on Brier by >=0.005)"
    elif delta_brier >= 0.005:
        verdict = "DROP weather (v4b strictly worse on Brier by >=0.005)"
    else:
        verdict = "DROP weather (no meaningful improvement, |ΔBrier| < 0.005)"
    LOGGER.info("VERDICT: %s", verdict)

    return 0


if __name__ == "__main__":
    sys.exit(main())
