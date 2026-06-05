"""One-off A/B: do Visual Crossing weather features help NFL ML/SP/TOT?

This is the retry of memory `weather-features-attempted`'s NFL chapter,
where Open-Meteo at 11km resolution showed a clean DROP (-2.5pts Brier
on TOTAL) across all attempted v3/v4a/v4b architectures. The hypothesis
for VC (1-4km grid, full historical depth) is that finer spatial
resolution captures the per-stadium weather variability that the 11km
grid averaged out.

NFL is the SHARPEST test of this hypothesis because:
  * NFL games have distinct, fixed stadiums (32 of them) with known
    coords — no city-string fuzz like soccer/tennis.
  * Total goals (TOTAL market) is the most weather-sensitive market
    in any major league sport — high wind kills passing + kicking,
    rain kills ball-handling, cold kills field goals.
  * NFL outcomes have lower noise than 1x2 soccer (fewer matches
    end as ties / unexpected upsets).

Tests three markets (ML / SP / TOT) in one run, each with its own
training frame + binary XGBoost. Decision per-market: KEEP when ΔBrier
≤ -0.005.

Sport-tuned thresholds (from the reverted compute_features_nfl.py):
  HIGH_WIND_KMH = 24  (~15mph; kicks become unreliable above this)
  WET_PRECIP_MM = 2   (lighter than soccer — ball handling sensitive)
  FREEZING_TEMP_C = 0 (frozen-pitch / numb-hands territory)
  HOT_TEMP_C    = 32  (heat-stress threshold for pad-wearing athletes)

Run on prod via:
    docker compose exec api python /app/scripts/ab_nfl_weather.py
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_nfl_weather")


HIGH_WIND_KMH = 24.0
WET_PRECIP_MM = 2.0
FREEZING_TEMP_C = 0.0
HOT_TEMP_C = 32.0


NFL_WEATHER_KEYS = (
    "weather_indoor",
    "weather_temp_c",
    "weather_wind_kmh",
    "weather_precip_mm",
    "weather_humidity_pct",
    "weather_high_wind",
    "weather_wet",
    "weather_freezing",
    "weather_hot",
)


# VC-only weather for finished NFL matches.
WEATHER_QUERY = """
    SELECT
        m.id::text AS match_id,
        mw.temperature_c,
        mw.wind_kmh,
        mw.precipitation_mm,
        mw.humidity_pct,
        vc.is_indoor AS weather_venue_indoor
    FROM matches m
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'nfl'
    LEFT JOIN LATERAL (
        SELECT data_kind, temperature_c, wind_kmh, precipitation_mm,
               humidity_pct, venue_coords_id
        FROM match_weather
        WHERE match_id = m.id
          AND data_kind LIKE 'vc_%%'
        ORDER BY created_at DESC
        LIMIT 1
    ) mw ON true
    LEFT JOIN venue_coords vc ON vc.id = mw.venue_coords_id
    WHERE m.status = 'finished'
"""


def fetch_weather_frame(database_url: str) -> pd.DataFrame:
    """Pull per-match VC weather snapshot + indoor flag. NaN for
    matches without a vc_* row (should be <1% with NFL's fully-
    seeded venue_coords)."""
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        raw = pd.read_sql(WEATHER_QUERY, engine)
    finally:
        engine.dispose()

    out = pd.DataFrame({"match_id": raw["match_id"]})
    indoor = raw["weather_venue_indoor"]
    out["weather_indoor"] = indoor.map(
        lambda v: 1.0 if v is True or v == 1 else (0.0 if v is False or v == 0 else np.nan)
    )
    is_indoor = out["weather_indoor"] == 1.0
    temp = raw["temperature_c"].astype(float)
    wind = raw["wind_kmh"].astype(float)
    precip = raw["precipitation_mm"].astype(float)
    humidity = raw["humidity_pct"].astype(float)
    out["weather_temp_c"] = temp.where(~is_indoor, np.nan)
    out["weather_wind_kmh"] = wind.where(~is_indoor, np.nan)
    out["weather_precip_mm"] = precip.where(~is_indoor, np.nan)
    out["weather_humidity_pct"] = humidity.where(~is_indoor, np.nan)
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
    out["weather_freezing"] = np.where(
        out["weather_temp_c"].notna(),
        (out["weather_temp_c"] < FREEZING_TEMP_C).astype(float),
        np.nan,
    )
    out["weather_hot"] = np.where(
        out["weather_temp_c"].notna(),
        (out["weather_temp_c"] > HOT_TEMP_C).astype(float),
        np.nan,
    )
    return out


def season_walk_forward_split(frame: pd.DataFrame, test_season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train on every season strictly earlier than test_season; test
    on test_season itself."""
    train = frame[frame["season"] < test_season].copy()
    test = frame[frame["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Walk-forward split for season={test_season!r} produced "
            f"train={len(train)} test={len(test)}. Available seasons: "
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
    database_url,
    weather,
    test_season,
):
    sys.path.insert(0, "/app/services/ml-models/src")
    from utils import training_data as td

    LOGGER.info("══════════════════ %s ══════════════════", label)
    base = getattr(td, loader_name)(database_url=database_url)
    LOGGER.info("Loaded %d %s matches.", len(base), label)

    base["match_id"] = base["match_id"].astype(str)
    merged = base.merge(weather, on="match_id", how="left")
    has_vc = merged["weather_temp_c"].notna() | (merged["weather_indoor"] == 1.0)
    LOGGER.info(
        "VC weather coverage: %d / %d matches (%.1f%%).",
        has_vc.sum(),
        len(merged),
        100.0 * has_vc.sum() / max(1, len(merged)),
    )

    train, test = season_walk_forward_split(merged, test_season)
    LOGGER.info(
        "Walk-forward: train=%d test=%d (test_season=%s).",
        len(train),
        len(test),
        test_season,
    )

    target = getattr(td, target_name)
    excluded = set(getattr(td, non_feature_set_name)) | {target}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1 = [c for c in numeric_cols if c not in excluded and c not in NFL_WEATHER_KEYS]
    v_vc = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1 features: %d  |  v_vc features: %d (+%d weather)",
        len(v1),
        len(v_vc),
        len(v_vc) - len(v1),
    )

    m_v1 = train_and_eval(train, test, v1, target)
    LOGGER.info(
        "v1-control:  acc=%.4f brier=%.4f ece=%.4f",
        m_v1["accuracy"],
        m_v1["brier"],
        m_v1["ece"],
    )
    m_vc = train_and_eval(train, test, v_vc, target)
    LOGGER.info(
        "v_vc:        acc=%.4f brier=%.4f ece=%.4f",
        m_vc["accuracy"],
        m_vc["brier"],
        m_vc["ece"],
    )

    d_acc = m_vc["accuracy"] - m_v1["accuracy"]
    d_brier = m_vc["brier"] - m_v1["brier"]
    d_ece = m_vc["ece"] - m_v1["ece"]
    LOGGER.info(
        "Δ acc: %+.4f  Δ brier: %+.4f  Δ ece: %+.4f",
        d_acc,
        d_brier,
        d_ece,
    )
    if d_brier <= -0.005:
        verdict = f"KEEP VC weather for {label}"
    elif d_brier >= 0.005:
        verdict = f"DROP VC weather for {label} (worse Brier)"
    else:
        verdict = f"DROP VC weather for {label} (|ΔBrier|<0.005)"
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

    LOGGER.info("Fetching VC weather snapshot for NFL...")
    weather = fetch_weather_frame(args.database_url)
    LOGGER.info("Weather rows: %d.", len(weather))

    run_market(
        "MONEYLINE",
        loader_name="load_nfl_moneyline_frame",
        non_feature_set_name="NFL_MONEYLINE_NON_FEATURE_COLUMNS",
        target_name="NFL_MONEYLINE_TARGET",
        database_url=args.database_url,
        weather=weather,
        test_season=args.test_season,
    )
    run_market(
        "SPREAD",
        loader_name="load_nfl_spread_frame",
        non_feature_set_name="NFL_SPREAD_NON_FEATURE_COLUMNS",
        target_name="NFL_SPREAD_TARGET",
        database_url=args.database_url,
        weather=weather,
        test_season=args.test_season,
    )
    run_market(
        "TOTAL",
        loader_name="load_nfl_total_frame",
        non_feature_set_name="NFL_TOTAL_NON_FEATURE_COLUMNS",
        target_name="NFL_TOTAL_TARGET",
        database_url=args.database_url,
        weather=weather,
        test_season=args.test_season,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
