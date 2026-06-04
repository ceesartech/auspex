"""One-off A/B: does adding Visual Crossing weather features help soccer 1x2?

This is the FIRST weather A/B for soccer — earlier attempts (NFL,
tennis) were blocked at 11km Open-Meteo resolution and soccer had no
venue data at all. With VC's 1-4km grid plus the team→stadium
fallback shipped this session, ~99.6% of soccer matches now have
weather coverage flowing in.

Methodology mirrors the prior NFL / tennis A/Bs:
  1. Load the existing soccer training frame (1x2, 3-class:
     home/draw/away) from features_cache.
  2. In-memory join of the latest VC weather snapshot per match.
     ONLY uses vc_* rows — Open-Meteo data is excluded so the A/B
     measures the VC delta, not a polluted blend.
  3. Walk-forward split: train on everything before the cutoff,
     test on everything after. Defaults to 2024-08-01 (start of
     the 2024-25 European season).
  4. Train two XGBoost multiclass models with IDENTICAL hyperparams:
       v1-control: existing features only
       v_vc      : existing + 8 weather features
  5. Report test accuracy, log-loss, per-class Brier average.

Soccer-tuned weather thresholds:
  HIGH_WIND_KMH = 30  (stronger than tennis 25kmh — outfield play
                       is less sensitive to wind than ball-toss)
  WET_PRECIP_MM = 5   (typical "match abandoned" threshold)
  HOT_TEMP_C    = 28  (UEFA cooling-break threshold at 32C; 28C
                       picks up the "uncomfortably warm" band)
  COLD_TEMP_C   = 5   (frozen-pitch territory)

Decision rule: KEEP when v_vc strictly improves log-loss by >=0.005.

Run on prod via:
    docker compose exec api python /app/scripts/ab_soccer_weather.py
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_soccer_weather")


HIGH_WIND_KMH = 30.0
WET_PRECIP_MM = 5.0
HOT_TEMP_C = 28.0
COLD_TEMP_C = 5.0


SOCCER_WEATHER_KEYS = (
    "weather_indoor",
    "weather_temp_c",
    "weather_wind_kmh",
    "weather_precip_mm",
    "weather_humidity_pct",
    "weather_high_wind",
    "weather_wet",
    "weather_hot",
    "weather_cold",
)


# Bulk VC-only weather query. data_kind LIKE 'vc_%' so Open-Meteo
# rows are excluded — this A/B measures the VC delta cleanly.
# match_weather_latest already takes the most recent row per match,
# but the data_kind filter guards against an Open-Meteo row beating
# a VC row to the latest slot on rare overlap.
WEATHER_QUERY = """
    SELECT
        m.id::text AS match_id,
        mw.temperature_c,
        mw.wind_kmh,
        mw.precipitation_mm,
        mw.humidity_pct,
        vc.is_indoor AS weather_venue_indoor
    FROM matches m
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
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
    """Pull per-match VC weather snapshot + indoor flag.
    Columns are NaN when no VC row exists (~0.4% of matches —
    the irreducible neutral-site residue)."""
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        raw = pd.read_sql(WEATHER_QUERY, engine)
    finally:
        engine.dispose()

    out = pd.DataFrame({"match_id": raw["match_id"]})

    # Indoor: VC stadium flag (most outdoor; some MLS / Tottenham
    # are indoor). Treat NaN as unknown — don't assume outdoor.
    indoor = raw["weather_venue_indoor"]
    out["weather_indoor"] = indoor.map(
        lambda v: 1.0 if v is True or v == 1 else (0.0 if v is False or v == 0 else np.nan)
    )

    # Indoor matches: keep weather_indoor=1, leave outdoor numerics
    # NaN. Outdoor venue → use the numerics.
    is_indoor = out["weather_indoor"] == 1.0
    temp = raw["temperature_c"].astype(float)
    wind = raw["wind_kmh"].astype(float)
    precip = raw["precipitation_mm"].astype(float)
    humidity = raw["humidity_pct"].astype(float)

    out["weather_temp_c"] = temp.where(~is_indoor, np.nan)
    out["weather_wind_kmh"] = wind.where(~is_indoor, np.nan)
    out["weather_precip_mm"] = precip.where(~is_indoor, np.nan)
    out["weather_humidity_pct"] = humidity.where(~is_indoor, np.nan)

    # Threshold flags. NaN-safe.
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
    out["weather_cold"] = np.where(
        out["weather_temp_c"].notna(),
        (out["weather_temp_c"] < COLD_TEMP_C).astype(float),
        np.nan,
    )

    return out


def date_walk_forward_split(
    frame: pd.DataFrame,
    cutoff_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match-date split. Train = before cutoff, test = on/after."""
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    if frame["match_date"].dt.tz is None:
        cutoff = cutoff.tz_localize(None)
    train = frame[frame["match_date"] < cutoff].copy()
    test = frame[frame["match_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError(f"Walk-forward at cutoff={cutoff_date}: train={len(train)} test={len(test)}")
    return train, test


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """3-class Brier: mean over rows of sum_c (P(c) - y_c)^2.
    Lower is better; 0 = perfect, 2 = perfectly wrong."""
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(((proba - one_hot) ** 2).sum(axis=1).mean())


def train_and_eval(train, test, feature_cols, target):
    from sklearn.metrics import accuracy_score, log_loss
    from xgboost import XGBClassifier

    X_train = train[feature_cols].values
    y_train = train[target].astype(int).values
    X_test = test[feature_cols].values
    y_test = test[target].astype(int).values

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
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
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    preds = proba.argmax(axis=1)
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1, 2])),
        "brier": multiclass_brier(y_test, proba),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--cutoff-date",
        default="2024-08-01",
        help="Walk-forward boundary (YYYY-MM-DD). Defaults to start of 2024-25 season.",
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
        LOGGER.error("DATABASE_URL not set.")
        return 1

    sys.path.insert(0, "/app/services/ml-models/src")
    from utils import training_data as td

    LOGGER.info("Loading soccer 1x2 training frame...")
    base = td.load_training_frame(database_url=args.database_url)
    LOGGER.info("Loaded %d soccer matches.", len(base))

    LOGGER.info("Fetching VC weather snapshot...")
    weather = fetch_weather_frame(args.database_url)
    coverage = weather["weather_temp_c"].notna().sum() + (weather["weather_indoor"] == 1.0).sum()
    LOGGER.info(
        "Weather rows: %d (indoor=%d, outdoor with numerics=%d, NaN-all=%d).",
        len(weather),
        int((weather["weather_indoor"] == 1.0).sum()),
        int(weather["weather_temp_c"].notna().sum()),
        int(len(weather) - coverage),
    )

    base["match_id"] = base["match_id"].astype(str)
    weather["match_id"] = weather["match_id"].astype(str)
    merged = base.merge(weather, on="match_id", how="left")
    has_vc = merged["weather_temp_c"].notna() | (merged["weather_indoor"] == 1.0)
    LOGGER.info(
        "VC weather coverage on training frame: %d / %d matches (%.1f%%).",
        has_vc.sum(),
        len(merged),
        100.0 * has_vc.sum() / max(1, len(merged)),
    )

    train, test = date_walk_forward_split(merged, args.cutoff_date)
    LOGGER.info(
        "Walk-forward: %d train, %d test (cutoff: %s).",
        len(train),
        len(test),
        args.cutoff_date,
    )

    # Feature column selection. v1-control drops the 8 weather keys;
    # v_vc keeps them.
    excluded = set(td.NON_FEATURE_COLUMNS) | {td.TARGET_COLUMN}
    numeric_cols = merged.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1_features = [c for c in numeric_cols if c not in excluded and c not in SOCCER_WEATHER_KEYS]
    v_vc_features = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1-control features: %d  |  v_vc features: %d (+%d weather)",
        len(v1_features),
        len(v_vc_features),
        len(v_vc_features) - len(v1_features),
    )

    LOGGER.info("Training v1-control (no weather)...")
    v1 = train_and_eval(train, test, v1_features, td.TARGET_COLUMN)
    LOGGER.info(
        "v1-control: acc=%.4f log_loss=%.4f brier=%.4f",
        v1["accuracy"],
        v1["log_loss"],
        v1["brier"],
    )

    LOGGER.info("Training v_vc (with weather)...")
    vc = train_and_eval(train, test, v_vc_features, td.TARGET_COLUMN)
    LOGGER.info(
        "v_vc:       acc=%.4f log_loss=%.4f brier=%.4f",
        vc["accuracy"],
        vc["log_loss"],
        vc["brier"],
    )

    d_acc = vc["accuracy"] - v1["accuracy"]
    d_ll = vc["log_loss"] - v1["log_loss"]
    d_brier = vc["brier"] - v1["brier"]

    LOGGER.info("")
    LOGGER.info("──────────────────── DELTAS (v_vc - v1) ────────────────────")
    LOGGER.info("Δ accuracy: %+.4f (positive favours v_vc)", d_acc)
    LOGGER.info("Δ log_loss: %+.4f (NEGATIVE favours v_vc)", d_ll)
    LOGGER.info("Δ brier:    %+.4f (NEGATIVE favours v_vc)", d_brier)

    if d_ll <= -0.005:
        verdict = "KEEP VC weather features for soccer"
    elif d_ll >= 0.005:
        verdict = "DROP VC weather features for soccer (worse log-loss)"
    else:
        verdict = "DROP VC weather features (|ΔLogLoss|<0.005)"
    LOGGER.info("VERDICT: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
