"""One-off A/B: do travel + venue-type features help NFL spread/total?

NFL spread + total were memory-flagged as the only markets where
cross-book features didn't help (PR #8 + PR #12). Memory's
``nfl-spread-total-efficient`` lists "QB injury data" as the
biggest signal, but that's a multi-week build. This script tests
the cheapest new-data lever available: TRAVEL distance from the
away team's home city to the game venue, and a binary DOME flag
for indoor stadiums. Both are derivable from team-city coordinates
+ a hardcoded NFL stadium list (no external data source).

Hypotheses:
  * Long-distance road games (e.g. New England → Las Vegas) put
    the away team at a fatigue + jet-lag disadvantage. Some books
    +/- 1.5 points effect documented in published research.
  * Dome teams (Detroit, New Orleans, Atlanta indoors, etc.) score
    higher than outdoor teams because indoor weather guarantees
    pass-friendly conditions.

Features added per match:
  away_travel_km — great-circle distance, away city → game venue
  away_timezones_crossed — abs(home_offset - away_offset), hours
  is_dome — 1.0 if game played indoors, 0.0 outdoors
  neutral_site — 1.0 if game venue is not the home team's stadium
                 (London, Mexico City, neutral playoff sites).

A/B mirrors scripts/ab_nfl_cross_book.py. XGBoost trained twice on
identical walk-forward split (last 2024-2025 season as test) on
both SPREAD and TOTAL.

Decision rule (per market, independent):
  ΔBrier ≤ -0.005 → KEEP travel features for that market.
"""

import argparse
import logging
import math
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ab_nfl_travel")


# NFL team home cities — (lat, lon, tz_offset_from_utc_hours, is_dome).
# Hardcoded since NFL has exactly 32 teams; the alternative (joining
# venue_coords) requires per-team stadium mapping which doesn't exist
# in our schema. Tz offset uses standard (winter) time — DST adds 1h
# in summer, but NFL season is mostly Sep-Jan when most cities are
# on standard time. is_dome=True covers permanent indoor stadiums
# plus retractable-roof teams (Arizona, Dallas, Houston, Atlanta,
# Indianapolis) — the roof is almost always closed during NFL games.
NFL_TEAM_HOME: dict[str, tuple[float, float, float, bool]] = {
    "Arizona Cardinals": (33.5276, -112.2626, -7.0, True),
    "Atlanta Falcons": (33.7553, -84.4006, -5.0, True),
    "Baltimore Ravens": (39.2779, -76.6227, -5.0, False),
    "Buffalo Bills": (42.7738, -78.7869, -5.0, False),
    "Carolina Panthers": (35.2258, -80.8528, -5.0, False),
    "Chicago Bears": (41.8623, -87.6167, -6.0, False),
    "Cincinnati Bengals": (39.0954, -84.5161, -5.0, False),
    "Cleveland Browns": (41.5061, -81.6995, -5.0, False),
    "Dallas Cowboys": (32.7475, -97.0945, -6.0, True),
    "Denver Broncos": (39.7439, -105.0201, -7.0, False),
    "Detroit Lions": (42.3400, -83.0457, -5.0, True),
    "Green Bay Packers": (44.5013, -88.0622, -6.0, False),
    "Houston Texans": (29.6847, -95.4107, -6.0, True),
    "Indianapolis Colts": (39.7601, -86.1639, -5.0, True),
    "Jacksonville Jaguars": (30.3239, -81.6373, -5.0, False),
    "Kansas City Chiefs": (39.0489, -94.4839, -6.0, False),
    "Las Vegas Raiders": (36.0908, -115.1834, -8.0, True),
    "Los Angeles Chargers": (33.9534, -118.3387, -8.0, False),
    "Los Angeles Rams": (33.9534, -118.3387, -8.0, False),
    "Miami Dolphins": (25.9580, -80.2389, -5.0, False),
    "Minnesota Vikings": (44.9740, -93.2580, -6.0, True),
    "New England Patriots": (42.0909, -71.2643, -5.0, False),
    "New Orleans Saints": (29.9509, -90.0815, -6.0, True),
    "New York Giants": (40.8135, -74.0745, -5.0, False),
    "New York Jets": (40.8135, -74.0745, -5.0, False),
    "Philadelphia Eagles": (39.9012, -75.1675, -5.0, False),
    "Pittsburgh Steelers": (40.4468, -80.0158, -5.0, False),
    "San Francisco 49ers": (37.4030, -121.9700, -8.0, False),
    "Seattle Seahawks": (47.5953, -122.3316, -8.0, False),
    "Tampa Bay Buccaneers": (27.9758, -82.5033, -5.0, False),
    "Tennessee Titans": (36.1665, -86.7713, -6.0, False),
    "Washington Commanders": (38.9077, -76.8645, -5.0, False),
}


def _great_circle_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Haversine distance in km. Mean Earth radius 6371."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_travel_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the 4 travel features to a frame with home_team + away_team
    columns. Missing teams (e.g. renamed franchises) fall back to
    neutral defaults so downstream NaN-handling doesn't break."""
    out = frame.copy()
    out["away_travel_km"] = 0.0
    out["away_timezones_crossed"] = 0.0
    out["is_dome"] = 0.5  # neutral default for missing
    out["neutral_site"] = 0.0

    for idx, row in out.iterrows():
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        if home_team not in NFL_TEAM_HOME or away_team not in NFL_TEAM_HOME:
            continue
        h_lat, h_lon, h_tz, h_dome = NFL_TEAM_HOME[home_team]
        a_lat, a_lon, a_tz, _ = NFL_TEAM_HOME[away_team]
        out.at[idx, "away_travel_km"] = _great_circle_km(
            a_lat,
            a_lon,
            h_lat,
            h_lon,
        )
        out.at[idx, "away_timezones_crossed"] = abs(h_tz - a_tz)
        out.at[idx, "is_dome"] = 1.0 if h_dome else 0.0
        # neutral_site stays 0 because we don't have explicit
        # neutral-venue metadata; future work can use matches.venue
        # to detect London / Mexico City / Wembley.
    return out


TRAVEL_KEYS = (
    "away_travel_km",
    "away_timezones_crossed",
    "is_dome",
    "neutral_site",
)


def season_walk_forward_split(
    frame: pd.DataFrame,
    test_season: str,
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
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
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
    test_season,
):
    sys.path.insert(0, "/app/services/ml-models/src")
    from utils import training_data as td

    LOGGER.info("══════════════════ %s ══════════════════", label)
    base = getattr(td, loader_name)(database_url=database_url)
    base = compute_travel_features(base)
    LOGGER.info(
        "Loaded %d %s matches. Travel cov: %.1f%% (mean km=%.0f).",
        len(base),
        label,
        100.0 * (base["away_travel_km"] > 0).sum() / max(1, len(base)),
        float(base["away_travel_km"][base["away_travel_km"] > 0].mean()),
    )

    train, test = season_walk_forward_split(base, test_season)
    LOGGER.info(
        "Walk-forward: train=%d test=%d (test_season=%s).",
        len(train),
        len(test),
        test_season,
    )

    target = getattr(td, target_name)
    excluded = set(getattr(td, non_feature_set_name)) | {target}
    numeric_cols = base.select_dtypes(include=[np.number, bool]).columns.tolist()
    v1 = [c for c in numeric_cols if c not in excluded and c not in TRAVEL_KEYS]
    vt = [c for c in numeric_cols if c not in excluded]
    LOGGER.info(
        "v1 features: %d  |  v_travel features: %d (+%d travel keys)",
        len(v1),
        len(vt),
        len(vt) - len(v1),
    )

    m_v1 = train_and_eval(train, test, v1, target)
    LOGGER.info(
        "v1-control:  acc=%.4f brier=%.4f ece=%.4f",
        m_v1["accuracy"],
        m_v1["brier"],
        m_v1["ece"],
    )
    m_t = train_and_eval(train, test, vt, target)
    LOGGER.info(
        "v_travel:    acc=%.4f brier=%.4f ece=%.4f",
        m_t["accuracy"],
        m_t["brier"],
        m_t["ece"],
    )

    d_acc = m_t["accuracy"] - m_v1["accuracy"]
    d_brier = m_t["brier"] - m_v1["brier"]
    d_ece = m_t["ece"] - m_v1["ece"]
    LOGGER.info(
        "Δ acc: %+.4f  Δ brier: %+.4f  Δ ece: %+.4f",
        d_acc,
        d_brier,
        d_ece,
    )
    if d_brier <= -0.005:
        verdict = f"KEEP travel features for {label}"
    elif d_brier >= 0.005:
        verdict = f"DROP travel features for {label} (worse Brier)"
    else:
        verdict = f"DROP travel features for {label} (|ΔBrier|<0.005)"
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

    run_market(
        "MONEYLINE",
        loader_name="load_nfl_moneyline_frame",
        non_feature_set_name="NFL_MONEYLINE_NON_FEATURE_COLUMNS",
        target_name="NFL_MONEYLINE_TARGET",
        database_url=args.database_url,
        test_season=args.test_season,
    )
    run_market(
        "SPREAD",
        loader_name="load_nfl_spread_frame",
        non_feature_set_name="NFL_SPREAD_NON_FEATURE_COLUMNS",
        target_name="NFL_SPREAD_TARGET",
        database_url=args.database_url,
        test_season=args.test_season,
    )
    run_market(
        "TOTAL",
        loader_name="load_nfl_total_frame",
        non_feature_set_name="NFL_TOTAL_NON_FEATURE_COLUMNS",
        target_name="NFL_TOTAL_TARGET",
        database_url=args.database_url,
        test_season=args.test_season,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
