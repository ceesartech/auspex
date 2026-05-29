"""Utilities for loading and validating model training data."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Pulls finished matches with closing odds joined in, plus the most recent
# features_cache row (when available). The odds-derived columns let us train
# a working baseline model even before features_cache has been populated for
# every historical match. Once compute_features.py has been run over the
# corpus, the `features` JSON adds the richer 250+ feature set.
DEFAULT_TRAINING_QUERY = """
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.league_id::text AS league_id,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        m.home_score,
        m.away_score,
        CASE
            WHEN m.home_score > m.away_score THEN 0
            WHEN m.home_score = m.away_score THEN 1
            ELSE 2
        END AS match_outcome,
        -- Closing 1x2 odds (Average of all books) and over/under 2.5.
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = '1x2'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = '1x2'
              AND o.selection = 'draw' AND NOT o.is_live) AS odds_draw,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = '1x2'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'over_under'
              AND o.selection = 'over'  AND o.line = 2.5 AND NOT o.is_live) AS odds_over25,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.market_type = 'over_under' AND o.match_id = m.id
              AND o.selection = 'under' AND o.line = 2.5 AND NOT o.is_live) AS odds_under25,
        fc.features
    FROM matches m
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.match_date ASC
"""

TARGET_COLUMN = "match_outcome"
# Columns that exist on the frame but should never be used as model
# inputs — either identifiers, outcome-leaking columns, or the raw
# features dict that we flatten elsewhere.
NON_FEATURE_COLUMNS = {
    "match_id",
    "match_date",
    "league_id",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    TARGET_COLUMN,
    "features",
}


# ============= NHL MONEYLINE TRAINING =============

# NHL moneyline target column. 0 = home won (incl. OT/SO), 1 = away won.
NHL_MONEYLINE_TARGET = "nhl_moneyline"

# NHL training query. Differs from the soccer DEFAULT_TRAINING_QUERY in
# three ways:
#   1. Joins on the NHL market types: 'moneyline', 'spread' (line ±1.5),
#      and 'total' (line 5.5) instead of '1x2' and 'over_under' (line 2.5).
#   2. Filters to leagues.sport = 'nhl' so the query never picks up
#      soccer rows even though they share the matches/odds tables.
#   3. Pulls features_cache rows keyed on feature_set = 'nhl_baseline'
#      (the Phase 2 set) — the soccer query takes the most-recent row
#      regardless of feature_set, which would mix incompatible schemas
#      for a sport that has both. NHL never overlaps but pinning by
#      feature_set is the load-bearing invariant going forward.
NHL_MONEYLINE_TRAINING_QUERY = """
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.season,
        m.league_id::text AS league_id,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        m.home_score,
        m.away_score,
        CASE
            WHEN m.home_score > m.away_score THEN 0
            ELSE 1
        END AS nhl_moneyline,
        -- Closing moneyline (averaged across books)
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        -- Puck line ±1.5 (averaged across books)
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'home' AND o.line = -1.5 AND NOT o.is_live) AS odds_home_pl15,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'away' AND o.line = 1.5 AND NOT o.is_live) AS odds_away_pl15,
        -- Totals 5.5 (canonical NHL line)
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'total'
              AND o.selection = 'over'  AND o.line = 5.5 AND NOT o.is_live) AS odds_over55,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'total'
              AND o.selection = 'under' AND o.line = 5.5 AND NOT o.is_live) AS odds_under55,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'nhl_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'nhl'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.home_score <> m.away_score  -- NHL always has a winner; ties = data error
    ORDER BY m.match_date ASC
"""

# Columns exposed by NHL_MONEYLINE_TRAINING_QUERY that must NOT be used
# as model inputs. Includes the target itself, identifiers, raw scores
# (outcome leakage), and the JSON features blob (flattened separately).
NHL_NON_FEATURE_COLUMNS = {
    "match_id",
    "match_date",
    "season",
    "league_id",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    NHL_MONEYLINE_TARGET,
    "features",
}


@dataclass(frozen=True)
class TrainingDataQuality:
    """Validation summary for a prepared training frame."""

    rows: int
    feature_count: int
    target_classes: int
    date_min: Optional[str]
    date_max: Optional[str]
    missing_feature_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "feature_count": self.feature_count,
            "target_classes": self.target_classes,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "missing_feature_rate": self.missing_feature_rate,
        }


def load_training_frame(
    *,
    database_url: Optional[str] = None,
    input_csv: Optional[str] = None,
    query: str = DEFAULT_TRAINING_QUERY,
) -> pd.DataFrame:
    """Load model training data from a CSV file or PostgreSQL."""
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        # SQLAlchemy engine to silence pandas' "only supports SQLAlchemy
        # connectable" warning and avoid the deprecated psycopg2 path.
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(query, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")

    return prepare_training_frame(raw)


def prepare_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten feature JSON and derive target + baseline columns."""
    if raw.empty:
        return raw.copy()

    frame = raw.copy()

    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)

    if TARGET_COLUMN not in frame.columns and {"home_score", "away_score"}.issubset(frame.columns):
        frame[TARGET_COLUMN] = np.select(
            [
                frame["home_score"] > frame["away_score"],
                frame["home_score"] == frame["away_score"],
            ],
            [0, 1],
            default=2,
        )

    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")

    # Even before features_cache is populated, we can derive a usable
    # baseline feature set from closing odds + rolling team form.
    frame = _add_implied_probabilities(frame)
    frame = _add_rolling_team_form(frame)

    return frame


def _add_implied_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized implied probabilities + market margin from 1x2 odds."""
    if not {"odds_home", "odds_draw", "odds_away"}.issubset(frame.columns):
        return frame
    inv_h = 1.0 / frame["odds_home"]
    inv_d = 1.0 / frame["odds_draw"]
    inv_a = 1.0 / frame["odds_away"]
    margin = inv_h + inv_d + inv_a
    frame["implied_prob_home"] = inv_h / margin
    frame["implied_prob_draw"] = inv_d / margin
    frame["implied_prob_away"] = inv_a / margin
    frame["bookie_margin"] = margin - 1.0  # overround; ~0.04-0.08 typical
    if {"odds_over25", "odds_under25"}.issubset(frame.columns):
        inv_o = 1.0 / frame["odds_over25"]
        inv_u = 1.0 / frame["odds_under25"]
        ou_margin = inv_o + inv_u
        frame["implied_prob_over25"] = inv_o / ou_margin
    return frame


def _add_rolling_team_form(frame: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add per-team rolling form (goals/points) computed from prior matches.

    Strictly pre-match: each row's rolling stats are computed from rows with
    earlier match_date. Avoids leakage.
    """
    needed = {"match_date", "home_team_id", "away_team_id", "home_score", "away_score"}
    if not needed.issubset(frame.columns):
        return frame

    # Build a long-format frame where each match contributes one row per
    # team (home perspective + away perspective), sorted chronologically.
    df = frame.sort_values("match_date").reset_index(drop=True)

    home = pd.DataFrame(
        {
            "match_date": df["match_date"],
            "team_id": df["home_team_id"],
            "goals_for": df["home_score"].astype(float),
            "goals_against": df["away_score"].astype(float),
        }
    )
    home["points"] = np.where(
        df["home_score"] > df["away_score"],
        3.0,
        np.where(df["home_score"] == df["away_score"], 1.0, 0.0),
    )
    away = pd.DataFrame(
        {
            "match_date": df["match_date"],
            "team_id": df["away_team_id"],
            "goals_for": df["away_score"].astype(float),
            "goals_against": df["home_score"].astype(float),
        }
    )
    away["points"] = np.where(
        df["away_score"] > df["home_score"],
        3.0,
        np.where(df["home_score"] == df["away_score"], 1.0, 0.0),
    )
    team_rows = pd.concat([home, away], ignore_index=True)
    team_rows = team_rows.sort_values(["team_id", "match_date"]).reset_index(drop=True)

    # Rolling mean over the prior `window` matches, EXCLUDING the current row
    # (shift(1) before rolling so we never peek at the current match's result).
    grouped = team_rows.groupby("team_id", group_keys=False)
    for col in ("goals_for", "goals_against", "points"):
        team_rows[f"roll_{col}"] = grouped[col].shift(1).rolling(window=window, min_periods=1).mean()

    # Join back to the original frame for home and away separately.
    home_stats = team_rows.merge(
        df[["match_date", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        on=["match_date", "team_id"],
        how="inner",
    )[["match_date", "team_id", "roll_goals_for", "roll_goals_against", "roll_points"]]
    away_stats = team_rows.merge(
        df[["match_date", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
        on=["match_date", "team_id"],
        how="inner",
    )[["match_date", "team_id", "roll_goals_for", "roll_goals_against", "roll_points"]]

    df = df.merge(
        home_stats.rename(
            columns={
                "team_id": "home_team_id",
                "roll_goals_for": "home_roll_goals_for",
                "roll_goals_against": "home_roll_goals_against",
                "roll_points": "home_roll_points",
            }
        ),
        on=["match_date", "home_team_id"],
        how="left",
    )
    df = df.merge(
        away_stats.rename(
            columns={
                "team_id": "away_team_id",
                "roll_goals_for": "away_roll_goals_for",
                "roll_goals_against": "away_roll_goals_against",
                "roll_points": "away_roll_points",
            }
        ),
        on=["match_date", "away_team_id"],
        how="left",
    )

    df["form_diff_points"] = df["home_roll_points"] - df["away_roll_points"]
    df["form_diff_goals"] = df["home_roll_goals_for"] - df["away_roll_goals_for"]
    return df


def get_feature_columns(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> List[str]:
    """Select numeric training features while avoiding result leakage columns."""
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


def validate_training_frame(
    frame: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
    min_samples: int = 100,
    min_feature_count: int = 5,
    min_target_classes: int = 2,
) -> TrainingDataQuality:
    """Validate that a frame is usable for model training."""
    if frame.empty:
        raise ValueError("Training data is empty")

    missing_columns = [column for column in ("match_date", target) if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing_columns)}")

    if len(frame) < min_samples:
        raise ValueError(f"Training data has {len(frame)} rows; at least {min_samples} are required")

    feature_columns = get_feature_columns(frame, target=target)
    if len(feature_columns) < min_feature_count:
        raise ValueError(
            f"Training data has {len(feature_columns)} numeric features; at least {min_feature_count} are required"
        )

    target_classes = int(frame[target].nunique(dropna=True))
    if target_classes < min_target_classes:
        raise ValueError(f"Training target has {target_classes} classes; at least {min_target_classes} are required")

    feature_values = frame[feature_columns]
    missing_feature_rate = float(feature_values.isna().sum().sum() / feature_values.size)

    if missing_feature_rate > 0.5:
        raise ValueError(f"Training features are {missing_feature_rate:.1%} missing; maximum allowed is 50.0%")

    date_min = frame["match_date"].min()
    date_max = frame["match_date"].max()

    return TrainingDataQuality(
        rows=len(frame),
        feature_count=len(feature_columns),
        target_classes=target_classes,
        date_min=date_min.isoformat() if pd.notna(date_min) else None,
        date_max=date_max.isoformat() if pd.notna(date_max) else None,
        missing_feature_rate=missing_feature_rate,
    )


def load_nhl_moneyline_frame(
    *,
    database_url: Optional[str] = None,
    input_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Load NHL moneyline training data from CSV or PostgreSQL.

    Returns a frame ready for model training: features_cache JSON
    flattened to columns, target column `nhl_moneyline` derived,
    match_date parsed. Unlike the soccer loader, this does NOT
    recompute rolling-form or implied-probability features in pandas —
    those already exist as columns in features_cache via
    compute_features_nhl.py, so re-deriving them would duplicate (and
    risk diverging from) the canonical pre-match values that prediction
    time will use.
    """
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(NHL_MONEYLINE_TRAINING_QUERY, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")

    return prepare_nhl_moneyline_frame(raw)


def prepare_nhl_moneyline_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten features_cache JSON, parse dates, ensure target column.

    Skips the soccer-specific implied-probability and rolling-form
    derivations — those live in features_cache as nhl_baseline columns,
    already devigged and rolling-aware (see scripts/compute_features_nhl.py).
    """
    if raw.empty:
        return raw.copy()

    frame = raw.copy()

    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)

    if NHL_MONEYLINE_TARGET not in frame.columns and {"home_score", "away_score"}.issubset(frame.columns):
        # SQL query derives this for us; this is a defensive fallback
        # for CSV inputs that bypass the query.
        frame[NHL_MONEYLINE_TARGET] = np.where(frame["home_score"] > frame["away_score"], 0, 1)

    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")

    return frame


def get_nhl_feature_columns(frame: pd.DataFrame, target: str = NHL_MONEYLINE_TARGET) -> List[str]:
    """Numeric training features for the NHL frame (excludes identifiers,
    raw scores, and the target)."""
    excluded = set(NHL_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


def _flatten_features(value: Any, prefix: str = "feature") -> Dict[str, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}

    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, dict):
        return {}

    flattened: Dict[str, float] = {}
    for key, item in value.items():
        safe_key = str(key).replace(".", "_")
        name = f"{prefix}__{safe_key}"
        if isinstance(item, dict):
            flattened.update(_flatten_features(item, prefix=name))
        elif isinstance(item, bool):
            flattened[name] = float(int(item))
        elif isinstance(item, (int, float, np.integer, np.floating)) and np.isfinite(item):
            flattened[name] = float(item)
    return flattened
