"""Utilities for loading and validating model training data."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Pulls finished SOCCER matches with closing odds joined in, plus the most
# recent features_cache row from the 'baseline' (soccer) feature set. The
# odds-derived columns let us train a working baseline model even before
# features_cache has been populated for every historical match. Once
# compute_features.py has been run over the corpus, the `features` JSON adds
# the richer 250+ feature set.
#
# The l.sport='soccer' filter + feature_set='baseline' pin matter: without
# them, finished NHL matches (which have NULL 1x2 odds and carry
# 'nhl_baseline' feature_set rows that flatten to NHL-only feature__* columns)
# would land in this frame as ~all-NaN rows. The naive missing-feature gate
# in validate_training_frame averages NaNs across every feature column, so
# even a few thousand NHL rows tipped the gate past 50% after Phase 3.
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
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'soccer'
      AND m.status = 'finished'
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


# ============= NHL REGULATION 3-WAY (60-minute outcome) =============

# NHL regulation 3-way target. 0 = home reg win, 1 = regulation tie
# (game went to OT/SO), 2 = away reg win. Distribution from recent
# seasons: home ~42%, tie ~22%, away ~36%.
NHL_REGULATION_TARGET = "nhl_regulation"

# NHL regulation training query. Differs from the moneyline query in
# four ways:
#   1. Target derived from matches.metadata->>'regulation_winner' (set
#      by load_nhl_historical when it sums periods 1-3 of the linescore),
#      not from final home_score/away_score (which include OT/SO).
#   2. Ties are KEPT — regulation ties are a valid third class, unlike
#      moneyline where every game has a winner.
#   3. WHERE filter requires metadata->>'regulation_winner' IS NOT NULL
#      so games loaded before the regulation_winner field was populated
#      (which would only matter if the schema evolves) get filtered out
#      cleanly rather than producing NULL targets.
#   4. The features_cache pin (feature_set='nhl_baseline') stays — the
#      same feature set serves both moneyline and regulation models;
#      only the target column changes.
NHL_REGULATION_TRAINING_QUERY = """
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
        CASE m.metadata->>'regulation_winner'
            WHEN 'home' THEN 0
            WHEN 'tie'  THEN 1
            WHEN 'away' THEN 2
        END AS nhl_regulation,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'home' AND o.line = -1.5 AND NOT o.is_live) AS odds_home_pl15,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'away' AND o.line = 1.5 AND NOT o.is_live) AS odds_away_pl15,
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
      AND m.metadata ? 'regulation_winner'
      AND m.metadata->>'regulation_winner' IN ('home', 'tie', 'away')
    ORDER BY m.match_date ASC
"""

# Same non-feature exclusion set as moneyline plus the new target.
# Built as a fresh set rather than aliasing NHL_NON_FEATURE_COLUMNS so
# adding a 4th task later doesn't accidentally inherit prior targets.
NHL_REGULATION_NON_FEATURE_COLUMNS = {
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
    NHL_REGULATION_TARGET,
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


def load_nhl_regulation_frame(
    *,
    database_url: Optional[str] = None,
    input_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Load NHL regulation 3-way training data from CSV or PostgreSQL.

    Returns a frame ready for model training: features_cache JSON
    flattened to columns, target column `nhl_regulation` derived from
    matches.metadata.regulation_winner, match_date parsed. Like the
    moneyline loader, this does NOT recompute rolling-form features in
    pandas — those already live in features_cache as nhl_baseline
    columns.
    """
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(NHL_REGULATION_TRAINING_QUERY, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")

    return prepare_nhl_regulation_frame(raw)


def prepare_nhl_regulation_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten features_cache JSON, parse dates. The target column is
    pre-computed in the SQL CASE so we don't need a fallback derivation
    here — unlike moneyline (which can be derived from scores), there's
    no way to recover the regulation winner from final scores alone
    once a game has gone to OT/SO.
    """
    if raw.empty:
        return raw.copy()

    frame = raw.copy()

    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)

    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")

    return frame


def get_nhl_regulation_feature_columns(frame: pd.DataFrame, target: str = NHL_REGULATION_TARGET) -> List[str]:
    """Numeric training features for the NHL regulation frame."""
    excluded = set(NHL_REGULATION_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


# ============= NHL PUCK LINE (home covers -1.5) =============

# Puck-line target. 0 = home covers -1.5 (won by 2+ goals INCLUDING OT/SO
# goals), 1 = home does not cover (won by 1 OR lost).
NHL_PUCK_LINE_TARGET = "nhl_puck_line"

# Identical to the moneyline query except for the CASE target —
# everything else (sport scoping, features_cache pin, market joins,
# defensive filters) carries over verbatim because the feature inputs
# don't change between NHL classification tasks. The target swap is
# what makes each task its own model.
NHL_PUCK_LINE_TRAINING_QUERY = """
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
            WHEN (m.home_score - m.away_score) >= 2 THEN 0
            ELSE 1
        END AS nhl_puck_line,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'home' AND o.line = -1.5 AND NOT o.is_live) AS odds_home_pl15,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'away' AND o.line = 1.5 AND NOT o.is_live) AS odds_away_pl15,
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
    ORDER BY m.match_date ASC
"""

NHL_PUCK_LINE_NON_FEATURE_COLUMNS = {
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
    NHL_PUCK_LINE_TARGET,
    "features",
}


# ============= NHL TOTAL (over 5.5) =============

# Total-goals target. 0 = over 5.5 (sum of final scores >= 6 INCLUDING
# OT/SO goals — the SO winner is credited with +1 toward the total per
# NHL convention), 1 = under 5.5 (sum <= 5).
NHL_TOTAL_TARGET = "nhl_total"

NHL_TOTAL_TRAINING_QUERY = """
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
            WHEN (m.home_score + m.away_score) >= 6 THEN 0
            ELSE 1
        END AS nhl_total,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'home' AND o.line = -1.5 AND NOT o.is_live) AS odds_home_pl15,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'spread'
              AND o.selection = 'away' AND o.line = 1.5 AND NOT o.is_live) AS odds_away_pl15,
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
    ORDER BY m.match_date ASC
"""

NHL_TOTAL_NON_FEATURE_COLUMNS = {
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
    NHL_TOTAL_TARGET,
    "features",
}


def _flatten_nhl_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Shared frame preparation for NHL puck-line and total tasks:
    flatten features_cache JSON, parse match_date. Targets are computed
    in the SQL query — no in-pandas fallback needed for either task
    because both are pure functions of home_score/away_score that the
    query CASE handles."""
    if raw.empty:
        return raw.copy()

    frame = raw.copy()

    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)

    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")

    return frame


def load_nhl_puck_line_frame(
    *,
    database_url: Optional[str] = None,
    input_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Load NHL puck-line (home covers -1.5) training data."""
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(NHL_PUCK_LINE_TRAINING_QUERY, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")

    return prepare_nhl_puck_line_frame(raw)


def prepare_nhl_puck_line_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return _flatten_nhl_frame(raw)


def get_nhl_puck_line_feature_columns(frame: pd.DataFrame, target: str = NHL_PUCK_LINE_TARGET) -> List[str]:
    excluded = set(NHL_PUCK_LINE_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


def load_nhl_total_frame(
    *,
    database_url: Optional[str] = None,
    input_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Load NHL total-goals (over 5.5) training data."""
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(NHL_TOTAL_TRAINING_QUERY, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")

    return prepare_nhl_total_frame(raw)


def prepare_nhl_total_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return _flatten_nhl_frame(raw)


def get_nhl_total_feature_columns(frame: pd.DataFrame, target: str = NHL_TOTAL_TARGET) -> List[str]:
    excluded = set(NHL_TOTAL_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


# ====================================================================
# NBA TRAINING QUERIES — 3 markets: moneyline, spread, total
# ====================================================================
#
# Architecture decision: NBA spread + total use LINE-AS-FEATURE design
# (the closing line is a numeric input column, not a fixed filter
# like NHL's puck_line=-1.5 / total=5.5). One trained model handles
# the full ladder of lines the book offers per game. See
# scripts/compute_features_nba.py for the matching feature-side
# implementation.
#
# Three queries, mirroring the NHL pattern:
#   * NBA_MONEYLINE_TRAINING_QUERY — 2-class home/away
#   * NBA_SPREAD_TRAINING_QUERY    — 2-class home-covered / not, derived
#                                    from (margin, closing_spread_home)
#   * NBA_TOTAL_TRAINING_QUERY     — 2-class over / under, derived from
#                                    (total, closing_total_line)
#
# Spread + total queries INNER JOIN to the odds table — matches without
# a real closing line are dropped from the training set (we never want
# to train on neutral-default lines). With current prod data (89%
# spread / 85% total coverage) this filters ~4125 finished matches
# down to ~3678 / ~3499 trainable rows. Plenty.

NBA_MONEYLINE_TARGET = "nba_moneyline"
NBA_SPREAD_TARGET = "nba_spread"
NBA_TOTAL_TARGET = "nba_total"

# Moneyline: home win = 0, away win = 1. NBA always has a winner
# after OT, so no tie row to filter.
NBA_MONEYLINE_TRAINING_QUERY = """
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
        CASE WHEN m.home_score > m.away_score THEN 0 ELSE 1 END AS nba_moneyline,
        -- Closing moneyline averaged across books
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'nba_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'nba'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.home_score <> m.away_score  -- NBA never ties; ties = data error
    ORDER BY m.match_date ASC
"""

# Spread: home covered = 0, away covered = 1. Cover condition:
# (home_score - away_score) > -closing_spread_home (the line is from
# home perspective; -7.5 means home favored by 7.5, so home covers iff
# margin > 7.5). The LATERAL spread.* subquery returns the avg closing
# line across books — INNER joined so matches without a closing spread
# are dropped from the training corpus.
NBA_SPREAD_TRAINING_QUERY = """
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
        spread.home_line AS closing_spread_home,
        spread.home_odds AS odds_spread_home,
        spread.away_odds AS odds_spread_away,
        CASE
            WHEN (m.home_score - m.away_score) > -spread.home_line THEN 0  -- home covered
            ELSE 1                                                          -- away covered
        END AS nba_spread,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    JOIN LATERAL (
        SELECT
            AVG(CASE WHEN o.selection = 'home' THEN o.line END) AS home_line,
            AVG(CASE WHEN o.selection = 'home' THEN o.odds_decimal END) AS home_odds,
            AVG(CASE WHEN o.selection = 'away' THEN o.odds_decimal END) AS away_odds
        FROM odds o
        WHERE o.match_id = m.id AND o.market_type = 'spread' AND NOT o.is_live
    ) spread ON spread.home_line IS NOT NULL
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'nba_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'nba'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.match_date ASC
"""

# Total: over = 0, under = 1. Cover condition: home_score + away_score
# > closing_total_line. INNER LATERAL filters to matches with a real
# total line (~85% of NBA matches in DB after the historical-odds
# backfill).
NBA_TOTAL_TRAINING_QUERY = """
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
        total.line AS closing_total_line,
        total.over_odds AS odds_total_over,
        total.under_odds AS odds_total_under,
        CASE
            WHEN (m.home_score + m.away_score) > total.line THEN 0  -- over
            ELSE 1                                                   -- under
        END AS nba_total,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    JOIN LATERAL (
        SELECT
            AVG(CASE WHEN o.selection = 'over'  THEN o.line END)        AS line,
            AVG(CASE WHEN o.selection = 'over'  THEN o.odds_decimal END) AS over_odds,
            AVG(CASE WHEN o.selection = 'under' THEN o.odds_decimal END) AS under_odds
        FROM odds o
        WHERE o.match_id = m.id AND o.market_type = 'total' AND NOT o.is_live
    ) total ON total.line IS NOT NULL
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'nba_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'nba'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.match_date ASC
"""

# Columns each query exposes that must NOT be used as model inputs.
# Same structure as NHL_NON_FEATURE_COLUMNS — IDs, raw scores (leakage),
# the target itself, and the features blob (flattened separately).
NBA_MONEYLINE_NON_FEATURE_COLUMNS = {
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
    NBA_MONEYLINE_TARGET,
    "features",
}
NBA_SPREAD_NON_FEATURE_COLUMNS = NBA_MONEYLINE_NON_FEATURE_COLUMNS - {NBA_MONEYLINE_TARGET} | {NBA_SPREAD_TARGET}
NBA_TOTAL_NON_FEATURE_COLUMNS = NBA_MONEYLINE_NON_FEATURE_COLUMNS - {NBA_MONEYLINE_TARGET} | {NBA_TOTAL_TARGET}


def _flatten_nba_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten features_cache JSON + parse match_date. Same shape as
    the NHL equivalent — no rolling-form recomputation in pandas
    because compute_features_nba.py already produced the canonical
    pre-match values that predict-time will use."""
    if raw.empty:
        return raw.copy()
    frame = raw.copy()
    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    return frame


def _load_nba(query: str, prepare, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(query, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")
    return prepare(raw)


def prepare_nba_moneyline_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_nba_frame(raw)
    # Defensive: derive target from scores if a CSV input bypassed SQL.
    if NBA_MONEYLINE_TARGET not in frame.columns and {"home_score", "away_score"}.issubset(frame.columns):
        frame[NBA_MONEYLINE_TARGET] = np.where(frame["home_score"] > frame["away_score"], 0, 1)
    return frame


def prepare_nba_spread_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_nba_frame(raw)
    if NBA_SPREAD_TARGET not in frame.columns and {"home_score", "away_score", "closing_spread_home"}.issubset(
        frame.columns
    ):
        margin = frame["home_score"] - frame["away_score"]
        frame[NBA_SPREAD_TARGET] = np.where(margin > -frame["closing_spread_home"], 0, 1)
    return frame


def prepare_nba_total_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_nba_frame(raw)
    if NBA_TOTAL_TARGET not in frame.columns and {"home_score", "away_score", "closing_total_line"}.issubset(
        frame.columns
    ):
        total = frame["home_score"] + frame["away_score"]
        frame[NBA_TOTAL_TARGET] = np.where(total > frame["closing_total_line"], 0, 1)
    return frame


def load_nba_moneyline_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_nba(NBA_MONEYLINE_TRAINING_QUERY, prepare_nba_moneyline_frame, database_url, input_csv)


def load_nba_spread_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_nba(NBA_SPREAD_TRAINING_QUERY, prepare_nba_spread_frame, database_url, input_csv)


def load_nba_total_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_nba(NBA_TOTAL_TRAINING_QUERY, prepare_nba_total_frame, database_url, input_csv)


def get_nba_moneyline_feature_columns(frame: pd.DataFrame, target: str = NBA_MONEYLINE_TARGET) -> List[str]:
    excluded = set(NBA_MONEYLINE_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


def get_nba_spread_feature_columns(frame: pd.DataFrame, target: str = NBA_SPREAD_TARGET) -> List[str]:
    excluded = set(NBA_SPREAD_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


def get_nba_total_feature_columns(frame: pd.DataFrame, target: str = NBA_TOTAL_TARGET) -> List[str]:
    excluded = set(NBA_TOTAL_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


# ====================================================================
# NFL TRAINING QUERIES — 3 markets: moneyline, spread, total
# ====================================================================
#
# Architecture mirrors NBA: line-as-feature for spread + total
# (closing_spread_home, closing_total_line in features_cache are
# numeric model inputs). One trained model handles every line the
# book offers per game — no fixed-line filters.
#
# NFL-specific filtering on the spread/total queries: same INNER
# JOIN to the odds table to drop matches without real closing lines.
# After the historical odds backfill, expect ~280 games × 17 weeks
# × 3 seasons ≈ 855 matches with high odds coverage.

NFL_MONEYLINE_TARGET = "nfl_moneyline"
NFL_SPREAD_TARGET = "nfl_spread"
NFL_TOTAL_TARGET = "nfl_total"

# Moneyline: home win = 0, away win = 1. NFL ties are possible but
# extremely rare (~1 per season) and create an edge case — we
# exclude them (home_score <> away_score) so the model trains on a
# clean 2-class target.
#
# Weather strategy (full Phase 14 history is worth understanding so
# this doesn't get re-attempted naively):
#   * v3 (commit 61d1d84) added weather features with NEUTRAL_DEFAULTS
#     for missing rows → -7pts OOS on NFL ML. The GBDT overfit
#     value=10.0 as a "missing-data sentinel." Reverted at 17b3eb6.
#   * v4a (commits 5e699f5 + a72a3b4) added an EXISTS-gate requiring
#     match_weather.data_kind='actual' + OMIT-on-missing in
#     compute_features → -13pts vs v1. The gate cut the training
#     corpus from ~620 to 363 pre-2024 rows, and inference broke
#     on test matches lacking weather columns. Walk-forward result
#     58.1% vs 71.3% v1 baseline.
#   * v4b (current): drop the gate. Train on the full corpus.
#     compute_features_*.fetch_weather now always emits the canonical
#     weather key set with None (→ NaN) for missing values. GBDTs
#     treat NaN as a "missing direction" in split learning — a
#     distinct concept from any literal default value, so no
#     leakage even though the corpus contains missing rows.
NFL_MONEYLINE_TRAINING_QUERY = """
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
        CASE WHEN m.home_score > m.away_score THEN 0 ELSE 1 END AS nfl_moneyline,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'nfl_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'nfl'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.home_score <> m.away_score
    ORDER BY m.match_date ASC
"""

# Spread: home covered = 0, away covered = 1. Same line-as-feature
# math as NBA. NFL spreads span -14.5 to +14.5 with 0.5-point steps,
# so pushes are rare but possible at half-point lines (in practice
# NFL lines are usually half-points to eliminate pushes).
NFL_SPREAD_TRAINING_QUERY = """
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
        spread.home_line AS closing_spread_home,
        spread.home_odds AS odds_spread_home,
        spread.away_odds AS odds_spread_away,
        CASE
            WHEN (m.home_score - m.away_score) > -spread.home_line THEN 0
            ELSE 1
        END AS nfl_spread,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    JOIN LATERAL (
        SELECT
            AVG(CASE WHEN o.selection = 'home' THEN o.line END) AS home_line,
            AVG(CASE WHEN o.selection = 'home' THEN o.odds_decimal END) AS home_odds,
            AVG(CASE WHEN o.selection = 'away' THEN o.odds_decimal END) AS away_odds
        FROM odds o
        WHERE o.match_id = m.id AND o.market_type = 'spread' AND NOT o.is_live
    ) spread ON spread.home_line IS NOT NULL
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'nfl_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'nfl'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.match_date ASC
"""

# Total: over = 0, under = 1. NFL totals span 38-55 typically.
NFL_TOTAL_TRAINING_QUERY = """
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
        total.line AS closing_total_line,
        total.over_odds AS odds_total_over,
        total.under_odds AS odds_total_under,
        CASE
            WHEN (m.home_score + m.away_score) > total.line THEN 0
            ELSE 1
        END AS nfl_total,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    JOIN LATERAL (
        SELECT
            AVG(CASE WHEN o.selection = 'over'  THEN o.line END)        AS line,
            AVG(CASE WHEN o.selection = 'over'  THEN o.odds_decimal END) AS over_odds,
            AVG(CASE WHEN o.selection = 'under' THEN o.odds_decimal END) AS under_odds
        FROM odds o
        WHERE o.match_id = m.id AND o.market_type = 'total' AND NOT o.is_live
    ) total ON total.line IS NOT NULL
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'nfl_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'nfl'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.match_date ASC
"""

# Same NON_FEATURE shape as NBA — IDs, raw scores (leakage), the
# target itself, and the features blob (flattened separately).
NFL_MONEYLINE_NON_FEATURE_COLUMNS = {
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
    NFL_MONEYLINE_TARGET,
    "features",
}
NFL_SPREAD_NON_FEATURE_COLUMNS = NFL_MONEYLINE_NON_FEATURE_COLUMNS - {NFL_MONEYLINE_TARGET} | {NFL_SPREAD_TARGET}
NFL_TOTAL_NON_FEATURE_COLUMNS = NFL_MONEYLINE_NON_FEATURE_COLUMNS - {NFL_MONEYLINE_TARGET} | {NFL_TOTAL_TARGET}


def _flatten_nfl_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten features_cache JSON + parse match_date. Mirror of the
    NBA equivalent — no rolling-form recomputation in pandas because
    compute_features_nfl.py already produced the canonical pre-match
    values predict-time will use."""
    if raw.empty:
        return raw.copy()
    frame = raw.copy()
    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    return frame


def _load_nfl(query: str, prepare, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(query, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")
    return prepare(raw)


def prepare_nfl_moneyline_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_nfl_frame(raw)
    if NFL_MONEYLINE_TARGET not in frame.columns and {"home_score", "away_score"}.issubset(frame.columns):
        frame[NFL_MONEYLINE_TARGET] = np.where(frame["home_score"] > frame["away_score"], 0, 1)
    return frame


def prepare_nfl_spread_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_nfl_frame(raw)
    if NFL_SPREAD_TARGET not in frame.columns and {"home_score", "away_score", "closing_spread_home"}.issubset(
        frame.columns
    ):
        margin = frame["home_score"] - frame["away_score"]
        frame[NFL_SPREAD_TARGET] = np.where(margin > -frame["closing_spread_home"], 0, 1)
    return frame


def prepare_nfl_total_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_nfl_frame(raw)
    if NFL_TOTAL_TARGET not in frame.columns and {"home_score", "away_score", "closing_total_line"}.issubset(
        frame.columns
    ):
        total = frame["home_score"] + frame["away_score"]
        frame[NFL_TOTAL_TARGET] = np.where(total > frame["closing_total_line"], 0, 1)
    return frame


def load_nfl_moneyline_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_nfl(NFL_MONEYLINE_TRAINING_QUERY, prepare_nfl_moneyline_frame, database_url, input_csv)


def load_nfl_spread_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_nfl(NFL_SPREAD_TRAINING_QUERY, prepare_nfl_spread_frame, database_url, input_csv)


def load_nfl_total_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_nfl(NFL_TOTAL_TRAINING_QUERY, prepare_nfl_total_frame, database_url, input_csv)


def get_nfl_moneyline_feature_columns(frame: pd.DataFrame, target: str = NFL_MONEYLINE_TARGET) -> List[str]:
    excluded = set(NFL_MONEYLINE_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


def get_nfl_spread_feature_columns(frame: pd.DataFrame, target: str = NFL_SPREAD_TARGET) -> List[str]:
    excluded = set(NFL_SPREAD_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


def get_nfl_total_feature_columns(frame: pd.DataFrame, target: str = NFL_TOTAL_TARGET) -> List[str]:
    excluded = set(NFL_TOTAL_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


# ── Tennis (Phase 11) ─────────────────────────────────────────────────
# Tennis is the first 1v1 sport. Single market in v1: moneyline (match
# winner). Total games + set-betting need linescore parsing — defer
# to v2. Ties don't exist in tennis (a match always produces a winner),
# so the WHERE clause doesn't need a tie-exclusion filter like NFL.

TENNIS_MONEYLINE_TARGET = "tennis_moneyline"

TENNIS_MONEYLINE_TRAINING_QUERY = """
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
        CASE WHEN m.home_score > m.away_score THEN 0 ELSE 1 END AS tennis_moneyline,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'tennis_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'tennis'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.home_score <> m.away_score
    ORDER BY m.match_date ASC
"""

TENNIS_MONEYLINE_NON_FEATURE_COLUMNS = {
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
    TENNIS_MONEYLINE_TARGET,
    "features",
}


def _flatten_tennis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten features_cache JSON + parse match_date. Mirror of the
    NFL equivalent — compute_features_tennis.py already produced the
    canonical pre-match values predict-time will use."""
    if raw.empty:
        return raw.copy()
    frame = raw.copy()
    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    return frame


def _load_tennis(
    query: str, prepare, database_url: Optional[str] = None, input_csv: Optional[str] = None
) -> pd.DataFrame:
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(query, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")
    return prepare(raw)


def prepare_tennis_moneyline_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_tennis_frame(raw)
    if TENNIS_MONEYLINE_TARGET not in frame.columns and {"home_score", "away_score"}.issubset(frame.columns):
        frame[TENNIS_MONEYLINE_TARGET] = np.where(frame["home_score"] > frame["away_score"], 0, 1)
    return frame


def load_tennis_moneyline_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_tennis(TENNIS_MONEYLINE_TRAINING_QUERY, prepare_tennis_moneyline_frame, database_url, input_csv)


def get_tennis_moneyline_feature_columns(frame: pd.DataFrame, target: str = TENNIS_MONEYLINE_TARGET) -> List[str]:
    excluded = set(TENNIS_MONEYLINE_NON_FEATURE_COLUMNS)
    excluded.add(target)
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [column for column in numeric if column not in excluded]


# ── MMA (Phase 12) ────────────────────────────────────────────────────
# MMA shares the 1v1 / single-moneyline-market shape with tennis. Same
# tie-exclusion filter (defensive — draws/data-corruption rows are
# already dropped at the ingest layer, but the WHERE keeps anything
# weird out of training).

MMA_MONEYLINE_TARGET = "mma_moneyline"

MMA_MONEYLINE_TRAINING_QUERY = """
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
        CASE WHEN m.home_score > m.away_score THEN 0 ELSE 1 END AS mma_moneyline,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'home' AND NOT o.is_live) AS odds_home_ml,
        (SELECT AVG(o.odds_decimal) FROM odds o
            WHERE o.match_id = m.id AND o.market_type = 'moneyline'
              AND o.selection = 'away' AND NOT o.is_live) AS odds_away_ml,
        fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    LEFT JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'mma_baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE l.sport = 'mma'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.home_score <> m.away_score
    ORDER BY m.match_date ASC
"""

MMA_MONEYLINE_NON_FEATURE_COLUMNS = {
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
    MMA_MONEYLINE_TARGET,
    "features",
}


def _flatten_mma_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    frame = raw.copy()
    if "features" in frame.columns:
        feature_rows = [_flatten_features(value) for value in frame["features"]]
        flattened = pd.DataFrame(feature_rows, index=frame.index)
        frame = pd.concat([frame.drop(columns=["features"]), flattened], axis=1)
    if "match_date" in frame.columns:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    return frame


def _load_mma(query: str, prepare, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    if input_csv:
        raw = pd.read_csv(input_csv)
    elif database_url:
        from sqlalchemy import create_engine

        engine = create_engine(database_url)
        try:
            raw = pd.read_sql(query, engine)
        finally:
            engine.dispose()
    else:
        raise ValueError("Provide input_csv or database_url")
    return prepare(raw)


def prepare_mma_moneyline_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_mma_frame(raw)
    if MMA_MONEYLINE_TARGET not in frame.columns and {"home_score", "away_score"}.issubset(frame.columns):
        frame[MMA_MONEYLINE_TARGET] = np.where(frame["home_score"] > frame["away_score"], 0, 1)
    return frame


def load_mma_moneyline_frame(*, database_url: Optional[str] = None, input_csv: Optional[str] = None) -> pd.DataFrame:
    return _load_mma(MMA_MONEYLINE_TRAINING_QUERY, prepare_mma_moneyline_frame, database_url, input_csv)


def get_mma_moneyline_feature_columns(frame: pd.DataFrame, target: str = MMA_MONEYLINE_TARGET) -> List[str]:
    excluded = set(MMA_MONEYLINE_NON_FEATURE_COLUMNS)
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
