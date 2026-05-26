"""Utilities for loading and validating model training data."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_TRAINING_QUERY = """
    SELECT
        m.id::text AS match_id,
        m.match_date,
        ht.name AS home_team,
        at.name AS away_team,
        m.home_score,
        m.away_score,
        CASE
            WHEN m.home_score > m.away_score THEN 0
            WHEN m.home_score = m.away_score THEN 1
            ELSE 2
        END AS match_outcome,
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
NON_FEATURE_COLUMNS = {
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    TARGET_COLUMN,
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
        import psycopg2

        with psycopg2.connect(database_url) as conn:
            raw = pd.read_sql(query, conn)
    else:
        raise ValueError("Provide input_csv or database_url")

    return prepare_training_frame(raw)


def prepare_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten feature JSON and derive target columns needed by model training."""
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

    return frame


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
