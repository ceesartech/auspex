"""Horse-racing training-data loader.

Horse racing is structurally different from team / 1v1 sports:
  * The unit of training is a RACE (with N runners), not a match.
  * Per-row features include both race-level context (duplicated
    across every entrant in the race) and per-entrant features.
  * The target is "did THIS entrant win THIS race" — a binary
    relevance score that LambdaMART / LightGBM Ranker consumes to
    learn relative ordering across the field.
  * The model adapter needs a `group` array telling it how many
    rows belong to each race. Within a group, the loss is computed
    pairwise (or listwise for LambdaRank), so the model learns
    relative strength conditioned on the field.

This loader pulls finished races with both:
  * a market_consensus_v1 prediction (so we have the consensus
    odds-implied prob as a feature — the model has to beat it AS
    INFORMATION, not in spite of it), AND
  * a recorded winner (finish_position=1 exists on some entrant).

Returns three artefacts ordered by race + program_number so the
group array aligns row-for-row with the DataFrame:

  X: pd.DataFrame  — numeric feature matrix (one row per entrant)
  y: np.ndarray    — binary target (1 if won, 0 otherwise)
  groups: np.ndarray — int array; sum(groups) == len(X)

The split-by-date helper preserves race-group integrity (you never
want a race split across train and test).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── SQL ─────────────────────────────────────────────────────────────


HORSE_RACING_TRAINING_QUERY = """
    SELECT
        r.id::text          AS race_id,
        r.race_date,
        r.track_name,
        r.race_number,
        e.id::text          AS entrant_id,
        e.program_number,
        e.morning_line_odds,
        e.starting_price,
        e.finish_position,
        e.scratched,
        e.disqualified,
        rp.confidence       AS consensus_prob,
        rfc.features        AS features_blob
    FROM races r
    JOIN race_entrants e ON e.race_id = r.id
    LEFT JOIN race_predictions rp
        ON rp.entrant_id = e.id
       AND rp.model_name = 'market_consensus_v1'
       AND rp.prediction_type = 'win'
    LEFT JOIN race_features_cache rfc
        ON rfc.race_id = r.id
       AND rfc.feature_set = 'horse_racing_baseline'
    WHERE r.status = 'finished'
      AND NOT e.scratched
      -- Restrict to races where SOME entrant has finish_position=1.
      -- Without a recorded winner the row is unusable as a target.
      AND EXISTS (
        SELECT 1 FROM race_entrants e2
        WHERE e2.race_id = r.id AND e2.finish_position = 1
      )
    ORDER BY r.race_date ASC, r.id::text, e.program_number ASC NULLS LAST
"""


# ── Quality summary (parallel to TrainingDataQuality for team sports) ──


@dataclass(frozen=True)
class HorseRacingDataQuality:
    rows: int
    races: int
    feature_count: int
    win_rate: float
    date_min: Optional[str]
    date_max: Optional[str]
    missing_feature_rate: float


# ── Helpers ─────────────────────────────────────────────────────────


def _extract_entrant_features(blob: Any, entrant_id: str) -> dict:
    """Pick THIS entrant's features dict out of the JSONB blob the
    feature script writes. The blob shape is:

        {
          "race_level": {...},
          "entrants": [
            {"entrant_id": "...", "features": {...}},
            ...
          ]
        }

    Returns {} when the blob is missing / malformed / has no entry
    for this entrant — the caller falls back to consensus odds +
    raw race fields so the row isn't entirely useless."""
    if blob is None:
        return {}
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except (TypeError, ValueError):
            return {}
    if not isinstance(blob, dict):
        return {}
    out: dict = {}
    race_level = blob.get("race_level")
    if isinstance(race_level, dict):
        # Prefix with race_ so race-level and per-entrant features
        # never collide on a generic key like "field_size".
        for k, v in race_level.items():
            out[f"race_{k}"] = v
    entrants = blob.get("entrants")
    if isinstance(entrants, list):
        for ent in entrants:
            if isinstance(ent, dict) and str(ent.get("entrant_id")) == str(entrant_id):
                feats = ent.get("features")
                if isinstance(feats, dict):
                    for k, v in feats.items():
                        out[k] = v
                break
    return out


def _consensus_implied_from_odds(ml: Any, sp: Any) -> Optional[float]:
    """Standalone 'best-available decimal → implied prob' so the model
    sees the same signal _consensus_decimal uses in the predictor.
    Picks ML when valid, falls back to SP. Returns None when neither
    is usable."""
    for raw in (ml, sp):
        if raw is None:
            continue
        try:
            d = float(raw)
        except (TypeError, ValueError):
            continue
        if d > 1.0:
            return 1.0 / d
    return None


# ── Loader ──────────────────────────────────────────────────────────


def load_training_frame(
    *,
    database_url: Optional[str] = None,
    input_csv: Optional[str] = None,
    query: str = HORSE_RACING_TRAINING_QUERY,
) -> pd.DataFrame:
    """Load + assemble. Returns one row per (race, entrant) ordered
    by race_date then race_id then program_number — that ordering is
    the load-bearing invariant for the group array."""
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

    if raw.empty:
        return raw.copy()

    return prepare_training_frame(raw)


def prepare_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Take the raw SQL output and flatten the features JSONB into
    columns. Adds:
      * `target` (1 if finish_position == 1, else 0)
      * `consensus_implied_prob` (from morning_line_odds / starting_price)
    Preserves race_id, race_date, program_number as identifier
    columns the trainer will exclude from features."""
    if raw.empty:
        return raw.copy()

    frame = raw.copy()

    # Flatten the JSONB blob per row. Avoids a left join in pandas
    # by handling it row-by-row from the already-keyed blob.
    feature_rows = []
    for blob, entrant_id in zip(frame["features_blob"], frame["entrant_id"]):
        feature_rows.append(_extract_entrant_features(blob, str(entrant_id)))
    flattened = pd.DataFrame(feature_rows, index=frame.index)
    frame = pd.concat([frame.drop(columns=["features_blob"]), flattened], axis=1)

    # Derived columns the trainer relies on.
    frame["target"] = (frame["finish_position"] == 1).astype(int)
    frame["consensus_implied_prob"] = [
        _consensus_implied_from_odds(ml, sp)
        for ml, sp in zip(frame["morning_line_odds"], frame["starting_price"])
    ]

    if "race_date" in frame.columns:
        frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce", utc=True)

    return frame


# ── Splitting + group-array helpers ─────────────────────────────────


NON_FEATURE_COLUMNS = {
    "race_id",
    "race_date",
    "track_name",
    "race_number",
    "entrant_id",
    "program_number",
    "finish_position",
    "scratched",
    "disqualified",
    "morning_line_odds",
    "starting_price",
    "target",
    "consensus_prob",
}


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns that aren't identifiers / target / raw odds.
    Keeps consensus_implied_prob (the model SHOULD see the market's
    view) but drops the raw decimals so the model isn't double-
    counting the same signal in two scales."""
    if frame.empty:
        return []
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [c for c in numeric if c not in NON_FEATURE_COLUMNS]


def split_by_date(
    frame: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward split: every row whose race_date < split_date
    goes to train; >= split_date to test. Race-group integrity is
    preserved because the query ordering keeps all entrants of a
    race contiguous AND every entrant in a race shares the same
    race_date (so no single race straddles the split)."""
    if "race_date" not in frame.columns:
        raise ValueError("Frame missing race_date column")
    split_ts = pd.to_datetime(split_date, utc=True)
    train_mask = frame["race_date"] < split_ts
    test_mask = ~train_mask
    return frame.loc[train_mask].reset_index(drop=True), frame.loc[
        test_mask
    ].reset_index(drop=True)


def group_array(frame: pd.DataFrame) -> np.ndarray:
    """Compute the per-race row-count array LightGBM Ranker expects.
    Returns an int array where sum(groups) == len(frame). The frame
    MUST be ordered so all entrants of one race are contiguous —
    the loader's ORDER BY guarantees that, but if you reshuffle in
    pandas (e.g. via dropna without sort) you must re-group first
    or training will silently treat the wrong rows as one race."""
    if frame.empty or "race_id" not in frame.columns:
        return np.array([], dtype=np.int64)
    # value_counts(sort=False) preserves the existing order, which is
    # what we need (Python dict preserves insertion order since 3.7).
    sizes = frame["race_id"].value_counts(sort=False).values
    return np.asarray(sizes, dtype=np.int64)


def validate_training_frame(
    frame: pd.DataFrame,
    *,
    min_races: int = 100,
    min_feature_count: int = 5,
) -> HorseRacingDataQuality:
    """Sanity-check a prepared frame. Raises ValueError when the
    corpus is unusable; otherwise returns a summary the caller logs."""
    if frame.empty:
        raise ValueError("Training data is empty")

    if "target" not in frame.columns:
        raise ValueError("Training data is missing the target column")

    races = int(frame["race_id"].nunique())
    if races < min_races:
        raise ValueError(f"Training data has {races} races; at least {min_races} required")

    feature_columns = get_feature_columns(frame)
    if len(feature_columns) < min_feature_count:
        raise ValueError(
            f"Training data has {len(feature_columns)} numeric features; "
            f"at least {min_feature_count} required"
        )

    # Spot-check group integrity: every race should have at least one
    # winner. If the EXISTS guard slipped (e.g., manual CSV input),
    # we want to catch it before the trainer silently learns garbage.
    winners_per_race = (
        frame.groupby("race_id")["target"].sum()
    )
    if (winners_per_race == 0).any():
        bad = int((winners_per_race == 0).sum())
        raise ValueError(f"{bad} races have no winner row; check the input")

    feature_values = frame[feature_columns]
    missing_rate = float(feature_values.isna().sum().sum() / max(feature_values.size, 1))
    win_rate = float(frame["target"].mean())

    return HorseRacingDataQuality(
        rows=len(frame),
        races=races,
        feature_count=len(feature_columns),
        win_rate=win_rate,
        date_min=frame["race_date"].min().isoformat()
        if "race_date" in frame.columns and pd.notna(frame["race_date"].min())
        else None,
        date_max=frame["race_date"].max().isoformat()
        if "race_date" in frame.columns and pd.notna(frame["race_date"].max())
        else None,
        missing_feature_rate=missing_rate,
    )
