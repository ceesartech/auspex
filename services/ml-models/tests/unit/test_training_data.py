"""Tests for training data loading and validation utilities."""

import pandas as pd
import pytest
from utils.training_data import get_feature_columns, prepare_training_frame, validate_training_frame


def test_prepare_training_frame_flattens_features_and_derives_target():
    raw = pd.DataFrame(
        [
            {
                "match_date": "2025-01-01T12:00:00Z",
                "home_score": 2,
                "away_score": 1,
                "features": {"team": {"form": 0.75}, "is_derby": True, "label": "ignored"},
            }
        ]
    )

    frame = prepare_training_frame(raw)

    assert frame.loc[0, "match_outcome"] == 0
    assert frame.loc[0, "feature__team__form"] == 0.75
    assert frame.loc[0, "feature__is_derby"] == 1.0
    assert "feature__label" not in frame.columns


def test_get_feature_columns_excludes_result_leakage_columns():
    frame = pd.DataFrame(
        {
            "match_date": pd.date_range("2025-01-01", periods=2),
            "home_score": [2, 1],
            "away_score": [1, 1],
            "match_outcome": [0, 1],
            "feature__form": [0.8, 0.5],
            "home_odds": [1.8, 2.0],
        }
    )

    assert get_feature_columns(frame) == ["feature__form", "home_odds"]


def test_validate_training_frame_reports_quality():
    frame = pd.DataFrame(
        {
            "match_date": pd.date_range("2025-01-01", periods=4),
            "match_outcome": [0, 1, 2, 0],
            "feature__form": [0.8, 0.5, 0.4, 0.7],
            "feature__odds": [1.8, 2.0, 3.1, 1.9],
        }
    )

    quality = validate_training_frame(frame, min_samples=4, min_feature_count=2)

    assert quality.rows == 4
    assert quality.feature_count == 2
    assert quality.target_classes == 3


def test_validate_training_frame_rejects_missing_features():
    frame = pd.DataFrame(
        {
            "match_date": pd.date_range("2025-01-01", periods=4),
            "match_outcome": [0, 1, 2, 0],
        }
    )

    with pytest.raises(ValueError, match="numeric features"):
        validate_training_frame(frame, min_samples=4, min_feature_count=1)
