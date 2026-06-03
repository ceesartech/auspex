"""Unit tests for the MMA training pipeline (Phase 12-3).

MMA shares the 1v1 / single-moneyline-market shape with tennis. Same
training-side structure (XGB + LGB + NN + Ensemble bundle, no
Poisson/Dixon-Coles, no team-columns requirement). Tests mirror the
tennis training tests with mma-specific fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predictors.model_config import (  # noqa: E402
    ENSEMBLE_MMA_MONEYLINE,
    LIGHTGBM_MMA_MONEYLINE,
    LIGHTGBM_NBA_MONEYLINE,
    NEURAL_NETWORK_MMA_MONEYLINE,
    NEURAL_NETWORK_NBA_MONEYLINE,
    XGBOOST_MMA_MONEYLINE,
    XGBOOST_NBA_MONEYLINE,
    ModelType,
    PredictionTask,
)
from training.train_all_models import MMA_MONEYLINE_BUNDLE, SPORT_BUNDLES  # noqa: E402
from utils.training_data import (  # noqa: E402
    MMA_MONEYLINE_NON_FEATURE_COLUMNS,
    MMA_MONEYLINE_TARGET,
    MMA_MONEYLINE_TRAINING_QUERY,
    get_mma_moneyline_feature_columns,
    prepare_mma_moneyline_frame,
)


class TestQueryScopedToMma:
    def test_sport_filter(self):
        assert "l.sport = 'mma'" in MMA_MONEYLINE_TRAINING_QUERY

    def test_features_cache_pinned(self):
        assert "feature_set = 'mma_baseline'" in MMA_MONEYLINE_TRAINING_QUERY

    def test_only_finished_matches(self):
        assert "m.status = 'finished'" in MMA_MONEYLINE_TRAINING_QUERY
        assert "home_score IS NOT NULL" in MMA_MONEYLINE_TRAINING_QUERY
        assert "away_score IS NOT NULL" in MMA_MONEYLINE_TRAINING_QUERY

    def test_tie_filter_present_defensively(self):
        # MMA draws are dropped at the ingest layer (extract_finished_fight
        # skips them) but the WHERE keeps any data corruption out.
        assert "m.home_score <> m.away_score" in MMA_MONEYLINE_TRAINING_QUERY


class TestPrepareMmaMoneylineFrame:
    def test_target_from_winner_flag(self):
        # home_score / away_score are 1/0 (binary winner flag) for MMA.
        # home_score > away_score → 0 (home won), else 1 (away won).
        raw = pd.DataFrame(
            {
                "home_score": [1, 0],
                "away_score": [0, 1],
                "match_date": ["2024-08-17", "2024-08-17"],
            }
        )
        frame = prepare_mma_moneyline_frame(raw)
        assert list(frame[MMA_MONEYLINE_TARGET]) == [0, 1]

    def test_features_json_flattened(self):
        raw = pd.DataFrame(
            [
                {
                    "match_date": "2024-08-17",
                    "home_score": 1,
                    "away_score": 0,
                    "features": {"odds_home_ml": 1.65, "h2h_balance": 0.0},
                }
            ]
        )
        frame = prepare_mma_moneyline_frame(raw)
        assert frame.loc[0, "feature__odds_home_ml"] == 1.65
        assert frame.loc[0, "feature__h2h_balance"] == 0.0


class TestFeatureColumnExclusion:
    def test_moneyline_excludes_target_and_scores(self):
        frame = pd.DataFrame(
            {
                MMA_MONEYLINE_TARGET: [0, 1],
                "home_score": [1, 0],
                "away_score": [0, 1],
                "match_id": ["a", "b"],
                "feature__odds_home_ml": [1.65, 2.40],
            }
        )
        cols = get_mma_moneyline_feature_columns(frame)
        assert "feature__odds_home_ml" in cols
        assert MMA_MONEYLINE_TARGET not in cols
        assert "home_score" not in cols
        assert "away_score" not in cols


class TestMmaModelConfigs:
    def test_xgb_two_class_softmax(self):
        cfg = XGBOOST_MMA_MONEYLINE
        assert cfg.hyperparameters["num_class"] == 2
        assert cfg.hyperparameters["objective"] == "multi:softprob"
        assert cfg.model_type == ModelType.XGBOOST
        assert cfg.prediction_task == PredictionTask.MMA_MONEYLINE
        assert cfg.target_column == "mma_moneyline"

    def test_lgb_two_class_multiclass(self):
        cfg = LIGHTGBM_MMA_MONEYLINE
        assert cfg.hyperparameters["num_class"] == 2
        assert cfg.model_type == ModelType.LIGHTGBM

    def test_nn_softmax_output(self):
        cfg = NEURAL_NETWORK_MMA_MONEYLINE
        assert cfg.hyperparameters["output_activation"] == "softmax"
        assert cfg.model_type == ModelType.NEURAL_NETWORK

    def test_ensemble_is_ensemble_type(self):
        assert ENSEMBLE_MMA_MONEYLINE.model_type == ModelType.ENSEMBLE
        assert ENSEMBLE_MMA_MONEYLINE.prediction_task == PredictionTask.MMA_MONEYLINE


class TestMmaHyperparamsMatchNba:
    """MMA corpus (~1500 fights) is between NFL (855) and NBA (4125).
    Bundles reuse NBA hyperparameter shape — no NFL-style shallowing
    needed."""

    def test_xgb_max_depth_matches_nba(self):
        assert XGBOOST_MMA_MONEYLINE.hyperparameters["max_depth"] == XGBOOST_NBA_MONEYLINE.hyperparameters["max_depth"]

    def test_lgb_num_leaves_matches_nba(self):
        assert (
            LIGHTGBM_MMA_MONEYLINE.hyperparameters["num_leaves"] == LIGHTGBM_NBA_MONEYLINE.hyperparameters["num_leaves"]
        )

    def test_nn_hidden_layers_match_nba(self):
        assert (
            NEURAL_NETWORK_MMA_MONEYLINE.hyperparameters["hidden_layers"]
            == NEURAL_NETWORK_NBA_MONEYLINE.hyperparameters["hidden_layers"]
        )


class TestMmaBundle:
    def test_bundle_registered_in_dispatch(self):
        assert SPORT_BUNDLES["mma_moneyline"] is MMA_MONEYLINE_BUNDLE
        assert MMA_MONEYLINE_BUNDLE.sport == "mma_moneyline"
        assert MMA_MONEYLINE_BUNDLE.ensemble_name == "ensemble_mma_ml"

    def test_no_poisson_or_dixon_coles(self):
        for spec in MMA_MONEYLINE_BUNDLE.base_models:
            assert "poisson" not in spec.name.lower()
            assert "dixon" not in spec.name.lower()

    def test_three_base_models(self):
        assert len(MMA_MONEYLINE_BUNDLE.base_models) == 3
        for spec in MMA_MONEYLINE_BUNDLE.base_models:
            assert spec.needs_team_columns is False


class TestNonFeatureColumns:
    def test_target_excluded(self):
        assert MMA_MONEYLINE_TARGET in MMA_MONEYLINE_NON_FEATURE_COLUMNS

    def test_scores_excluded(self):
        assert "home_score" in MMA_MONEYLINE_NON_FEATURE_COLUMNS
        assert "away_score" in MMA_MONEYLINE_NON_FEATURE_COLUMNS


_ = np
_ = pytest
