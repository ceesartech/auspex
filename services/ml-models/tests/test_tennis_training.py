"""Unit tests for the tennis training pipeline (Phase 11-3).

Tennis is the first 1v1 sport — single market in v1 (moneyline).
Differs from NBA/NFL training in:

  * No tie-exclusion filter needed in the training query (tennis
    matches always produce a winner; no draws/ties).
  * No spread or total bundle in v1. Total games needs linescore
    parsing not in the ESPN backfill; set-betting is also v2.
  * No Poisson / Dixon-Coles — tennis scoring is set-level binary,
    not Poisson-friendly.
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
    ENSEMBLE_TENNIS_MONEYLINE,
    LIGHTGBM_NBA_MONEYLINE,
    LIGHTGBM_TENNIS_MONEYLINE,
    NEURAL_NETWORK_NBA_MONEYLINE,
    NEURAL_NETWORK_TENNIS_MONEYLINE,
    XGBOOST_NBA_MONEYLINE,
    XGBOOST_TENNIS_MONEYLINE,
    ModelType,
    PredictionTask,
)
from training.train_all_models import SPORT_BUNDLES, TENNIS_MONEYLINE_BUNDLE  # noqa: E402
from utils.training_data import (  # noqa: E402
    TENNIS_MONEYLINE_NON_FEATURE_COLUMNS,
    TENNIS_MONEYLINE_TARGET,
    TENNIS_MONEYLINE_TRAINING_QUERY,
    get_tennis_moneyline_feature_columns,
    prepare_tennis_moneyline_frame,
)

# ── Training query: scope + features_cache pin ──────────────────────


class TestQueryScopedToTennis:
    def test_sport_filter(self):
        assert "l.sport = 'tennis'" in TENNIS_MONEYLINE_TRAINING_QUERY

    def test_features_cache_pinned(self):
        # Locked to tennis_baseline — without this we'd pick up
        # soccer/nhl/nba/nfl features_cache rows by accident.
        assert "feature_set = 'tennis_baseline'" in TENNIS_MONEYLINE_TRAINING_QUERY

    def test_only_finished_matches(self):
        assert "m.status = 'finished'" in TENNIS_MONEYLINE_TRAINING_QUERY
        assert "home_score IS NOT NULL" in TENNIS_MONEYLINE_TRAINING_QUERY
        assert "away_score IS NOT NULL" in TENNIS_MONEYLINE_TRAINING_QUERY

    def test_tie_filter_present_defensively(self):
        # Tennis matches always produce a winner so this is belt-and-
        # suspenders — but if a data quality bug ever wrote a 0-0
        # row (retirement-before-start), the filter keeps it out of
        # training.
        assert "m.home_score <> m.away_score" in TENNIS_MONEYLINE_TRAINING_QUERY

    def test_query_gates_on_actual_weather(self):
        # Tennis is an outdoor sport (except for a small share of
        # indoor finals). Same default-bias risk as NFL: training on
        # rows without real weather would push the model toward the
        # neutral defaults instead of actual conditions. Gate the
        # corpus to matches with match_weather.data_kind='actual' so
        # the GBDT never sees a default during training. Drop the
        # gate when every tracked tour-level match has a weather row.
        assert "EXISTS" in TENNIS_MONEYLINE_TRAINING_QUERY
        assert "match_weather" in TENNIS_MONEYLINE_TRAINING_QUERY
        assert "mw.data_kind = 'actual'" in TENNIS_MONEYLINE_TRAINING_QUERY
        assert "mw.match_id = m.id" in TENNIS_MONEYLINE_TRAINING_QUERY


# ── Target derivation: pure pandas fallback ─────────────────────────


class TestPrepareTennisMoneylineFrame:
    def test_target_from_scores(self):
        # CSV path bypasses the SQL CASE → pandas fallback derives target.
        # home_score > away_score → 0 (home wins), else 1 (away wins).
        raw = pd.DataFrame(
            {
                "home_score": [3, 1, 2],
                "away_score": [1, 3, 0],
                "match_date": ["2024-09-09", "2024-09-10", "2024-09-11"],
            }
        )
        frame = prepare_tennis_moneyline_frame(raw)
        assert list(frame[TENNIS_MONEYLINE_TARGET]) == [0, 1, 0]

    def test_features_json_flattened(self):
        raw = pd.DataFrame(
            [
                {
                    "match_date": "2024-09-09",
                    "home_score": 3,
                    "away_score": 1,
                    "features": {"odds_home_ml": 1.45, "h2h_balance": 2.0},
                }
            ]
        )
        frame = prepare_tennis_moneyline_frame(raw)
        assert frame.loc[0, "feature__odds_home_ml"] == 1.45
        assert frame.loc[0, "feature__h2h_balance"] == 2.0


# ── Feature column selectors exclude target + IDs ───────────────────


class TestFeatureColumnExclusion:
    def test_moneyline_excludes_target_and_scores(self):
        frame = pd.DataFrame(
            {
                TENNIS_MONEYLINE_TARGET: [0, 1],
                "home_score": [3, 1],
                "away_score": [1, 3],
                "match_id": ["a", "b"],
                "feature__odds_home_ml": [1.45, 2.10],
            }
        )
        cols = get_tennis_moneyline_feature_columns(frame)
        assert "feature__odds_home_ml" in cols
        assert TENNIS_MONEYLINE_TARGET not in cols
        assert "home_score" not in cols
        assert "away_score" not in cols


# ── Model config invariants (2-class softmax) ───────────────────────


class TestTennisModelConfigs:
    def test_xgb_two_class_softmax(self):
        cfg = XGBOOST_TENNIS_MONEYLINE
        assert cfg.hyperparameters["num_class"] == 2
        assert cfg.hyperparameters["objective"] == "multi:softprob"
        assert cfg.model_type == ModelType.XGBOOST
        assert cfg.prediction_task == PredictionTask.TENNIS_MONEYLINE
        assert cfg.target_column == "tennis_moneyline"

    def test_lgb_two_class_multiclass(self):
        cfg = LIGHTGBM_TENNIS_MONEYLINE
        assert cfg.hyperparameters["num_class"] == 2
        assert cfg.hyperparameters["objective"] == "multiclass"
        assert cfg.model_type == ModelType.LIGHTGBM

    def test_nn_softmax_output(self):
        cfg = NEURAL_NETWORK_TENNIS_MONEYLINE
        assert cfg.hyperparameters["output_activation"] == "softmax"
        assert cfg.loss_function == "categorical_crossentropy"
        assert cfg.model_type == ModelType.NEURAL_NETWORK

    def test_ensemble_is_ensemble_type(self):
        assert ENSEMBLE_TENNIS_MONEYLINE.model_type == ModelType.ENSEMBLE
        assert ENSEMBLE_TENNIS_MONEYLINE.prediction_task == PredictionTask.TENNIS_MONEYLINE


class TestTennisHyperparamsMatchNba:
    """Tennis corpus (~12-15k matches) is comparable to NBA, so the
    bundles reuse NBA hyperparameter shape (no NFL-style shallowing).
    Locked so a future tweak that adopts NFL hyperparameters trips
    the test."""

    def test_xgb_max_depth_matches_nba(self):
        assert (
            XGBOOST_TENNIS_MONEYLINE.hyperparameters["max_depth"] == XGBOOST_NBA_MONEYLINE.hyperparameters["max_depth"]
        )

    def test_lgb_num_leaves_matches_nba(self):
        assert (
            LIGHTGBM_TENNIS_MONEYLINE.hyperparameters["num_leaves"]
            == LIGHTGBM_NBA_MONEYLINE.hyperparameters["num_leaves"]
        )

    def test_nn_hidden_layers_match_nba(self):
        assert (
            NEURAL_NETWORK_TENNIS_MONEYLINE.hyperparameters["hidden_layers"]
            == NEURAL_NETWORK_NBA_MONEYLINE.hyperparameters["hidden_layers"]
        )


# ── SportBundle wiring ──────────────────────────────────────────────


class TestTennisBundle:
    def test_bundle_registered_in_dispatch(self):
        assert SPORT_BUNDLES["tennis_moneyline"] is TENNIS_MONEYLINE_BUNDLE
        assert TENNIS_MONEYLINE_BUNDLE.sport == "tennis_moneyline"
        assert TENNIS_MONEYLINE_BUNDLE.ensemble_name == "ensemble_tennis_ml"

    def test_no_poisson_or_dixon_coles(self):
        # Tennis scoring is set-level binary, not Poisson-friendly.
        # Locked so nobody adds hockey_poisson to a tennis bundle by
        # accident.
        for spec in TENNIS_MONEYLINE_BUNDLE.base_models:
            assert "poisson" not in spec.name.lower()
            assert "dixon" not in spec.name.lower()

    def test_three_base_models(self):
        # XGB + LGB + NN, no team-columns requirement.
        assert len(TENNIS_MONEYLINE_BUNDLE.base_models) == 3
        for spec in TENNIS_MONEYLINE_BUNDLE.base_models:
            assert spec.needs_team_columns is False


# ── Non-feature column set ──────────────────────────────────────────


class TestNonFeatureColumns:
    def test_target_excluded(self):
        assert TENNIS_MONEYLINE_TARGET in TENNIS_MONEYLINE_NON_FEATURE_COLUMNS

    def test_scores_excluded(self):
        assert "home_score" in TENNIS_MONEYLINE_NON_FEATURE_COLUMNS
        assert "away_score" in TENNIS_MONEYLINE_NON_FEATURE_COLUMNS

    def test_match_metadata_excluded(self):
        for col in ("match_id", "match_date", "season", "league_id", "features"):
            assert col in TENNIS_MONEYLINE_NON_FEATURE_COLUMNS


# Quiet the unused-pytest import lint when only fixtures are used.
_ = np
_ = pytest
