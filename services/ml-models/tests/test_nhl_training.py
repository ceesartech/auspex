"""Unit tests for the NHL moneyline training pipeline (Phase 3a).

Covers the pure derivation paths (no DB, no model fit) and a few
structural invariants the orchestrator depends on:

  * NHL_MONEYLINE_TRAINING_QUERY scopes to leagues.sport='nhl' and
    pins the features_cache row to feature_set='nhl_baseline'.
  * prepare_nhl_moneyline_frame flattens features JSON, derives the
    2-class target, and parses match_date.
  * get_nhl_feature_columns excludes the target + identifiers.
  * NHL model configs declare num_class=2 (XGB, LGBM) and softmax (NN),
    NOT a binary objective — required because the ensemble indexes
    proba[:, y] which needs 2D output.
  * NHL_MONEYLINE_BUNDLE wires the right loader / configs / model
    list with no Poisson/Dixon-Coles (soccer-specific).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predictors.model_config import (  # noqa: E402
    ENSEMBLE_NHL_MONEYLINE,
    ENSEMBLE_NHL_REGULATION,
    LIGHTGBM_NHL_MONEYLINE,
    LIGHTGBM_NHL_REGULATION,
    NEURAL_NETWORK_NHL_MONEYLINE,
    NEURAL_NETWORK_NHL_REGULATION,
    NHL_MONEYLINE_CONFIGS,
    NHL_REGULATION_CONFIGS,
    XGBOOST_NHL_MONEYLINE,
    XGBOOST_NHL_REGULATION,
    ModelType,
    PredictionTask,
)
from training.train_all_models import (  # noqa: E402
    NHL_MONEYLINE_BUNDLE,
    NHL_REGULATION_BUNDLE,
    SOCCER_BUNDLE,
    SPORT_BUNDLES,
)
from utils.training_data import (  # noqa: E402
    NHL_MONEYLINE_TARGET,
    NHL_MONEYLINE_TRAINING_QUERY,
    NHL_NON_FEATURE_COLUMNS,
    NHL_REGULATION_TARGET,
    NHL_REGULATION_TRAINING_QUERY,
    get_nhl_feature_columns,
    get_nhl_regulation_feature_columns,
    prepare_nhl_moneyline_frame,
    prepare_nhl_regulation_frame,
)

# ── Query shape ──────────────────────────────────────────────────────


class TestNhlMoneylineQuery:
    def test_scopes_to_nhl_sport(self):
        assert "l.sport = 'nhl'" in NHL_MONEYLINE_TRAINING_QUERY

    def test_pins_features_cache_to_nhl_baseline(self):
        assert "feature_set = 'nhl_baseline'" in NHL_MONEYLINE_TRAINING_QUERY

    def test_drops_ties_at_query_time(self):
        # NHL games always have a winner — a tie row would be a data
        # integrity bug, not a valid training example.
        assert "m.home_score <> m.away_score" in NHL_MONEYLINE_TRAINING_QUERY

    @pytest.mark.parametrize(
        "market_clause",
        [
            "market_type = 'moneyline'",
            "market_type = 'spread'",
            "market_type = 'total'",
        ],
    )
    def test_pulls_all_three_nhl_markets(self, market_clause):
        assert market_clause in NHL_MONEYLINE_TRAINING_QUERY

    def test_emits_2class_target(self):
        # Target is 0 (home) or 1 (away) — defined by the CASE in the
        # SELECT clause.
        assert "AS nhl_moneyline" in NHL_MONEYLINE_TRAINING_QUERY
        assert "WHEN m.home_score > m.away_score THEN 0" in NHL_MONEYLINE_TRAINING_QUERY


# ── Frame preparation ────────────────────────────────────────────────


class TestPrepareNhlMoneylineFrame:
    def test_empty_in_empty_out(self):
        empty = pd.DataFrame()
        assert prepare_nhl_moneyline_frame(empty).empty

    def test_flattens_features_json(self):
        raw = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": ["2025-01-15T00:00:00Z"],
                "home_score": [3],
                "away_score": [2],
                "nhl_moneyline": [0],
                "features": [json.dumps({"odds_home_ml": 1.85, "home_roll_goals_for": 3.1})],
            }
        )
        out = prepare_nhl_moneyline_frame(raw)
        # _flatten_features uses "feature__" prefix.
        assert "feature__odds_home_ml" in out.columns
        assert "feature__home_roll_goals_for" in out.columns
        assert out["feature__odds_home_ml"].iloc[0] == pytest.approx(1.85)

    def test_derives_target_from_scores_when_missing(self):
        # CSV path: target may not be pre-computed by the SQL query.
        raw = pd.DataFrame(
            {
                "match_id": ["m1", "m2"],
                "match_date": ["2025-01-15", "2025-01-16"],
                "home_score": [3, 1],
                "away_score": [2, 4],
            }
        )
        out = prepare_nhl_moneyline_frame(raw)
        assert NHL_MONEYLINE_TARGET in out.columns
        assert out[NHL_MONEYLINE_TARGET].tolist() == [0, 1]  # home win, away win

    def test_parses_match_date(self):
        raw = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": ["2025-01-15T00:00:00Z"],
                "home_score": [3],
                "away_score": [2],
                "nhl_moneyline": [0],
            }
        )
        out = prepare_nhl_moneyline_frame(raw)
        assert pd.api.types.is_datetime64_any_dtype(out["match_date"])


# ── Feature column selection ─────────────────────────────────────────


class TestGetNhlFeatureColumns:
    def test_excludes_target_and_identifiers(self):
        frame = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": pd.to_datetime(["2025-01-15"]),
                "season": ["2024-2025"],
                "league_id": ["l1"],
                "home_team_id": ["t1"],
                "away_team_id": ["t2"],
                "home_team": ["Toronto Maple Leafs"],
                "away_team": ["Boston Bruins"],
                "home_score": [3],
                "away_score": [2],
                "nhl_moneyline": [0],
                # Real features:
                "feature__odds_home_ml": [1.85],
                "feature__home_roll_goals_for": [3.1],
                "feature__away_roll_save_pct": [0.910],
            }
        )
        cols = get_nhl_feature_columns(frame)
        assert "nhl_moneyline" not in cols
        assert "home_score" not in cols
        assert "away_score" not in cols
        for excluded in NHL_NON_FEATURE_COLUMNS:
            assert excluded not in cols
        # Real features make it through:
        assert set(cols) == {
            "feature__odds_home_ml",
            "feature__home_roll_goals_for",
            "feature__away_roll_save_pct",
        }


# ── Model configs ────────────────────────────────────────────────────


class TestNhlModelConfigs:
    def test_xgboost_is_2_class_softmax_not_binary(self):
        # The ensemble's loss function indexes proba[:, y_encoded[j]],
        # which REQUIRES 2D output. multi:softprob with num_class=2
        # gives (N, 2); a binary:logistic objective would give (N,)
        # and break the ensemble's optimizer. Regression guard.
        assert XGBOOST_NHL_MONEYLINE.hyperparameters["objective"] == "multi:softprob"
        assert XGBOOST_NHL_MONEYLINE.hyperparameters["num_class"] == 2

    def test_lightgbm_is_2_class_multiclass_not_binary(self):
        # Same reason as XGBoost. The LightGBM wrapper has an _is_binary
        # path that returns 1D for OVER_UNDER / BTTS — NHL_MONEYLINE
        # deliberately stays out of that path.
        assert LIGHTGBM_NHL_MONEYLINE.hyperparameters["objective"] == "multiclass"
        assert LIGHTGBM_NHL_MONEYLINE.hyperparameters["num_class"] == 2

    def test_neural_network_uses_softmax(self):
        # NN num_classes is derived from label_encoder.classes_ at
        # train time, so the only thing we need to lock here is the
        # output activation.
        assert NEURAL_NETWORK_NHL_MONEYLINE.hyperparameters["output_activation"] == "softmax"

    @pytest.mark.parametrize(
        "config",
        [XGBOOST_NHL_MONEYLINE, LIGHTGBM_NHL_MONEYLINE, NEURAL_NETWORK_NHL_MONEYLINE, ENSEMBLE_NHL_MONEYLINE],
    )
    def test_all_configs_target_nhl_moneyline(self, config):
        assert config.prediction_task == PredictionTask.NHL_MONEYLINE
        assert config.target_column == NHL_MONEYLINE_TARGET

    def test_configs_dict_complete(self):
        # All four configs must be reachable by name from the dict so
        # the prediction service (Phase 4) can look them up.
        assert set(NHL_MONEYLINE_CONFIGS.keys()) == {
            "xgboost_nhl_ml",
            "lightgbm_nhl_ml",
            "neural_network_nhl_ml",
            "ensemble_nhl_ml",
        }


# ── Sport bundles ────────────────────────────────────────────────────


class TestSportBundles:
    def test_all_sports_registered(self):
        # Lock the exact set so a sport added but not threaded through
        # the orchestrator fails this test rather than ghost-loading at
        # train time.
        assert set(SPORT_BUNDLES.keys()) == {"soccer", "nhl_moneyline", "nhl_regulation"}

    def test_nhl_bundle_uses_nhl_loader_and_target(self):
        assert NHL_MONEYLINE_BUNDLE.target_column == NHL_MONEYLINE_TARGET
        # The loader is the NHL-specific function, not the soccer one.
        assert NHL_MONEYLINE_BUNDLE.load_frame.__name__ == "load_nhl_moneyline_frame"
        assert NHL_MONEYLINE_BUNDLE.feature_columns.__name__ == "get_nhl_feature_columns"

    def test_nhl_bundle_excludes_poisson_and_dixon_coles(self):
        # Both Poisson-family models assume soccer scoring rules; NHL
        # ensemble drops them entirely in v1. Hockey-Poisson port is
        # Phase 3d work.
        names = {m.name for m in NHL_MONEYLINE_BUNDLE.base_models}
        assert "poisson" not in names
        assert "dixon_coles" not in names
        # And by model type — same check, different angle:
        types = {m.config.model_type for m in NHL_MONEYLINE_BUNDLE.base_models}
        assert ModelType.POISSON not in types
        assert ModelType.DIXON_COLES not in types

    def test_nhl_bundle_includes_three_base_models(self):
        names = [m.name for m in NHL_MONEYLINE_BUNDLE.base_models]
        assert names == ["xgboost_nhl_ml", "lightgbm_nhl_ml", "neural_network_nhl_ml"]

    def test_nhl_ensemble_name_and_config_match(self):
        assert NHL_MONEYLINE_BUNDLE.ensemble_name == "ensemble_nhl_ml"
        assert NHL_MONEYLINE_BUNDLE.ensemble_config is ENSEMBLE_NHL_MONEYLINE

    def test_soccer_bundle_unchanged(self):
        # Soccer is the regression-risk surface for this phase — the
        # registry name, target, and 5-model list must stay identical
        # so the existing prod soccer pipeline doesn't drift.
        assert SOCCER_BUNDLE.target_column == "match_outcome"
        assert SOCCER_BUNDLE.ensemble_name == "ensemble"
        names = [m.name for m in SOCCER_BUNDLE.base_models]
        assert names == ["xgboost", "lightgbm", "neural_network", "poisson", "dixon_coles"]


# ── Regulation 3-way: query shape, prep, configs, bundle ─────────────


class TestNhlRegulationQuery:
    def test_scopes_to_nhl_sport(self):
        assert "l.sport = 'nhl'" in NHL_REGULATION_TRAINING_QUERY

    def test_pins_features_cache_to_nhl_baseline(self):
        # Reuses the moneyline feature set — only the target differs.
        assert "feature_set = 'nhl_baseline'" in NHL_REGULATION_TRAINING_QUERY

    def test_keeps_regulation_ties(self):
        # Unlike moneyline (which excludes ties — every NHL game has a
        # winner), regulation 3-way KEEPS ties as the second class. The
        # query must NOT contain the moneyline tie-exclusion clause.
        assert "m.home_score <> m.away_score" not in NHL_REGULATION_TRAINING_QUERY

    def test_target_derived_from_metadata_regulation_winner(self):
        # Target comes from matches.metadata->>'regulation_winner', not
        # from final home_score/away_score (those include OT/SO).
        assert "metadata->>'regulation_winner'" in NHL_REGULATION_TRAINING_QUERY

    def test_filters_to_games_with_regulation_winner(self):
        # Defensive: skip rows where regulation_winner wasn't recorded
        # (shouldn't happen with current loader but a schema-evolution
        # safety net).
        assert "metadata ? 'regulation_winner'" in NHL_REGULATION_TRAINING_QUERY

    def test_encodes_3_classes(self):
        # 0=home, 1=tie, 2=away — same numeric convention as soccer's
        # MATCH_OUTCOME so the LabelEncoder + ensemble indexer behave
        # identically across the two 3-class tasks.
        assert "WHEN 'home' THEN 0" in NHL_REGULATION_TRAINING_QUERY
        assert "WHEN 'tie'  THEN 1" in NHL_REGULATION_TRAINING_QUERY
        assert "WHEN 'away' THEN 2" in NHL_REGULATION_TRAINING_QUERY


class TestPrepareNhlRegulationFrame:
    def test_empty_in_empty_out(self):
        assert prepare_nhl_regulation_frame(pd.DataFrame()).empty

    def test_flattens_features_json(self):
        raw = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": ["2025-01-15T00:00:00Z"],
                "nhl_regulation": [1],  # tie
                "features": [json.dumps({"odds_home_ml": 1.85})],
            }
        )
        out = prepare_nhl_regulation_frame(raw)
        assert "feature__odds_home_ml" in out.columns

    def test_parses_match_date(self):
        raw = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": ["2025-01-15T00:00:00Z"],
                "nhl_regulation": [0],
            }
        )
        out = prepare_nhl_regulation_frame(raw)
        assert pd.api.types.is_datetime64_any_dtype(out["match_date"])

    def test_does_not_derive_target_from_scores(self):
        # Unlike moneyline, the regulation target cannot be inferred
        # from home_score/away_score (those include OT/SO). The CSV
        # path requires the target column to be present already.
        raw = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": ["2025-01-15"],
                "home_score": [4],
                "away_score": [3],
            }
        )
        out = prepare_nhl_regulation_frame(raw)
        assert NHL_REGULATION_TARGET not in out.columns


class TestGetNhlRegulationFeatureColumns:
    def test_excludes_target_and_identifiers(self):
        frame = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": pd.to_datetime(["2025-01-15"]),
                "home_team_id": ["t1"],
                "away_team_id": ["t2"],
                "home_score": [3],
                "away_score": [2],
                "nhl_regulation": [1],
                "feature__odds_home_ml": [1.85],
                "feature__home_roll_save_pct": [0.91],
            }
        )
        cols = get_nhl_regulation_feature_columns(frame)
        assert "nhl_regulation" not in cols
        assert "home_score" not in cols
        assert set(cols) == {"feature__odds_home_ml", "feature__home_roll_save_pct"}


class TestNhlRegulationConfigs:
    def test_xgboost_is_3_class_softmax(self):
        # Same softmax-not-binary reason as moneyline (ensemble indexes
        # proba[:, y], needs 2D), but with num_class=3 for the
        # home/tie/away classes.
        assert XGBOOST_NHL_REGULATION.hyperparameters["objective"] == "multi:softprob"
        assert XGBOOST_NHL_REGULATION.hyperparameters["num_class"] == 3

    def test_lightgbm_is_3_class_multiclass(self):
        assert LIGHTGBM_NHL_REGULATION.hyperparameters["objective"] == "multiclass"
        assert LIGHTGBM_NHL_REGULATION.hyperparameters["num_class"] == 3

    def test_neural_network_uses_softmax(self):
        assert NEURAL_NETWORK_NHL_REGULATION.hyperparameters["output_activation"] == "softmax"

    @pytest.mark.parametrize(
        "config",
        [
            XGBOOST_NHL_REGULATION,
            LIGHTGBM_NHL_REGULATION,
            NEURAL_NETWORK_NHL_REGULATION,
            ENSEMBLE_NHL_REGULATION,
        ],
    )
    def test_all_configs_target_nhl_regulation(self, config):
        assert config.prediction_task == PredictionTask.NHL_REGULATION
        assert config.target_column == NHL_REGULATION_TARGET

    def test_configs_dict_complete(self):
        assert set(NHL_REGULATION_CONFIGS.keys()) == {
            "xgboost_nhl_reg",
            "lightgbm_nhl_reg",
            "neural_network_nhl_reg",
            "ensemble_nhl_reg",
        }


class TestNhlRegulationBundle:
    def test_uses_regulation_loader_and_target(self):
        assert NHL_REGULATION_BUNDLE.target_column == NHL_REGULATION_TARGET
        assert NHL_REGULATION_BUNDLE.load_frame.__name__ == "load_nhl_regulation_frame"
        assert NHL_REGULATION_BUNDLE.feature_columns.__name__ == "get_nhl_regulation_feature_columns"

    def test_excludes_poisson_dixon_coles(self):
        # Same reasoning as moneyline — Poisson-family models assume
        # soccer scoring. Hockey-Poisson port is Phase 3d.
        types = {m.config.model_type for m in NHL_REGULATION_BUNDLE.base_models}
        assert ModelType.POISSON not in types
        assert ModelType.DIXON_COLES not in types

    def test_three_base_models(self):
        names = [m.name for m in NHL_REGULATION_BUNDLE.base_models]
        assert names == ["xgboost_nhl_reg", "lightgbm_nhl_reg", "neural_network_nhl_reg"]

    def test_ensemble_name_and_config_match(self):
        assert NHL_REGULATION_BUNDLE.ensemble_name == "ensemble_nhl_reg"
        assert NHL_REGULATION_BUNDLE.ensemble_config is ENSEMBLE_NHL_REGULATION
