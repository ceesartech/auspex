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
    ENSEMBLE_NHL_PUCK_LINE,
    ENSEMBLE_NHL_REGULATION,
    ENSEMBLE_NHL_TOTAL,
    LIGHTGBM_NHL_MONEYLINE,
    LIGHTGBM_NHL_PUCK_LINE,
    LIGHTGBM_NHL_REGULATION,
    LIGHTGBM_NHL_TOTAL,
    NEURAL_NETWORK_NHL_MONEYLINE,
    NEURAL_NETWORK_NHL_PUCK_LINE,
    NEURAL_NETWORK_NHL_REGULATION,
    NEURAL_NETWORK_NHL_TOTAL,
    NHL_MONEYLINE_CONFIGS,
    NHL_PUCK_LINE_CONFIGS,
    NHL_REGULATION_CONFIGS,
    NHL_TOTAL_CONFIGS,
    XGBOOST_NHL_MONEYLINE,
    XGBOOST_NHL_PUCK_LINE,
    XGBOOST_NHL_REGULATION,
    XGBOOST_NHL_TOTAL,
    ModelType,
    PredictionTask,
)
from training.train_all_models import (  # noqa: E402
    NHL_MONEYLINE_BUNDLE,
    NHL_PUCK_LINE_BUNDLE,
    NHL_REGULATION_BUNDLE,
    NHL_TOTAL_BUNDLE,
    SOCCER_BUNDLE,
    SPORT_BUNDLES,
)
from utils.training_data import (  # noqa: E402
    NHL_MONEYLINE_TARGET,
    NHL_MONEYLINE_TRAINING_QUERY,
    NHL_NON_FEATURE_COLUMNS,
    NHL_PUCK_LINE_TARGET,
    NHL_PUCK_LINE_TRAINING_QUERY,
    NHL_REGULATION_TARGET,
    NHL_REGULATION_TRAINING_QUERY,
    NHL_TOTAL_TARGET,
    NHL_TOTAL_TRAINING_QUERY,
    get_nhl_feature_columns,
    get_nhl_regulation_feature_columns,
    prepare_nhl_moneyline_frame,
    prepare_nhl_puck_line_frame,
    prepare_nhl_regulation_frame,
    prepare_nhl_total_frame,
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
        # train time. Every key follows the <sport>_<market> convention.
        assert set(SPORT_BUNDLES.keys()) == {
            "soccer_match_result",
            "nhl_moneyline",
            "nhl_regulation",
            "nhl_puck_line",
            "nhl_total",
            "nba_moneyline",
            "nba_spread",
            "nba_total",
        }

    def test_nhl_bundle_uses_nhl_loader_and_target(self):
        assert NHL_MONEYLINE_BUNDLE.target_column == NHL_MONEYLINE_TARGET
        # The loader is the NHL-specific function, not the soccer one.
        assert NHL_MONEYLINE_BUNDLE.load_frame.__name__ == "load_nhl_moneyline_frame"
        assert NHL_MONEYLINE_BUNDLE.feature_columns.__name__ == "get_nhl_feature_columns"

    def test_nhl_bundle_includes_hockey_poisson_not_dixon_coles(self):
        # Phase 3d added hockey-Poisson (NHL-tuned variant) to the NHL
        # ensembles. Soccer's Dixon-Coles is still excluded — it bakes
        # in soccer-specific tau corrections for low-scoring games that
        # don't apply to NHL.
        names = {m.name for m in NHL_MONEYLINE_BUNDLE.base_models}
        assert "hockey_poisson_nhl_ml" in names
        assert "dixon_coles" not in names
        # Soccer's PoissonMatchPredictor (the base class) shouldn't be
        # in the NHL bundle either — only the hockey-tuned subclass.
        types = {m.config.model_type for m in NHL_MONEYLINE_BUNDLE.base_models}
        assert ModelType.POISSON in types  # via hockey-Poisson
        assert ModelType.DIXON_COLES not in types

    def test_nhl_bundle_includes_four_base_models(self):
        names = [m.name for m in NHL_MONEYLINE_BUNDLE.base_models]
        assert names == [
            "xgboost_nhl_ml",
            "lightgbm_nhl_ml",
            "neural_network_nhl_ml",
            "hockey_poisson_nhl_ml",
        ]

    def test_nhl_ensemble_name_and_config_match(self):
        assert NHL_MONEYLINE_BUNDLE.ensemble_name == "ensemble_nhl_ml"
        assert NHL_MONEYLINE_BUNDLE.ensemble_config is ENSEMBLE_NHL_MONEYLINE

    def test_soccer_bundle_uses_explicit_match_result_naming(self):
        # Soccer follows the same *_<sport>_<market> convention as NHL.
        # Locks in the rename so future edits don't accidentally drift
        # back to the bare "ensemble" / "xgboost" forms.
        assert SOCCER_BUNDLE.sport == "soccer_match_result"
        assert SOCCER_BUNDLE.target_column == "match_outcome"
        assert SOCCER_BUNDLE.ensemble_name == "ensemble_soccer_match_result"
        names = [m.name for m in SOCCER_BUNDLE.base_models]
        assert names == [
            "xgboost_soccer_match_result",
            "lightgbm_soccer_match_result",
            "neural_network_soccer_match_result",
            "poisson_soccer_match_result",
            "dixon_coles_soccer_match_result",
        ]


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

    def test_includes_hockey_poisson_not_dixon_coles(self):
        # Phase 3d: hockey-Poisson derives P(home reg / tie / away reg)
        # by integrating the joint goal distribution. Dixon-Coles still
        # excluded.
        types = {m.config.model_type for m in NHL_REGULATION_BUNDLE.base_models}
        assert ModelType.POISSON in types
        assert ModelType.DIXON_COLES not in types

    def test_four_base_models(self):
        names = [m.name for m in NHL_REGULATION_BUNDLE.base_models]
        assert names == [
            "xgboost_nhl_reg",
            "lightgbm_nhl_reg",
            "neural_network_nhl_reg",
            "hockey_poisson_nhl_reg",
        ]

    def test_ensemble_name_and_config_match(self):
        assert NHL_REGULATION_BUNDLE.ensemble_name == "ensemble_nhl_reg"
        assert NHL_REGULATION_BUNDLE.ensemble_config is ENSEMBLE_NHL_REGULATION


# ── Puck-line (binary, home covers -1.5) ─────────────────────────────


class TestNhlPuckLineQuery:
    def test_scopes_to_nhl_sport(self):
        assert "l.sport = 'nhl'" in NHL_PUCK_LINE_TRAINING_QUERY

    def test_pins_features_cache_to_nhl_baseline(self):
        assert "feature_set = 'nhl_baseline'" in NHL_PUCK_LINE_TRAINING_QUERY

    def test_target_uses_score_margin_threshold(self):
        # Bet settles on FINAL score (incl. OT/SO), so the target uses
        # m.home_score - m.away_score, not regulation-only scores. The
        # threshold is 2 because puck-line is -1.5 — home must win by
        # 2+ to cover.
        assert "(m.home_score - m.away_score) >= 2" in NHL_PUCK_LINE_TRAINING_QUERY

    def test_does_not_drop_close_games(self):
        # Unlike moneyline (which dropped ties for being data errors),
        # puck-line keeps every finished NHL game — 1-goal games are
        # the modal "does not cover" outcome and we need them for the
        # negative class.
        assert "m.home_score <> m.away_score" not in NHL_PUCK_LINE_TRAINING_QUERY


class TestNhlPuckLineFramePrep:
    def test_empty_in_empty_out(self):
        assert prepare_nhl_puck_line_frame(pd.DataFrame()).empty

    def test_flattens_features_json(self):
        raw = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": ["2025-01-15T00:00:00Z"],
                "nhl_puck_line": [0],
                "features": [json.dumps({"odds_home_pl15": 2.30})],
            }
        )
        out = prepare_nhl_puck_line_frame(raw)
        assert "feature__odds_home_pl15" in out.columns


class TestNhlPuckLineConfigs:
    def test_xgboost_is_2_class_softmax(self):
        assert XGBOOST_NHL_PUCK_LINE.hyperparameters["objective"] == "multi:softprob"
        assert XGBOOST_NHL_PUCK_LINE.hyperparameters["num_class"] == 2

    def test_lightgbm_is_2_class_multiclass(self):
        assert LIGHTGBM_NHL_PUCK_LINE.hyperparameters["objective"] == "multiclass"
        assert LIGHTGBM_NHL_PUCK_LINE.hyperparameters["num_class"] == 2

    @pytest.mark.parametrize(
        "config",
        [
            XGBOOST_NHL_PUCK_LINE,
            LIGHTGBM_NHL_PUCK_LINE,
            NEURAL_NETWORK_NHL_PUCK_LINE,
            ENSEMBLE_NHL_PUCK_LINE,
        ],
    )
    def test_all_configs_target_nhl_puck_line(self, config):
        assert config.prediction_task == PredictionTask.NHL_PUCK_LINE
        assert config.target_column == NHL_PUCK_LINE_TARGET

    def test_configs_dict_complete(self):
        assert set(NHL_PUCK_LINE_CONFIGS.keys()) == {
            "xgboost_nhl_pl",
            "lightgbm_nhl_pl",
            "neural_network_nhl_pl",
            "ensemble_nhl_pl",
        }


class TestNhlPuckLineBundle:
    def test_uses_puck_line_loader_and_target(self):
        assert NHL_PUCK_LINE_BUNDLE.target_column == NHL_PUCK_LINE_TARGET
        assert NHL_PUCK_LINE_BUNDLE.load_frame.__name__ == "load_nhl_puck_line_frame"
        assert NHL_PUCK_LINE_BUNDLE.feature_columns.__name__ == "get_nhl_puck_line_feature_columns"

    def test_four_base_models_with_hockey_poisson(self):
        types = {m.config.model_type for m in NHL_PUCK_LINE_BUNDLE.base_models}
        assert ModelType.POISSON in types  # hockey-Poisson, the principled answer for puck-line
        assert ModelType.DIXON_COLES not in types
        names = [m.name for m in NHL_PUCK_LINE_BUNDLE.base_models]
        assert names == [
            "xgboost_nhl_pl",
            "lightgbm_nhl_pl",
            "neural_network_nhl_pl",
            "hockey_poisson_nhl_pl",
        ]

    def test_ensemble_name_and_config(self):
        assert NHL_PUCK_LINE_BUNDLE.ensemble_name == "ensemble_nhl_pl"
        assert NHL_PUCK_LINE_BUNDLE.ensemble_config is ENSEMBLE_NHL_PUCK_LINE


# ── Total (binary, over 5.5) ─────────────────────────────────────────


class TestNhlTotalQuery:
    def test_scopes_to_nhl_sport(self):
        assert "l.sport = 'nhl'" in NHL_TOTAL_TRAINING_QUERY

    def test_pins_features_cache_to_nhl_baseline(self):
        assert "feature_set = 'nhl_baseline'" in NHL_TOTAL_TRAINING_QUERY

    def test_target_uses_score_sum_threshold(self):
        # NHL convention: SO winner is credited with +1 goal, so the
        # bet settles on sum of final home_score + away_score. Line is
        # 5.5, so threshold is 6 for "over".
        assert "(m.home_score + m.away_score) >= 6" in NHL_TOTAL_TRAINING_QUERY


class TestNhlTotalFramePrep:
    def test_empty_in_empty_out(self):
        assert prepare_nhl_total_frame(pd.DataFrame()).empty

    def test_flattens_features_json(self):
        raw = pd.DataFrame(
            {
                "match_id": ["m1"],
                "match_date": ["2025-01-15T00:00:00Z"],
                "nhl_total": [0],
                "features": [json.dumps({"odds_over55": 1.95})],
            }
        )
        out = prepare_nhl_total_frame(raw)
        assert "feature__odds_over55" in out.columns


class TestNhlTotalConfigs:
    def test_xgboost_is_2_class_softmax(self):
        assert XGBOOST_NHL_TOTAL.hyperparameters["objective"] == "multi:softprob"
        assert XGBOOST_NHL_TOTAL.hyperparameters["num_class"] == 2

    def test_lightgbm_is_2_class_multiclass(self):
        assert LIGHTGBM_NHL_TOTAL.hyperparameters["objective"] == "multiclass"
        assert LIGHTGBM_NHL_TOTAL.hyperparameters["num_class"] == 2

    @pytest.mark.parametrize(
        "config",
        [
            XGBOOST_NHL_TOTAL,
            LIGHTGBM_NHL_TOTAL,
            NEURAL_NETWORK_NHL_TOTAL,
            ENSEMBLE_NHL_TOTAL,
        ],
    )
    def test_all_configs_target_nhl_total(self, config):
        assert config.prediction_task == PredictionTask.NHL_TOTAL
        assert config.target_column == NHL_TOTAL_TARGET

    def test_configs_dict_complete(self):
        assert set(NHL_TOTAL_CONFIGS.keys()) == {
            "xgboost_nhl_tot",
            "lightgbm_nhl_tot",
            "neural_network_nhl_tot",
            "ensemble_nhl_tot",
        }


class TestNhlTotalBundle:
    def test_uses_total_loader_and_target(self):
        assert NHL_TOTAL_BUNDLE.target_column == NHL_TOTAL_TARGET
        assert NHL_TOTAL_BUNDLE.load_frame.__name__ == "load_nhl_total_frame"
        assert NHL_TOTAL_BUNDLE.feature_columns.__name__ == "get_nhl_total_feature_columns"

    def test_four_base_models_with_hockey_poisson(self):
        types = {m.config.model_type for m in NHL_TOTAL_BUNDLE.base_models}
        assert ModelType.POISSON in types  # hockey-Poisson directly integrates P(total >= 6)
        assert ModelType.DIXON_COLES not in types
        names = [m.name for m in NHL_TOTAL_BUNDLE.base_models]
        assert names == [
            "xgboost_nhl_tot",
            "lightgbm_nhl_tot",
            "neural_network_nhl_tot",
            "hockey_poisson_nhl_tot",
        ]

    def test_ensemble_name_and_config(self):
        assert NHL_TOTAL_BUNDLE.ensemble_name == "ensemble_nhl_tot"
        assert NHL_TOTAL_BUNDLE.ensemble_config is ENSEMBLE_NHL_TOTAL
