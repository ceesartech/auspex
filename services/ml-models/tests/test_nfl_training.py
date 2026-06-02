"""Unit tests for the NFL training pipeline (Phase 10 E3).

Mirrors test_nba_training but adapted for NFL — same line-as-feature
design for spread + total (the closing line is BOTH a feature AND a
target input), same 3-base-model + ensemble shape (no Poisson /
Dixon-Coles — NFL scoring is too matchup-specific for a discrete
prior to help). NFL-specific deltas:

  * Queries scope to sport='nfl' and pin feature_set='nfl_baseline'.
  * Moneyline excludes ties (home_score <> away_score in the SQL
    WHERE clause) since NFL ties are rare but mess up 2-class labels.
  * Configs use shallower trees + smaller NN than NBA (fewer training
    rows: ~285 games/season vs NBA's ~1230).
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
    ENSEMBLE_NFL_MONEYLINE,
    ENSEMBLE_NFL_SPREAD,
    ENSEMBLE_NFL_TOTAL,
    LIGHTGBM_NBA_MONEYLINE,
    LIGHTGBM_NFL_MONEYLINE,
    LIGHTGBM_NFL_SPREAD,
    LIGHTGBM_NFL_TOTAL,
    NEURAL_NETWORK_NBA_MONEYLINE,
    NEURAL_NETWORK_NFL_MONEYLINE,
    NEURAL_NETWORK_NFL_SPREAD,
    NEURAL_NETWORK_NFL_TOTAL,
    XGBOOST_NBA_MONEYLINE,
    XGBOOST_NFL_MONEYLINE,
    XGBOOST_NFL_SPREAD,
    XGBOOST_NFL_TOTAL,
    ModelType,
    PredictionTask,
)
from training.train_all_models import (  # noqa: E402
    NFL_MONEYLINE_BUNDLE,
    NFL_SPREAD_BUNDLE,
    NFL_TOTAL_BUNDLE,
    SPORT_BUNDLES,
)
from utils.training_data import (  # noqa: E402
    NFL_MONEYLINE_NON_FEATURE_COLUMNS,
    NFL_MONEYLINE_TARGET,
    NFL_MONEYLINE_TRAINING_QUERY,
    NFL_SPREAD_NON_FEATURE_COLUMNS,
    NFL_SPREAD_TARGET,
    NFL_SPREAD_TRAINING_QUERY,
    NFL_TOTAL_NON_FEATURE_COLUMNS,
    NFL_TOTAL_TARGET,
    NFL_TOTAL_TRAINING_QUERY,
    get_nfl_moneyline_feature_columns,
    get_nfl_spread_feature_columns,
    get_nfl_total_feature_columns,
    prepare_nfl_moneyline_frame,
    prepare_nfl_spread_frame,
    prepare_nfl_total_frame,
)

# ── Training queries: scope + features_cache pin ────────────────────


class TestQueriesScopedToNfl:
    """Every NFL query must filter to sport='nfl' AND pin
    features_cache to feature_set='nfl_baseline'. Without either,
    we'd cross-contaminate with NBA or NHL rows that share the
    matches table."""

    @pytest.mark.parametrize(
        "query",
        [NFL_MONEYLINE_TRAINING_QUERY, NFL_SPREAD_TRAINING_QUERY, NFL_TOTAL_TRAINING_QUERY],
    )
    def test_sport_filter(self, query):
        assert "l.sport = 'nfl'" in query

    @pytest.mark.parametrize(
        "query",
        [NFL_MONEYLINE_TRAINING_QUERY, NFL_SPREAD_TRAINING_QUERY, NFL_TOTAL_TRAINING_QUERY],
    )
    def test_features_cache_pinned(self, query):
        assert "feature_set = 'nfl_baseline'" in query

    @pytest.mark.parametrize(
        "query",
        [NFL_MONEYLINE_TRAINING_QUERY, NFL_SPREAD_TRAINING_QUERY, NFL_TOTAL_TRAINING_QUERY],
    )
    def test_only_finished_matches(self, query):
        assert "m.status = 'finished'" in query
        assert "home_score IS NOT NULL" in query
        assert "away_score IS NOT NULL" in query


class TestMoneylineExcludesTies:
    """NFL allows regular-season ties (~1-2/year) but a 2-class
    classifier can't represent them. The moneyline query MUST drop
    them from training, or the labels become arbitrary."""

    def test_query_excludes_ties(self):
        assert "m.home_score <> m.away_score" in NFL_MONEYLINE_TRAINING_QUERY


class TestLineAsFeatureFiltering:
    """Spread + total queries must INNER JOIN odds so matches without
    real closing lines are dropped. Same invariant as NBA."""

    def test_spread_query_requires_real_line(self):
        assert "JOIN LATERAL" in NFL_SPREAD_TRAINING_QUERY
        assert "spread.home_line IS NOT NULL" in NFL_SPREAD_TRAINING_QUERY

    def test_total_query_requires_real_line(self):
        assert "JOIN LATERAL" in NFL_TOTAL_TRAINING_QUERY
        assert "total.line IS NOT NULL" in NFL_TOTAL_TRAINING_QUERY

    def test_spread_query_exposes_line_as_column(self):
        assert "spread.home_line AS closing_spread_home" in NFL_SPREAD_TRAINING_QUERY

    def test_total_query_exposes_line_as_column(self):
        assert "total.line AS closing_total_line" in NFL_TOTAL_TRAINING_QUERY


# ── Target derivation: pure pandas fallback ─────────────────────────


class TestPrepareNflMoneylineFrame:
    def test_target_from_scores(self):
        raw = pd.DataFrame(
            {
                "home_score": [31, 17],
                "away_score": [24, 24],
                "match_date": ["2025-01-12", "2025-01-19"],
            }
        )
        frame = prepare_nfl_moneyline_frame(raw)
        # home win → 0, away win → 1.
        assert list(frame[NFL_MONEYLINE_TARGET]) == [0, 1]

    def test_features_json_flattened(self):
        raw = pd.DataFrame(
            [
                {
                    "match_date": "2025-01-12",
                    "home_score": 31,
                    "away_score": 24,
                    "features": {"odds_home_ml": 1.65, "home_roll_pts_scored": 27.5},
                }
            ]
        )
        frame = prepare_nfl_moneyline_frame(raw)
        assert frame.loc[0, "feature__odds_home_ml"] == 1.65
        assert frame.loc[0, "feature__home_roll_pts_scored"] == 27.5


class TestPrepareNflSpreadFrame:
    def test_target_from_margin_vs_line(self):
        # Same cover formula as NBA: home covers iff margin > -line.
        #   line=-7 (home favored by 7), margin=10 → covered (0)
        #   line=-7,                     margin=5  → no       (1)
        #   line=+3 (home dog by 3),     margin=-1 → covered (0)  (-1 > -3)
        #   line=+3,                     margin=-7 → no      (1)  (-7 > -3 false)
        raw = pd.DataFrame(
            {
                "home_score": [31, 24, 20, 17],
                "away_score": [21, 19, 21, 24],
                "closing_spread_home": [-7.0, -7.0, 3.0, 3.0],
                "match_date": ["2025-01-12"] * 4,
            }
        )
        frame = prepare_nfl_spread_frame(raw)
        assert list(frame[NFL_SPREAD_TARGET]) == [0, 1, 0, 1]


class TestPrepareNflTotalFrame:
    def test_target_from_total_vs_line(self):
        raw = pd.DataFrame(
            {
                "home_score": [31, 17, 24],
                "away_score": [24, 14, 21],
                "closing_total_line": [44.5, 44.5, 45.0],
                "match_date": ["2025-01-12", "2025-01-19", "2025-01-26"],
            }
        )
        # totals: 55, 31, 45.
        # 55 > 44.5 → over (0)
        # 31 > 44.5 → no  → under (1)
        # 45 > 45   → no (strict) → under (1)
        frame = prepare_nfl_total_frame(raw)
        assert list(frame[NFL_TOTAL_TARGET]) == [0, 1, 1]


# ── Feature column selectors exclude target + IDs ───────────────────


class TestFeatureColumnExclusion:
    def test_moneyline_excludes_target_and_scores(self):
        frame = pd.DataFrame(
            {
                NFL_MONEYLINE_TARGET: [0, 1],
                "home_score": [31, 17],
                "away_score": [24, 24],
                "match_id": ["a", "b"],
                "feature__odds_home_ml": [1.65, 2.40],
            }
        )
        cols = get_nfl_moneyline_feature_columns(frame)
        assert "feature__odds_home_ml" in cols
        assert NFL_MONEYLINE_TARGET not in cols
        assert "home_score" not in cols
        assert "away_score" not in cols

    def test_spread_keeps_closing_line_as_feature(self):
        frame = pd.DataFrame(
            {
                NFL_SPREAD_TARGET: [0, 1],
                "closing_spread_home": [-7.5, -3.0],
                "home_score": [24, 21],
                "away_score": [17, 24],
            }
        )
        cols = get_nfl_spread_feature_columns(frame)
        assert "closing_spread_home" in cols
        assert NFL_SPREAD_TARGET not in cols

    def test_total_keeps_closing_line_as_feature(self):
        frame = pd.DataFrame(
            {
                NFL_TOTAL_TARGET: [0, 1],
                "closing_total_line": [44.5, 42.0],
                "home_score": [31, 17],
                "away_score": [24, 14],
            }
        )
        cols = get_nfl_total_feature_columns(frame)
        assert "closing_total_line" in cols
        assert NFL_TOTAL_TARGET not in cols


# ── Model config invariants (2-class softmax) ───────────────────────


@pytest.mark.parametrize(
    "xgb,lgb,nn,ens,task,target",
    [
        (
            XGBOOST_NFL_MONEYLINE,
            LIGHTGBM_NFL_MONEYLINE,
            NEURAL_NETWORK_NFL_MONEYLINE,
            ENSEMBLE_NFL_MONEYLINE,
            PredictionTask.NFL_MONEYLINE,
            "nfl_moneyline",
        ),
        (
            XGBOOST_NFL_SPREAD,
            LIGHTGBM_NFL_SPREAD,
            NEURAL_NETWORK_NFL_SPREAD,
            ENSEMBLE_NFL_SPREAD,
            PredictionTask.NFL_SPREAD,
            "nfl_spread",
        ),
        (
            XGBOOST_NFL_TOTAL,
            LIGHTGBM_NFL_TOTAL,
            NEURAL_NETWORK_NFL_TOTAL,
            ENSEMBLE_NFL_TOTAL,
            PredictionTask.NFL_TOTAL,
            "nfl_total",
        ),
    ],
)
class TestNflModelConfigs:
    def test_xgb_two_class_softmax(self, xgb, lgb, nn, ens, task, target):
        assert xgb.hyperparameters["num_class"] == 2
        assert xgb.hyperparameters["objective"] == "multi:softprob"
        assert xgb.model_type == ModelType.XGBOOST
        assert xgb.prediction_task == task
        assert xgb.target_column == target

    def test_lgb_two_class_multiclass(self, xgb, lgb, nn, ens, task, target):
        assert lgb.hyperparameters["num_class"] == 2
        assert lgb.hyperparameters["objective"] == "multiclass"
        assert lgb.model_type == ModelType.LIGHTGBM

    def test_nn_softmax_output(self, xgb, lgb, nn, ens, task, target):
        assert nn.hyperparameters["output_activation"] == "softmax"
        assert nn.loss_function == "categorical_crossentropy"
        assert nn.model_type == ModelType.NEURAL_NETWORK

    def test_ensemble_is_ensemble_type(self, xgb, lgb, nn, ens, task, target):
        assert ens.model_type == ModelType.ENSEMBLE
        assert ens.prediction_task == task


class TestNflShallowerThanNba:
    """NFL has FEWER training rows than NBA (~855 across 3 seasons
    vs NBA's 4125), so its hyperparameters are tuned down to match.
    Locked here so a future tweak doesn't accidentally make NFL deeper
    than NBA on a smaller corpus."""

    def test_xgb_shallower(self):
        assert XGBOOST_NFL_MONEYLINE.hyperparameters["max_depth"] < XGBOOST_NBA_MONEYLINE.hyperparameters["max_depth"]

    def test_lgb_fewer_leaves(self):
        assert (
            LIGHTGBM_NFL_MONEYLINE.hyperparameters["num_leaves"] < LIGHTGBM_NBA_MONEYLINE.hyperparameters["num_leaves"]
        )

    def test_nn_smaller_hidden_layers(self):
        nfl_layers = NEURAL_NETWORK_NFL_MONEYLINE.hyperparameters["hidden_layers"]
        nba_layers = NEURAL_NETWORK_NBA_MONEYLINE.hyperparameters["hidden_layers"]
        # Sum of layer widths is the cheap proxy for total parameter count.
        assert sum(nfl_layers) < sum(nba_layers)


# ── SportBundle wiring ──────────────────────────────────────────────


class TestNflBundles:
    @pytest.mark.parametrize(
        "bundle,key,ensemble_name",
        [
            (NFL_MONEYLINE_BUNDLE, "nfl_moneyline", "ensemble_nfl_ml"),
            (NFL_SPREAD_BUNDLE, "nfl_spread", "ensemble_nfl_sp"),
            (NFL_TOTAL_BUNDLE, "nfl_total", "ensemble_nfl_tot"),
        ],
    )
    def test_bundle_registered_in_dispatch(self, bundle, key, ensemble_name):
        assert SPORT_BUNDLES[key] is bundle
        assert bundle.sport == key
        assert bundle.ensemble_name == ensemble_name

    @pytest.mark.parametrize("bundle", [NFL_MONEYLINE_BUNDLE, NFL_SPREAD_BUNDLE, NFL_TOTAL_BUNDLE])
    def test_no_poisson_or_dixon_coles(self, bundle):
        # NFL scoring is too matchup-specific for a discrete Poisson
        # prior to add signal over GBDT + NN. Locked so nobody adds
        # hockey_poisson to an NFL bundle by accident.
        for spec in bundle.base_models:
            assert "poisson" not in spec.name.lower(), f"Unexpected poisson in {bundle.sport}: {spec.name}"
            assert "dixon" not in spec.name.lower(), f"Unexpected dixon in {bundle.sport}: {spec.name}"

    @pytest.mark.parametrize("bundle", [NFL_MONEYLINE_BUNDLE, NFL_SPREAD_BUNDLE, NFL_TOTAL_BUNDLE])
    def test_three_base_models(self, bundle):
        assert len(bundle.base_models) == 3
        for spec in bundle.base_models:
            assert spec.needs_team_columns is False


# ── Non-feature column sets ─────────────────────────────────────────


class TestNonFeatureColumnSets:
    def test_each_target_excluded_from_its_own_set(self):
        assert NFL_MONEYLINE_TARGET in NFL_MONEYLINE_NON_FEATURE_COLUMNS
        assert NFL_SPREAD_TARGET in NFL_SPREAD_NON_FEATURE_COLUMNS
        assert NFL_TOTAL_TARGET in NFL_TOTAL_NON_FEATURE_COLUMNS

    def test_targets_dont_cross_contaminate(self):
        assert NFL_SPREAD_TARGET not in NFL_MONEYLINE_NON_FEATURE_COLUMNS
        assert NFL_TOTAL_TARGET not in NFL_SPREAD_NON_FEATURE_COLUMNS
        assert NFL_MONEYLINE_TARGET not in NFL_TOTAL_NON_FEATURE_COLUMNS

    def test_scores_excluded_everywhere(self):
        for col_set in (
            NFL_MONEYLINE_NON_FEATURE_COLUMNS,
            NFL_SPREAD_NON_FEATURE_COLUMNS,
            NFL_TOTAL_NON_FEATURE_COLUMNS,
        ):
            assert "home_score" in col_set
            assert "away_score" in col_set


# Quiet the unused-pytest import lint when only fixtures are used.
_ = np
