"""Unit tests for the NBA training pipeline (Phase 6 NBA-3).

Mirrors test_nhl_training but adapted for NBA's line-as-feature
design — spread + total queries derive targets from
(margin, closing_spread_home) and (total, closing_total_line),
so the line itself must show up as both a feature AND a target
input. Tests lock down:

  * NBA_*_TRAINING_QUERY scopes to leagues.sport='nba' and pins
    features_cache to feature_set='nba_baseline'.
  * Spread + total queries INNER JOIN to odds — matches without a
    real closing line are dropped from training (we never train on
    neutral-default lines).
  * Target derivation math: home covered iff margin > -line; over
    iff (home + away) > line.
  * prepare_nba_*_frame flattens features JSON, parses match_date,
    and derives the target as a fallback for CSV inputs that bypass
    SQL.
  * NBA model configs declare num_class=2 (XGB, LGBM) and softmax (NN)
    — required because the ensemble indexes proba[:, y].
  * NBA bundles wire the right loader / configs with NO
    Poisson/Dixon-Coles (basketball scoring is too high-variance).
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
    ENSEMBLE_NBA_MONEYLINE,
    ENSEMBLE_NBA_SPREAD,
    ENSEMBLE_NBA_TOTAL,
    LIGHTGBM_NBA_MONEYLINE,
    LIGHTGBM_NBA_SPREAD,
    LIGHTGBM_NBA_TOTAL,
    NEURAL_NETWORK_NBA_MONEYLINE,
    NEURAL_NETWORK_NBA_SPREAD,
    NEURAL_NETWORK_NBA_TOTAL,
    XGBOOST_NBA_MONEYLINE,
    XGBOOST_NBA_SPREAD,
    XGBOOST_NBA_TOTAL,
    ModelType,
    PredictionTask,
)
from training.train_all_models import (  # noqa: E402
    NBA_MONEYLINE_BUNDLE,
    NBA_SPREAD_BUNDLE,
    NBA_TOTAL_BUNDLE,
    SPORT_BUNDLES,
)
from utils.training_data import (  # noqa: E402
    NBA_MONEYLINE_NON_FEATURE_COLUMNS,
    NBA_MONEYLINE_TARGET,
    NBA_MONEYLINE_TRAINING_QUERY,
    NBA_SPREAD_NON_FEATURE_COLUMNS,
    NBA_SPREAD_TARGET,
    NBA_SPREAD_TRAINING_QUERY,
    NBA_TOTAL_NON_FEATURE_COLUMNS,
    NBA_TOTAL_TARGET,
    NBA_TOTAL_TRAINING_QUERY,
    get_nba_moneyline_feature_columns,
    get_nba_spread_feature_columns,
    get_nba_total_feature_columns,
    prepare_nba_moneyline_frame,
    prepare_nba_spread_frame,
    prepare_nba_total_frame,
)


# ── Training queries: scope + features_cache pin ────────────────────


class TestQueriesScopedToNba:
    """Every NBA query must (a) filter to sport='nba' and (b) pin
    features_cache to feature_set='nba_baseline'. Without either,
    we'd cross-contaminate with other sports' rows."""

    @pytest.mark.parametrize(
        "query",
        [NBA_MONEYLINE_TRAINING_QUERY, NBA_SPREAD_TRAINING_QUERY, NBA_TOTAL_TRAINING_QUERY],
    )
    def test_sport_filter(self, query):
        assert "l.sport = 'nba'" in query

    @pytest.mark.parametrize(
        "query",
        [NBA_MONEYLINE_TRAINING_QUERY, NBA_SPREAD_TRAINING_QUERY, NBA_TOTAL_TRAINING_QUERY],
    )
    def test_features_cache_pinned(self, query):
        assert "feature_set = 'nba_baseline'" in query

    @pytest.mark.parametrize(
        "query",
        [NBA_MONEYLINE_TRAINING_QUERY, NBA_SPREAD_TRAINING_QUERY, NBA_TOTAL_TRAINING_QUERY],
    )
    def test_only_finished_matches(self, query):
        # Don't ever train on un-played games — would inject NULL
        # scores into the target.
        assert "m.status = 'finished'" in query
        assert "home_score IS NOT NULL" in query
        assert "away_score IS NOT NULL" in query


class TestLineAsFeatureFiltering:
    """Spread + total queries must INNER JOIN odds so matches without
    real closing lines are dropped. Otherwise we'd train on rows
    where the line column is NULL or, worse, default → garbage
    target derivation."""

    def test_spread_query_requires_real_line(self):
        # The INNER LATERAL `spread.home_line IS NOT NULL` is the
        # filter that drops no-odds matches.
        assert "JOIN LATERAL" in NBA_SPREAD_TRAINING_QUERY
        assert "spread.home_line IS NOT NULL" in NBA_SPREAD_TRAINING_QUERY
        # And NOT a LEFT JOIN — that would let nulls through.
        spread_block = NBA_SPREAD_TRAINING_QUERY.split("JOIN LATERAL (")[1].split("ON")[1]
        # The first ON after the spread LATERAL is the filter condition
        assert "IS NOT NULL" in spread_block.split("\n")[0]

    def test_total_query_requires_real_line(self):
        assert "JOIN LATERAL" in NBA_TOTAL_TRAINING_QUERY
        assert "total.line IS NOT NULL" in NBA_TOTAL_TRAINING_QUERY

    def test_spread_query_exposes_line_as_column(self):
        # The line must appear in the SELECT as closing_spread_home
        # so it lands in the trainable feature matrix.
        assert "spread.home_line AS closing_spread_home" in NBA_SPREAD_TRAINING_QUERY

    def test_total_query_exposes_line_as_column(self):
        assert "total.line AS closing_total_line" in NBA_TOTAL_TRAINING_QUERY


# ── Target derivation: pure pandas fallback ─────────────────────────


class TestPrepareNbaMoneylineFrame:
    def test_target_from_scores(self):
        # CSV path bypasses the SQL CASE → pandas fallback derives target.
        raw = pd.DataFrame(
            {
                "home_score": [110, 95, 100],
                "away_score": [100, 108, 100],  # last is a tie, NBA shouldn't have these
                "match_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            }
        )
        frame = prepare_nba_moneyline_frame(raw)
        # home win → 0, away win → 1. Tie row uses fallback np.where,
        # which evaluates (home > away) as False → 1 (away).
        assert list(frame[NBA_MONEYLINE_TARGET]) == [0, 1, 1]

    def test_features_json_flattened(self):
        raw = pd.DataFrame(
            [
                {
                    "match_date": "2025-01-01",
                    "home_score": 110,
                    "away_score": 100,
                    "features": {"odds_home_ml": 1.85, "home_roll_pts_scored": 115.0},
                }
            ]
        )
        frame = prepare_nba_moneyline_frame(raw)
        assert frame.loc[0, "feature__odds_home_ml"] == 1.85
        assert frame.loc[0, "feature__home_roll_pts_scored"] == 115.0


class TestPrepareNbaSpreadFrame:
    def test_target_from_margin_vs_line(self):
        # Cover formula: home covered iff margin > -line.
        # Concrete cases:
        #   line=-7 (home favored by 7), margin=10 → 10 > 7 → home covered (0)
        #   line=-7,                     margin=5  → 5  > 7 → no       (1)
        #   line=+7 (home dog),          margin=-5 → -5 > -7 → home cov(0)
        #   line=+7,                     margin=-10 → -10 > -7 → no    (1)
        raw = pd.DataFrame(
            {
                "home_score": [110, 105, 100, 100],
                "away_score": [100, 100, 105, 110],
                "closing_spread_home": [-7.0, -7.0, 7.0, 7.0],
                "match_date": ["2025-01-01"] * 4,
            }
        )
        frame = prepare_nba_spread_frame(raw)
        assert list(frame[NBA_SPREAD_TARGET]) == [0, 1, 0, 1]


class TestPrepareNbaTotalFrame:
    def test_target_from_total_vs_line(self):
        raw = pd.DataFrame(
            {
                "home_score": [120, 100, 110],
                "away_score": [110, 105, 110],
                "closing_total_line": [225.0, 225.0, 220.0],
                "match_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            }
        )
        # totals: 230, 205, 220.
        # 230 > 225 → over (0)
        # 205 > 225 → no  → under (1)
        # 220 > 220 → no (strict gt) → under (1)
        frame = prepare_nba_total_frame(raw)
        assert list(frame[NBA_TOTAL_TARGET]) == [0, 1, 1]


# ── Feature column selectors exclude target + IDs ───────────────────


class TestFeatureColumnExclusion:
    def test_moneyline_excludes_target_and_scores(self):
        frame = pd.DataFrame(
            {
                NBA_MONEYLINE_TARGET: [0, 1],
                "home_score": [100, 110],
                "away_score": [95, 100],
                "match_id": ["a", "b"],
                "feature__odds_home_ml": [1.8, 1.9],
            }
        )
        cols = get_nba_moneyline_feature_columns(frame)
        assert "feature__odds_home_ml" in cols
        assert NBA_MONEYLINE_TARGET not in cols
        assert "home_score" not in cols
        assert "away_score" not in cols

    def test_spread_keeps_closing_line_as_feature(self):
        # Line-as-feature design — closing_spread_home MUST land in
        # the feature columns so the model conditions on it.
        frame = pd.DataFrame(
            {
                NBA_SPREAD_TARGET: [0, 1],
                "closing_spread_home": [-7.5, -3.0],
                "home_score": [100, 110],
                "away_score": [95, 100],
            }
        )
        cols = get_nba_spread_feature_columns(frame)
        assert "closing_spread_home" in cols
        assert NBA_SPREAD_TARGET not in cols

    def test_total_keeps_closing_line_as_feature(self):
        frame = pd.DataFrame(
            {
                NBA_TOTAL_TARGET: [0, 1],
                "closing_total_line": [225.0, 220.0],
                "home_score": [120, 100],
                "away_score": [110, 105],
            }
        )
        cols = get_nba_total_feature_columns(frame)
        assert "closing_total_line" in cols
        assert NBA_TOTAL_TARGET not in cols


# ── Model config invariants (2-class softmax, not binary) ───────────


@pytest.mark.parametrize(
    "xgb,lgb,nn,ens,task,target",
    [
        (
            XGBOOST_NBA_MONEYLINE,
            LIGHTGBM_NBA_MONEYLINE,
            NEURAL_NETWORK_NBA_MONEYLINE,
            ENSEMBLE_NBA_MONEYLINE,
            PredictionTask.NBA_MONEYLINE,
            "nba_moneyline",
        ),
        (
            XGBOOST_NBA_SPREAD,
            LIGHTGBM_NBA_SPREAD,
            NEURAL_NETWORK_NBA_SPREAD,
            ENSEMBLE_NBA_SPREAD,
            PredictionTask.NBA_SPREAD,
            "nba_spread",
        ),
        (
            XGBOOST_NBA_TOTAL,
            LIGHTGBM_NBA_TOTAL,
            NEURAL_NETWORK_NBA_TOTAL,
            ENSEMBLE_NBA_TOTAL,
            PredictionTask.NBA_TOTAL,
            "nba_total",
        ),
    ],
)
class TestNbaModelConfigs:
    def test_xgb_two_class_softmax(self, xgb, lgb, nn, ens, task, target):
        # Critical: num_class=2 + multi:softprob (NOT binary). The
        # ensemble layer indexes proba[:, y] which requires a 2D
        # output array. A binary objective would return 1D and break
        # the ensemble blend silently.
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


# ── SportBundle wiring ──────────────────────────────────────────────


class TestNbaBundles:
    @pytest.mark.parametrize(
        "bundle,key,ensemble_name",
        [
            (NBA_MONEYLINE_BUNDLE, "nba_moneyline", "ensemble_nba_ml"),
            (NBA_SPREAD_BUNDLE, "nba_spread", "ensemble_nba_sp"),
            (NBA_TOTAL_BUNDLE, "nba_total", "ensemble_nba_tot"),
        ],
    )
    def test_bundle_registered_in_dispatch(self, bundle, key, ensemble_name):
        assert SPORT_BUNDLES[key] is bundle
        assert bundle.sport == key
        assert bundle.ensemble_name == ensemble_name

    @pytest.mark.parametrize("bundle", [NBA_MONEYLINE_BUNDLE, NBA_SPREAD_BUNDLE, NBA_TOTAL_BUNDLE])
    def test_no_poisson_or_dixon_coles(self, bundle):
        # Basketball is too high-variance / continuous for a discrete
        # Poisson prior to add signal over GBDT + NN. Locked here so
        # someone doesn't add hockey_poisson to an NBA bundle by
        # accident.
        for spec in bundle.base_models:
            assert "poisson" not in spec.name.lower(), f"Unexpected poisson in {bundle.sport}: {spec.name}"
            assert "dixon" not in spec.name.lower(), f"Unexpected dixon in {bundle.sport}: {spec.name}"

    @pytest.mark.parametrize("bundle", [NBA_MONEYLINE_BUNDLE, NBA_SPREAD_BUNDLE, NBA_TOTAL_BUNDLE])
    def test_three_base_models(self, bundle):
        # XGB + LGB + NN, no team-columns requirement.
        assert len(bundle.base_models) == 3
        for spec in bundle.base_models:
            assert spec.needs_team_columns is False


# ── Non-feature column sets ─────────────────────────────────────────


class TestNonFeatureColumnSets:
    def test_each_target_excluded_from_its_own_set(self):
        assert NBA_MONEYLINE_TARGET in NBA_MONEYLINE_NON_FEATURE_COLUMNS
        assert NBA_SPREAD_TARGET in NBA_SPREAD_NON_FEATURE_COLUMNS
        assert NBA_TOTAL_TARGET in NBA_TOTAL_NON_FEATURE_COLUMNS

    def test_targets_dont_cross_contaminate(self):
        # Each market's non-feature set should NOT exclude another
        # market's target — otherwise valid features could be dropped.
        assert NBA_SPREAD_TARGET not in NBA_MONEYLINE_NON_FEATURE_COLUMNS
        assert NBA_TOTAL_TARGET not in NBA_SPREAD_NON_FEATURE_COLUMNS
        assert NBA_MONEYLINE_TARGET not in NBA_TOTAL_NON_FEATURE_COLUMNS

    def test_scores_excluded_everywhere(self):
        # home_score / away_score are leakage — must be excluded from
        # every market's feature set.
        for col_set in (
            NBA_MONEYLINE_NON_FEATURE_COLUMNS,
            NBA_SPREAD_NON_FEATURE_COLUMNS,
            NBA_TOTAL_NON_FEATURE_COLUMNS,
        ):
            assert "home_score" in col_set
            assert "away_score" in col_set


# Quiet the unused-pytest import lint when only fixtures are used.
_ = np
