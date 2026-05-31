"""Unit tests for HockeyPoissonPredictor (Phase 3d).

Covers the math: train fits team strengths, predict_proba derives
task-specific (N, num_classes) probability arrays from the joint goal
distribution, and the outputs sum to 1 per row for every task. No DB
or ensemble integration — those land in the broader training run.
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
    HOCKEY_POISSON_NHL_MONEYLINE,
    HOCKEY_POISSON_NHL_PUCK_LINE,
    HOCKEY_POISSON_NHL_REGULATION,
    HOCKEY_POISSON_NHL_TOTAL,
    ModelType,
    PredictionTask,
)
from predictors.poisson_models import HockeyPoissonPredictor  # noqa: E402

# A tiny synthetic NHL-ish corpus: 4 teams, mix of home/away wins,
# enough scoreline variety to fit meaningful attack/defense strengths.
SYNTHETIC_GAMES = pd.DataFrame(
    [
        {"home_team": "Toronto", "away_team": "Boston", "home_score": 4, "away_score": 2},
        {"home_team": "Toronto", "away_team": "Montreal", "home_score": 3, "away_score": 1},
        {"home_team": "Toronto", "away_team": "Detroit", "home_score": 5, "away_score": 3},
        {"home_team": "Boston", "away_team": "Toronto", "home_score": 2, "away_score": 3},
        {"home_team": "Boston", "away_team": "Montreal", "home_score": 4, "away_score": 1},
        {"home_team": "Boston", "away_team": "Detroit", "home_score": 3, "away_score": 2},
        {"home_team": "Montreal", "away_team": "Toronto", "home_score": 2, "away_score": 4},
        {"home_team": "Montreal", "away_team": "Boston", "home_score": 1, "away_score": 3},
        {"home_team": "Montreal", "away_team": "Detroit", "home_score": 2, "away_score": 2},
        {"home_team": "Detroit", "away_team": "Toronto", "home_score": 2, "away_score": 5},
        {"home_team": "Detroit", "away_team": "Boston", "home_score": 1, "away_score": 4},
        {"home_team": "Detroit", "away_team": "Montreal", "home_score": 3, "away_score": 1},
    ]
)


@pytest.fixture(scope="module")
def fitted_models() -> dict:
    """Train one HockeyPoissonPredictor per task. Module-scoped so the
    MLE fit runs once even though each task config produces a separate
    predict_proba shape."""
    models = {}
    for cfg in (
        HOCKEY_POISSON_NHL_MONEYLINE,
        HOCKEY_POISSON_NHL_REGULATION,
        HOCKEY_POISSON_NHL_PUCK_LINE,
        HOCKEY_POISSON_NHL_TOTAL,
    ):
        m = HockeyPoissonPredictor(cfg)
        m.train(SYNTHETIC_GAMES.copy())
        models[cfg.prediction_task] = m
    return models


# ── Config shape ─────────────────────────────────────────────────────


class TestHockeyPoissonConfigs:
    @pytest.mark.parametrize(
        "config, task",
        [
            (HOCKEY_POISSON_NHL_MONEYLINE, PredictionTask.NHL_MONEYLINE),
            (HOCKEY_POISSON_NHL_REGULATION, PredictionTask.NHL_REGULATION),
            (HOCKEY_POISSON_NHL_PUCK_LINE, PredictionTask.NHL_PUCK_LINE),
            (HOCKEY_POISSON_NHL_TOTAL, PredictionTask.NHL_TOTAL),
        ],
    )
    def test_per_task_config(self, config, task):
        assert config.model_type == ModelType.POISSON
        assert config.prediction_task == task
        # Hockey-tuned hyperparameters
        assert config.hyperparameters["max_goals"] == 10
        assert config.hyperparameters["home_advantage"] == 0.15

    def test_smaller_home_advantage_than_soccer(self):
        # Locked in here so a refactor that copies soccer's 0.25 breaks
        # the test rather than silently degrading NHL predictions.
        soccer_ha = 0.25  # POISSON_CONFIG default
        hockey_ha = HOCKEY_POISSON_NHL_MONEYLINE.hyperparameters["home_advantage"]
        assert hockey_ha < soccer_ha


# ── Training fit ──────────────────────────────────────────────────────


class TestHockeyPoissonTraining:
    def test_fits_attack_defense_for_all_teams(self, fitted_models):
        m = fitted_models[PredictionTask.NHL_MONEYLINE]
        assert m.is_fitted
        for team in ("Toronto", "Boston", "Montreal", "Detroit"):
            assert team in m.team_attack
            assert team in m.team_defense

    def test_attack_defense_strengths_have_sensible_ordering(self, fitted_models):
        # Toronto wins every game; Detroit loses most. Toronto's attack
        # should be > Detroit's; Detroit's defense (goals allowed) should
        # be > Toronto's defense. defense rating is "goals allowed",
        # so higher = worse.
        m = fitted_models[PredictionTask.NHL_MONEYLINE]
        assert m.team_attack["Toronto"] > m.team_attack["Detroit"]
        assert m.team_defense["Detroit"] > m.team_defense["Toronto"]


# ── Output shape + normalization ──────────────────────────────────────


PREDICT_X = pd.DataFrame(
    [
        {"home_team": "Toronto", "away_team": "Boston"},
        {"home_team": "Detroit", "away_team": "Toronto"},
        {"home_team": "Montreal", "away_team": "Detroit"},
    ]
)


class TestPredictProbaShape:
    def test_moneyline_is_2_class(self, fitted_models):
        proba = fitted_models[PredictionTask.NHL_MONEYLINE].predict_proba(PREDICT_X)
        assert proba.shape == (3, 2)

    def test_regulation_is_3_class(self, fitted_models):
        proba = fitted_models[PredictionTask.NHL_REGULATION].predict_proba(PREDICT_X)
        assert proba.shape == (3, 3)

    def test_puck_line_is_2_class(self, fitted_models):
        proba = fitted_models[PredictionTask.NHL_PUCK_LINE].predict_proba(PREDICT_X)
        assert proba.shape == (3, 2)

    def test_total_is_2_class(self, fitted_models):
        proba = fitted_models[PredictionTask.NHL_TOTAL].predict_proba(PREDICT_X)
        assert proba.shape == (3, 2)

    @pytest.mark.parametrize(
        "task",
        [
            PredictionTask.NHL_MONEYLINE,
            PredictionTask.NHL_REGULATION,
            PredictionTask.NHL_PUCK_LINE,
            PredictionTask.NHL_TOTAL,
        ],
    )
    def test_proba_rows_sum_to_one(self, fitted_models, task):
        proba = fitted_models[task].predict_proba(PREDICT_X)
        row_sums = proba.sum(axis=1)
        # Tight tolerance — the model normalizes explicitly so any drift
        # would point to a derivation bug (e.g., missed mask, double-count).
        assert np.allclose(row_sums, 1.0, atol=1e-9)

    @pytest.mark.parametrize(
        "task",
        [
            PredictionTask.NHL_MONEYLINE,
            PredictionTask.NHL_REGULATION,
            PredictionTask.NHL_PUCK_LINE,
            PredictionTask.NHL_TOTAL,
        ],
    )
    def test_probabilities_in_valid_range(self, fitted_models, task):
        proba = fitted_models[task].predict_proba(PREDICT_X)
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()


# ── Derivation correctness ────────────────────────────────────────────


class TestDerivationCorrectness:
    """Spot-check that the task-specific derivations are sensible against
    the underlying joint distribution. Toronto > Detroit (strong attack
    + weak opponent defense) should be a heavy favorite across tasks."""

    def test_strong_home_favorite_wins_moneyline(self, fitted_models):
        # Toronto at home vs Detroit — strongest team vs weakest. Should
        # be > 70% to win the moneyline.
        proba = fitted_models[PredictionTask.NHL_MONEYLINE].predict_proba(
            pd.DataFrame([{"home_team": "Toronto", "away_team": "Detroit"}])
        )
        p_home_win = proba[0, 0]
        assert p_home_win > 0.7, f"Toronto vs Detroit P(home win) was {p_home_win:.3f}"

    def test_regulation_3way_tie_class_is_smallest(self, fitted_models):
        # NHL regulation ties happen ~22% league-wide but in a lopsided
        # matchup the regulation outcome leans heavily to the favorite.
        # P(tie) should be the smallest of the three classes here.
        proba = fitted_models[PredictionTask.NHL_REGULATION].predict_proba(
            pd.DataFrame([{"home_team": "Toronto", "away_team": "Detroit"}])
        )
        p_home, p_tie, p_away = proba[0]
        assert p_tie < p_home, "regulation tie should be less likely than the favorite's win"
        assert p_tie < 0.4, f"P(reg tie) {p_tie:.3f} suspiciously high for a lopsided matchup"

    def test_moneyline_includes_ot_so_half_of_tie_mass(self, fitted_models):
        # Construct the analytical comparison: regulation P(home), P(tie),
        # P(away) from the regulation model. Moneyline P(home win) should
        # equal P(home reg) + 0.5 * P(reg tie), within tolerance for the
        # underlying-model identity (same MLE state).
        x = pd.DataFrame([{"home_team": "Boston", "away_team": "Montreal"}])
        reg_proba = fitted_models[PredictionTask.NHL_REGULATION].predict_proba(x)
        ml_proba = fitted_models[PredictionTask.NHL_MONEYLINE].predict_proba(x)
        expected_home_ml = reg_proba[0, 0] + 0.5 * reg_proba[0, 1]
        assert ml_proba[0, 0] == pytest.approx(expected_home_ml, abs=1e-6)

    def test_puck_line_strictly_below_moneyline_for_home(self, fitted_models):
        # P(home covers -1.5) must be <= P(home wins) for every matchup —
        # covering is a subset of winning. Tight inequality holds because
        # 1-goal wins (in regulation OR OT/SO) don't cover.
        x = pd.DataFrame(
            [
                {"home_team": "Toronto", "away_team": "Detroit"},
                {"home_team": "Detroit", "away_team": "Toronto"},
            ]
        )
        pl = fitted_models[PredictionTask.NHL_PUCK_LINE].predict_proba(x)
        ml = fitted_models[PredictionTask.NHL_MONEYLINE].predict_proba(x)
        for i in range(len(x)):
            assert pl[i, 0] <= ml[i, 0] + 1e-9, "puck-line cover prob > moneyline win prob"

    def test_total_over_under_complement(self, fitted_models):
        # Over + under must sum to 1 exactly (no third option for the
        # 5.5 line — a 5-goal game is under, 6-goal game is over).
        proba = fitted_models[PredictionTask.NHL_TOTAL].predict_proba(PREDICT_X)
        assert np.allclose(proba[:, 0] + proba[:, 1], 1.0, atol=1e-9)


# ── Graceful fallback when team unseen ───────────────────────────────


class TestUnseenTeamFallback:
    def test_unseen_team_uses_neutral_prior(self, fitted_models):
        # A team not in the training corpus falls back to attack=defense=1.
        # The resulting probabilities should still be valid (sum to 1,
        # in [0,1]).
        m = fitted_models[PredictionTask.NHL_MONEYLINE]
        x = pd.DataFrame([{"home_team": "NeverHeardOf", "away_team": "Toronto"}])
        proba = m.predict_proba(x)
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()
