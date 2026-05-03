"""Tests for Ensemble and Poisson/Dixon-Coles models."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.ensemble import EnsemblePredictor
from models.poisson_models import (
    DixonColesPredictor,
    PoissonMatchPredictor,
    dixon_coles_tau,
    poisson_pmf,
)


class TestPoissonPMF:
    """Tests for Poisson probability mass function."""

    def test_pmf_k0(self):
        assert abs(poisson_pmf(0, 1.5) - np.exp(-1.5)) < 1e-10

    def test_pmf_positive(self):
        p = poisson_pmf(2, 1.5)
        assert 0 < p < 1

    def test_pmf_zero_lambda(self):
        assert poisson_pmf(0, 0) == 1.0
        assert poisson_pmf(1, 0) == 0.0

    def test_pmf_sums_to_one(self):
        lam = 2.0
        total = sum(poisson_pmf(k, lam) for k in range(20))
        assert abs(total - 1.0) < 0.001


class TestDixonColesTau:
    """Tests for Dixon-Coles correction factor."""

    def test_tau_0_0(self):
        tau = dixon_coles_tau(0, 0, 1.5, 1.0, -0.1)
        assert tau == 1.0 - 1.5 * 1.0 * (-0.1)

    def test_tau_0_1(self):
        tau = dixon_coles_tau(0, 1, 1.5, 1.0, -0.1)
        assert tau == 1.0 + 1.5 * (-0.1)

    def test_tau_1_0(self):
        tau = dixon_coles_tau(1, 0, 1.5, 1.0, -0.1)
        assert tau == 1.0 + 1.0 * (-0.1)

    def test_tau_1_1(self):
        tau = dixon_coles_tau(1, 1, 1.5, 1.0, -0.1)
        assert tau == 1.0 - (-0.1)

    def test_tau_high_scores(self):
        # For scores > 1, tau should be 1.0
        assert dixon_coles_tau(2, 2, 1.5, 1.0, -0.1) == 1.0
        assert dixon_coles_tau(3, 0, 1.5, 1.0, -0.1) == 1.0


class TestPoissonMatchPredictor:
    """Tests for Poisson model."""

    def test_init(self, poisson_config):
        model = PoissonMatchPredictor(poisson_config)
        assert model.home_advantage == 0.25
        assert model.max_goals == 4
        assert model.is_fitted is False

    def test_prepare_data(self, poisson_config, sample_match_data):
        model = PoissonMatchPredictor(poisson_config)
        X, y = model.prepare_data(sample_match_data, "match_outcome")
        assert "home_team" in X.columns
        assert "away_team" in X.columns

    def test_prepare_data_missing_columns(self, poisson_config):
        model = PoissonMatchPredictor(poisson_config)
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Missing required column"):
            model.prepare_data(df, "a")

    def test_train(self, poisson_config, sample_match_data):
        model = PoissonMatchPredictor(poisson_config)
        result = model.train(sample_match_data)

        assert model.is_fitted is True
        assert "team_count" in result
        assert "league_avg_goals" in result
        assert len(model.team_attack) > 0
        assert len(model.team_defense) > 0

    def test_predict_proba(self, poisson_config, sample_match_data):
        model = PoissonMatchPredictor(poisson_config)
        model.train(sample_match_data)

        proba = model.predict_proba(sample_match_data.head(10))
        assert proba.shape == (10, 3)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_predict(self, poisson_config, sample_match_data):
        model = PoissonMatchPredictor(poisson_config)
        model.train(sample_match_data)

        preds = model.predict(sample_match_data.head(10))
        assert len(preds) == 10
        assert all(p in [0, 1, 2] for p in preds)

    def test_predict_score(self, poisson_config, sample_match_data):
        model = PoissonMatchPredictor(poisson_config)
        model.train(sample_match_data)

        result = model.predict_score("TeamA", "TeamB")
        assert "home_lambda" in result
        assert "away_lambda" in result
        assert "probabilities" in result
        assert "top_scores" in result
        assert "over_2_5" in result
        assert "btts" in result
        assert 0 <= result["over_2_5"] <= 1
        assert 0 <= result["btts"] <= 1

    def test_save_load(self, poisson_config, sample_match_data):
        model = PoissonMatchPredictor(poisson_config)
        model.train(sample_match_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "poisson.json")
            model.save(path)

            model2 = PoissonMatchPredictor(poisson_config)
            model2.load(path)

            assert model2.is_fitted is True
            assert model2.team_attack == model.team_attack
            assert model2.team_defense == model.team_defense


class TestDixonColesPredictor:
    """Tests for Dixon-Coles model."""

    def test_init(self, dixon_coles_config):
        model = DixonColesPredictor(dixon_coles_config)
        assert model.rho == -0.1
        assert model.time_decay == 0.001

    def test_train(self, dixon_coles_config, sample_match_data):
        model = DixonColesPredictor(dixon_coles_config)
        result = model.train(sample_match_data)

        assert model.is_fitted is True
        assert "rho" in result
        assert isinstance(result["rho"], float)

    def test_predict_proba(self, dixon_coles_config, sample_match_data):
        model = DixonColesPredictor(dixon_coles_config)
        model.train(sample_match_data)

        proba = model.predict_proba(sample_match_data.head(5))
        assert proba.shape == (5, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_save_load(self, dixon_coles_config, sample_match_data):
        model = DixonColesPredictor(dixon_coles_config)
        model.train(sample_match_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dc.json")
            model.save(path)

            model2 = DixonColesPredictor(dixon_coles_config)
            model2.load(path)

            assert model2.rho == model.rho
            assert model2.is_fitted is True


class TestEnsemblePredictor:
    """Tests for ensemble model."""

    def _make_mock_model(self, n_classes=3):
        """Create a mock model that returns random probabilities."""
        mock = MagicMock()
        mock.config = MagicMock()
        mock.config.model_type = MagicMock()
        mock.config.model_type.value = "mock"
        mock.config.to_dict.return_value = {}

        def mock_predict_proba(X):
            n = len(X)
            probs = np.random.dirichlet([1] * n_classes, n)
            return probs

        def mock_predict(X):
            probs = mock_predict_proba(X)
            return np.argmax(probs, axis=1)

        mock.predict_proba = mock_predict_proba
        mock.predict = mock_predict
        return mock

    def test_init(self, ensemble_config):
        model = EnsemblePredictor(ensemble_config)
        assert len(model.models) == 0
        assert len(model.weights) == 0

    def test_add_remove_model(self, ensemble_config):
        model = EnsemblePredictor(ensemble_config)
        mock = self._make_mock_model()

        model.add_model("model_a", mock, weight=0.6)
        assert "model_a" in model.models
        assert model.weights["model_a"] == 0.6

        model.remove_model("model_a")
        assert "model_a" not in model.models

    def test_predict_proba_equal_weights(self, ensemble_config, sample_match_data):
        model = EnsemblePredictor(ensemble_config)
        np.random.seed(42)

        model.add_model("a", self._make_mock_model(), weight=1.0)
        model.add_model("b", self._make_mock_model(), weight=1.0)

        proba = model.predict_proba(sample_match_data.head(5))
        assert proba.shape == (5, 3)
        # Should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.05)

    def test_predict(self, ensemble_config, sample_match_data):
        model = EnsemblePredictor(ensemble_config)
        model.add_model("a", self._make_mock_model(), weight=1.0)

        preds = model.predict(sample_match_data.head(5))
        assert len(preds) == 5
        assert all(p in [0, 1, 2] for p in preds)

    def test_predict_no_models(self, ensemble_config, sample_match_data):
        model = EnsemblePredictor(ensemble_config)
        with pytest.raises(ValueError, match="No models in ensemble"):
            model.predict_proba(sample_match_data.head(5))

    def test_train_no_val(self, ensemble_config):
        model = EnsemblePredictor(ensemble_config)
        model.add_model("a", self._make_mock_model(), weight=1.0)
        model.add_model("b", self._make_mock_model(), weight=1.0)

        result = model.train(pd.DataFrame(), val_df=None)
        assert model.is_fitted is True
        # Equal weights when no val data
        assert abs(model.weights["a"] - 0.5) < 0.01
        assert abs(model.weights["b"] - 0.5) < 0.01

    def test_predict_with_disagreement(self, ensemble_config, sample_match_data):
        model = EnsemblePredictor(ensemble_config)
        model.add_model("a", self._make_mock_model(), weight=1.0)
        model.add_model("b", self._make_mock_model(), weight=1.0)

        result = model.predict_with_disagreement(sample_match_data.head(5))
        assert "ensemble_proba" in result
        assert "individual_predictions" in result
        assert "disagreement" in result
        assert len(result["disagreement"]) == 5

    def test_save_load(self, ensemble_config):
        model = EnsemblePredictor(ensemble_config)
        model.add_model("a", self._make_mock_model(), weight=0.6)
        model.add_model("b", self._make_mock_model(), weight=0.4)
        model.is_fitted = True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ensemble.json")
            model.save(path)

            model2 = EnsemblePredictor(ensemble_config)
            model2.load(path)

            assert model2.is_fitted is True
            assert model2.weights == model.weights
