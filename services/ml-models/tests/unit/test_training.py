"""Tests for training pipeline: cross-validation, calibration, hyperparameter optimization."""

import numpy as np
import pandas as pd
import pytest
from training.calibration import ProbabilityCalibrator
from training.cross_validation import TimeSeriesCV


class TestTimeSeriesCV:
    """Tests for TimeSeriesCV."""

    def test_split_basic(self):
        df = pd.DataFrame(
            {
                "match_date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "feature": np.random.randn(100),
                "match_outcome": np.random.randint(0, 3, 100),
            }
        )

        cv = TimeSeriesCV(n_splits=5, date_column="match_date")
        folds = cv.split(df)

        assert len(folds) == 5
        for fold in folds:
            assert len(fold["train"]) > 0
            assert len(fold["val"]) > 0
            # Train dates should be before val dates
            train_max = fold["train"]["match_date"].max()
            val_min = fold["val"]["match_date"].min()
            assert train_max <= val_min

    def test_split_with_min_train_size(self):
        df = pd.DataFrame(
            {
                "match_date": pd.date_range("2023-01-01", periods=50, freq="D"),
                "feature": np.random.randn(50),
                "match_outcome": np.random.randint(0, 3, 50),
            }
        )

        cv = TimeSeriesCV(n_splits=5, min_train_size=30, date_column="match_date")
        folds = cv.split(df)

        # Some folds may be skipped due to min train size
        for fold in folds:
            assert fold["train_size"] >= 30

    def test_split_with_gap(self):
        df = pd.DataFrame(
            {
                "match_date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "feature": np.random.randn(100),
                "match_outcome": np.random.randint(0, 3, 100),
            }
        )

        cv = TimeSeriesCV(n_splits=3, gap=5, date_column="match_date")
        folds = cv.split(df)

        assert len(folds) == 3


class TestProbabilityCalibrator:
    """Tests for ProbabilityCalibrator."""

    def test_isotonic_fit_calibrate(self):
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 3, n)
        # Simulate somewhat calibrated probabilities
        y_proba = np.random.dirichlet([2, 2, 2], n)

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_proba, y_true)

        assert cal.is_fitted is True
        # June-2026 rewrite: fitted maps are stored as serializable
        # isotonic knots in .params (JSON round-trip for the ensemble
        # metadata), not sklearn objects in .calibrators.
        assert len(cal.params) == 3

        calibrated = cal.calibrate(y_proba)
        assert calibrated.shape == (n, 3)
        # Should sum to ~1
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=0.01)
        # Should be non-negative
        assert (calibrated >= 0).all()

    def test_platt_fit_calibrate(self):
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 3, n)
        y_proba = np.random.dirichlet([2, 2, 2], n)

        cal = ProbabilityCalibrator(method="platt")
        cal.fit(y_proba, y_true)

        assert cal.is_fitted is True

        calibrated = cal.calibrate(y_proba)
        assert calibrated.shape == (n, 3)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=0.01)

    def test_calibrate_not_fitted(self):
        cal = ProbabilityCalibrator()
        y_proba = np.array([[0.5, 0.3, 0.2]])
        with pytest.raises(ValueError, match="Calibrator not fitted"):
            cal.calibrate(y_proba)

    def test_calibration_error(self):
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 3, n)
        y_proba = np.random.dirichlet([2, 2, 2], n)

        cal = ProbabilityCalibrator()
        error = cal.calibration_error(y_proba, y_true)

        assert "ece" in error
        assert "mce" in error
        assert error["ece"] >= 0
        assert error["mce"] >= 0
        assert error["mce"] >= error["ece"]


class TestPerLeagueBaselineGuard:
    """A Poisson-family model configured for per-league baselines that is
    handed a frame with no league column must STOP the run.

    Skipping it looks safer and is not: nothing downstream notices an absent
    member. The ensemble's surviving-weight guard only fires for members that
    are in `self.models` and fail at predict time, so a member that was never
    trained contributes nothing to `attempted_weight` and no blend reads as
    degraded. From there: no dixon_coles_* in trained_models -> the
    derived-market guard has nothing to score -> "not checked" -> the bundle
    promotes on its 1x2 Brier alone -> precompute_predictions finds no DC
    member and writes ZERO rows for all ~15 derived soccer markets.
    """

    def _frame(self, n=120, with_league=False):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "match_date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "home_team": [f"T{i % 8}" for i in range(n)],
                "away_team": [f"T{(i + 3) % 8}" for i in range(n)],
                "home_score": rng.integers(0, 4, n).astype(float),
                "away_score": rng.integers(0, 4, n).astype(float),
            }
        )
        frame["match_outcome"] = np.where(
            frame["home_score"] > frame["away_score"], 0, np.where(frame["home_score"] == frame["away_score"], 1, 2)
        )
        if with_league:
            frame["league_id"] = ["L1"] * (n // 2) + ["L2"] * (n - n // 2)
        return frame

    def _orchestrator(self, tmp_path):
        from training.train_all_models import TrainingOrchestrator

        return TrainingOrchestrator(registry_dir=str(tmp_path / "registry"), run_cv=False, calibrate=False)

    def test_a_missing_league_column_raises_instead_of_silently_skipping(self, tmp_path):
        orch = self._orchestrator(tmp_path)
        df = self._frame()
        with pytest.raises(ValueError) as exc:
            orch.train_all(df, df, features=[], target="match_outcome", model_types=["dixon_coles"])
        assert "per_league_baselines" in str(exc.value)
        assert "league" in str(exc.value)
        # And it did not quietly leave a half-built bundle behind.
        assert "dixon_coles_soccer_match_result" not in orch.trained_models

    def test_the_same_frame_with_a_league_column_trains(self, tmp_path):
        orch = self._orchestrator(tmp_path)
        df = self._frame(with_league=True)
        orch.train_all(df, df, features=[], target="match_outcome", model_types=["dixon_coles"])
        model = orch.trained_models["dixon_coles_soccer_match_result"]
        assert model.is_fitted
        assert model.per_league_baselines is True
        assert set(model.league_baselines) >= {"L1", "L2"}
