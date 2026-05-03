"""Tests for training pipeline: cross-validation, calibration, hyperparameter optimization."""

import numpy as np
import pandas as pd
import pytest

from training.calibration import ProbabilityCalibrator
from training.cross_validation import TimeSeriesCV


class TestTimeSeriesCV:
    """Tests for TimeSeriesCV."""

    def test_split_basic(self):
        df = pd.DataFrame({
            "match_date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "feature": np.random.randn(100),
            "match_outcome": np.random.randint(0, 3, 100),
        })

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
        df = pd.DataFrame({
            "match_date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "feature": np.random.randn(50),
            "match_outcome": np.random.randint(0, 3, 50),
        })

        cv = TimeSeriesCV(n_splits=5, min_train_size=30, date_column="match_date")
        folds = cv.split(df)

        # Some folds may be skipped due to min train size
        for fold in folds:
            assert fold["train_size"] >= 30

    def test_split_with_gap(self):
        df = pd.DataFrame({
            "match_date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "feature": np.random.randn(100),
            "match_outcome": np.random.randint(0, 3, 100),
        })

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
        assert len(cal.calibrators) == 3

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
