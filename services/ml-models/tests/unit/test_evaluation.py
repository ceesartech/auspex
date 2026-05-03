"""Tests for evaluation metrics, comparison, and performance tracking."""

import tempfile

import numpy as np

from evaluation.metrics import compute_all_metrics, compute_roi, ranked_probability_score
from evaluation.model_comparison import ABTestTracker
from evaluation.performance_tracker import PerformanceTracker
from models.model_config import PredictionTask


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_multiclass_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 0])
        y_proba = np.array([
            [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.6, 0.2],
            [0.9, 0.05, 0.05], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
        ])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.8  # 8/10
        assert "log_loss" in metrics
        assert metrics["log_loss"] > 0
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "brier_score" in metrics
        assert "rps" in metrics

    def test_binary_metrics(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        y_proba = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8],
            [0.6, 0.4], [0.7, 0.3], [0.3, 0.7], [0.4, 0.6],
        ])

        metrics = compute_all_metrics(
            y_true, y_pred, y_proba, task=PredictionTask.OVER_UNDER
        )

        assert "accuracy" in metrics
        assert "roc_auc" in metrics

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        ])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 1.0


class TestRankedProbabilityScore:
    """Tests for RPS."""

    def test_perfect_rps(self):
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rps = ranked_probability_score(y_true, y_proba)
        assert rps == 0.0

    def test_worst_rps(self):
        y_true = np.array([0])
        y_proba = np.array([[0.0, 0.0, 1.0]])
        rps = ranked_probability_score(y_true, y_proba)
        assert rps > 0

    def test_rps_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 50)
        y_proba = np.random.dirichlet([1, 1, 1], 50)
        rps = ranked_probability_score(y_true, y_proba)
        assert 0 <= rps <= 1


class TestComputeROI:
    """Tests for ROI calculation."""

    def test_basic_roi(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 0, 0])
        odds = np.array([2.0, 2.0, 2.0])

        result = compute_roi(y_true, y_pred, odds)

        assert result["bets_placed"] == 3
        assert result["bets_won"] == 2
        assert result["total_staked"] == 3.0
        assert result["total_return"] == 4.0  # 2 wins at 2.0 odds
        assert result["profit"] == 1.0
        assert abs(result["roi"] - 1.0 / 3.0) < 1e-10

    def test_roi_all_wrong(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        odds = np.array([2.0, 2.0, 2.0])

        result = compute_roi(y_true, y_pred, odds)
        assert result["bets_won"] == 0
        assert result["roi"] == -1.0

    def test_roi_with_confidence_filter(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 0, 0])
        odds = np.array([2.0, 2.0, 2.0])
        y_proba = np.array([[0.8, 0.1, 0.1], [0.4, 0.3, 0.3], [0.7, 0.2, 0.1]])

        result = compute_roi(
            y_true, y_pred, odds, min_confidence=0.6, y_pred_proba=y_proba
        )
        # Only bets where max prob >= 0.6: indices 0 and 2
        assert result["bets_placed"] == 2

    def test_roi_no_bets(self):
        y_true = np.array([0])
        y_pred = np.array([0])
        odds = np.array([2.0])
        y_proba = np.array([[0.4, 0.3, 0.3]])

        result = compute_roi(
            y_true, y_pred, odds, min_confidence=0.9, y_pred_proba=y_proba
        )
        assert result["bets_placed"] == 0
        assert result["roi"] == 0.0


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_log_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PerformanceTracker(storage_path=tmpdir)

            tracker.log_metrics("model_a", {"accuracy": 0.55, "log_loss": 0.8})
            tracker.log_metrics("model_a", {"accuracy": 0.60, "log_loss": 0.75})

            trend = tracker.get_metric_trend("model_a", "accuracy")
            assert trend == [0.55, 0.60]

    def test_detect_drift_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PerformanceTracker(storage_path=tmpdir)
            tracker.log_metrics("model_a", {"accuracy": 0.5})

            result = tracker.detect_drift("model_a", "accuracy", window=5)
            assert result["has_drift"] is False
            assert "Not enough data" in result["reason"]

    def test_detect_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PerformanceTracker(storage_path=tmpdir)

            # Historical: high accuracy
            for _ in range(10):
                tracker.log_metrics("model_a", {"accuracy": 0.65})
            # Recent: low accuracy
            for _ in range(10):
                tracker.log_metrics("model_a", {"accuracy": 0.50})

            result = tracker.detect_drift("model_a", "accuracy", window=10, threshold=0.05)
            assert result["has_drift"] is True
            assert result["difference"] < 0

    def test_get_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PerformanceTracker(storage_path=tmpdir)
            tracker.log_metrics("model_a", {"accuracy": 0.5, "log_loss": 0.8})
            tracker.log_metrics("model_a", {"accuracy": 0.6, "log_loss": 0.7})

            summary = tracker.get_summary("model_a")
            assert summary["n_evaluations"] == 2
            assert "accuracy_mean" in summary
            assert "accuracy_best" in summary

    def test_get_summary_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PerformanceTracker(storage_path=tmpdir)
            summary = tracker.get_summary("unknown_model")
            assert summary == {}


class TestABTestTracker:
    """Tests for ABTestTracker."""

    def test_record_and_retrieve(self):
        tracker = ABTestTracker()
        tracker.record_prediction("test1", "model_a", "m1", 0, actual=0, odds=2.0)
        tracker.record_prediction("test1", "model_b", "m1", 1, actual=0, odds=2.0)
        tracker.record_prediction("test1", "model_a", "m2", 1, actual=1, odds=3.0)
        tracker.record_prediction("test1", "model_b", "m2", 1, actual=1, odds=3.0)

        df = tracker.get_results("test1")
        assert len(df) == 4

    def test_compute_test_metrics(self):
        tracker = ABTestTracker()
        tracker.record_prediction("test1", "model_a", "m1", 0, actual=0, odds=2.0)
        tracker.record_prediction("test1", "model_a", "m2", 1, actual=1, odds=3.0)
        tracker.record_prediction("test1", "model_b", "m1", 1, actual=0, odds=2.0)
        tracker.record_prediction("test1", "model_b", "m2", 0, actual=1, odds=3.0)

        metrics = tracker.compute_test_metrics("test1")
        assert "model_a" in metrics
        assert "model_b" in metrics
        assert metrics["model_a"]["accuracy"] == 1.0
        assert metrics["model_b"]["accuracy"] == 0.0
