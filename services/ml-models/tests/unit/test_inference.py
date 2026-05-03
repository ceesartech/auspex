"""Tests for inference components: RealTimePredictor, BatchPredictor, ONNXConverter."""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from inference.batch_predictor import BatchPredictor
from inference.real_time_predictor import RealTimePredictor


class TestRealTimePredictor:
    """Tests for RealTimePredictor."""

    def _make_mock_model(self):
        mock = MagicMock()
        mock.feature_names = ["f1", "f2", "f3"]

        def mock_predict_proba(X):
            n = len(X)
            return np.array([[0.5, 0.3, 0.2]] * n)

        mock.predict_proba = mock_predict_proba
        return mock

    def test_init(self):
        model = self._make_mock_model()
        predictor = RealTimePredictor(model)
        assert predictor.model == model
        assert predictor.cache_ttl == 300
        assert len(predictor.latency_history) == 0

    def test_predict_no_cache(self):
        model = self._make_mock_model()
        predictor = RealTimePredictor(model)

        result = predictor.predict(
            {"f1": 0.5, "f2": 0.3, "f3": 0.1},
            use_cache=False,
        )

        assert "predicted_class" in result
        assert result["predicted_class"] == 0
        assert "probabilities" in result
        assert "confidence" in result
        assert result["confidence"] == 0.5
        assert "latency_ms" in result
        assert result["cache_hit"] is False

    def test_predict_with_redis_cache(self):
        model = self._make_mock_model()

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        predictor = RealTimePredictor(model, redis_client=mock_redis, cache_ttl=60)

        result = predictor.predict(
            {"f1": 1.0, "f2": 0.5, "f3": 0.2},
            match_id="match_123",
        )

        assert result["cache_hit"] is False
        # Should have called setex to cache
        mock_redis.setex.assert_called_once()

    def test_predict_cache_hit(self):
        model = self._make_mock_model()

        cached_data = json.dumps(
            {
                "predicted_class": 1,
                "probabilities": {"home_win": 0.3, "draw": 0.5, "away_win": 0.2},
                "confidence": 0.5,
            }
        )
        mock_redis = MagicMock()
        mock_redis.get.return_value = cached_data

        predictor = RealTimePredictor(model, redis_client=mock_redis)

        result = predictor.predict(
            {"f1": 1.0, "f2": 0.5, "f3": 0.2},
            match_id="match_123",
        )

        assert result["cache_hit"] is True
        assert result["predicted_class"] == 1

    def test_predict_redis_failure_graceful(self):
        model = self._make_mock_model()

        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis down")
        mock_redis.setex.side_effect = Exception("Redis down")

        predictor = RealTimePredictor(model, redis_client=mock_redis)

        # Should still work, just without cache
        result = predictor.predict(
            {"f1": 1.0, "f2": 0.5, "f3": 0.2},
            match_id="match_123",
        )

        assert result["predicted_class"] == 0
        assert result["cache_hit"] is False

    def test_latency_stats(self):
        model = self._make_mock_model()
        predictor = RealTimePredictor(model)

        # Make multiple predictions
        for _ in range(10):
            predictor.predict({"f1": 0.5, "f2": 0.3, "f3": 0.1}, use_cache=False)

        stats = predictor.get_latency_stats()
        assert "mean_ms" in stats
        assert "p50_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert stats["n_predictions"] == 10
        assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]

    def test_latency_stats_empty(self):
        model = self._make_mock_model()
        predictor = RealTimePredictor(model)
        stats = predictor.get_latency_stats()
        assert stats == {}

    def test_onnx_fallback(self):
        model = self._make_mock_model()

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="features")]
        mock_session.run.return_value = [np.array([[0.6, 0.2, 0.2]])]

        predictor = RealTimePredictor(model, onnx_session=mock_session)

        result = predictor.predict({"f1": 1.0, "f2": 0.5, "f3": 0.2}, use_cache=False)
        assert result["predicted_class"] == 0
        assert result["probabilities"]["home_win"] == 0.6


class TestBatchPredictor:
    """Tests for BatchPredictor."""

    def _make_mock_model(self):
        mock = MagicMock()
        mock.predict = MagicMock(side_effect=lambda X: np.zeros(len(X), dtype=int))
        mock.predict_proba = MagicMock(side_effect=lambda X: np.array([[0.5, 0.3, 0.2]] * len(X)))
        return mock

    def test_init(self):
        model = self._make_mock_model()
        predictor = BatchPredictor(model, batch_size=64)
        assert predictor.batch_size == 64

    def test_predict_batch(self):
        model = self._make_mock_model()
        predictor = BatchPredictor(model, batch_size=50)

        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        result = predictor.predict_batch(df)

        assert len(result) == 100
        assert "prediction" in result.columns
        assert "prob_home_win" in result.columns
        assert "prob_draw" in result.columns
        assert "prob_away_win" in result.columns
        assert "confidence" in result.columns

    def test_predict_batch_no_probabilities(self):
        model = self._make_mock_model()
        predictor = BatchPredictor(model, batch_size=50)

        df = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        result = predictor.predict_batch(df, include_probabilities=False)

        assert len(result) == 30
        assert "prediction" in result.columns
        assert "prob_home_win" not in result.columns

    def test_predict_with_value(self):
        model = self._make_mock_model()
        predictor = BatchPredictor(model)

        df = pd.DataFrame(
            {
                "f1": np.random.randn(20),
                "f2": np.random.randn(20),
                "home_odds": np.random.uniform(1.5, 3.0, 20),
                "draw_odds": np.random.uniform(2.5, 4.0, 20),
                "away_odds": np.random.uniform(2.0, 5.0, 20),
            }
        )

        odds_columns = {0: "home_odds", 1: "draw_odds", 2: "away_odds"}
        result = predictor.predict_with_value(df, odds_columns=odds_columns)

        assert "value_home_win" in result.columns
        assert "is_value_home_win" in result.columns
        assert result["is_value_home_win"].dtype == bool

    def test_predict_with_value_no_odds(self):
        model = self._make_mock_model()
        predictor = BatchPredictor(model)

        df = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        result = predictor.predict_with_value(df)

        # Should still work, just without value columns
        assert "prediction" in result.columns

    def test_batch_size_respected(self):
        """Ensure the model is called multiple times for large batches."""
        model = self._make_mock_model()
        predictor = BatchPredictor(model, batch_size=10)

        df = pd.DataFrame(np.random.randn(35, 3), columns=["a", "b", "c"])
        predictor.predict_batch(df)

        # 35 rows / 10 batch_size = 4 calls
        assert model.predict.call_count == 4
        assert model.predict_proba.call_count == 4
