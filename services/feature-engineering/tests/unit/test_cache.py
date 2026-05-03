"""Tests for feature cache manager."""

import json

import pytest
from src.core.cache import FeatureCacheManager


class TestFeatureCacheManager:

    @pytest.fixture
    def cache(self, mock_config, mock_db, mock_redis):
        return FeatureCacheManager(mock_config, mock_db, mock_redis)

    @pytest.fixture
    def sample_features(self):
        return {"f1": 0.5, "f2": 1.0, "f3": None}

    def test_make_key(self, cache):
        key = cache._make_key("match-123", "full")
        assert key == "features:v1:match-123:full"

    def test_get_redis_hit(self, cache, mock_redis, sample_features):
        mock_redis.get.return_value = json.dumps(sample_features)
        result = cache.get("match-123")
        assert result == sample_features

    def test_get_redis_miss_db_hit(self, cache, mock_redis, mock_db, sample_features):
        mock_redis.get.return_value = None
        mock_db.execute_query.return_value = [
            {"feature_data": json.dumps(sample_features), "computed_at": "2025-01-15"}
        ]

        result = cache.get("match-123")
        assert result == sample_features
        # Should warm up Redis
        mock_redis.setex.assert_called_once()

    def test_get_both_miss(self, cache, mock_redis, mock_db):
        mock_redis.get.return_value = None
        mock_db.execute_query.return_value = []

        result = cache.get("match-123")
        assert result is None

    def test_set(self, cache, mock_redis, mock_db, sample_features):
        cache.set("match-123", sample_features)

        # Should write to Redis
        mock_redis.setex.assert_called_once()
        # Should write to DB
        mock_db.execute_query.assert_called_once()

    def test_invalidate(self, cache, mock_redis, mock_db):
        cache.invalidate("match-123")
        mock_redis.delete.assert_called_once()
        mock_db.execute_query.assert_called_once()

    def test_redis_failure_graceful(self, cache, mock_redis, mock_db, sample_features):
        """Should not crash if Redis fails."""
        mock_redis.get.side_effect = Exception("Redis down")
        mock_db.execute_query.return_value = []

        result = cache.get("match-123")
        assert result is None  # Graceful degradation

    def test_db_failure_graceful(self, cache, mock_redis, mock_db, sample_features):
        """Should not crash if DB cache fails."""
        mock_redis.get.return_value = None
        mock_db.execute_query.side_effect = Exception("DB error")

        result = cache.get("match-123")
        assert result is None

    def test_set_redis_failure_still_writes_db(self, cache, mock_redis, mock_db, sample_features):
        mock_redis.setex.side_effect = Exception("Redis write failed")
        cache.set("match-123", sample_features)
        # DB write should still happen
        mock_db.execute_query.assert_called_once()
