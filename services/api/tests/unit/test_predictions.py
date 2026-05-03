"""Tests for prediction endpoints"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("config.settings") as mock:
        mock.JWT_SECRET = "test-secret-key-for-testing-only"
        mock.JWT_ALGORITHM = "HS256"
        mock.JWT_EXPIRATION_HOURS = 24
        mock.USER_DOB = "1994-05-09"
        mock.DATABASE_URL = "postgresql://test:test@localhost:5432/test_db"
        mock.REDIS_URL = "redis://localhost:6379/15"
        mock.MODEL_PATH = "/tmp/models"
        mock.REDIS_CACHE_TTL = 3600
        yield mock


class TestPredictionService:
    """Test PredictionService business logic"""

    def test_predict_match_not_found(self, mock_db, mock_settings):
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchone.return_value = None

        service = PredictionService(mock_db)

        with pytest.raises(ValueError, match="not found"):
            service.predict_match(match_id=str(uuid.uuid4()))

    def test_predict_match_with_stored_prediction(self, mock_db, mock_row, mock_settings):
        from services.prediction_service import PredictionService

        match_id = str(uuid.uuid4())

        # Mock match data query
        match_result = mock_row(
            id=uuid.UUID(match_id),
            league_name="Premier League",
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=datetime(2025, 3, 15, 15, 0),
            venue="Emirates Stadium",
        )

        # Mock features query
        features_result = mock_row(features={"team__home__form__wins__last5": 3})

        # Mock stored prediction
        prediction_result = mock_row(
            predicted_outcome="home",
            confidence=0.72,
            probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
        )

        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:  # Match data
                result.fetchone.return_value = match_result
            elif call_count[0] == 2:  # Features
                result.fetchone.return_value = features_result
            elif call_count[0] == 3:  # Stored prediction
                result.fetchone.return_value = prediction_result
            else:
                result.fetchone.return_value = None
            return result

        mock_db.execute.side_effect = side_effect

        service = PredictionService(mock_db)
        response = service.predict_match(match_id=match_id)

        assert response.predicted_outcome == "home"
        assert response.confidence == 0.72
        assert response.match_info.home_team == "Arsenal"

    def test_get_upcoming_predictions(self, mock_db, mock_row, mock_settings):
        from services.prediction_service import PredictionService

        rows = [
            mock_row(
                id=uuid.uuid4(),
                match_id=uuid.uuid4(),
                predicted_outcome="home",
                confidence=0.72,
                probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
                model_name="ensemble",
                model_version="v1.0",
                features_used={},
                feature_importance={},
                match_date=datetime(2025, 3, 15, 15, 0),
                venue="Emirates Stadium",
                league_name="Premier League",
                home_team="Arsenal",
                away_team="Chelsea",
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = rows

        service = PredictionService(mock_db)
        predictions = service.get_upcoming_predictions(limit=10)

        assert len(predictions) == 1
        assert predictions[0].predicted_outcome == "home"
        assert predictions[0].match_info.home_team == "Arsenal"

    def test_get_live_predictions_empty(self, mock_db, mock_settings):
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []

        service = PredictionService(mock_db)
        predictions = service.get_live_predictions()

        assert predictions == []

    def test_store_prediction(self, mock_db, mock_settings):
        from services.prediction_service import PredictionService

        service = PredictionService(mock_db)

        prediction = {
            "predicted_label": "home",
            "confidence": 0.72,
            "probabilities": {"home": 0.55, "draw": 0.25, "away": 0.20},
        }

        service._store_prediction(str(uuid.uuid4()), prediction)

        mock_db.execute.assert_called()
        mock_db.commit.assert_called()


class TestCacheService:
    """Test CacheService"""

    def test_get_prediction_cache_miss(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis.get.return_value = None
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            result = service.get_prediction("match123")
            assert result is None

    def test_get_prediction_cache_hit(self, mock_settings):
        import json

        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            cached_data = {"predicted_outcome": "home", "confidence": 0.72}
            mock_redis.get.return_value = json.dumps(cached_data)
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            result = service.get_prediction("match123")
            assert result["predicted_outcome"] == "home"

    def test_set_prediction(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            service.set_prediction("match123", {"outcome": "home"})
            mock_redis.setex.assert_called_once()

    def test_invalidate_prediction(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            service.invalidate_prediction("match123")
            mock_redis.delete.assert_called_once_with("prediction:match123")

    def test_cache_graceful_redis_failure(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis.get.side_effect = Exception("Connection refused")
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            result = service.get_prediction("match123")
            assert result is None  # Graceful degradation
