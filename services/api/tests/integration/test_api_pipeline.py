"""Integration tests - end-to-end API flow tests"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock
import uuid
import json

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
        mock.RATE_LIMIT_REQUESTS = 100
        mock.RATE_LIMIT_WINDOW = 60
        yield mock


class TestAuthToPredictionFlow:
    """Test full auth -> predict flow"""

    def test_login_generates_valid_token(self, mock_settings):
        """Login with correct DOB returns a valid JWT"""
        from auth.jwt_handler import (
            create_access_token,
            verify_token,
            verify_date_of_birth,
        )

        # Step 1: Verify DOB
        assert verify_date_of_birth("1994-05-09") is True

        # Step 2: Create token
        token = create_access_token(
            data={"user_id": "owner", "username": "ceesar"}
        )

        # Step 3: Token is valid
        payload = verify_token(token)
        assert payload["user_id"] == "owner"

    def test_invalid_dob_blocks_login(self, mock_settings):
        """Wrong DOB prevents token creation"""
        from auth.jwt_handler import verify_date_of_birth

        assert verify_date_of_birth("2000-01-01") is False

    def test_prediction_requires_auth(self, mock_settings):
        """Prediction endpoint requires valid auth"""
        from auth.dependencies import require_auth
        from fastapi import HTTPException

        # Missing user_id in payload
        with pytest.raises(HTTPException) as exc_info:
            require_auth({"username": "anonymous"})
        assert exc_info.value.status_code == 401


class TestPredictionToRecommendationFlow:
    """Test prediction -> recommendation pipeline"""

    def test_prediction_feeds_recommendation(self, mock_row, mock_settings):
        """Predictions are used to generate recommendations"""
        from services.recommendation_service import RecommendationService

        mock_db = MagicMock()

        # Simulate high-EV recommendations from predictions
        recs = [
            mock_row(
                recommendation_id=uuid.uuid4(),
                match_date=datetime(2025, 3, 15, 15, 0),
                league_name="Premier League",
                home_team="Arsenal",
                away_team="Chelsea",
                bet_type="1x2",
                selection="home",
                odds_at_recommendation=1.85,
                bookmaker="bet365",
                confidence_rating="high",
                expected_value=0.12,
                recommended_stake=25.0,
                reasoning="Model confidence 72%, positive EV",
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = recs

        service = RecommendationService(mock_db)
        results = service.get_high_value_recommendations(min_ev=0.05)

        assert len(results) == 1
        assert results[0].expected_value == 0.12
        assert results[0].confidence_level == "high"


class TestNotificationFlow:
    """Test notification creation and retrieval"""

    def test_high_value_bet_creates_notification(self, mock_row, mock_settings):
        from services.notification_service import NotificationService

        mock_db = MagicMock()

        # Simulate high-value bets found
        bets = [
            mock_row(
                id=uuid.uuid4(),
                bet_type="1x2",
                selection="home",
                odds_at_recommendation=1.85,
                expected_value=0.15,
                bookmaker="bet365",
                match_date=datetime(2025, 3, 15, 15, 0),
                home_team="Arsenal",
                away_team="Chelsea",
                league_name="Premier League",
            ),
        ]

        call_count = [0]
        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.fetchall.return_value = bets
            else:
                result.fetchone.return_value = mock_row(id=uuid.uuid4())
            return result

        mock_db.execute.side_effect = side_effect

        service = NotificationService(mock_db)
        alerts = service.check_high_value_bets(min_ev=0.10)

        assert len(alerts) == 1
        assert alerts[0]["expected_value"] == 0.15

    def test_model_drift_detection(self, mock_row, mock_settings):
        from services.notification_service import NotificationService

        mock_db = MagicMock()

        # Simulate drift detected
        drift_rows = [
            mock_row(
                model_name="ensemble",
                model_version="v1.0",
                current_metrics={"accuracy": 0.60},
                previous_metrics={"accuracy": 0.72},
                evaluation_date=date(2025, 3, 15),
            ),
        ]

        call_count = [0]
        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.fetchall.return_value = drift_rows
            else:
                result.fetchone.return_value = mock_row(id=uuid.uuid4())
            return result

        mock_db.execute.side_effect = side_effect

        service = NotificationService(mock_db)
        drift_alerts = service.check_model_drift(threshold=0.05)

        assert len(drift_alerts) == 1
        assert drift_alerts[0]["drift"] == pytest.approx(0.12)
        assert drift_alerts[0]["model_name"] == "ensemble"


class TestAccumulatorFlow:
    """Test accumulator build flow"""

    def test_accumulator_confidence_levels(self, mock_settings):
        from services.recommendation_service import RecommendationService

        assert RecommendationService._get_accumulator_confidence(0.6) == "high"
        assert RecommendationService._get_accumulator_confidence(0.35) == "medium"
        assert RecommendationService._get_accumulator_confidence(0.15) == "low"


class TestRequestValidation:
    """Test request model validation"""

    def test_accumulator_request_valid(self, mock_settings):
        from models.requests import AccumulatorRequest

        req = AccumulatorRequest(
            min_legs=2,
            max_legs=5,
            min_total_odds=2.0,
            max_total_odds=10.0,
            confidence_threshold=0.65,
        )
        assert req.min_legs == 2
        assert req.max_legs == 5

    def test_accumulator_request_invalid_legs(self, mock_settings):
        from models.requests import AccumulatorRequest

        with pytest.raises(ValueError, match="max_legs must be >= min_legs"):
            AccumulatorRequest(
                min_legs=5,
                max_legs=2,  # Invalid: less than min_legs
            )

    def test_bulk_prediction_request_limits(self, mock_settings):
        from models.requests import BulkPredictionRequest

        # Valid
        req = BulkPredictionRequest(match_ids=["id1", "id2"])
        assert len(req.match_ids) == 2

    def test_user_preferences_risk_tolerance_values(self, mock_settings):
        from models.requests import UserPreferencesUpdate

        # Valid values
        for level in ["LOW", "MEDIUM", "HIGH"]:
            update = UserPreferencesUpdate(risk_tolerance=level)
            assert update.risk_tolerance == level
