"""Tests for recommendation service and endpoints"""

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import uuid

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
        yield mock


class TestRecommendationService:
    """Test RecommendationService"""

    def test_get_active_recommendations(self, mock_db, mock_row, mock_settings):
        from services.recommendation_service import RecommendationService

        rows = [
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
                reasoning="Strong home form",
                status="pending",
                model_name="ensemble",
                model_confidence=0.72,
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = rows

        service = RecommendationService(mock_db)
        recs = service.get_active_recommendations(limit=10)

        assert len(recs) == 1
        assert recs[0].market_type == "1x2"
        assert recs[0].outcome == "home"
        assert recs[0].recommended_odds == 1.85
        assert recs[0].confidence_level == "high"

    def test_get_active_recommendations_empty(self, mock_db, mock_settings):
        from services.recommendation_service import RecommendationService

        mock_db.execute.return_value.fetchall.return_value = []

        service = RecommendationService(mock_db)
        recs = service.get_active_recommendations()

        assert recs == []

    def test_get_active_recommendations_with_filters(self, mock_db, mock_settings):
        from services.recommendation_service import RecommendationService

        mock_db.execute.return_value.fetchall.return_value = []

        service = RecommendationService(mock_db)
        service.get_active_recommendations(
            confidence_level="HIGH",
            market_type="1x2",
            min_odds=1.5,
            max_odds=3.0,
        )

        # Verify query was executed with params
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["confidence_level"] == "high"
        assert params["market_type"] == "1x2"

    def test_get_high_value_recommendations(self, mock_db, mock_row, mock_settings):
        from services.recommendation_service import RecommendationService

        rows = [
            mock_row(
                recommendation_id=uuid.uuid4(),
                match_date=datetime(2025, 3, 15, 15, 0),
                league_name="La Liga",
                home_team="Barcelona",
                away_team="Real Madrid",
                bet_type="over_under",
                selection="over_2.5",
                odds_at_recommendation=1.95,
                bookmaker="betmgm",
                confidence_rating="very_high",
                expected_value=0.18,
                recommended_stake=50.0,
                reasoning="Both teams score a lot",
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = rows

        service = RecommendationService(mock_db)
        recs = service.get_high_value_recommendations(min_ev=0.10)

        assert len(recs) == 1
        assert recs[0].expected_value == 0.18

    def test_build_accumulator_not_enough_legs(self, mock_db, mock_settings):
        from services.recommendation_service import RecommendationService

        mock_db.execute.return_value.fetchall.return_value = []

        service = RecommendationService(mock_db)

        with pytest.raises(ValueError, match="Not enough qualifying"):
            service.build_accumulator(min_legs=3)

    def test_build_accumulator_success(self, mock_db, mock_row, mock_settings):
        from services.recommendation_service import RecommendationService

        legs = [
            mock_row(
                recommendation_id=uuid.uuid4(),
                match_id=uuid.uuid4(),
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
                reasoning="Strong form",
                model_confidence=0.72,
            ),
            mock_row(
                recommendation_id=uuid.uuid4(),
                match_id=uuid.uuid4(),
                match_date=datetime(2025, 3, 15, 17, 30),
                league_name="La Liga",
                home_team="Barcelona",
                away_team="Sevilla",
                bet_type="over_under",
                selection="over_2.5",
                odds_at_recommendation=1.65,
                bookmaker="betmgm",
                confidence_rating="high",
                expected_value=0.08,
                recommended_stake=20.0,
                reasoning="High-scoring teams",
                model_confidence=0.75,
            ),
        ]

        # First call returns legs, second returns bankroll
        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:  # Fetch legs
                result.fetchall.return_value = legs
            elif call_count[0] == 2:  # Get bankroll
                result.fetchone.return_value = mock_row(
                    preference_value={"value": 5000}
                )
            elif call_count[0] == 3:  # Insert accumulator
                result.fetchone.return_value = None
            return result

        mock_db.execute.side_effect = side_effect

        service = RecommendationService(mock_db)
        acc = service.build_accumulator(min_legs=2, max_legs=5)

        assert len(acc.legs) == 2
        assert acc.total_odds == round(1.85 * 1.65, 4)
        assert acc.combined_probability == round(0.72 * 0.75, 6)

    def test_accumulator_confidence_rating(self):
        from services.recommendation_service import RecommendationService

        assert RecommendationService._get_accumulator_confidence(0.6) == "high"
        assert RecommendationService._get_accumulator_confidence(0.4) == "medium"
        assert RecommendationService._get_accumulator_confidence(0.2) == "low"
