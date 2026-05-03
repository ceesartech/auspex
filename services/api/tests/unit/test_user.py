"""Tests for user endpoints - preferences, betting history, dashboard"""

import sys
from pathlib import Path
from datetime import date
from unittest.mock import patch
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


class TestLoginFlow:
    """Test DOB-based login"""

    def test_login_request_validation(self, mock_settings):
        from models.requests import LoginRequest

        # Valid request
        req = LoginRequest(
            username="ceesar",
            password="password123",
            date_of_birth=date(1994, 5, 9),
        )
        assert req.username == "ceesar"
        assert req.date_of_birth == date(1994, 5, 9)

    def test_login_request_underage_rejected(self, mock_settings):
        from models.requests import LoginRequest

        with pytest.raises(ValueError, match="21 years"):
            LoginRequest(
                username="younguser",
                password="password123",
                date_of_birth=date(2010, 1, 1),  # Too young
            )

    def test_login_correct_dob(self, mock_settings):
        from auth.jwt_handler import verify_date_of_birth

        assert verify_date_of_birth("1994-05-09") is True

    def test_login_wrong_dob(self, mock_settings):
        from auth.jwt_handler import verify_date_of_birth

        assert verify_date_of_birth("1990-01-01") is False

    def test_token_created_on_login(self, mock_settings):
        from auth.jwt_handler import create_access_token, verify_token

        token = create_access_token(data={"user_id": "owner", "username": "ceesar"})
        payload = verify_token(token)

        assert payload["user_id"] == "owner"
        assert payload["username"] == "ceesar"


class TestUserPreferences:
    """Test user preferences CRUD"""

    def test_preferences_update_model(self, mock_settings):
        from models.requests import UserPreferencesUpdate

        update = UserPreferencesUpdate(
            bankroll=5000.0,
            risk_tolerance="MEDIUM",
            confidence_threshold=0.65,
            kelly_fraction=0.25,
            favorite_sports=["soccer", "nfl"],
        )

        assert update.bankroll == 5000.0
        assert update.risk_tolerance == "MEDIUM"
        assert len(update.favorite_sports) == 2

    def test_preferences_update_partial(self, mock_settings):
        from models.requests import UserPreferencesUpdate

        update = UserPreferencesUpdate(bankroll=3000.0)

        data = update.model_dump(exclude_none=True)
        assert "bankroll" in data
        assert "risk_tolerance" not in data

    def test_preferences_invalid_risk_tolerance(self, mock_settings):
        from models.requests import UserPreferencesUpdate

        with pytest.raises(ValueError):
            UserPreferencesUpdate(risk_tolerance="EXTREME")

    def test_preferences_confidence_threshold_bounds(self, mock_settings):
        from models.requests import UserPreferencesUpdate

        with pytest.raises(ValueError):
            UserPreferencesUpdate(confidence_threshold=1.5)

        with pytest.raises(ValueError):
            UserPreferencesUpdate(confidence_threshold=-0.1)


class TestBettingHistory:
    """Test betting history recording"""

    def test_betting_history_record_model(self, mock_settings):
        from models.requests import BettingHistoryRecord

        record = BettingHistoryRecord(
            bookmaker="bet365",
            bet_type="1x2",
            selection="home",
            odds=1.85,
            stake=25.0,
            notes="Confident pick",
        )

        assert record.bookmaker == "bet365"
        assert record.odds == 1.85
        assert record.stake == 25.0

    def test_betting_history_record_with_recommendation(self, mock_settings):
        from models.requests import BettingHistoryRecord

        rec_id = str(uuid.uuid4())
        match_id = str(uuid.uuid4())

        record = BettingHistoryRecord(
            recommendation_id=rec_id,
            match_id=match_id,
            bookmaker="betmgm",
            bet_type="over_under",
            selection="over_2.5",
            odds=1.95,
            stake=50.0,
        )

        assert record.recommendation_id == rec_id
        assert record.match_id == match_id

    def test_betting_history_invalid_odds(self, mock_settings):
        from models.requests import BettingHistoryRecord

        with pytest.raises(ValueError):
            BettingHistoryRecord(
                bookmaker="bet365",
                bet_type="1x2",
                selection="home",
                odds=0.5,  # Invalid: must be > 1.0
                stake=25.0,
            )

    def test_betting_history_invalid_stake(self, mock_settings):
        from models.requests import BettingHistoryRecord

        with pytest.raises(ValueError):
            BettingHistoryRecord(
                bookmaker="bet365",
                bet_type="1x2",
                selection="home",
                odds=1.85,
                stake=-10.0,  # Invalid: must be > 0
            )


class TestDashboard:
    """Test dashboard statistics"""

    def test_dashboard_stats_formatting(self, mock_row, mock_settings):
        """Verify dashboard response structure"""

        stats = mock_row(
            total_bets=50,
            active_bets=5,
            total_staked=2500.0,
            total_returns=2800.0,
            profit_loss=300.0,
            roi_pct=12.0,
            win_rate=55.0,
        )

        dashboard = {
            "total_bets": stats.total_bets,
            "active_bets": stats.active_bets,
            "total_staked": float(stats.total_staked),
            "total_returns": float(stats.total_returns),
            "profit_loss": float(stats.profit_loss),
            "roi_pct": float(stats.roi_pct),
            "win_rate": float(stats.win_rate),
        }

        assert dashboard["total_bets"] == 50
        assert dashboard["profit_loss"] == 300.0
        assert dashboard["roi_pct"] == 12.0

    def test_dashboard_zero_bets(self, mock_row, mock_settings):
        """Dashboard with no betting history"""

        stats = mock_row(
            total_bets=0,
            active_bets=0,
            total_staked=0,
            total_returns=0,
            profit_loss=0,
            roi_pct=0,
            win_rate=0,
        )

        assert stats.total_bets == 0
        assert stats.roi_pct == 0
