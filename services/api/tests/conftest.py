"""Test fixtures for API service"""

import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Patch settings before any imports that use them
with patch.dict(
    "os.environ",
    {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        "REDIS_URL": "redis://localhost:6379/15",
        "JWT_SECRET": "test-secret-key-for-testing-only",
        "USER_DOB": "1994-05-09",
    },
):
    from config import Settings

    test_settings = Settings()


# ==================== Auth Fixtures ====================


@pytest.fixture
def jwt_secret():
    return "test-secret-key-for-testing-only"


@pytest.fixture
def valid_token(jwt_secret):
    """Create a valid JWT token for testing"""
    payload = {
        "user_id": "owner",
        "username": "testuser",
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    return pyjwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def expired_token(jwt_secret):
    """Create an expired JWT token"""
    payload = {
        "user_id": "owner",
        "username": "testuser",
        "exp": datetime.utcnow() - timedelta(hours=1),
    }
    return pyjwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def auth_headers(valid_token):
    """Authorization headers with valid token"""
    return {"Authorization": f"Bearer {valid_token}"}


# ==================== Database Fixtures ====================


@pytest.fixture
def mock_db():
    """Mock database session"""
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []
    db.execute.return_value.fetchone.return_value = None
    return db


# ==================== Sample Data Fixtures ====================


@pytest.fixture
def sample_match_id():
    return str(uuid.uuid4())


@pytest.fixture
def sample_match_data(sample_match_id):
    """Sample match data as returned from DB"""
    return {
        "match_id": sample_match_id,
        "league_name": "Premier League",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": datetime(2025, 3, 15, 15, 0),
        "venue": "Emirates Stadium",
    }


@pytest.fixture
def sample_prediction_response(sample_match_data):
    """Sample prediction response"""
    return {
        "match_info": sample_match_data,
        "predicted_outcome": "home",
        "probabilities": {"home": 0.55, "draw": 0.25, "away": 0.20},
        "confidence": 0.72,
        "model_version": "ensemble_v1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_recommendation_data():
    """Sample recommendation data"""
    return {
        "recommendation_id": str(uuid.uuid4()),
        "match_date": datetime(2025, 3, 15, 15, 0),
        "league_name": "Premier League",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bet_type": "1x2",
        "selection": "home",
        "odds_at_recommendation": 1.85,
        "bookmaker": "bet365",
        "confidence_rating": "high",
        "expected_value": 0.12,
        "recommended_stake": 25.0,
        "reasoning": "Strong home form, H2H advantage",
    }


@pytest.fixture
def sample_odds_data(sample_match_id):
    """Sample odds data"""
    return [
        {
            "id": str(uuid.uuid4()),
            "match_id": sample_match_id,
            "bookmaker": "bet365",
            "market_type": "1x2",
            "selection": "home",
            "odds_decimal": 1.85,
            "implied_probability": 0.5405,
            "timestamp": datetime.utcnow(),
            "is_live": False,
        },
        {
            "id": str(uuid.uuid4()),
            "match_id": sample_match_id,
            "bookmaker": "bet365",
            "market_type": "1x2",
            "selection": "draw",
            "odds_decimal": 3.50,
            "implied_probability": 0.2857,
            "timestamp": datetime.utcnow(),
            "is_live": False,
        },
    ]


@pytest.fixture
def sample_lottery_draw():
    """Sample lottery draw"""
    return {
        "id": str(uuid.uuid4()),
        "game": "powerball",
        "draw_date": datetime(2025, 3, 12),
        "numbers": [5, 12, 23, 45, 67],
        "bonus_number": 15,
        "multiplier": 2,
        "jackpot_amount": 150000000.00,
    }


@pytest.fixture
def sample_user_preferences():
    """Sample user preferences"""
    return {
        "bankroll": {"value": 5000},
        "risk_tolerance": {"value": "MEDIUM"},
        "confidence_threshold": {"value": 0.65},
        "kelly_fraction": {"value": 0.25},
        "favorite_sports": {"value": ["soccer", "nfl"]},
    }


# ==================== Mock Row Helper ====================


class MockRow:
    """Helper to create mock database row objects with attribute access"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def mock_row():
    """Factory fixture to create mock DB rows"""
    return MockRow
