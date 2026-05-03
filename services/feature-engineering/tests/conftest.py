"""Shared test fixtures for feature engineering tests."""

from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from redis import Redis
from src.categories.base import MatchContext
from src.core.config import FeatureConfig
from src.core.database import DatabaseManager

# ---------- IDs ----------

HOME_TEAM_ID = UUID("11111111-1111-1111-1111-111111111111")
AWAY_TEAM_ID = UUID("22222222-2222-2222-2222-222222222222")
MATCH_ID = UUID("33333333-3333-3333-3333-333333333333")
LEAGUE_ID = UUID("44444444-4444-4444-4444-444444444444")
PLAYER_1_ID = UUID("55555555-5555-5555-5555-555555555555")
PLAYER_2_ID = UUID("66666666-6666-6666-6666-666666666666")


@pytest.fixture
def mock_config():
    """Feature engineering configuration for tests."""
    config = Mock(spec=FeatureConfig)
    config.database_url = "postgresql://test:test@localhost:5432/test_db"
    config.redis_url = "redis://localhost:6379/15"
    config.db_min_connections = 1
    config.db_max_connections = 5
    config.redis_cache_ttl = 3600
    config.db_cache_ttl = 86400
    config.feature_version = "v1"
    config.form_windows = [3, 5, 10, 20]
    config.h2h_max_matches = 20
    config.max_missing_rate = 0.3
    config.outlier_std_threshold = 4.0
    config.max_workers = 6
    config.log_level = "DEBUG"
    return config


@pytest.fixture
def mock_db():
    """Mock database manager."""
    db = Mock(spec=DatabaseManager)
    db.execute_query = Mock(return_value=[])
    db.call_function = Mock(return_value=[])
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = Mock(spec=Redis)
    redis.get = Mock(return_value=None)
    redis.setex = Mock()
    redis.delete = Mock()
    redis.exists = Mock(return_value=False)
    return redis


@pytest.fixture
def match_context():
    """Standard match context for testing."""
    return MatchContext(
        match_id=MATCH_ID,
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
        league_id=LEAGUE_ID,
        season="2024-2025",
        match_date=datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc),
        venue="Home Stadium",
        referee="Michael Oliver",
        home_team_name="Home FC",
        away_team_name="Away United",
        league_name="Premier League",
        country="England",
    )


@pytest.fixture
def sample_team_matches():
    """Sample match data for team performance tests."""
    from datetime import timedelta

    base_date = datetime(2025, 1, 15, tzinfo=timezone.utc)
    base_matches = []
    for i in range(20):
        home_score = (i % 3) + 1  # 1, 2, 3, 1, 2, ...
        away_score = i % 2  # 0, 1, 0, 1, ...
        base_matches.append(
            {
                "match_id": str(uuid4()),
                "match_date": base_date - timedelta(days=i),
                "home_team_id": str(HOME_TEAM_ID) if i % 2 == 0 else str(uuid4()),
                "away_team_id": str(uuid4()) if i % 2 == 0 else str(HOME_TEAM_ID),
                "home_score": home_score,
                "away_score": away_score,
                "season": "2024-2025",
                "possession_home": 55.0,
                "possession_away": 45.0,
                "shots_home": 12,
                "shots_away": 8,
                "shots_on_target_home": 5,
                "shots_on_target_away": 3,
                "corners_home": 6,
                "corners_away": 4,
                "fouls_home": 10,
                "fouls_away": 12,
                "yellow_cards_home": 2,
                "yellow_cards_away": 3,
                "red_cards_home": 0,
                "red_cards_away": 0,
                "xg_home": 1.5,
                "xg_away": 0.8,
            }
        )
    return base_matches


@pytest.fixture
def sample_h2h_matches():
    """Sample head-to-head match data."""
    from datetime import timedelta

    base_date = datetime(2024, 12, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(10):
        matches.append(
            {
                "match_id": str(uuid4()),
                "match_date": base_date - timedelta(days=i * 30),
                "home_team_id": str(HOME_TEAM_ID) if i % 2 == 0 else str(AWAY_TEAM_ID),
                "away_team_id": str(AWAY_TEAM_ID) if i % 2 == 0 else str(HOME_TEAM_ID),
                "home_score": 2 if i % 3 == 0 else 1,
                "away_score": 1 if i % 3 != 2 else 2,
                "possession_home": 52.0,
                "possession_away": 48.0,
                "shots_home": 10,
                "shots_away": 9,
                "xg_home": 1.3,
                "xg_away": 1.1,
            }
        )
    return matches


@pytest.fixture
def sample_squad():
    """Sample squad data."""
    return [
        {
            "player_id": str(PLAYER_1_ID),
            "name": "Star Player",
            "position": "FW",
            "market_value": 50000000,
            "date_of_birth": datetime(1995, 5, 10).date(),
        },
        {
            "player_id": str(PLAYER_2_ID),
            "name": "Key Midfielder",
            "position": "MF",
            "market_value": 30000000,
            "date_of_birth": datetime(1997, 8, 22).date(),
        },
        {
            "player_id": str(uuid4()),
            "name": "Goalkeeper",
            "position": "GK",
            "market_value": 10000000,
            "date_of_birth": datetime(1993, 1, 5).date(),
        },
    ]


@pytest.fixture
def sample_availability():
    """Sample player availability data."""
    return [
        {
            "player_id": str(PLAYER_2_ID),
            "status": "injured",
            "injury_type": "hamstring",
            "expected_return": datetime(2025, 2, 1).date(),
        },
    ]


@pytest.fixture
def sample_player_stats():
    """Sample recent player match stats."""
    return [
        {
            "player_id": str(PLAYER_1_ID),
            "minutes_played": 90,
            "goals": 1,
            "assists": 0,
            "xg": 0.8,
            "xa": 0.1,
            "rating": 7.5,
            "key_passes": 2,
            "shots": 3,
            "match_date": datetime(2025, 1, 10, tzinfo=timezone.utc),
        },
        {
            "player_id": str(PLAYER_1_ID),
            "minutes_played": 85,
            "goals": 0,
            "assists": 1,
            "xg": 0.3,
            "xa": 0.5,
            "rating": 7.2,
            "key_passes": 3,
            "shots": 2,
            "match_date": datetime(2025, 1, 5, tzinfo=timezone.utc),
        },
    ]


@pytest.fixture
def sample_odds():
    """Sample odds data."""
    return [
        {
            "bookmaker": "bet365",
            "market_type": "1x2",
            "selection": "home",
            "odds_decimal": 2.10,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "bet365",
            "market_type": "1x2",
            "selection": "draw",
            "odds_decimal": 3.40,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "bet365",
            "market_type": "1x2",
            "selection": "away",
            "odds_decimal": 3.50,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "betmgm",
            "market_type": "1x2",
            "selection": "home",
            "odds_decimal": 2.15,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "betmgm",
            "market_type": "1x2",
            "selection": "draw",
            "odds_decimal": 3.30,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "betmgm",
            "market_type": "1x2",
            "selection": "away",
            "odds_decimal": 3.40,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "bet365",
            "market_type": "over_under_25",
            "selection": "over",
            "odds_decimal": 1.85,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "bet365",
            "market_type": "over_under_25",
            "selection": "under",
            "odds_decimal": 1.95,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
        {
            "bookmaker": "bet365",
            "market_type": "btts",
            "selection": "yes",
            "odds_decimal": 1.80,
            "timestamp": datetime(2025, 1, 14),
            "is_opening": False,
        },
    ]


@pytest.fixture
def sample_opening_odds():
    """Sample opening odds data."""
    return [
        {"bookmaker": "bet365", "market_type": "1x2", "selection": "home", "odds_decimal": 2.00},
        {"bookmaker": "bet365", "market_type": "1x2", "selection": "draw", "odds_decimal": 3.50},
        {"bookmaker": "bet365", "market_type": "1x2", "selection": "away", "odds_decimal": 3.60},
    ]


@pytest.fixture
def sample_standings():
    """Sample league standings data."""
    return [
        {
            "team_id": str(HOME_TEAM_ID),
            "team_name": "Home FC",
            "played": 20,
            "won": 12,
            "drawn": 4,
            "lost": 4,
            "goals_for": 35,
            "goals_against": 18,
            "goal_difference": 17,
            "points": 40,
        },
        {
            "team_id": str(uuid4()),
            "team_name": "Other Team",
            "played": 20,
            "won": 11,
            "drawn": 5,
            "lost": 4,
            "goals_for": 30,
            "goals_against": 20,
            "goal_difference": 10,
            "points": 38,
        },
        {
            "team_id": str(uuid4()),
            "team_name": "Third Team",
            "played": 20,
            "won": 10,
            "drawn": 5,
            "lost": 5,
            "goals_for": 28,
            "goals_against": 22,
            "goal_difference": 6,
            "points": 35,
        },
        {
            "team_id": str(AWAY_TEAM_ID),
            "team_name": "Away United",
            "played": 20,
            "won": 8,
            "drawn": 6,
            "lost": 6,
            "goals_for": 25,
            "goals_against": 25,
            "goal_difference": 0,
            "points": 30,
        },
    ]


@pytest.fixture
def sample_league_stats():
    """Sample league season stats."""
    return [
        {
            "avg_goals": 2.8,
            "avg_home_goals": 1.6,
            "avg_away_goals": 1.2,
            "std_goals": 1.5,
            "home_win_rate": 0.45,
            "draw_rate": 0.25,
        }
    ]


@pytest.fixture
def sample_referee_stats():
    """Sample referee statistics."""
    return [
        {
            "matches_refereed": 50,
            "avg_yellows": 4.2,
            "avg_reds": 0.15,
            "avg_fouls": 22.5,
            "avg_goals": 2.7,
        }
    ]
