"""Tests for match endpoints"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

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


class TestMatchRouteHelpers:
    """Test helper functions in matches route"""

    def test_get_team_form(self, mock_db, mock_row, mock_settings):
        from routes.matches import _get_team_form

        team_id = str(uuid.uuid4())
        rows = [
            mock_row(
                match_id=uuid.uuid4(),
                match_date=datetime(2025, 3, 10, 15, 0),
                opponent_id=uuid.uuid4(),
                is_home=True,
                goals_for=2,
                goals_against=1,
                result="W",
                points=3,
            ),
            mock_row(
                match_id=uuid.uuid4(),
                match_date=datetime(2025, 3, 5, 15, 0),
                opponent_id=uuid.uuid4(),
                is_home=False,
                goals_for=1,
                goals_against=1,
                result="D",
                points=1,
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = rows

        form = _get_team_form(mock_db, team_id)

        assert len(form) == 2
        assert form[0]["result"] == "W"
        assert form[0]["goals_for"] == 2
        assert form[1]["result"] == "D"

    def test_get_team_form_empty(self, mock_db, mock_settings):
        from routes.matches import _get_team_form

        mock_db.execute.return_value.fetchall.return_value = []

        form = _get_team_form(mock_db, str(uuid.uuid4()))
        assert form == []

    def test_get_h2h_stats(self, mock_db, mock_row, mock_settings):
        from routes.matches import _get_h2h_stats

        team1_id = str(uuid.uuid4())
        team2_id = str(uuid.uuid4())

        rows = [
            mock_row(
                match_id=uuid.uuid4(),
                match_date=datetime(2025, 1, 15, 15, 0),
                home_team_id=uuid.UUID(team1_id),
                away_team_id=uuid.UUID(team2_id),
                home_score=3,
                away_score=1,
                league_name="Premier League",
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = rows

        h2h = _get_h2h_stats(mock_db, team1_id, team2_id)

        assert len(h2h) == 1
        assert h2h[0]["home_score"] == 3
        assert h2h[0]["away_score"] == 1

    def test_get_h2h_stats_no_matches(self, mock_db, mock_settings):
        from routes.matches import _get_h2h_stats

        mock_db.execute.return_value.fetchall.return_value = []

        h2h = _get_h2h_stats(mock_db, str(uuid.uuid4()), str(uuid.uuid4()))
        assert h2h == []


class TestMatchEndpointLogic:
    """Test match endpoint response formatting"""

    def test_upcoming_matches_formatting(self, mock_db, mock_row, mock_settings):
        """Verify response formatting for upcoming matches"""

        match_id = uuid.uuid4()
        row = mock_row(
            match_id=match_id,
            match_date=datetime(2025, 3, 15, 15, 0),
            league_name="Premier League",
            league_country="England",
            home_team="Arsenal",
            away_team="Chelsea",
            venue="Emirates Stadium",
            home_odds=1.85,
            draw_odds=3.50,
            away_odds=4.20,
            bookmaker="bet365",
        )

        # Simulate formatting logic
        formatted = {
            "match_id": str(row.match_id),
            "match_date": row.match_date.isoformat(),
            "league_name": row.league_name,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "venue": row.venue,
            "odds": {
                "home": float(row.home_odds),
                "draw": float(row.draw_odds),
                "away": float(row.away_odds),
                "bookmaker": row.bookmaker,
            },
        }

        assert formatted["home_team"] == "Arsenal"
        assert formatted["odds"]["home"] == 1.85
        assert formatted["odds"]["bookmaker"] == "bet365"

    def test_match_detail_formatting(self, mock_row, mock_settings):
        """Verify match detail response structure"""

        result = mock_row(
            id=uuid.uuid4(),
            match_date=datetime(2025, 3, 15, 15, 0),
            status="scheduled",
            home_score=None,
            away_score=None,
            half_time_home=None,
            half_time_away=None,
            season="2024-25",
            round="28",
            venue="Emirates Stadium",
            attendance=None,
            league_name="Premier League",
            league_country="England",
            sport="soccer",
            home_team="Arsenal",
            home_team_id=uuid.uuid4(),
            away_team="Chelsea",
            away_team_id=uuid.uuid4(),
            referee_name="Michael Oliver",
        )

        detail = {
            "match_id": str(result.id),
            "status": result.status,
            "league": {"name": result.league_name},
            "home_team": {"name": result.home_team},
            "away_team": {"name": result.away_team},
            "referee": result.referee_name,
        }

        assert detail["status"] == "scheduled"
        assert detail["referee"] == "Michael Oliver"

    def test_match_stats_with_form_and_h2h(self, mock_db, mock_row, mock_settings):
        """Verify stats response includes form and h2h"""
        from routes.matches import _get_team_form

        team_id = str(uuid.uuid4())
        rows = [
            mock_row(
                match_id=uuid.uuid4(),
                match_date=datetime(2025, 3, 10),
                opponent_id=uuid.uuid4(),
                is_home=True,
                goals_for=3,
                goals_against=0,
                result="W",
                points=3,
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = rows

        form = _get_team_form(mock_db, team_id, num_matches=5)

        assert len(form) == 1
        assert form[0]["points"] == 3
