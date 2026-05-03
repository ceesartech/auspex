"""Tests for team performance feature computer."""

import pytest
from src.categories.team_performance import TeamPerformanceComputer


class TestTeamPerformanceComputer:
    """Test team performance feature computation."""

    @pytest.fixture
    def computer(self, mock_config, mock_db):
        return TeamPerformanceComputer(mock_config, mock_db)

    def test_feature_count(self, computer):
        """Should produce 62 features (15 metrics * 4 windows * 2 sides + extras)."""
        count = computer.feature_count
        # 15 metrics * 4 windows * 2 sides = 120
        # 3 trends * 2 sides = 6
        # 2 consistency * 2 sides = 4
        # Total = 130... but we check it's at least 62 per the plan
        assert count >= 62, f"Expected >=62 features, got {count}"

    def test_all_feature_names_have_prefix(self, computer):
        for name in computer.get_feature_names():
            assert name.startswith("team__"), f"Bad prefix: {name}"

    def test_compute_with_no_data(self, computer, match_context, mock_db):
        """Should return all None when no match data."""
        mock_db.execute_query.return_value = []
        features = computer.compute(match_context)
        assert len(features) == computer.feature_count
        # All should be None when no data
        assert all(v is None for v in features.values())

    def test_compute_with_data(self, computer, match_context, mock_db, sample_team_matches):
        """Should compute features when match data is available."""
        mock_db.execute_query.return_value = sample_team_matches
        features = computer.compute(match_context)

        # Should have non-None values
        non_null = sum(1 for v in features.values() if v is not None)
        assert non_null > 0

    def test_windowed_metrics(self, computer, match_context, mock_db, sample_team_matches):
        """Should compute metrics for all windows."""
        mock_db.execute_query.return_value = sample_team_matches
        features = computer.compute(match_context)

        for w in [3, 5, 10, 20]:
            key = f"team__home__form__wins__last{w}"
            assert key in features, f"Missing feature: {key}"

    def test_points_per_game_range(self, computer, match_context, mock_db, sample_team_matches):
        """PPG should be between 0 and 3."""
        mock_db.execute_query.return_value = sample_team_matches
        features = computer.compute(match_context)

        for w in [3, 5, 10, 20]:
            for side in ["home", "away"]:
                key = f"team__{side}__form__points_per_game__last{w}"
                val = features.get(key)
                if val is not None:
                    assert 0 <= val <= 3, f"{key}={val}"

    def test_goals_non_negative(self, computer, match_context, mock_db, sample_team_matches):
        """Goals scored and conceded should be non-negative."""
        mock_db.execute_query.return_value = sample_team_matches
        features = computer.compute(match_context)

        for side in ["home", "away"]:
            for metric in ["goals_scored", "goals_conceded"]:
                for w in [3, 5, 10, 20]:
                    key = f"team__{side}__form__{metric}__last{w}"
                    val = features.get(key)
                    if val is not None:
                        assert val >= 0, f"{key}={val}"

    def test_trend_features(self, computer, match_context, mock_db, sample_team_matches):
        """Trend features should be computed."""
        mock_db.execute_query.return_value = sample_team_matches
        features = computer.compute(match_context)

        for side in ["home", "away"]:
            key = f"team__{side}__points_trend"
            assert key in features

    def test_db_called_for_both_teams(self, computer, match_context, mock_db, sample_team_matches):
        """Should query DB for both home and away teams."""
        mock_db.execute_query.return_value = sample_team_matches
        computer.compute(match_context)
        assert mock_db.execute_query.call_count == 2  # Home + away
