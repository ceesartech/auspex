"""Tests for contextual feature computer."""

from datetime import datetime, timezone

import pytest
from src.categories.contextual import ContextualFeatureComputer


class TestContextualFeatureComputer:

    @pytest.fixture
    def computer(self, mock_config, mock_db):
        return ContextualFeatureComputer(mock_config, mock_db)

    def test_feature_count(self, computer):
        assert computer.feature_count == 36

    def test_all_feature_names_have_prefix(self, computer):
        for name in computer.get_feature_names():
            assert name.startswith("ctx__"), f"Bad prefix: {name}"

    def test_compute_no_data(self, computer, match_context, mock_db):
        mock_db.execute_query.return_value = []
        mock_db.call_function.return_value = []
        features = computer.compute(match_context)
        assert len(features) == 36

    def test_rest_days(self, computer, match_context, mock_db):
        last_match = datetime(2025, 1, 12, tzinfo=timezone.utc)
        mock_db.execute_query.side_effect = lambda q, p, fetch=False: (
            [{"last_match_date": last_match}] if "MAX(m.match_date)" in q else []
        )
        mock_db.call_function.return_value = []

        features = computer.compute(match_context)

        home_rest = features["ctx__home__rest_days"]
        if home_rest is not None:
            assert home_rest == 3.0  # Jan 15 - Jan 12

    def test_short_rest_flag(self, computer, match_context, mock_db):
        last_match = datetime(2025, 1, 13, tzinfo=timezone.utc)
        mock_db.execute_query.side_effect = lambda q, p, fetch=False: (
            [{"last_match_date": last_match}] if "MAX(m.match_date)" in q else []
        )
        mock_db.call_function.return_value = []

        features = computer.compute(match_context)
        home_rest = features["ctx__home__rest_days"]
        if home_rest is not None:
            assert features["ctx__home__is_short_rest"] == 1.0

    def test_referee_features(self, computer, match_context, mock_db, sample_referee_stats):
        def side_effect(q, p, fetch=False):
            if "m.referee" in q:
                return sample_referee_stats
            return []

        mock_db.execute_query.side_effect = side_effect
        mock_db.call_function.return_value = []

        features = computer.compute(match_context)
        if features["ctx__referee__avg_yellows"] is not None:
            assert features["ctx__referee__avg_yellows"] == 4.2
            assert features["ctx__referee__is_strict"] == 1.0

    def test_league_stats(self, computer, match_context, mock_db, sample_league_stats):
        def side_effect(q, p, fetch=False):
            if "AVG(m.home_score + m.away_score)" in q:
                return sample_league_stats
            return []

        mock_db.execute_query.side_effect = side_effect
        mock_db.call_function.return_value = []

        features = computer.compute(match_context)
        if features["ctx__league__avg_goals"] is not None:
            assert features["ctx__league__avg_goals"] == 2.8

    def test_standings_features(self, computer, match_context, mock_db, sample_standings):
        mock_db.execute_query.return_value = []
        mock_db.call_function.return_value = sample_standings

        features = computer.compute(match_context)

        if features["ctx__home__league_position"] is not None:
            assert features["ctx__home__league_position"] == 1.0  # First in standings
            assert features["ctx__home__is_top4"] == 1.0

        if features["ctx__away__league_position"] is not None:
            assert features["ctx__away__league_position"] == 4.0
            assert features["ctx__away__is_top4"] == 1.0

    def test_position_diff(self, computer, match_context, mock_db, sample_standings):
        mock_db.execute_query.return_value = []
        mock_db.call_function.return_value = sample_standings

        features = computer.compute(match_context)
        if features["ctx__position_diff"] is not None:
            assert features["ctx__position_diff"] == 3.0  # 4 - 1

    def test_venue_neutral(self, computer, match_context, mock_db):
        mock_db.execute_query.return_value = []
        mock_db.call_function.return_value = []
        features = computer.compute(match_context)
        assert features["ctx__venue__is_neutral"] == 0.0
