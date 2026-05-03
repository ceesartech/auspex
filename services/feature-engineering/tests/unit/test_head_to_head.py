"""Tests for head-to-head feature computer."""

import pytest
from src.categories.head_to_head import HeadToHeadComputer


class TestHeadToHeadComputer:

    @pytest.fixture
    def computer(self, mock_config, mock_db):
        return HeadToHeadComputer(mock_config, mock_db)

    def test_feature_count(self, computer):
        assert computer.feature_count == 22

    def test_all_feature_names_have_prefix(self, computer):
        for name in computer.get_feature_names():
            assert name.startswith("h2h__"), f"Bad prefix: {name}"

    def test_compute_no_h2h_data(self, computer, match_context, mock_db):
        mock_db.execute_query.return_value = []
        features = computer.compute(match_context)
        assert features["h2h__total_matches"] == 0.0

    def test_compute_with_data(self, computer, match_context, mock_db, sample_h2h_matches):
        mock_db.execute_query.return_value = sample_h2h_matches
        features = computer.compute(match_context)

        assert features["h2h__total_matches"] == 10.0
        assert features["h2h__home_wins"] is not None
        assert features["h2h__away_wins"] is not None
        assert features["h2h__draws"] is not None

        # Sum of results should equal total matches
        total = features["h2h__home_wins"] + features["h2h__away_wins"] + features["h2h__draws"]
        assert total == 10.0

    def test_rates_sum_to_one(self, computer, match_context, mock_db, sample_h2h_matches):
        mock_db.execute_query.return_value = sample_h2h_matches
        features = computer.compute(match_context)

        rate_sum = features["h2h__home_win_rate"] + features["h2h__away_win_rate"] + features["h2h__draw_rate"]
        assert abs(rate_sum - 1.0) < 0.01

    def test_rates_in_range(self, computer, match_context, mock_db, sample_h2h_matches):
        mock_db.execute_query.return_value = sample_h2h_matches
        features = computer.compute(match_context)

        for key in ["h2h__home_win_rate", "h2h__away_win_rate", "h2h__draw_rate", "h2h__btts_rate", "h2h__over25_rate"]:
            val = features[key]
            assert 0 <= val <= 1, f"{key}={val}"

    def test_recent3_max_values(self, computer, match_context, mock_db, sample_h2h_matches):
        mock_db.execute_query.return_value = sample_h2h_matches
        features = computer.compute(match_context)

        assert 0 <= features["h2h__recent3__home_wins"] <= 3
        assert 0 <= features["h2h__recent3__away_wins"] <= 3

    def test_days_since_last(self, computer, match_context, mock_db, sample_h2h_matches):
        mock_db.execute_query.return_value = sample_h2h_matches
        features = computer.compute(match_context)
        assert features["h2h__days_since_last"] >= 0

    def test_ewa_goals_non_negative(self, computer, match_context, mock_db, sample_h2h_matches):
        mock_db.execute_query.return_value = sample_h2h_matches
        features = computer.compute(match_context)
        assert features["h2h__ewa_home_goals"] >= 0
        assert features["h2h__ewa_away_goals"] >= 0
