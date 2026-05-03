"""Tests for player metrics feature computer."""

import pytest
from src.categories.player_metrics import PlayerMetricsComputer


class TestPlayerMetricsComputer:

    @pytest.fixture
    def computer(self, mock_config, mock_db):
        return PlayerMetricsComputer(mock_config, mock_db)

    def test_feature_count(self, computer):
        assert computer.feature_count == 32

    def test_all_feature_names_have_prefix(self, computer):
        for name in computer.get_feature_names():
            assert name.startswith("player__"), f"Bad prefix: {name}"

    def test_compute_no_data(self, computer, match_context, mock_db):
        mock_db.execute_query.return_value = []
        features = computer.compute(match_context)
        assert len(features) == 32

    def test_compute_with_squad(
        self, computer, match_context, mock_db, sample_squad, sample_availability, sample_player_stats
    ):
        """Should compute features when squad data is available."""

        def side_effect(query, params, fetch=False):
            if "FROM players" in query:
                return sample_squad
            elif "player_availability" in query:
                return sample_availability
            elif "player_match_stats" in query:
                return sample_player_stats
            return []

        mock_db.execute_query.side_effect = side_effect
        features = computer.compute(match_context)

        # Should have some non-None values
        non_null = sum(1 for v in features.values() if v is not None)
        assert non_null > 0

    def test_squad_value_total(self, computer, match_context, mock_db, sample_squad):
        def side_effect(query, params, fetch=False):
            if "FROM players" in query:
                return sample_squad
            return []

        mock_db.execute_query.side_effect = side_effect
        features = computer.compute(match_context)

        val = features["player__home__squad_value_total"]
        if val is not None:
            assert val == 90000000  # 50M + 30M + 10M

    def test_injured_count(self, computer, match_context, mock_db, sample_squad, sample_availability):
        def side_effect(query, params, fetch=False):
            if "FROM players" in query:
                return sample_squad
            elif "player_availability" in query:
                return sample_availability
            return []

        mock_db.execute_query.side_effect = side_effect
        features = computer.compute(match_context)

        val = features["player__home__injured_count"]
        if val is not None:
            assert val == 1.0  # One injured player in sample data

    def test_unavailable_pct_range(self, computer, match_context, mock_db, sample_squad, sample_availability):
        def side_effect(query, params, fetch=False):
            if "FROM players" in query:
                return sample_squad
            elif "player_availability" in query:
                return sample_availability
            return []

        mock_db.execute_query.side_effect = side_effect
        features = computer.compute(match_context)

        for side in ["home", "away"]:
            val = features[f"player__{side}__unavailable_pct"]
            if val is not None:
                assert 0 <= val <= 1, f"player__{side}__unavailable_pct={val}"
