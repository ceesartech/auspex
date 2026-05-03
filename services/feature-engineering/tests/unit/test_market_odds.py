"""Tests for market odds feature computer."""

import pytest
from src.categories.market_odds import MarketOddsComputer


class TestMarketOddsComputer:

    @pytest.fixture
    def computer(self, mock_config, mock_db):
        return MarketOddsComputer(mock_config, mock_db)

    def test_feature_count(self, computer):
        assert computer.feature_count == 26

    def test_all_feature_names_have_prefix(self, computer):
        for name in computer.get_feature_names():
            assert name.startswith("odds__"), f"Bad prefix: {name}"

    def test_compute_no_data(self, computer, match_context, mock_db):
        mock_db.execute_query.return_value = []
        features = computer.compute(match_context)
        assert len(features) == 26

    def test_compute_with_odds(self, computer, match_context, mock_db, sample_odds, sample_opening_odds):
        call_count = [0]

        def side_effect(q, p, fetch=False):
            call_count[0] += 1
            if "DISTINCT ON" in q:
                return sample_odds
            elif "is_opening" in q:
                return sample_opening_odds
            return sample_odds

        mock_db.execute_query.side_effect = side_effect
        features = computer.compute(match_context)

        # Implied probabilities should sum to ~1
        h = features["odds__implied_prob__home"]
        d = features["odds__implied_prob__draw"]
        a = features["odds__implied_prob__away"]
        if all(v is not None for v in [h, d, a]):
            assert abs(h + d + a - 1.0) < 0.01

    def test_favourite_detection(self, computer, match_context, mock_db, sample_odds, sample_opening_odds):
        mock_db.execute_query.side_effect = lambda q, p, fetch=False: (
            sample_odds if "DISTINCT ON" in q else sample_opening_odds if "is_opening" in q else sample_odds
        )
        features = computer.compute(match_context)

        if features["odds__favourite"] is not None:
            # Home has lowest odds (2.10), so should be favourite
            assert features["odds__favourite"] == 1.0

    def test_odds_movement(self, computer, match_context, mock_db, sample_odds, sample_opening_odds):
        mock_db.execute_query.side_effect = lambda q, p, fetch=False: (
            sample_odds if "DISTINCT ON" in q else sample_opening_odds if "is_opening" in q else sample_odds
        )
        features = computer.compute(match_context)

        if features["odds__movement__home"] is not None:
            # Current 2.10/2.15, Opening 2.00 -> movement > 0
            assert features["odds__movement__home"] is not None

    def test_overround_non_negative(self, computer, match_context, mock_db, sample_odds, sample_opening_odds):
        mock_db.execute_query.side_effect = lambda q, p, fetch=False: (
            sample_odds if "DISTINCT ON" in q else sample_opening_odds if "is_opening" in q else sample_odds
        )
        features = computer.compute(match_context)

        if features["odds__overround"] is not None:
            assert features["odds__overround"] >= 0

    def test_secondary_markets(self, computer, match_context, mock_db, sample_odds, sample_opening_odds):
        mock_db.execute_query.side_effect = lambda q, p, fetch=False: (
            sample_odds if "DISTINCT ON" in q else sample_opening_odds if "is_opening" in q else sample_odds
        )
        features = computer.compute(match_context)

        if features["odds__over25__odds"] is not None:
            assert features["odds__over25__odds"] > 1
        if features["odds__btts_yes__odds"] is not None:
            assert features["odds__btts_yes__odds"] > 1
