"""Tests for derived feature computer."""

import pytest
from src.categories.derived import DerivedFeatureComputer


class TestDerivedFeatureComputer:

    @pytest.fixture
    def computer(self, mock_config, mock_db):
        return DerivedFeatureComputer(mock_config, mock_db)

    @pytest.fixture
    def prior_features(self):
        """Simulated phase-1 output features."""
        return {
            # Team performance
            "team__home__form__points_per_game__last3": 2.33,
            "team__home__form__points_per_game__last5": 2.0,
            "team__home__form__points_per_game__last10": 1.8,
            "team__home__form__goals_scored__last3": 7.0,
            "team__home__form__goals_scored__last5": 10.0,
            "team__home__form__goals_conceded__last3": 2.0,
            "team__home__form__goals_conceded__last5": 5.0,
            "team__home__form__avg_xg__last3": 2.1,
            "team__home__form__avg_xg__last5": 1.8,
            "team__home__form__avg_xg_against__last5": 0.9,
            "team__home__form__avg_shots__last5": 14.0,
            "team__home__form__avg_possession__last5": 58.0,
            "team__home__form__clean_sheets__last5": 2.0,
            "team__away__form__points_per_game__last3": 1.33,
            "team__away__form__points_per_game__last5": 1.4,
            "team__away__form__points_per_game__last10": 1.5,
            "team__away__form__goals_scored__last3": 3.0,
            "team__away__form__goals_scored__last5": 6.0,
            "team__away__form__goals_conceded__last3": 4.0,
            "team__away__form__goals_conceded__last5": 8.0,
            "team__away__form__avg_xg__last3": 1.1,
            "team__away__form__avg_xg__last5": 1.2,
            "team__away__form__avg_xg_against__last5": 1.5,
            "team__away__form__avg_shots__last5": 10.0,
            "team__away__form__avg_possession__last5": 45.0,
            "team__away__form__clean_sheets__last5": 1.0,
            # H2H
            "h2h__home_win_rate": 0.6,
            "h2h__away_win_rate": 0.2,
            # Odds
            "odds__implied_prob__home": 0.48,
            "odds__implied_prob__draw": 0.27,
            "odds__implied_prob__away": 0.25,
            # Context
            "ctx__league__avg_goals": 2.8,
            "ctx__home__rest_days": 7.0,
            "ctx__away__rest_days": 4.0,
            "ctx__position_diff": 5.0,
            "ctx__home__league_position": 3.0,
            "ctx__away__league_position": 8.0,
            # Player
            "player__home__squad_value_total": 500000000,
            "player__away__squad_value_total": 200000000,
            "player__home__unavailable_pct": 0.05,
            "player__away__unavailable_pct": 0.15,
        }

    def test_feature_count(self, computer):
        # Should have ~95 derived features
        assert computer.feature_count == 60

    def test_all_feature_names_have_prefix(self, computer):
        for name in computer.get_feature_names():
            assert name.startswith("derived__"), f"Bad prefix: {name}"

    def test_compute_empty_priors(self, computer, match_context):
        """Should handle empty prior features gracefully."""
        features = computer.compute(match_context, prior_features={})
        assert len(features) == computer.feature_count

    def test_elo_ratings(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)

        # Home has better PPG, should have higher Elo
        home_elo = features["derived__elo__home_rating"]
        away_elo = features["derived__elo__away_rating"]
        assert home_elo is not None
        assert away_elo is not None
        assert home_elo > away_elo

        # Expected score should be > 0.5 for better team
        assert features["derived__elo__home_expected"] > 0.5

    def test_poisson_probabilities(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)

        p_home = features["derived__poisson__home_win"]
        p_draw = features["derived__poisson__draw"]
        p_away = features["derived__poisson__away_win"]

        if all(v is not None for v in [p_home, p_draw, p_away]):
            # Should sum to ~1
            assert abs(p_home + p_draw + p_away - 1.0) < 0.01
            # All probabilities in [0, 1]
            assert 0 <= p_home <= 1
            assert 0 <= p_draw <= 1
            assert 0 <= p_away <= 1

    def test_poisson_over25(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)
        over25 = features["derived__poisson__over25"]
        if over25 is not None:
            assert 0 <= over25 <= 1

    def test_momentum_features(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)

        for side in ["home", "away"]:
            assert features[f"derived__momentum__{side}__attack"] is not None
            assert features[f"derived__momentum__{side}__defense"] is not None

    def test_differentials(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)

        xg_diff = features["derived__diff__xg"]
        assert xg_diff is not None
        # Home xG 1.8, Away xG 1.2, diff = 0.6
        assert xg_diff == pytest.approx(0.6, abs=0.01)

    def test_squad_value_ratio(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)
        ratio = features["derived__value__squad_ratio"]
        assert ratio is not None
        assert ratio == pytest.approx(2.5, abs=0.01)

    def test_model_agreement(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)
        for key in [
            "derived__agreement__elo_vs_odds",
            "derived__agreement__poisson_vs_odds",
        ]:
            val = features[key]
            if val is not None:
                assert 0 <= val <= 1, f"{key}={val}"

    def test_composite_scores(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)
        h_str = features["derived__composite__home_strength"]
        a_str = features["derived__composite__away_strength"]
        if h_str is not None and a_str is not None:
            assert h_str > a_str  # Home team is stronger

    def test_data_completeness(self, computer, match_context, prior_features):
        features = computer.compute(match_context, prior_features=prior_features)
        completeness = features["derived__confidence__data_completeness"]
        assert completeness is not None
        assert 0 <= completeness <= 1
        assert completeness > 0.5  # Most priors are non-null
