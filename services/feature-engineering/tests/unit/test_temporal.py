"""Tests for temporal feature computer."""

from datetime import datetime, timezone

import pytest

from src.categories.temporal import TemporalFeatureComputer
from src.categories.base import MatchContext


class TestTemporalFeatureComputer:
    """Test temporal feature computation."""

    @pytest.fixture
    def computer(self, mock_config, mock_db):
        return TemporalFeatureComputer(mock_config, mock_db)

    def test_feature_count(self, computer):
        """Should produce exactly 15 temporal features."""
        assert computer.feature_count == 15

    def test_all_feature_names_have_prefix(self, computer):
        """All features should start with temporal__."""
        for name in computer.get_feature_names():
            assert name.startswith("temporal__"), f"Bad prefix: {name}"

    def test_compute_returns_all_features(self, computer, match_context):
        features = computer.compute(match_context)
        assert len(features) == 15
        assert all(v is not None for v in features.values())

    def test_day_of_week(self, computer, match_context):
        """Jan 15, 2025 is a Wednesday (dow=2)."""
        features = computer.compute(match_context)
        assert features["temporal__dow__raw"] == 2.0
        assert features["temporal__dow__is_weekend"] == 0.0

    def test_weekend_detection(self, computer, mock_config, mock_db):
        """Saturday should be detected as weekend."""
        ctx = MatchContext(
            match_id="00000000-0000-0000-0000-000000000000",
            home_team_id="11111111-1111-1111-1111-111111111111",
            away_team_id="22222222-2222-2222-2222-222222222222",
            league_id="44444444-4444-4444-4444-444444444444",
            season="2024-2025",
            match_date=datetime(2025, 1, 18, 15, 0, tzinfo=timezone.utc),  # Saturday
        )
        features = computer.compute(ctx)
        assert features["temporal__dow__raw"] == 5.0
        assert features["temporal__dow__is_weekend"] == 1.0

    def test_cyclical_encoding_ranges(self, computer, match_context):
        features = computer.compute(match_context)
        for key in ["temporal__dow__sin", "temporal__dow__cos",
                    "temporal__month__sin", "temporal__month__cos",
                    "temporal__hour__sin", "temporal__hour__cos",
                    "temporal__doy__sin", "temporal__doy__cos"]:
            assert -1.0 <= features[key] <= 1.0, f"{key}={features[key]}"

    def test_month(self, computer, match_context):
        features = computer.compute(match_context)
        assert features["temporal__month__raw"] == 1.0  # January

    def test_hour(self, computer, match_context):
        features = computer.compute(match_context)
        assert features["temporal__hour__raw"] == 15.0  # 3 PM

    def test_season_phase_mid_season(self, computer, match_context):
        """January should be mid-season."""
        features = computer.compute(match_context)
        phase = features["temporal__season__phase"]
        assert 0.4 < phase < 0.7  # Mid-season
        assert features["temporal__season__is_early"] == 0.0
        assert features["temporal__season__is_late"] == 0.0

    def test_season_phase_early(self, computer, mock_config, mock_db):
        ctx = MatchContext(
            match_id="00000000-0000-0000-0000-000000000000",
            home_team_id="11111111-1111-1111-1111-111111111111",
            away_team_id="22222222-2222-2222-2222-222222222222",
            league_id="44444444-4444-4444-4444-444444444444",
            season="2024-2025",
            match_date=datetime(2024, 8, 20, 15, 0, tzinfo=timezone.utc),  # August
        )
        features = computer.compute(ctx)
        assert features["temporal__season__is_early"] == 1.0

    def test_season_phase_late(self, computer, mock_config, mock_db):
        ctx = MatchContext(
            match_id="00000000-0000-0000-0000-000000000000",
            home_team_id="11111111-1111-1111-1111-111111111111",
            away_team_id="22222222-2222-2222-2222-222222222222",
            league_id="44444444-4444-4444-4444-444444444444",
            season="2024-2025",
            match_date=datetime(2025, 5, 10, 15, 0, tzinfo=timezone.utc),  # May
        )
        features = computer.compute(ctx)
        assert features["temporal__season__is_late"] == 1.0
