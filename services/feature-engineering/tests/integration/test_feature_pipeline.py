"""Integration test for the full feature pipeline.

Uses mocked DB/Redis but exercises the full computation flow
from orchestrator through all categories and derived features.
"""

import json
from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.core.config import FeatureConfig
from src.core.database import DatabaseManager
from src.orchestrator import RealTimeFeatureComputer
from tests.conftest import (
    AWAY_TEAM_ID,
    HOME_TEAM_ID,
    LEAGUE_ID,
    MATCH_ID,
    PLAYER_1_ID,
)


class TestFeaturePipelineIntegration:
    """Full pipeline integration test with mock data."""

    @pytest.fixture
    def full_db_mock(
        self,
        sample_team_matches,
        sample_h2h_matches,
        sample_squad,
        sample_availability,
        sample_player_stats,
        sample_odds,
        sample_opening_odds,
        sample_standings,
        sample_league_stats,
        sample_referee_stats,
    ):
        """DB mock that returns appropriate data based on query content."""
        db = Mock(spec=DatabaseManager)

        def side_effect(query, params=None, fetch=False):
            q = query.strip()

            # Match details
            if "JOIN leagues" in q and "JOIN teams" in q:
                return [{
                    "match_id": MATCH_ID,
                    "home_team_id": HOME_TEAM_ID,
                    "away_team_id": AWAY_TEAM_ID,
                    "league_id": LEAGUE_ID,
                    "season": "2024-2025",
                    "match_date": datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc),
                    "venue": "Home Stadium",
                    "referee": "Michael Oliver",
                    "home_team_name": "Home FC",
                    "away_team_name": "Away United",
                    "league_name": "Premier League",
                    "country": "England",
                }]
            # Team matches
            if "FROM matches m" in q and "LEFT JOIN match_stats" in q and "home_team_id = %s OR" in q:
                return sample_team_matches
            # H2H
            if "home_team_id = %s AND m.away_team_id = %s" in q:
                return sample_h2h_matches
            # Squad
            if "FROM players p" in q and "team_id" in q:
                return sample_squad
            # Availability
            if "player_availability" in q:
                return sample_availability
            # Player recent stats
            if "player_match_stats" in q:
                return sample_player_stats
            # Latest odds
            if "DISTINCT ON" in q:
                return sample_odds
            # Opening odds
            if "is_opening" in q:
                return sample_opening_odds
            # All odds
            if "FROM odds" in q:
                return sample_odds
            # Last match date
            if "MAX(m.match_date)" in q:
                return [{"last_match_date": datetime(2025, 1, 10, tzinfo=timezone.utc)}]
            # Referee
            if "m.referee" in q:
                return sample_referee_stats
            # League stats
            if "AVG(m.home_score + m.away_score)" in q:
                return sample_league_stats
            # Cache queries
            if "features_cache" in q:
                return []

            return []

        db.execute_query.side_effect = side_effect
        db.call_function.return_value = sample_standings
        return db

    @pytest.fixture
    def pipeline(self, mock_config, full_db_mock, mock_redis):
        mock_redis.get.return_value = None  # No cache hit
        return RealTimeFeatureComputer(mock_config, full_db_mock, mock_redis)

    def test_full_pipeline_feature_count(self, pipeline):
        """Orchestrator should report 250+ registered features."""
        assert pipeline.get_feature_count() >= 300

    def test_full_pipeline_computation(self, pipeline):
        """Full pipeline should compute features without errors."""
        features = pipeline.compute_features(
            str(MATCH_ID), use_cache=False, validate=True
        )
        assert isinstance(features, dict)
        assert len(features) >= 300

    def test_full_pipeline_no_none_heavy(self, pipeline):
        """Most features should be non-None with full mock data."""
        features = pipeline.compute_features(
            str(MATCH_ID), use_cache=False
        )
        non_null = sum(1 for v in features.values() if v is not None)
        total = len(features)
        ratio = non_null / total
        # With full mock data, at least 40% should be non-null
        assert ratio >= 0.4, f"Only {ratio:.1%} non-null ({non_null}/{total})"

    def test_full_pipeline_categories_present(self, pipeline):
        """Features from all categories should be present."""
        features = pipeline.compute_features(
            str(MATCH_ID), use_cache=False
        )
        prefixes = {"team__", "h2h__", "player__", "ctx__", "odds__",
                     "temporal__", "derived__"}
        found_prefixes = set()
        for name in features:
            for prefix in prefixes:
                if name.startswith(prefix):
                    found_prefixes.add(prefix)

        assert found_prefixes == prefixes, (
            f"Missing prefixes: {prefixes - found_prefixes}"
        )

    def test_full_pipeline_temporal_always_present(self, pipeline):
        """Temporal features should always be non-None."""
        features = pipeline.compute_features(
            str(MATCH_ID), use_cache=False
        )
        temporal = {k: v for k, v in features.items() if k.startswith("temporal__")}
        assert all(v is not None for v in temporal.values()), (
            f"Some temporal features are None: "
            f"{[k for k, v in temporal.items() if v is None]}"
        )

    def test_full_pipeline_cache_write(self, pipeline, mock_redis):
        """Should write to cache after computation."""
        pipeline.compute_features(str(MATCH_ID), use_cache=True)
        # Redis setex should have been called (cache set)
        assert mock_redis.setex.called

    def test_feature_names_all_valid_format(self, pipeline):
        """All feature names should follow naming convention."""
        features = pipeline.compute_features(
            str(MATCH_ID), use_cache=False
        )
        for name in features:
            parts = name.split("__")
            assert len(parts) >= 2, f"Feature name '{name}' has < 2 parts"
