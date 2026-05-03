"""Tests for the feature orchestrator."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from src.orchestrator import RealTimeFeatureComputer
from tests.conftest import HOME_TEAM_ID, AWAY_TEAM_ID, LEAGUE_ID, MATCH_ID


class TestRealTimeFeatureComputer:

    @pytest.fixture
    def match_row(self):
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

    @pytest.fixture
    def orchestrator(self, mock_config, mock_db, mock_redis):
        return RealTimeFeatureComputer(mock_config, mock_db, mock_redis)

    def test_total_feature_count(self, orchestrator):
        """Should register 250+ features across all categories."""
        count = orchestrator.get_feature_count()
        assert count >= 300, f"Expected >=300 features, got {count}"

    def test_category_counts(self, orchestrator):
        counts = orchestrator.get_category_counts()
        assert "team" in counts
        assert "h2h" in counts
        assert "player" in counts
        assert "ctx" in counts
        assert "odds" in counts
        assert "temporal" in counts
        assert "derived" in counts

    def test_feature_names_unique(self, orchestrator):
        names = orchestrator.get_feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_compute_features_with_cache_hit(
        self, orchestrator, mock_redis, mock_db
    ):
        """Should return cached features without computing."""
        cached = {"f1": 1.0, "f2": 2.0}
        mock_redis.get.return_value = json.dumps(cached)

        result = orchestrator.compute_features(str(MATCH_ID))
        assert result == cached
        # Should not query for match details
        # (only cache get query, no MATCH_DETAILS query)

    def test_compute_features_cache_miss(
        self, orchestrator, mock_redis, mock_db, match_row
    ):
        """Should compute features when cache misses."""
        mock_redis.get.return_value = None

        # First call returns match details, subsequent return empty
        call_count = [0]

        def db_side_effect(q, p=None, fetch=False):
            call_count[0] += 1
            if "FROM matches m" in q and "JOIN leagues" in q:
                return match_row
            return []

        mock_db.execute_query.side_effect = db_side_effect
        mock_db.call_function.return_value = []

        result = orchestrator.compute_features(str(MATCH_ID))
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_compute_features_no_match(
        self, orchestrator, mock_redis, mock_db
    ):
        """Should return empty dict if match not found."""
        mock_redis.get.return_value = None
        mock_db.execute_query.return_value = []

        result = orchestrator.compute_features(str(MATCH_ID))
        assert result == {}

    def test_compute_features_no_cache(
        self, orchestrator, mock_redis, mock_db, match_row
    ):
        """Should skip cache when use_cache=False."""
        mock_db.execute_query.side_effect = lambda q, p=None, fetch=False: (
            match_row if "JOIN leagues" in q else []
        )
        mock_db.call_function.return_value = []

        result = orchestrator.compute_features(
            str(MATCH_ID), use_cache=False
        )
        # Redis get should not be called
        mock_redis.get.assert_not_called()

    def test_build_context(self, orchestrator, mock_db, match_row):
        mock_db.execute_query.return_value = match_row
        ctx = orchestrator._build_context(str(MATCH_ID))

        assert ctx is not None
        assert ctx.match_id == MATCH_ID
        assert ctx.home_team_name == "Home FC"
        assert ctx.season == "2024-2025"

    def test_build_context_not_found(self, orchestrator, mock_db):
        mock_db.execute_query.return_value = []
        ctx = orchestrator._build_context(str(MATCH_ID))
        assert ctx is None

    def test_safe_compute_handles_error(self, orchestrator, match_context):
        """Category failure should return empty features, not crash."""
        broken_computer = Mock()
        broken_computer.compute.side_effect = Exception("Boom")
        broken_computer._empty_features.return_value = {"f1": None}
        broken_computer.get_feature_names.return_value = ["f1"]

        result = orchestrator._safe_compute("broken", broken_computer, match_context)
        assert result == {"f1": None}

    def test_compute_batch(self, orchestrator, mock_redis, mock_db, match_row):
        mock_redis.get.return_value = None
        mock_db.execute_query.side_effect = lambda q, p=None, fetch=False: (
            match_row if "JOIN leagues" in q else []
        )
        mock_db.call_function.return_value = []

        results = orchestrator.compute_batch([str(MATCH_ID)])
        assert str(MATCH_ID) in results
