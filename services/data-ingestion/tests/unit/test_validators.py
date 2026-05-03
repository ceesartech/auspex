"""Unit tests for data validators"""

from services.data_ingestion.src.validators.data_validator import DataValidator


class TestDataValidator:
    """Tests for DataValidator class"""

    def setup_method(self):
        """Create validator instance for each test"""
        self.validator = DataValidator()

    # --- Odds Validation ---

    def test_validate_odds_valid(self):
        """Test validation passes for valid odds data"""
        result = self.validator.validate_odds(
            {"match_id": 1, "bookmaker": "bet365", "market_type": "1X2", "outcome": "home", "odds_decimal": 2.50}
        )
        assert result is True

    def test_validate_odds_missing_field(self):
        """Test validation fails for missing required field"""
        result = self.validator.validate_odds(
            {
                "match_id": 1,
                "bookmaker": "bet365",
                "market_type": "1X2",
                # missing outcome and odds_decimal
            }
        )
        assert result is False

    def test_validate_odds_too_low(self):
        """Test validation fails for odds below minimum"""
        result = self.validator.validate_odds(
            {"match_id": 1, "bookmaker": "bet365", "market_type": "1X2", "outcome": "home", "odds_decimal": 0.50}
        )
        assert result is False

    def test_validate_odds_too_high(self):
        """Test validation fails for odds above maximum"""
        result = self.validator.validate_odds(
            {"match_id": 1, "bookmaker": "bet365", "market_type": "1X2", "outcome": "home", "odds_decimal": 1500}
        )
        assert result is False

    def test_validate_odds_invalid_bookmaker(self):
        """Test validation fails for invalid bookmaker"""
        result = self.validator.validate_odds(
            {
                "match_id": 1,
                "bookmaker": "invalid_bookie",
                "market_type": "1X2",
                "outcome": "home",
                "odds_decimal": 2.50,
            }
        )
        assert result is False

    def test_validate_odds_betmgm(self):
        """Test validation passes for BetMGM"""
        result = self.validator.validate_odds(
            {"match_id": 1, "bookmaker": "betmgm", "market_type": "1X2", "outcome": "draw", "odds_decimal": 3.25}
        )
        assert result is True

    # --- Match Validation ---

    def test_validate_match_valid(self):
        """Test validation passes for valid match data"""
        result = self.validator.validate_match(
            {"home_team_id": 1, "away_team_id": 2, "match_date": "2024-12-15", "league_id": 1}
        )
        assert result is True

    def test_validate_match_missing_field(self):
        """Test validation fails for missing required field"""
        result = self.validator.validate_match(
            {
                "home_team_id": 1,
                # missing away_team_id, match_date, league_id
            }
        )
        assert result is False

    def test_validate_match_same_teams(self):
        """Test validation fails when home and away teams are the same"""
        result = self.validator.validate_match(
            {"home_team_id": 1, "away_team_id": 1, "match_date": "2024-12-15", "league_id": 1}
        )
        assert result is False

    def test_validate_match_different_teams(self):
        """Test validation passes for different teams"""
        result = self.validator.validate_match(
            {"home_team_id": 5, "away_team_id": 10, "match_date": "2024-12-15", "league_id": 3}
        )
        assert result is True
