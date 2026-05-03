"""Data validation using Great Expectations"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate scraped data before database insertion"""

    def validate_odds(self, odds_data: Dict) -> bool:
        """Validate odds data"""

        required_fields = ["match_id", "bookmaker", "market_type", "outcome", "odds_decimal"]

        # Check required fields
        for field in required_fields:
            if field not in odds_data:
                logger.error(f"Missing required field: {field}")
                return False

        # Validate odds range
        if odds_data["odds_decimal"] < 1.01 or odds_data["odds_decimal"] > 1000:
            logger.error(f"Invalid odds: {odds_data['odds_decimal']}")
            return False

        # Validate bookmaker
        valid_bookmakers = ["bet365", "betmgm", "draftkings", "fanduel"]
        if odds_data["bookmaker"] not in valid_bookmakers:
            logger.error(f"Invalid bookmaker: {odds_data['bookmaker']}")
            return False

        return True

    def validate_match(self, match_data: Dict) -> bool:
        """Validate match data"""

        required_fields = ["home_team_id", "away_team_id", "match_date", "league_id"]

        for field in required_fields:
            if field not in match_data:
                logger.error(f"Missing required field: {field}")
                return False

        # Check teams are different
        if match_data["home_team_id"] == match_data["away_team_id"]:
            logger.error("Home and away teams cannot be the same")
            return False

        return True
