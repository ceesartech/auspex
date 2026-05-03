"""BetMGM odds scraper"""

import json
import logging
from datetime import datetime
from typing import Dict

from ..core.config import BetMGMConfig
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class BetMGMScraper(BaseScraper):
    """Scrape odds from BetMGM"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: BetMGMConfig = self.config

    def scrape(self) -> int:
        """Scrape all odds from BetMGM"""

        # BetMGM has a JSON API that can be accessed
        api_url = "https://sports.co.betmgm.com/cds-api/bettingoffer/fixtures"

        params = {
            "x-bwin-accessid": "YjFkZGM1ODEtYzI2Ni00YTU5LWEzNTAtODVhMzQ2NmI4Nzhm",
            "lang": "en-us",
            "country": "US",
            "offerMapping": "All",
            "scoreboardMode": "Full",
            "fixtureTypes": "Standard",
            "sportIds": "1",  # Soccer
        }

        html = self._fetch_page(api_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()]))

        try:
            data = json.loads(html)
            fixtures = data.get("fixtures", [])

            odds_count = 0
            for fixture in fixtures:
                odds_count += self._process_fixture(fixture)

            return odds_count

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse BetMGM JSON: {e}")
            return 0

    def _process_fixture(self, fixture: Dict) -> int:
        """Process a single fixture and extract odds"""

        try:
            # Extract match info
            home_team = fixture["participants"][0]["name"]
            away_team = fixture["participants"][1]["name"]
            match_time = datetime.fromisoformat(fixture["startDate"].replace("Z", "+00:00"))

            # Get or create match
            match_id = self._get_or_create_match(home_team, away_team, match_time)

            # Extract odds from markets
            markets = fixture.get("optionMarkets", [])
            odds_count = 0

            for market in markets:
                market_name = market.get("name", {}).get("value", "")

                if "1X2" in market_name or "Match Result" in market_name:
                    odds_count += self._save_1x2_odds(match_id, market)

                elif "Total Goals" in market_name or "Over/Under" in market_name:
                    odds_count += self._save_over_under_odds(match_id, market)

                elif "Both Teams to Score" in market_name:
                    odds_count += self._save_btts_odds(match_id, market)

            return odds_count

        except Exception as e:
            logger.error(f"Error processing BetMGM fixture: {e}")
            return 0

    def _get_or_create_match(self, home_team: str, away_team: str, match_time: datetime) -> int:
        """Get match_id or create match"""
        from datetime import timedelta

        query = """
            SELECT m.match_id
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            WHERE ht.name = %s
                AND at.name = %s
                AND m.match_date BETWEEN %s AND %s
        """

        date_start = match_time - timedelta(hours=2)
        date_end = match_time + timedelta(hours=2)

        result = self.db.execute_query(query, (home_team, away_team, date_start, date_end), fetch=True)

        if result:
            return result[0]["match_id"]

        home_team_id = self.db.get_or_create_team(home_team, league_id=1)
        away_team_id = self.db.get_or_create_team(away_team, league_id=1)

        insert_query = """
            INSERT INTO matches (home_team_id, away_team_id, match_date, status, external_id)
            VALUES (%s, %s, %s, 'scheduled', %s)
            RETURNING match_id
        """

        external_id = f"betmgm_{home_team}_{away_team}_{match_time.strftime('%Y%m%d')}"

        result = self.db.execute_query(insert_query, (home_team_id, away_team_id, match_time, external_id), fetch=True)

        return result[0]["match_id"]

    def _save_1x2_odds(self, match_id: int, market: Dict) -> int:
        """Save 1X2 market odds"""

        outcomes = market.get("options", [])
        timestamp = datetime.utcnow()

        for outcome in outcomes:
            outcome_name = outcome.get("name", {}).get("value", "").lower()
            odds_decimal = outcome.get("price", {}).get("dec")

            if odds_decimal and outcome_name:
                if "home" in outcome_name or "1" in outcome_name:
                    outcome_type = "home"
                elif "draw" in outcome_name or "x" in outcome_name:
                    outcome_type = "draw"
                elif "away" in outcome_name or "2" in outcome_name:
                    outcome_type = "away"
                else:
                    continue

                self._insert_odds(match_id, "1X2", outcome_type, odds_decimal, timestamp)

        return len(outcomes)

    def _save_over_under_odds(self, match_id: int, market: Dict) -> int:
        """Save Over/Under market odds"""
        outcomes = market.get("options", [])
        timestamp = datetime.utcnow()
        count = 0

        for outcome in outcomes:
            outcome_name = outcome.get("name", {}).get("value", "").lower()
            odds_decimal = outcome.get("price", {}).get("dec")

            if odds_decimal and outcome_name:
                if "over" in outcome_name:
                    outcome_type = "over"
                elif "under" in outcome_name:
                    outcome_type = "under"
                else:
                    continue

                # Extract line value (e.g., 2.5 from "Over 2.5")
                import re

                line_match = re.search(r"([\d.]+)", outcome_name)
                line = line_match.group(1) if line_match else "2.5"

                self._insert_odds(match_id, f"over_under_{line}", outcome_type, odds_decimal, timestamp)
                count += 1

        return count

    def _save_btts_odds(self, match_id: int, market: Dict) -> int:
        """Save Both Teams To Score odds"""
        outcomes = market.get("options", [])
        timestamp = datetime.utcnow()
        count = 0

        for outcome in outcomes:
            outcome_name = outcome.get("name", {}).get("value", "").lower()
            odds_decimal = outcome.get("price", {}).get("dec")

            if odds_decimal and outcome_name:
                if "yes" in outcome_name:
                    outcome_type = "yes"
                elif "no" in outcome_name:
                    outcome_type = "no"
                else:
                    continue

                self._insert_odds(match_id, "btts", outcome_type, odds_decimal, timestamp)
                count += 1

        return count

    def _insert_odds(self, match_id: int, market_type: str, outcome: str, odds_decimal: float, timestamp: datetime):
        """Insert odds into database"""

        query = """
            INSERT INTO odds (match_id, bookmaker, market_type, outcome, odds_decimal, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """

        self.db.execute_query(query, (match_id, "betmgm", market_type, outcome, odds_decimal, timestamp))
