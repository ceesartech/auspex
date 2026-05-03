"""Bet365 odds scraper"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

from ..core.config import Bet365Config
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class Bet365Scraper(BaseScraper):
    """Scrape odds from Bet365"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: Bet365Config = self.config

    def scrape(self) -> int:
        """Scrape all odds from Bet365"""
        total_odds = 0

        # Scrape soccer odds
        total_odds += self._scrape_soccer_odds()

        # Scrape other sports as needed
        # total_odds += self._scrape_nfl_odds()
        # total_odds += self._scrape_nhl_odds()

        return total_odds

    def _scrape_soccer_odds(self) -> int:
        """Scrape soccer odds"""
        logger.info("Scraping Bet365 soccer odds")

        # This is a simplified example - actual implementation would be more complex
        # Bet365 uses heavy JavaScript, so Selenium is required

        html = self._fetch_page(self.config.soccer_url, use_selenium=True)
        soup = self._parse_html(html)

        odds_data = []

        # Parse matches and odds
        # NOTE: Bet365's HTML structure changes frequently, so selectors need to be updated
        matches = soup.find_all("div", class_="match-container")  # Example selector

        for match in matches:
            try:
                odds_entry = self._parse_match_odds(match)
                if odds_entry:
                    odds_data.append(odds_entry)
            except Exception as e:
                logger.error(f"Error parsing match: {e}")
                continue

        # Bulk insert odds
        if odds_data:
            self._save_odds(odds_data)

        return len(odds_data)

    def _parse_match_odds(self, match_element) -> Dict[str, Any]:
        """Parse odds from a match element"""
        # This is a simplified example - actual implementation depends on HTML structure

        try:
            # Extract teams
            home_team = match_element.find("span", class_="home-team").text.strip()
            away_team = match_element.find("span", class_="away-team").text.strip()

            # Extract match time
            match_time_str = match_element.find("span", class_="match-time").text.strip()
            match_date = self._parse_match_time(match_time_str)

            # Extract odds
            home_odds = match_element.find("span", class_="home-odds").text.strip()
            draw_odds = match_element.find("span", class_="draw-odds").text.strip()
            away_odds = match_element.find("span", class_="away-odds").text.strip()

            # Get or create match in database
            match_id = self._get_or_create_match(home_team, away_team, match_date)

            return {
                "match_id": match_id,
                "home_odds": float(home_odds),
                "draw_odds": float(draw_odds),
                "away_odds": float(away_odds),
                "timestamp": datetime.utcnow(),
            }
        except Exception as e:
            logger.error(f"Error parsing match element: {e}")
            return None

    def _parse_match_time(self, time_str: str) -> datetime:
        """Parse match time string to datetime"""
        # Example: "Today 15:00" or "Tomorrow 20:00" or "05 Mar 15:00"

        now = datetime.utcnow()

        if "today" in time_str.lower():
            time_part = time_str.split()[-1]
            hour, minute = map(int, time_part.split(":"))
            return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        elif "tomorrow" in time_str.lower():
            time_part = time_str.split()[-1]
            hour, minute = map(int, time_part.split(":"))
            tomorrow = now + timedelta(days=1)
            return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)

        else:
            # Parse date like "05 Mar 15:00"
            # Implementation depends on format
            pass

        return now

    def _get_or_create_match(self, home_team: str, away_team: str, match_date: datetime) -> int:
        """Get match_id or create match"""

        # Try to find existing match
        query = """
            SELECT m.match_id
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            WHERE ht.name = %s
                AND at.name = %s
                AND m.match_date BETWEEN %s AND %s
        """

        date_start = match_date - timedelta(hours=2)
        date_end = match_date + timedelta(hours=2)

        result = self.db.execute_query(query, (home_team, away_team, date_start, date_end), fetch=True)

        if result:
            return result[0]["match_id"]

        # Create new match
        # First get or create teams
        home_team_id = self.db.get_or_create_team(home_team, league_id=1)  # Default league
        away_team_id = self.db.get_or_create_team(away_team, league_id=1)

        insert_query = """
            INSERT INTO matches (home_team_id, away_team_id, match_date, status, external_id)
            VALUES (%s, %s, %s, 'scheduled', %s)
            RETURNING match_id
        """

        external_id = f"bet365_{home_team}_{away_team}_{match_date.strftime('%Y%m%d')}"

        result = self.db.execute_query(insert_query, (home_team_id, away_team_id, match_date, external_id), fetch=True)

        return result[0]["match_id"]

    def _save_odds(self, odds_data: List[Dict]):
        """Save odds to database"""

        for odds in odds_data:
            # Save 1X2 odds
            self._save_market_odds(odds["match_id"], "1X2", "home", odds["home_odds"], odds["timestamp"])
            self._save_market_odds(odds["match_id"], "1X2", "draw", odds["draw_odds"], odds["timestamp"])
            self._save_market_odds(odds["match_id"], "1X2", "away", odds["away_odds"], odds["timestamp"])

    def _save_market_odds(
        self, match_id: int, market_type: str, outcome: str, odds_decimal: float, timestamp: datetime
    ):
        """Save individual odds entry"""

        data = {
            "match_id": match_id,
            "bookmaker": "bet365",
            "market_type": market_type,
            "outcome": outcome,
            "odds_decimal": Decimal(str(odds_decimal)),
            "timestamp": timestamp,
        }

        # Check for duplicates
        checksum = self._calculate_checksum(data)
        if self._is_duplicate(checksum):
            return

        # Insert odds
        columns = list(data.keys())
        values = [tuple(data[col] for col in columns)]

        self.db.insert_many("odds", columns, values)
