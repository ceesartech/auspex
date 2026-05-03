"""Understat xG and advanced statistics scraper"""

import json
import logging
import re
from decimal import Decimal
from typing import Dict, List

from ..core.config import UnderstatConfig
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class UnderstatScraper(BaseScraper):
    """Scrape xG data from Understat"""

    LEAGUE_MAP = {
        "EPL": "EPL",
        "La_liga": "La_liga",
        "Bundesliga": "Bundesliga",
        "Serie_A": "Serie_A",
        "Ligue_1": "Ligue_1",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: UnderstatConfig = self.config

    def scrape(self) -> int:
        """Scrape xG data for all leagues"""
        total_matches = 0

        for league_key, league_url in self.LEAGUE_MAP.items():
            try:
                matches = self._scrape_league(league_url)
                total_matches += matches
            except Exception as e:
                logger.error(f"Error scraping {league_key}: {e}")
                continue

        return total_matches

    def _scrape_league(self, league: str) -> int:
        """Scrape xG data for a specific league"""
        from datetime import datetime

        current_year = datetime.now().year
        url = f"{self.config.base_url}/league/{league}/{current_year}"

        logger.info(f"Scraping Understat {league} from {url}")

        html = self._fetch_page(url)

        # Understat stores data as JSON in script tags
        matches_data = self._extract_json_data(html, "datesData")
        if not matches_data:
            logger.warning(f"No match data found for {league}")
            return 0

        matches_scraped = 0
        for match in matches_data:
            try:
                self._process_match(match, league)
                matches_scraped += 1
            except Exception as e:
                logger.error(f"Error processing Understat match: {e}")
                continue

        return matches_scraped

    def _extract_json_data(self, html: str, variable_name: str) -> List[Dict]:
        """Extract JSON data from Understat's embedded script tags"""
        pattern = rf"var\s+{variable_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)

        if not match:
            return []

        json_str = match.group(1)
        # Understat encodes special chars
        json_str = json_str.encode().decode("unicode_escape")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Understat JSON: {e}")
            return []

    def _process_match(self, match: Dict, league: str):
        """Process a single match's xG data"""
        home_team = match.get("h", {}).get("title", "")
        away_team = match.get("a", {}).get("title", "")
        home_xg = match.get("xG", {}).get("h")
        away_xg = match.get("xG", {}).get("a")
        match_date = match.get("datetime")

        if not all([home_team, away_team, match_date]):
            return

        # Check for duplicates
        checksum = self._calculate_checksum(
            {
                "source": "understat",
                "home": home_team,
                "away": away_team,
                "date": match_date,
                "home_xg": home_xg,
                "away_xg": away_xg,
            }
        )

        if self._is_duplicate(checksum):
            return

        # Find match in database and update xG stats
        query = """
            SELECT ms.id, m.id as match_id
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            LEFT JOIN match_stats ms ON ms.match_id = m.id AND ms.team_id = ht.id
            WHERE ht.normalized_name ILIKE %s
                AND at.normalized_name ILIKE %s
                AND m.match_date::date = %s::date
        """

        result = self.db.execute_query(query, (f"%{home_team}%", f"%{away_team}%", match_date), fetch=True)

        if result and home_xg is not None:
            match_id = result[0]["match_id"]
            # Upsert home xG
            self.db.upsert(
                "match_stats",
                {
                    "match_id": match_id,
                    "team_id": match_id,  # Will need proper team_id lookup
                    "expected_goals": Decimal(str(home_xg)),
                    "expected_goals_against": Decimal(str(away_xg)) if away_xg else None,
                },
                conflict_columns=["match_id", "team_id"],
                update_columns=["expected_goals", "expected_goals_against"],
            )
