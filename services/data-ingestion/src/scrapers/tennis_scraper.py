"""Tennis data scraper"""

import logging
from datetime import datetime
from typing import Dict

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class TennisScraper(BaseScraper):
    """Scrape tennis match data and rankings from ATP/WTA"""

    BASE_URL = "https://www.atptour.com"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scrape(self) -> int:
        """Scrape tennis data"""
        total = 0

        # Scrape upcoming matches
        total += self._scrape_schedule()

        # Scrape recent results
        total += self._scrape_results()

        # Scrape rankings
        total += self._scrape_rankings()

        return total

    def _scrape_schedule(self) -> int:
        """Scrape upcoming tennis matches"""
        url = f"{self.BASE_URL}/en/scores/current"

        logger.info("Scraping tennis schedule")

        html = self._fetch_page(url)
        soup = self._parse_html(html)

        count = 0
        match_items = soup.find_all("div", class_="day-table")

        for day in match_items:
            matches = day.find_all("tr", class_="match")
            for match in matches:
                try:
                    match_data = self._parse_match(match)
                    if match_data:
                        self._save_match(match_data)
                        count += 1
                except Exception as e:
                    logger.error(f"Error parsing tennis match: {e}")
                    continue

        return count

    def _parse_match(self, match_element) -> Dict:
        """Parse a tennis match element"""
        try:
            player1 = match_element.find("td", class_="player1")
            player2 = match_element.find("td", class_="player2")

            if not player1 or not player2:
                return None

            player1_name = player1.find("a").text.strip() if player1.find("a") else player1.text.strip()
            player2_name = player2.find("a").text.strip() if player2.find("a") else player2.text.strip()

            tournament = match_element.find("td", class_="tournament")
            tournament_name = tournament.text.strip() if tournament else "Unknown"

            score_elem = match_element.find("td", class_="score")
            score = score_elem.text.strip() if score_elem else None

            return {
                "player1": player1_name,
                "player2": player2_name,
                "tournament": tournament_name,
                "score": score,
                "sport": "tennis",
                "date": datetime.now().date(),
            }
        except Exception as e:
            logger.error(f"Error parsing tennis match: {e}")
            return None

    def _save_match(self, match_data: Dict):
        """Save tennis match to database"""
        checksum = self._calculate_checksum(match_data)
        if self._is_duplicate(checksum):
            return

        league_id = self.db.get_or_create_league(
            name=match_data["tournament"], country="International", sport="tennis", season=str(datetime.now().year)
        )

        player1_id = self.db.get_or_create_team(match_data["player1"], league_id)
        player2_id = self.db.get_or_create_team(match_data["player2"], league_id)

        self.db.upsert(
            "matches",
            {
                "home_team_id": player1_id,
                "away_team_id": player2_id,
                "league_id": league_id,
                "match_date": str(match_data["date"]),
                "status": "finished" if match_data.get("score") else "scheduled",
                "season": str(datetime.now().year),
            },
            conflict_columns=["home_team_id", "away_team_id", "match_date"],
            update_columns=["status"],
        )

    def _scrape_results(self) -> int:
        """Scrape recent tennis results"""
        f"{self.BASE_URL}/en/scores/results-archive"
        logger.info("Scraping tennis results")
        # Similar implementation to _scrape_schedule
        return 0

    def _scrape_rankings(self) -> int:
        """Scrape ATP rankings"""
        f"{self.BASE_URL}/en/rankings/singles"
        logger.info("Scraping tennis rankings")
        # Rankings are stored as team metadata
        return 0
