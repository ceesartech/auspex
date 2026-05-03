"""ESPN sports data scraper for NFL, Boxing, MMA"""

import json
import logging
from datetime import datetime
from typing import Dict

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ESPNScraper(BaseScraper):
    """Scrape sports data from ESPN's public API"""

    API_URL = "https://site.api.espn.com/apis/site/v2/sports"

    SPORT_ENDPOINTS = {
        "nfl": ("football", "nfl"),
        "boxing": ("boxing", ""),
        "mma": ("mma", "ufc"),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scrape(self) -> int:
        """Scrape data from ESPN for all configured sports"""
        total = 0

        for sport_key, (sport, league) in self.SPORT_ENDPOINTS.items():
            try:
                count = self._scrape_sport(sport, league, sport_key)
                total += count
            except Exception as e:
                logger.error(f"Error scraping ESPN {sport_key}: {e}")
                continue

        return total

    def _scrape_sport(self, sport: str, league: str, sport_key: str) -> int:
        """Scrape a specific sport from ESPN API"""
        if league:
            url = f"{self.API_URL}/{sport}/{league}/scoreboard"
        else:
            url = f"{self.API_URL}/{sport}/scoreboard"

        logger.info(f"Scraping ESPN {sport_key} from {url}")

        html = self._fetch_page(url)

        try:
            data = json.loads(html)
            events = data.get("events", [])

            count = 0
            for event in events:
                try:
                    self._process_event(event, sport_key)
                    count += 1
                except Exception as e:
                    logger.error(f"Error processing ESPN event: {e}")
                    continue

            return count

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ESPN JSON: {e}")
            return 0

    def _process_event(self, event: Dict, sport_key: str):
        """Process a single ESPN event"""
        event_id = event.get("id")
        event.get("name", "")
        date_str = event.get("date", "")
        status = event.get("status", {}).get("type", {}).get("state", "pre")

        competitions = event.get("competitions", [])
        if not competitions:
            return

        competition = competitions[0]
        competitors = competition.get("competitors", [])

        if len(competitors) < 2:
            return

        home_competitor = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_competitor = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_team = home_competitor.get("team", {}).get("displayName", "")
        away_team = away_competitor.get("team", {}).get("displayName", "")
        home_score = home_competitor.get("score")
        away_score = away_competitor.get("score")

        venue_info = competition.get("venue", {})
        venue = venue_info.get("fullName", "")

        # Map ESPN status to our status
        status_map = {
            "pre": "scheduled",
            "in": "live",
            "post": "finished",
        }
        mapped_status = status_map.get(status, "scheduled")

        checksum = self._calculate_checksum(
            {
                "source": "espn",
                "event_id": event_id,
                "home": home_team,
                "away": away_team,
                "date": date_str,
                "score": f"{home_score}-{away_score}",
            }
        )

        if self._is_duplicate(checksum):
            return

        # Get or create league
        league_id = self.db.get_or_create_league(
            name=sport_key.upper(), country="USA", sport=sport_key, season=str(datetime.now().year)
        )

        # Get or create teams
        home_team_id = self.db.get_or_create_team(home_team, league_id)
        away_team_id = self.db.get_or_create_team(away_team, league_id)

        # Upsert match
        match_data = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "league_id": league_id,
            "match_date": date_str,
            "status": mapped_status,
            "venue": venue,
            "season": str(datetime.now().year),
        }

        if home_score is not None:
            match_data["home_score"] = int(home_score)
        if away_score is not None:
            match_data["away_score"] = int(away_score)

        self.db.upsert(
            "matches",
            match_data,
            conflict_columns=["home_team_id", "away_team_id", "match_date"],
            update_columns=["status", "home_score", "away_score"],
        )
