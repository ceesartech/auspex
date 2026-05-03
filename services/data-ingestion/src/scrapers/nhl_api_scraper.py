"""NHL API data scraper"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from .base_scraper import BaseScraper
from ..core.config import ScraperConfig

logger = logging.getLogger(__name__)

class NHLAPIScraper(BaseScraper):
    """Scrape data from the NHL official API"""

    API_URL = "https://api-web.nhle.com/v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scrape(self) -> int:
        """Scrape NHL schedule and game data"""
        total = 0

        # Scrape today's schedule
        total += self._scrape_schedule()

        # Scrape standings
        total += self._scrape_standings()

        return total

    def _scrape_schedule(self) -> int:
        """Scrape NHL schedule"""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"{self.API_URL}/schedule/{today}"

        logger.info(f"Scraping NHL schedule for {today}")

        html = self._fetch_page(url)

        try:
            data = json.loads(html)
            game_weeks = data.get('gameWeek', [])

            count = 0
            for day in game_weeks:
                games = day.get('games', [])
                for game in games:
                    try:
                        self._process_game(game)
                        count += 1
                    except Exception as e:
                        logger.error(f"Error processing NHL game: {e}")
                        continue

            return count

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse NHL API JSON: {e}")
            return 0

    def _process_game(self, game: Dict):
        """Process a single NHL game"""
        game_id = game.get('id')
        home_team_info = game.get('homeTeam', {})
        away_team_info = game.get('awayTeam', {})

        home_team = home_team_info.get('placeName', {}).get('default', '') + ' ' + home_team_info.get('commonName', {}).get('default', '')
        away_team = away_team_info.get('placeName', {}).get('default', '') + ' ' + away_team_info.get('commonName', {}).get('default', '')
        home_team = home_team.strip()
        away_team = away_team.strip()

        game_date = game.get('gameDate', '')
        game_state = game.get('gameState', 'FUT')
        home_score = home_team_info.get('score')
        away_score = away_team_info.get('score')

        venue = game.get('venue', {}).get('default', '')

        # Map game state
        state_map = {
            'FUT': 'scheduled',
            'PRE': 'scheduled',
            'LIVE': 'live',
            'CRIT': 'live',
            'FINAL': 'finished',
            'OFF': 'finished',
        }
        status = state_map.get(game_state, 'scheduled')

        checksum = self._calculate_checksum({
            'source': 'nhl',
            'game_id': game_id,
            'home': home_team,
            'away': away_team,
            'date': game_date,
            'score': f"{home_score}-{away_score}",
            'state': game_state
        })

        if self._is_duplicate(checksum):
            return

        # Get or create league
        league_id = self.db.get_or_create_league(
            name='NHL',
            country='USA',
            sport='hockey',
            season=str(datetime.now().year)
        )

        # Get or create teams
        home_team_id = self.db.get_or_create_team(home_team, league_id)
        away_team_id = self.db.get_or_create_team(away_team, league_id)

        # Upsert match
        match_data = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'league_id': league_id,
            'match_date': game_date,
            'status': status,
            'venue': venue,
            'season': str(datetime.now().year),
        }

        if home_score is not None:
            match_data['home_score'] = int(home_score)
        if away_score is not None:
            match_data['away_score'] = int(away_score)

        self.db.upsert(
            'matches',
            match_data,
            conflict_columns=['home_team_id', 'away_team_id', 'match_date'],
            update_columns=['status', 'home_score', 'away_score']
        )

    def _scrape_standings(self) -> int:
        """Scrape NHL standings"""
        url = f"{self.API_URL}/standings/now"

        logger.info("Scraping NHL standings")

        html = self._fetch_page(url)

        try:
            data = json.loads(html)
            standings = data.get('standings', [])

            count = 0
            for team_standing in standings:
                try:
                    team_name = team_standing.get('teamName', {}).get('default', '')
                    if team_name:
                        count += 1
                except Exception as e:
                    logger.error(f"Error processing NHL standing: {e}")
                    continue

            return count

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse NHL standings JSON: {e}")
            return 0
