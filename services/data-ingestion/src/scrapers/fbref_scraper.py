"""FBref soccer statistics scraper"""

import logging
from typing import Dict
from datetime import datetime
import re
from decimal import Decimal

from .base_scraper import BaseScraper
from ..core.config import FBrefConfig

logger = logging.getLogger(__name__)


class FBrefScraper(BaseScraper):
    """Scrape soccer statistics from FBref"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: FBrefConfig = self.config

    def scrape(self) -> int:
        """Scrape statistics for all configured leagues"""

        total_matches = 0

        for league in self.config.leagues:
            try:
                matches = self._scrape_league(league)
                total_matches += matches
            except Exception as e:
                logger.error(f"Error scraping {league}: {e}")
                continue

        return total_matches

    def _scrape_league(self, league: str) -> int:
        """Scrape a specific league"""

        # Get current season
        current_year = datetime.now().year
        season = f"{current_year}-{current_year + 1}"

        # Construct URL
        league_url = f"{self.config.base_url}/en/comps/9/{season}/{season}-{league}-Stats"

        logger.info(f"Scraping {league} from {league_url}")

        html = self._fetch_page(league_url)
        soup = self._parse_html(html)

        # Find scores & fixtures table
        fixtures_table = soup.find('table', {'id': 'sched'})

        if not fixtures_table:
            logger.warning(f"No fixtures table found for {league}")
            return 0

        matches_scraped = 0

        # Parse each row
        rows = fixtures_table.find('tbody').find_all('tr')

        for row in rows:
            try:
                match_data = self._parse_match_row(row, league, season)
                if match_data:
                    self._save_match_stats(match_data)
                    matches_scraped += 1
            except Exception as e:
                logger.error(f"Error parsing match row: {e}")
                continue

        return matches_scraped

    def _parse_match_row(self, row, league: str, season: str) -> Dict:
        """Parse a match row from FBref table"""

        cells = row.find_all('td')

        if len(cells) < 10:
            return None

        try:
            # Extract data
            date_str = cells[1].text.strip()
            home_team = cells[3].text.strip()
            away_team = cells[5].text.strip()
            score = cells[4].text.strip()

            # Parse score
            if '\u2013' in score or '-' in score:
                home_score, away_score = re.split('[\u2013-]', score)
                home_score = int(home_score.strip())
                away_score = int(away_score.strip())
            else:
                # Match hasn't been played yet
                return None

            # Parse date
            match_date = datetime.strptime(date_str, '%Y-%m-%d')

            # Extract xG if available
            xg_cell = cells[6] if len(cells) > 6 else None
            home_xg = float(xg_cell.text.strip()) if xg_cell and xg_cell.text.strip() else None

            xga_cell = cells[7] if len(cells) > 7 else None
            away_xg = float(xga_cell.text.strip()) if xga_cell and xga_cell.text.strip() else None

            return {
                'league': league,
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'home_score': home_score,
                'away_score': away_score,
                'home_xg': home_xg,
                'away_xg': away_xg
            }

        except Exception as e:
            logger.error(f"Error parsing match row: {e}")
            return None

    def _save_match_stats(self, match_data: Dict):
        """Save match and statistics to database"""

        # Get or create league
        league_id = self.db.get_or_create_league(
            name=match_data['league'],
            country='International',
            sport='soccer',
            season=match_data['season']
        )

        # Get or create teams
        home_team_id = self.db.get_or_create_team(
            name=match_data['home_team'],
            league_id=league_id
        )

        away_team_id = self.db.get_or_create_team(
            name=match_data['away_team'],
            league_id=league_id
        )

        # Check if match exists
        query = """
            SELECT match_id FROM matches
            WHERE home_team_id = %s
                AND away_team_id = %s
                AND match_date = %s
        """

        result = self.db.execute_query(
            query,
            (home_team_id, away_team_id, match_data['match_date']),
            fetch=True
        )

        if result:
            match_id = result[0]['match_id']

            # Update match with scores
            update_query = """
                UPDATE matches
                SET home_score = %s, away_score = %s, status = 'finished'
                WHERE match_id = %s
            """
            self.db.execute_query(
                update_query,
                (match_data['home_score'], match_data['away_score'], match_id)
            )
        else:
            # Create new match
            insert_query = """
                INSERT INTO matches
                (league_id, home_team_id, away_team_id, match_date,
                 home_score, away_score, status, season)
                VALUES (%s, %s, %s, %s, %s, %s, 'finished', %s)
                RETURNING match_id
            """

            result = self.db.execute_query(
                insert_query,
                (league_id, home_team_id, away_team_id, match_data['match_date'],
                 match_data['home_score'], match_data['away_score'], match_data['season']),
                fetch=True
            )

            match_id = result[0]['match_id']

        # Save xG stats if available
        if match_data.get('home_xg') is not None:
            self._save_xg_stats(match_id, home_team_id, away_team_id,
                                match_data['home_xg'], match_data['away_xg'])

    def _save_xg_stats(
        self,
        match_id: int,
        home_team_id: int,
        away_team_id: int,
        home_xg: float,
        away_xg: float
    ):
        """Save expected goals statistics"""

        # Home team stats
        self.db.upsert(
            'match_stats',
            {
                'match_id': match_id,
                'team_id': home_team_id,
                'is_home': True,
                'xg': Decimal(str(home_xg)),
                'xga': Decimal(str(away_xg))
            },
            conflict_columns=['match_id', 'team_id'],
            update_columns=['xg', 'xga']
        )

        # Away team stats
        self.db.upsert(
            'match_stats',
            {
                'match_id': match_id,
                'team_id': away_team_id,
                'is_home': False,
                'xg': Decimal(str(away_xg)),
                'xga': Decimal(str(home_xg))
            },
            conflict_columns=['match_id', 'team_id'],
            update_columns=['xg', 'xga']
        )
