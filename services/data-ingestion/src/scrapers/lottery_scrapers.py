"""Lottery draw scrapers (Powerball and Mega Millions)"""

import logging
from typing import List, Dict
from datetime import datetime
import re

from .base_scraper import BaseScraper
from ..core.config import ScraperConfig

logger = logging.getLogger(__name__)

class PowerballScraper(BaseScraper):
    """Scrape Powerball lottery draws"""

    def scrape(self) -> int:
        """Scrape latest Powerball draws"""

        url = "https://www.powerball.com/previous-results"
        html = self._fetch_page(url)
        soup = self._parse_html(html)

        draws_scraped = 0

        # Find all draw results
        draw_items = soup.find_all('div', class_='item')

        for item in draw_items:
            try:
                draw_data = self._parse_powerball_draw(item)
                if draw_data:
                    self._save_lottery_draw(draw_data)
                    draws_scraped += 1
            except Exception as e:
                logger.error(f"Error parsing Powerball draw: {e}")
                continue

        return draws_scraped

    def _parse_powerball_draw(self, item) -> Dict:
        """Parse Powerball draw data"""

        date_str = item.find('h5', class_='card-title').text.strip()
        draw_date = datetime.strptime(date_str, '%B %d, %Y').date()

        # Extract numbers
        numbers = []
        number_divs = item.find_all('div', class_='item-powerball')

        for num_div in number_divs[:-1]:  # Last one is powerball
            numbers.append(int(num_div.text.strip()))

        powerball = int(number_divs[-1].text.strip())

        # Extract jackpot
        jackpot_str = item.find('div', class_='jackpot-amount').text.strip()
        jackpot = self._parse_jackpot(jackpot_str)

        return {
            'lottery_type': 'powerball',
            'draw_date': draw_date,
            'numbers': numbers,
            'powerball': powerball,
            'jackpot': jackpot
        }

    def _parse_jackpot(self, jackpot_str: str) -> int:
        """Parse jackpot string to cents"""
        # "$50 Million" -> 5000000000 cents
        match = re.search(r'\$?([\d.]+)\s*(million|billion)?', jackpot_str, re.I)

        if not match:
            return 0

        amount = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else ''

        if 'million' in unit:
            return int(amount * 1_000_000 * 100)
        elif 'billion' in unit:
            return int(amount * 1_000_000_000 * 100)
        else:
            return int(amount * 100)

    def _save_lottery_draw(self, draw_data: Dict):
        """Save lottery draw to database"""

        query = """
            INSERT INTO lottery_draws
            (lottery_type, draw_date, numbers, powerball, megaball, jackpot)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (lottery_type, draw_date) DO NOTHING
        """

        self.db.execute_query(
            query,
            (
                draw_data['lottery_type'],
                draw_data['draw_date'],
                draw_data['numbers'],
                draw_data.get('powerball'),
                draw_data.get('megaball'),
                draw_data['jackpot']
            )
        )


class MegaMillionsScraper(BaseScraper):
    """Scrape Mega Millions lottery draws"""

    def scrape(self) -> int:
        """Scrape latest Mega Millions draws"""

        url = "https://www.megamillions.com/winning-numbers/previous-drawings"
        html = self._fetch_page(url)
        soup = self._parse_html(html)

        draws_scraped = 0

        draw_items = soup.find_all('div', class_='drawing-results')

        for item in draw_items:
            try:
                draw_data = self._parse_mega_millions_draw(item)
                if draw_data:
                    self._save_lottery_draw(draw_data)
                    draws_scraped += 1
            except Exception as e:
                logger.error(f"Error parsing Mega Millions draw: {e}")
                continue

        return draws_scraped

    def _parse_mega_millions_draw(self, item) -> Dict:
        """Parse Mega Millions draw data"""

        date_str = item.find('h5', class_='card-title').text.strip()
        draw_date = datetime.strptime(date_str, '%B %d, %Y').date()

        # Extract numbers
        numbers = []
        number_divs = item.find_all('li', class_='ball')

        for num_div in number_divs[:-1]:  # Last one is megaball
            numbers.append(int(num_div.text.strip()))

        megaball = int(number_divs[-1].text.strip())

        # Extract jackpot
        jackpot_elem = item.find('div', class_='jackpot-amount')
        jackpot = self._parse_jackpot(jackpot_elem.text.strip()) if jackpot_elem else 0

        return {
            'lottery_type': 'mega_millions',
            'draw_date': draw_date,
            'numbers': numbers,
            'megaball': megaball,
            'jackpot': jackpot
        }

    def _parse_jackpot(self, jackpot_str: str) -> int:
        """Parse jackpot string to cents"""
        match = re.search(r'\$?([\d.]+)\s*(million|billion)?', jackpot_str, re.I)

        if not match:
            return 0

        amount = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else ''

        if 'million' in unit:
            return int(amount * 1_000_000 * 100)
        elif 'billion' in unit:
            return int(amount * 1_000_000_000 * 100)
        else:
            return int(amount * 100)

    def _save_lottery_draw(self, draw_data: Dict):
        """Save lottery draw to database"""

        query = """
            INSERT INTO lottery_draws
            (lottery_type, draw_date, numbers, powerball, megaball, jackpot)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (lottery_type, draw_date) DO NOTHING
        """

        self.db.execute_query(
            query,
            (
                draw_data['lottery_type'],
                draw_data['draw_date'],
                draw_data['numbers'],
                draw_data.get('powerball'),
                draw_data.get('megaball'),
                draw_data['jackpot']
            )
        )
