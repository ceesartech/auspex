"""Lottery draw scrapers (Powerball and Mega Millions)"""

import logging
import re
from datetime import datetime
from typing import Dict

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


def _parse_jackpot_dollars(jackpot_str: str) -> float:
    """Parse a jackpot string (e.g. '$50 Million', '$1.2 Billion') to dollars,
    matching lottery_draws.jackpot_amount which is DECIMAL dollars."""
    match = re.search(r"\$?([\d.]+)\s*(million|billion)?", jackpot_str, re.I)
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else ""
    if "million" in unit:
        return amount * 1_000_000
    if "billion" in unit:
        return amount * 1_000_000_000
    return amount


def _save_lottery_draw(db, draw_data: Dict) -> None:
    """Insert a draw into lottery_draws using the actual schema columns
    (game, draw_date, numbers, bonus_number, jackpot_amount). Idempotent on
    (game, draw_date)."""
    query = """
        INSERT INTO lottery_draws (game, draw_date, numbers, bonus_number, jackpot_amount)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (game, draw_date) DO NOTHING
    """
    db.execute_query(
        query,
        (
            draw_data["game"],
            draw_data["draw_date"],
            draw_data["numbers"],
            draw_data["bonus_number"],
            draw_data.get("jackpot_amount"),
        ),
    )


class PowerballScraper(BaseScraper):
    """Scrape Powerball lottery draws"""

    def scrape(self) -> int:
        """Scrape latest Powerball draws"""

        url = "https://www.powerball.com/previous-results"
        html = self._fetch_page(url)
        soup = self._parse_html(html)

        draws_scraped = 0

        # Find all draw results
        draw_items = soup.find_all("div", class_="item")

        for item in draw_items:
            try:
                draw_data = self._parse_powerball_draw(item)
                if draw_data:
                    _save_lottery_draw(self.db, draw_data)
                    draws_scraped += 1
            except Exception as e:
                logger.error(f"Error parsing Powerball draw: {e}")
                continue

        return draws_scraped

    def _parse_powerball_draw(self, item) -> Dict:
        """Parse Powerball draw data"""

        date_str = item.find("h5", class_="card-title").text.strip()
        draw_date = datetime.strptime(date_str, "%B %d, %Y").date()

        # Extract numbers
        numbers = []
        number_divs = item.find_all("div", class_="item-powerball")

        for num_div in number_divs[:-1]:  # Last one is powerball
            numbers.append(int(num_div.text.strip()))

        powerball = int(number_divs[-1].text.strip())

        # Extract jackpot
        jackpot_str = item.find("div", class_="jackpot-amount").text.strip()
        jackpot = _parse_jackpot_dollars(jackpot_str)

        return {
            "game": "powerball",
            "draw_date": draw_date,
            "numbers": numbers,
            "bonus_number": powerball,
            "jackpot_amount": jackpot,
        }


class MegaMillionsScraper(BaseScraper):
    """Scrape Mega Millions lottery draws"""

    def scrape(self) -> int:
        """Scrape latest Mega Millions draws"""

        url = "https://www.megamillions.com/winning-numbers/previous-drawings"
        html = self._fetch_page(url)
        soup = self._parse_html(html)

        draws_scraped = 0

        draw_items = soup.find_all("div", class_="drawing-results")

        for item in draw_items:
            try:
                draw_data = self._parse_mega_millions_draw(item)
                if draw_data:
                    _save_lottery_draw(self.db, draw_data)
                    draws_scraped += 1
            except Exception as e:
                logger.error(f"Error parsing Mega Millions draw: {e}")
                continue

        return draws_scraped

    def _parse_mega_millions_draw(self, item) -> Dict:
        """Parse Mega Millions draw data"""

        date_str = item.find("h5", class_="card-title").text.strip()
        draw_date = datetime.strptime(date_str, "%B %d, %Y").date()

        # Extract numbers
        numbers = []
        number_divs = item.find_all("li", class_="ball")

        for num_div in number_divs[:-1]:  # Last one is megaball
            numbers.append(int(num_div.text.strip()))

        megaball = int(number_divs[-1].text.strip())

        # Extract jackpot
        jackpot_elem = item.find("div", class_="jackpot-amount")
        jackpot = _parse_jackpot_dollars(jackpot_elem.text.strip()) if jackpot_elem else 0.0

        return {
            "game": "mega_millions",
            "draw_date": draw_date,
            "numbers": numbers,
            "bonus_number": megaball,
            "jackpot_amount": jackpot,
        }
