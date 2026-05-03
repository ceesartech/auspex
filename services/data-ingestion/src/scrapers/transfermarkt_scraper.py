"""Transfermarkt player and team data scraper"""

import logging
import re
from datetime import datetime
from decimal import Decimal
from typing import Dict

from ..core.config import TransfermarktConfig
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class TransfermarktScraper(BaseScraper):
    """Scrape player values, transfers, and injuries from Transfermarkt"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: TransfermarktConfig = self.config

    def scrape(self) -> int:
        """Scrape all data from Transfermarkt"""
        total = 0

        # Scrape player injuries
        total += self._scrape_injuries()

        # Scrape transfers
        total += self._scrape_transfers()

        return total

    def _scrape_injuries(self) -> int:
        """Scrape current injuries for major leagues"""
        logger.info("Scraping Transfermarkt injuries")

        url = f"{self.config.base_url}/premier-league/verletztespieler/wettbewerb/GB1"
        html = self._fetch_page(url)
        soup = self._parse_html(html)

        injuries_scraped = 0

        injury_table = soup.find("table", class_="items")
        if not injury_table:
            logger.warning("No injury table found")
            return 0

        rows = injury_table.find("tbody").find_all("tr")

        for row in rows:
            try:
                injury_data = self._parse_injury_row(row)
                if injury_data:
                    self._save_injury(injury_data)
                    injuries_scraped += 1
            except Exception as e:
                logger.error(f"Error parsing injury row: {e}")
                continue

        return injuries_scraped

    def _parse_injury_row(self, row) -> Dict:
        """Parse an injury row from the table"""
        cells = row.find_all("td")
        if len(cells) < 5:
            return None

        try:
            player_name = cells[0].find("a").text.strip() if cells[0].find("a") else cells[0].text.strip()
            team_name = cells[1].find("a").text.strip() if cells[1].find("a") else cells[1].text.strip()
            injury_type = cells[2].text.strip()
            since_str = cells[3].text.strip()
            until_str = cells[4].text.strip() if len(cells) > 4 else None

            return {
                "player_name": player_name,
                "team_name": team_name,
                "injury_type": injury_type,
                "since": since_str,
                "until": until_str,
                "status": "injured",
            }
        except Exception as e:
            logger.error(f"Error parsing injury: {e}")
            return None

    def _save_injury(self, injury_data: Dict):
        """Save injury data to player_availability table"""
        checksum = self._calculate_checksum(injury_data)
        if self._is_duplicate(checksum):
            return

        # Find player in database
        query = """
            SELECT id FROM players
            WHERE normalized_name ILIKE %s
            LIMIT 1
        """
        result = self.db.execute_query(query, (f"%{injury_data['player_name']}%",), fetch=True)

        if result:
            player_id = result[0]["id"]
            self.db.upsert(
                "player_availability",
                {
                    "player_id": player_id,
                    "status": "injured",
                    "reason": injury_data["injury_type"],
                    "start_date": injury_data["since"],
                    "expected_return": injury_data.get("until"),
                    "source": "transfermarkt",
                },
                conflict_columns=["player_id", "start_date"],
                update_columns=["status", "reason", "expected_return"],
            )

    def _scrape_transfers(self) -> int:
        """Scrape recent transfers"""
        logger.info("Scraping Transfermarkt transfers")

        url = (
            f"{self.config.base_url}/transfers/transfertagedetail/statistik/top/plus/0"
            "/galession//land_id//wettbewerb_id//minMarktwert//maxMarktwert//veression//betrag/"
        )
        html = self._fetch_page(url)
        soup = self._parse_html(html)

        transfers_scraped = 0

        transfer_table = soup.find("table", class_="items")
        if not transfer_table:
            return 0

        rows = transfer_table.find("tbody").find_all("tr")

        for row in rows:
            try:
                transfer_data = self._parse_transfer_row(row)
                if transfer_data:
                    self._save_transfer(transfer_data)
                    transfers_scraped += 1
            except Exception as e:
                logger.error(f"Error parsing transfer row: {e}")
                continue

        return transfers_scraped

    def _parse_transfer_row(self, row) -> Dict:
        """Parse a transfer row"""
        cells = row.find_all("td")
        if len(cells) < 6:
            return None

        try:
            player_name = cells[0].find("a").text.strip() if cells[0].find("a") else cells[0].text.strip()
            from_team = cells[2].find("a").text.strip() if cells[2].find("a") else cells[2].text.strip()
            to_team = cells[4].find("a").text.strip() if cells[4].find("a") else cells[4].text.strip()
            fee_str = cells[5].text.strip()

            # Parse fee
            fee = self._parse_fee(fee_str)

            return {
                "player_name": player_name,
                "from_team": from_team,
                "to_team": to_team,
                "fee": fee,
                "transfer_date": datetime.now().date(),
                "transfer_type": "loan" if "loan" in fee_str.lower() else "permanent",
            }
        except Exception as e:
            logger.error(f"Error parsing transfer: {e}")
            return None

    def _parse_fee(self, fee_str: str) -> Decimal:
        """Parse transfer fee string"""
        match = re.search(r"([\d.]+)\s*(m|k)?", fee_str.replace(",", "."), re.I)
        if not match:
            return Decimal("0")

        amount = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else ""

        if unit == "m":
            return Decimal(str(amount * 1_000_000))
        elif unit == "k":
            return Decimal(str(amount * 1_000))
        return Decimal(str(amount))

    def _save_transfer(self, transfer_data: Dict):
        """Save transfer to database"""
        checksum = self._calculate_checksum(transfer_data)
        if self._is_duplicate(checksum):
            return

        query = """
            INSERT INTO transfers (player_id, from_team_id, to_team_id, transfer_date, fee, transfer_type)
            SELECT
                p.id,
                ft.id,
                tt.id,
                %s,
                %s,
                %s
            FROM players p
            LEFT JOIN teams ft ON ft.normalized_name ILIKE %s
            LEFT JOIN teams tt ON tt.normalized_name ILIKE %s
            WHERE p.normalized_name ILIKE %s
            ON CONFLICT DO NOTHING
        """

        self.db.execute_query(
            query,
            (
                transfer_data["transfer_date"],
                transfer_data["fee"],
                transfer_data["transfer_type"],
                f"%{transfer_data['from_team']}%",
                f"%{transfer_data['to_team']}%",
                f"%{transfer_data['player_name']}%",
            ),
        )
