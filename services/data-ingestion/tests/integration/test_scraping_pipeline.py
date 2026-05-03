"""Integration tests for the scraping pipeline."""

import pytest

from services.data_ingestion.src.scrapers.bet365_scraper import Bet365Scraper
from services.data_ingestion.src.scrapers.betmgm_scraper import BetMGMScraper
from services.data_ingestion.src.scrapers.fbref_scraper import FBrefScraper
from services.data_ingestion.src.validators.data_validator import DataValidator


@pytest.mark.integration
class TestScrapingPipeline:
    """Integration tests for the end-to-end scraping pipeline."""

    def test_bet365_scraper_initialization(self, mock_config, mock_db, mock_redis):
        """Test that Bet365 scraper can be initialized"""
        scraper = Bet365Scraper(mock_config, mock_db, mock_redis)
        assert scraper.scraper_name == "Bet365Scraper"
        assert scraper.config == mock_config

    def test_betmgm_scraper_initialization(self, mock_config, mock_db, mock_redis):
        """Test that BetMGM scraper can be initialized"""
        scraper = BetMGMScraper(mock_config, mock_db, mock_redis)
        assert scraper.scraper_name == "BetMGMScraper"
        assert scraper.config == mock_config

    def test_fbref_scraper_initialization(self, mock_config, mock_db, mock_redis):
        """Test that FBref scraper can be initialized"""
        scraper = FBrefScraper(mock_config, mock_db, mock_redis)
        assert scraper.scraper_name == "FBrefScraper"

    def test_scraper_has_session(self, mock_config, mock_db, mock_redis):
        """Test that scraper creates a requests session"""
        scraper = Bet365Scraper(mock_config, mock_db, mock_redis)
        assert scraper.session is not None

    def test_scraper_headers(self, mock_config, mock_db, mock_redis):
        """Test that scraper sets proper headers"""
        scraper = Bet365Scraper(mock_config, mock_db, mock_redis)
        headers = scraper._get_headers()
        assert 'User-Agent' in headers
        assert 'Accept' in headers
        assert 'DNT' in headers

    def test_data_validator_odds(self):
        """Test data validator with valid odds"""
        validator = DataValidator()
        result = validator.validate_odds({
            'match_id': 1,
            'bookmaker': 'bet365',
            'market_type': '1X2',
            'outcome': 'home',
            'odds_decimal': 2.50
        })
        assert result is True

    def test_data_validator_odds_invalid(self):
        """Test data validator rejects invalid odds"""
        validator = DataValidator()
        result = validator.validate_odds({
            'match_id': 1,
            'bookmaker': 'bet365',
            'market_type': '1X2',
            'outcome': 'home',
            'odds_decimal': 0.50  # Too low
        })
        assert result is False

    def test_checksum_deduplication(self, mock_config, mock_db, mock_redis):
        """Test that checksum-based deduplication works"""
        mock_redis.exists.return_value = False

        scraper = Bet365Scraper(mock_config, mock_db, mock_redis)

        data = {'key': 'value', 'number': 42}
        checksum = scraper._calculate_checksum(data)

        assert len(checksum) == 32  # MD5 hex digest length
        assert not scraper._is_duplicate(checksum)

        # Second call - should be marked as duplicate
        mock_redis.exists.return_value = True
        assert scraper._is_duplicate(checksum)
