"""Unit tests for scrapers"""

from unittest.mock import Mock, patch

from src.scrapers.bet365_scraper import Bet365Scraper


class TestBet365Scraper:
    """Test Bet365 scraper"""

    def test_initialization(self, mock_config, mock_db, mock_redis):
        """Test scraper initialization"""
        scraper = Bet365Scraper(mock_config, mock_db, mock_redis)

        assert scraper.config == mock_config
        assert scraper.db == mock_db
        assert scraper.redis == mock_redis

    @patch("src.scrapers.base_scraper.requests.Session")
    def test_fetch_page(self, mock_session, mock_config, mock_db, mock_redis):
        """Test page fetching"""
        mock_response = Mock()
        mock_response.text = "<html>Test</html>"
        mock_session.return_value.get.return_value = mock_response

        scraper = Bet365Scraper(mock_config, mock_db, mock_redis)
        html = scraper._fetch_with_requests("https://test.com")

        assert html == "<html>Test</html>"

    def test_parse_match_time_today(self, mock_config, mock_db, mock_redis):
        """Test parsing match time for today"""
        scraper = Bet365Scraper(mock_config, mock_db, mock_redis)

        result = scraper._parse_match_time("Today 15:00")

        assert result.hour == 15
        assert result.minute == 0
