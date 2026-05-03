"""Configuration for data ingestion service"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import os


class ScraperConfig(BaseSettings):
    """Base configuration for all scrapers"""

    # Database
    database_url: str = Field(default_factory=lambda: os.getenv('DATABASE_URL'))

    # Redis
    redis_url: str = Field(default_factory=lambda: os.getenv('REDIS_URL'))

    # Scraping behavior
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    backoff_factor: float = 2.0

    # Rate limiting
    min_delay: float = 1.0  # seconds between requests
    max_delay: float = 3.0

    # User agents
    rotate_user_agents: bool = True

    # Proxies (optional)
    use_proxies: bool = False
    proxy_list_url: Optional[str] = None

    # Selenium/Browser settings
    headless: bool = True
    disable_images: bool = True
    disable_css: bool = False

    # Chrome driver
    chrome_driver_path: Optional[str] = None

    # Data validation
    validate_data: bool = True
    raise_on_validation_error: bool = False

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "/var/log/betting-system"

    class Config:
        env_file = ".env"
        case_sensitive = False

# Scraper-specific configs


class Bet365Config(ScraperConfig):
    """Bet365 scraper configuration"""
    base_url: str = "https://www.bet365.com"
    soccer_url: str = "https://www.bet365.com/#/AC/B1/C1/D13/E0/F2/"
    login_required: bool = False
    scrape_interval_minutes: int = 1  # Every minute for odds


class BetMGMConfig(ScraperConfig):
    """BetMGM scraper configuration"""
    base_url: str = "https://sports.co.betmgm.com"
    soccer_url: str = "https://sports.co.betmgm.com/en/sports/soccer-4"
    scrape_interval_minutes: int = 1


class FBrefConfig(ScraperConfig):
    """FBref scraper configuration"""
    base_url: str = "https://fbref.com"
    scrape_interval_hours: int = 24  # Daily
    leagues: List[str] = [
        "Premier-League",
        "La-Liga",
        "Serie-A",
        "Bundesliga",
        "Ligue-1",
        "Champions-League",
        "MLS"
    ]


class UnderstatConfig(ScraperConfig):
    """Understat scraper configuration"""
    base_url: str = "https://understat.com"
    scrape_interval_hours: int = 24


class TransfermarktConfig(ScraperConfig):
    """Transfermarkt scraper configuration"""
    base_url: str = "https://www.transfermarkt.com"
    scrape_interval_hours: int = 24
