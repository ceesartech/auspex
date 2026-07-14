"""Configuration for the data-ingestion service.

`ScraperConfig` predates the current architecture — it was the base config for
the Selenium/Playwright scrapers removed in the 2026-07 audit. It survives
because `DatabaseManager` takes it as its connection config (it only reads
`database_url`) and the e2e fixtures construct it. The browser/proxy fields and
the per-site subclasses are gone; the connection + rate-limit fields remain.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings


class ScraperConfig(BaseSettings):
    """Connection + fetch-behaviour config for the ingestion service."""

    # Database
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL"))

    # Redis
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL"))

    # Fetch behavior
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    backoff_factor: float = 2.0

    # Rate limiting
    min_delay: float = 1.0  # seconds between requests
    max_delay: float = 3.0

    # User agents
    rotate_user_agents: bool = True

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
        extra = "ignore"
