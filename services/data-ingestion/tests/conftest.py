"""Pytest configuration and fixtures"""

from unittest.mock import Mock

import pytest
from redis import Redis
from services.data_ingestion.src.core.config import ScraperConfig
from services.data_ingestion.src.core.database import DatabaseManager


@pytest.fixture
def mock_config():
    """Mock scraper configuration"""
    config = Mock(spec=ScraperConfig)
    config.database_url = "postgresql://test:test@localhost:5432/test_db"
    config.redis_url = "redis://localhost:6379/15"
    config.request_timeout = 30
    config.max_retries = 3
    config.min_delay = 0.1
    config.max_delay = 0.2
    config.rotate_user_agents = True
    config.use_proxies = False
    config.headless = True
    return config


@pytest.fixture
def mock_db(mock_config):
    """Mock database manager"""
    return Mock(spec=DatabaseManager)


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    return Mock(spec=Redis)
