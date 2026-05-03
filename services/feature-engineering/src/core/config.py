"""Feature engineering service configuration."""

import os
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class FeatureConfig(BaseSettings):
    """Configuration for the feature engineering service."""

    # Database
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://betting:betting@localhost:5432/betting_db")
    )

    # Redis
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # Connection pool
    db_min_connections: int = 1
    db_max_connections: int = 10

    # Cache TTLs (seconds)
    redis_cache_ttl: int = 3600  # 1 hour
    db_cache_ttl: int = 86400  # 24 hours

    # Feature version (for cache key namespacing)
    feature_version: str = "v1"

    # Form windows for rolling aggregation
    form_windows: List[int] = [3, 5, 10, 20]

    # H2H settings
    h2h_max_matches: int = 20

    # Validation thresholds
    max_missing_rate: float = 0.3  # 30% missing features triggers warning
    outlier_std_threshold: float = 4.0  # Z-score threshold for outlier detection

    # Parallel execution
    max_workers: int = 6  # One per independent category

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
