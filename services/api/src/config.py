"""API configuration"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""

    # API
    API_TITLE: str = "Betting System API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://betting:betting@localhost:5432/betting_system")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # User
    USER_DOB: str = os.getenv("USER_DOB", "1994-05-09")

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://yourdomain.com",
    ]

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Celery
    CELERY_BROKER_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("REDIS_URL", "redis://localhost:6379/2")

    # Model paths
    MODEL_PATH: str = "/app/models"

    # Feature flags
    ENABLE_WEBSOCKET: bool = True
    ENABLE_PREDICTIONS: bool = True
    ENABLE_NOTIFICATIONS: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
