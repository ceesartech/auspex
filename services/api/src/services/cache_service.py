"""Redis caching service"""

import logging
from typing import Optional
import json
from redis import Redis

from config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching predictions and data"""

    def __init__(self):
        try:
            self.redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.ttl = settings.REDIS_CACHE_TTL
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis = None
            self.ttl = settings.REDIS_CACHE_TTL

    def get_prediction(self, match_id: str) -> Optional[dict]:
        """Get cached prediction"""
        if not self.redis:
            return None

        key = f"prediction:{match_id}"

        try:
            cached = self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    def set_prediction(self, match_id: str, prediction: dict, ttl: int = None):
        """Cache prediction"""
        if not self.redis:
            return

        key = f"prediction:{match_id}"
        ttl = ttl or self.ttl

        try:
            self.redis.setex(key, ttl, json.dumps(prediction, default=str))
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def invalidate_prediction(self, match_id: str):
        """Invalidate cached prediction"""
        if not self.redis:
            return

        key = f"prediction:{match_id}"
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")

    def get(self, key: str) -> Optional[str]:
        """Generic cache get"""
        if not self.redis:
            return None
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = None):
        """Generic cache set"""
        if not self.redis:
            return
        try:
            self.redis.setex(key, ttl or self.ttl, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
