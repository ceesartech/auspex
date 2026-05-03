"""Dual-layer feature cache: Redis (fast, short TTL) + DB (slower, long TTL)."""

import json
import logging
from typing import Dict, Optional

from redis import Redis

from ..utils.sql_queries import CACHE_CLEANUP, CACHE_DELETE, CACHE_GET, CACHE_SET
from .config import FeatureConfig
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class FeatureCacheManager:
    """Two-tier cache for computed features.

    Layer 1: Redis — 1-hour TTL, fast reads for live/real-time.
    Layer 2: PostgreSQL features_cache table — 24-hour TTL, persistent.
    """

    def __init__(self, config: FeatureConfig, db_manager: DatabaseManager, redis_client: Redis):
        self.config = config
        self.db = db_manager
        self.redis = redis_client

    def _make_key(self, match_id: str, feature_set: str = "full") -> str:
        return f"features:{self.config.feature_version}:{match_id}:{feature_set}"

    def get(self, match_id: str, feature_set: str = "full") -> Optional[Dict[str, Optional[float]]]:
        """Try to get features from cache (Redis first, then DB)."""
        key = self._make_key(match_id, feature_set)

        # Layer 1: Redis
        try:
            cached = self.redis.get(key)
            if cached:
                logger.debug(f"Cache hit (Redis): {key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")

        # Layer 2: DB
        try:
            result = self.db.execute_query(CACHE_GET, (key, self.config.db_cache_ttl), fetch=True)
            if result:
                data = result[0].get("feature_data")
                if data:
                    features = json.loads(data) if isinstance(data, str) else data
                    # Warm up Redis cache
                    self._set_redis(key, features)
                    logger.debug(f"Cache hit (DB): {key}")
                    return features
        except Exception as e:
            logger.warning(f"DB cache read failed: {e}")

        return None

    def set(
        self,
        match_id: str,
        features: Dict[str, Optional[float]],
        feature_set: str = "full",
    ) -> None:
        """Store features in both cache layers."""
        key = self._make_key(match_id, feature_set)

        # Layer 1: Redis
        self._set_redis(key, features)

        # Layer 2: DB
        try:
            self.db.execute_query(CACHE_SET, (key, json.dumps(features)))
            logger.debug(f"Cache set (DB): {key}")
        except Exception as e:
            logger.warning(f"DB cache write failed: {e}")

    def invalidate(self, match_id: str, feature_set: str = "full") -> None:
        """Remove features from both caches."""
        key = self._make_key(match_id, feature_set)

        try:
            self.redis.delete(key)
        except Exception as e:
            logger.warning(f"Redis cache delete failed: {e}")

        try:
            self.db.execute_query(CACHE_DELETE, (key,))
        except Exception as e:
            logger.warning(f"DB cache delete failed: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired entries from DB cache. Returns count deleted."""
        try:
            return self.db.execute_query(CACHE_CLEANUP, (self.config.db_cache_ttl,))
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            return 0

    def _set_redis(self, key: str, features: Dict) -> None:
        try:
            self.redis.setex(key, self.config.redis_cache_ttl, json.dumps(features))
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")
