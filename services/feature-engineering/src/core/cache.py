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

    def _redis_key(self, match_id: str, feature_set: str = "full") -> str:
        """Redis key. Composite (no SQL constraint) — namespaced to feature_version
        so a model retrain that bumps the version invalidates the cache."""
        return f"features:{self.config.feature_version}:{match_id}:{feature_set}"

    def get(self, match_id: str, feature_set: str = "full") -> Optional[Dict[str, Optional[float]]]:
        """Try to get features from cache (Redis first, then DB)."""
        redis_key = self._redis_key(match_id, feature_set)

        # Layer 1: Redis
        try:
            cached = self.redis.get(redis_key)
            if cached:
                logger.debug(f"Cache hit (Redis): {redis_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")

        # Layer 2: DB. The features_cache table's natural key is the
        # 3-tuple (match_id, feature_set, feature_version); we pass it
        # explicitly to CACHE_GET.
        try:
            result = self.db.execute_query(
                CACHE_GET,
                (match_id, feature_set, self.config.feature_version),
                fetch=True,
            )
            if result:
                data = result[0].get("feature_data")
                if data:
                    features = json.loads(data) if isinstance(data, str) else data
                    # Warm up Redis cache
                    self._set_redis(redis_key, features)
                    logger.debug("Cache hit (DB): match=%s set=%s", match_id, feature_set)
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
        redis_key = self._redis_key(match_id, feature_set)

        # Layer 1: Redis
        self._set_redis(redis_key, features)

        # Layer 2: DB
        try:
            self.db.execute_query(
                CACHE_SET,
                (
                    match_id,
                    feature_set,
                    self.config.feature_version,
                    json.dumps(features),
                    str(self.config.db_cache_ttl),
                ),
            )
            logger.debug("Cache set (DB): match=%s set=%s", match_id, feature_set)
        except Exception as e:
            logger.warning(f"DB cache write failed: {e}")

    def invalidate(self, match_id: str, feature_set: str = "full") -> None:
        """Remove features from both caches."""
        redis_key = self._redis_key(match_id, feature_set)

        try:
            self.redis.delete(redis_key)
        except Exception as e:
            logger.warning(f"Redis cache delete failed: {e}")

        try:
            self.db.execute_query(
                CACHE_DELETE,
                (match_id, feature_set, self.config.feature_version),
            )
        except Exception as e:
            logger.warning(f"DB cache delete failed: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired entries from DB cache. Returns count deleted."""
        try:
            # CACHE_CLEANUP now uses the schema's expires_at column rather
            # than a Python-passed TTL.
            return self.db.execute_query(CACHE_CLEANUP)
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            return 0

    def _set_redis(self, key: str, features: Dict) -> None:
        try:
            self.redis.setex(key, self.config.redis_cache_ttl, json.dumps(features))
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")
