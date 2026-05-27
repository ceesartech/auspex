"""Redis caching service"""

import hashlib
import json
import logging
from typing import Any, Optional

from config import settings
from redis import Redis

logger = logging.getLogger(__name__)


def _feature_hash(features: Optional[Any]) -> str:
    """Stable short hash of the feature payload for cache-key composition.

    If features is None or unhashable, returns 'nofeat' so callers fall back
    to a feature-agnostic cache slot.
    """
    if features is None:
        return "nofeat"
    try:
        canonical = json.dumps(features, sort_keys=True, default=str)
    except Exception:
        return "nofeat"
    return hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


class CacheService:
    """Service for caching predictions and data"""

    def __init__(self):
        try:
            # Hiredis is the C-backed parser; if installed it cuts round-trip
            # parsing cost roughly in half. decode_responses=True keeps callers
            # working with str rather than bytes.
            self.redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.ttl = settings.REDIS_CACHE_TTL
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis = None
            self.ttl = settings.REDIS_CACHE_TTL

    def _prediction_key(
        self,
        match_id: str,
        model_version: Optional[str] = None,
        features: Optional[Any] = None,
    ) -> str:
        """Build a cache key. Keys are scoped by model_version + feature_hash
        so that a model retrain or a feature recompute invalidates the cache
        automatically without an explicit purge step.
        """
        version = model_version or "v0"
        return f"prediction:{match_id}:{version}:{_feature_hash(features)}"

    def get_prediction(
        self,
        match_id: str,
        model_version: Optional[str] = None,
        features: Optional[Any] = None,
    ) -> Optional[dict]:
        """Get cached prediction"""
        if not self.redis:
            return None

        key = self._prediction_key(match_id, model_version, features)

        try:
            cached = self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    def set_prediction(
        self,
        match_id: str,
        prediction: dict,
        ttl: Optional[int] = None,
        model_version: Optional[str] = None,
        features: Optional[Any] = None,
    ):
        """Cache prediction"""
        if not self.redis:
            return

        key = self._prediction_key(match_id, model_version, features)
        ttl = ttl or self.ttl

        try:
            self.redis.setex(key, ttl, json.dumps(prediction, default=str))
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def invalidate_prediction(self, match_id: str):
        """Invalidate all cached predictions for a match across versions/feature
        hashes. Uses SCAN so this is safe to run against large keyspaces."""
        if not self.redis:
            return

        pattern = f"prediction:{match_id}:*"
        try:
            for key in self.redis.scan_iter(match=pattern, count=200):
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

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Generic cache set"""
        if not self.redis:
            return
        try:
            self.redis.setex(key, ttl or self.ttl, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
