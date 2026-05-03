"""Proxy rotation and management"""

import requests
from typing import List, Dict, Optional
import random
import logging
from redis import Redis
import json

logger = logging.getLogger(__name__)

class ProxyManager:
    """Manage proxy rotation"""

    def __init__(self, redis_client: Redis, proxy_list_url: Optional[str] = None):
        self.redis = redis_client
        self.proxy_list_url = proxy_list_url
        self.proxy_key = "scrapers:proxies"
        self._load_proxies()

    def _load_proxies(self):
        """Load proxies from URL or cache"""
        # Try to get from cache
        cached = self.redis.get(self.proxy_key)
        if cached:
            return json.loads(cached)

        if not self.proxy_list_url:
            logger.warning("No proxy list URL configured")
            return []

        try:
            response = requests.get(self.proxy_list_url, timeout=10)
            proxies = response.text.strip().split('\n')

            # Cache for 1 hour
            self.redis.setex(self.proxy_key, 3600, json.dumps(proxies))
            logger.info(f"Loaded {len(proxies)} proxies")
            return proxies
        except Exception as e:
            logger.error(f"Failed to load proxies: {e}")
            return []

    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get a random proxy"""
        proxies = self._load_proxies()
        if not proxies:
            return None

        proxy_url = random.choice(proxies)
        return {
            'http': f'http://{proxy_url}',
            'https': f'http://{proxy_url}'
        }

    def mark_proxy_bad(self, proxy_url: str):
        """Remove a non-working proxy"""
        proxies = self._load_proxies()
        if proxy_url in proxies:
            proxies.remove(proxy_url)
            self.redis.setex(self.proxy_key, 3600, json.dumps(proxies))
