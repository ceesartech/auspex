"""Base scraper class with common functionality"""

import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from redis import Redis
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..core.config import ScraperConfig
from ..core.database import DatabaseManager
from ..utils.proxy_manager import ProxyManager
from ..utils.retry_logic import retry_with_backoff

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all scrapers"""

    def __init__(self, config: ScraperConfig, db_manager: DatabaseManager, redis_client: Redis):
        self.config = config
        self.db = db_manager
        self.redis = redis_client
        self.ua = UserAgent()
        self.proxy_manager = ProxyManager(redis_client, config.proxy_list_url) if config.use_proxies else None
        self.session = self._create_session()
        self.scraper_name = self.__class__.__name__

    def _create_session(self) -> requests.Session:
        """Create requests session with headers"""
        session = requests.Session()
        session.headers.update(self._get_headers())
        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get random headers to avoid detection"""
        return {
            "User-Agent": self.ua.random if self.config.rotate_user_agents else self.ua.chrome,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }

    def _random_delay(self):
        """Add random delay between requests"""
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
        time.sleep(delay)

    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def _fetch_page(self, url: str, use_selenium: bool = False) -> Optional[str]:
        """Fetch page content with retry logic"""
        logger.info(f"Fetching: {url}")

        try:
            if use_selenium:
                return self._fetch_with_selenium(url)
            else:
                return self._fetch_with_requests(url)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise

    def _fetch_with_requests(self, url: str) -> str:
        """Fetch with requests library"""
        proxies = self.proxy_manager.get_proxy() if self.proxy_manager else None

        response = self.session.get(url, timeout=self.config.request_timeout, proxies=proxies)
        response.raise_for_status()

        self._random_delay()
        return response.text

    def _fetch_with_selenium(self, url: str) -> str:
        """Fetch with Selenium for JavaScript-heavy sites"""
        options = Options()

        if self.config.headless:
            options.add_argument("--headless")

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={self.ua.random}")

        if self.config.disable_images:
            prefs = {"profile.managed_default_content_settings.images": 2}
            options.add_experimental_option("prefs", prefs)

        # Anti-detection
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
            },
        )

        try:
            driver.get(url)
            # Wait for page to load
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            html = driver.page_source
            return html
        finally:
            driver.quit()
            self._random_delay()

    def _parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML with BeautifulSoup"""
        return BeautifulSoup(html, "lxml")

    def _calculate_checksum(self, data: Dict) -> str:
        """Calculate MD5 checksum for deduplication"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _is_duplicate(self, checksum: str) -> bool:
        """Check if data already scraped using Redis"""
        key = f"checksum:{self.scraper_name}:{checksum}"
        exists = self.redis.exists(key)
        if not exists:
            # Store checksum for 7 days
            self.redis.setex(key, 604800, "1")
        return bool(exists)

    def _log_scraping_result(self, status: str, records_scraped: int, error: str = None):
        """Log scraping results to database"""
        query = """
            INSERT INTO scraping_logs
            (scraper_name, status, records_scraped, error_message, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        self.db.execute_query(query, (self.scraper_name, status, records_scraped, error, datetime.utcnow()))

    @abstractmethod
    def scrape(self) -> int:
        """Main scraping method - must be implemented by subclasses"""

    def run(self) -> int:
        """Run scraper with error handling and logging"""
        start_time = time.time()
        logger.info(f"Starting {self.scraper_name}")

        try:
            records_scraped = self.scrape()
            duration = time.time() - start_time

            logger.info(f"{self.scraper_name} completed: {records_scraped} records " f"in {duration:.2f}s")

            self._log_scraping_result("success", records_scraped)
            return records_scraped

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{self.scraper_name} failed after {duration:.2f}s: {e}", exc_info=True)
            self._log_scraping_result("failed", 0, str(e))
            raise
