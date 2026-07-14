"""Pytest configuration.

Puts the service root on sys.path so tests can `from src.…` import. The
scraper mock fixtures (mock_config/mock_db/mock_redis) were removed with the
Selenium scrapers in the 2026-07 audit — nothing surviving used them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
