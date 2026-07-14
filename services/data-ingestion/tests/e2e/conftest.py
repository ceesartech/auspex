"""End-to-end test fixtures for data-ingestion.

Tests in this directory hit a real Postgres + Redis (provided by CI service
containers, or by `docker compose up` locally). They are auto-marked
`integration` so unit-test runs filter them out.

Required env vars (defaults match CI):

    DATABASE_URL  postgresql://test_user:test_password@localhost:5432/betting_system_test
    REDIS_URL     redis://localhost:6379/15
"""

import os
import socket
import sys
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test_user:test_password@localhost:5432/betting_system_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

# Match the unit-test conftest setup so `from src.X import Y` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "db" / "migrations"


def pytest_collection_modifyitems(config, items):
    """Auto-mark every test under tests/e2e/ as integration."""
    e2e_dir = str(Path(__file__).parent.resolve())
    integration_marker = pytest.mark.integration
    for item in items:
        if str(Path(item.fspath).resolve()).startswith(e2e_dir):
            item.add_marker(integration_marker)


def _tcp_reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_services() -> None:
    if not _tcp_reachable(os.environ["DATABASE_URL"], 5432):
        pytest.skip(f"Postgres not reachable at {os.environ['DATABASE_URL']}", allow_module_level=True)
    if not _tcp_reachable(os.environ["REDIS_URL"], 6379):
        pytest.skip(f"Redis not reachable at {os.environ['REDIS_URL']}", allow_module_level=True)


@pytest.fixture(scope="session")
def engine():
    from sqlalchemy import create_engine

    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _apply_migrations(engine) -> None:
    """Apply SQL migrations once per session, skipping if schema already exists.

    Migration 002 isn't idempotent (no `IF NOT EXISTS` on CREATE INDEX), so
    we detect the `leagues` table and no-op when it's there. CI's separate
    migration step runs before pytest.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        already = conn.execute(text("SELECT to_regclass('public.leagues') AS t")).scalar_one()
    if already is not None:
        return

    for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
        sql = migration.read_text()
        with engine.begin() as conn:
            conn.execute(text(sql))


@pytest.fixture
def clean_tables(engine) -> Iterator:
    """Truncate scraping_logs + matches/teams/leagues before the test."""
    from sqlalchemy import text

    tables = [
        "odds",
        "match_stats",
        "matches",
        "teams",
        "leagues",
        "scraping_logs",
        "data_quality_logs",
    ]
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE " + ", ".join(tables) + " RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture
def db_config():
    """A minimal ingestion config for tests (DatabaseManager only reads
    database_url; the extra fields exercise the rate-limit defaults)."""
    from src.core.config import ScraperConfig

    return ScraperConfig(
        database_url=os.environ["DATABASE_URL"],
        redis_url=os.environ["REDIS_URL"],
        request_timeout=5,
        max_retries=1,
        min_delay=0.0,
        max_delay=0.0,
        rotate_user_agents=False,
    )


@pytest.fixture
def db_manager(db_config, clean_tables):
    """A real DatabaseManager pointing at the test DB."""
    from src.core.database import DatabaseManager

    return DatabaseManager(db_config)


@pytest.fixture
def redis_client() -> Iterator:
    """Real Redis client; flushed before and after each test."""
    from redis import Redis

    client = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()
        client.close()
