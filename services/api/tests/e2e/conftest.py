"""End-to-end test fixtures for the API service.

These tests exercise the real FastAPI application against a real Postgres and
Redis. They are marked `integration` so the unit-test run filters them out.

Required environment variables (CI provides them; locally use docker compose):

    DATABASE_URL  postgresql://user:pass@host:port/db
    REDIS_URL     redis://host:port/db_index
    JWT_SECRET    any non-empty string (defaults to a fixed test value)
    USER_DOB      YYYY-MM-DD (defaults to 1994-05-09)

Each test runs against a freshly-truncated database, so order does not matter.
"""

import os
import sys
from pathlib import Path
from typing import Iterator

import pytest

# The default DOB and JWT secret used by the tests when not set in the
# environment. We set them BEFORE importing the app so config.Settings picks
# them up at module load.
os.environ.setdefault("JWT_SECRET", "e2e-test-secret-key")
os.environ.setdefault("USER_DOB", "1994-05-09")
os.environ.setdefault("DATABASE_URL", "postgresql://test_user:test_password@localhost:5432/betting_system_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

# Make `from main import app` resolve the same way the unit-test conftest does.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def pytest_collection_modifyitems(config, items):
    """Mark every test under tests/e2e/ as `integration` automatically."""
    e2e_dir = str(Path(__file__).parent.resolve())
    integration_marker = pytest.mark.integration
    for item in items:
        if str(Path(item.fspath).resolve()).startswith(e2e_dir):
            item.add_marker(integration_marker)


MIGRATIONS_DIR = Path(__file__).parent.parent.parent.parent / "data-ingestion" / "db" / "migrations"

# Tables we truncate between tests. Ordered so children come before parents
# (TRUNCATE ... CASCADE handles FK chains, but listing them keeps it explicit).
TABLES_TO_TRUNCATE = [
    "accumulator_legs",
    "accumulators",
    "betting_history",
    "betting_recommendations",
    "predictions",
    "user_preferences",
    "odds",
    "match_stats",
    "matches",
    "teams",
    "leagues",
    "lottery_draws",
    "weather",
    "referee_stats",
    "referees",
    "team_managers",
    "managers",
    "transfers",
    "player_match_stats",
    "player_availability",
    "players",
    "model_performance_logs",
    "model_retraining_logs",
    "data_quality_logs",
    "scraping_logs",
]


def _db_reachable(url: str) -> bool:
    """Return True iff we can open a TCP connection to the Postgres in `url`."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _redis_reachable(url: str) -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_services() -> None:
    """Skip the whole E2E suite if Postgres or Redis aren't available."""
    if not _db_reachable(os.environ["DATABASE_URL"]):
        pytest.skip(f"Postgres not reachable at {os.environ['DATABASE_URL']}", allow_module_level=True)
    if not _redis_reachable(os.environ["REDIS_URL"]):
        pytest.skip(f"Redis not reachable at {os.environ['REDIS_URL']}", allow_module_level=True)


@pytest.fixture(scope="session")
def engine():
    """A SQLAlchemy engine pointing at the test database."""
    from sqlalchemy import create_engine

    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _apply_migrations(engine) -> None:
    """Apply SQL migrations once per session, skipping if schema already exists.

    Migration 002 (indexes) is not idempotent — `CREATE INDEX` without
    `IF NOT EXISTS` raises if the index is already there. CI's separate
    migration step runs before pytest, so when running under CI the
    schema is already present and we no-op.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        already = conn.execute(text("SELECT to_regclass('public.leagues') AS t")).scalar_one()
    if already is not None:
        return  # schema is already in place

    for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
        sql = migration.read_text()
        with engine.begin() as conn:
            conn.execute(text(sql))


@pytest.fixture
def db(engine) -> Iterator:
    """A SQLAlchemy session with all tables truncated before the test runs."""
    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with engine.begin() as conn:
        # RESTART IDENTITY resets sequences; CASCADE drops dependent rows too.
        conn.execute(text("TRUNCATE TABLE " + ", ".join(TABLES_TO_TRUNCATE) + " RESTART IDENTITY CASCADE"))

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def redis_client() -> Iterator:
    """A real Redis client connected to the test DB index, flushed before use."""
    from redis import Redis

    client = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()
        client.close()


@pytest.fixture
def client(_apply_migrations, redis_client):
    """A FastAPI TestClient bound to the real application + lifespan."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client) -> dict:
    """Hit the real /login endpoint and return Authorization headers."""
    response = client.post(
        "/api/v1/user/login",
        json={
            "username": "owner",
            "password": "any-password-of-eight-or-more",
            "date_of_birth": os.environ["USER_DOB"],
        },
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
