"""Shared fixtures for end-to-end integration tests."""

import os
import time

import psycopg2
import pytest
import redis as redis_lib
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://betting_user:change_me_in_production_MUST_BE_STRONG@localhost:5432/betting_system",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Demo credentials seeded by seed_dev_data.py
DEMO_EMAIL = "demo@betting-system.local"
DEMO_PASSWORD = "demo1234"


@pytest.fixture(scope="session")
def api_url():
    return API_BASE


@pytest.fixture(scope="session")
def db():
    """Live database connection for the test session."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def redis_client():
    client = redis_lib.from_url(REDIS_URL)
    yield client
    client.close()


@pytest.fixture(scope="session")
def wait_for_api(api_url):
    """Block until the API /health endpoint responds."""
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            resp = requests.get(f"{api_url}/health", timeout=3)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    pytest.fail(f"API at {api_url} did not become ready within 60 seconds")


@pytest.fixture(scope="session")
def auth_token(api_url, wait_for_api):
    """Return a valid JWT for the demo user."""
    resp = requests.post(
        f"{api_url}/api/v1/auth/login",
        json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD},
        timeout=10,
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def authed(api_url, auth_token):
    """Return a requests.Session pre-configured with the auth token."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {auth_token}"
    session.headers["Content-Type"] = "application/json"
    return session
