"""E2E: Authentication flow."""

import pytest
import requests


def test_login_valid_credentials(api_url, wait_for_api):
    resp = requests.post(
        f"{api_url}/api/v1/auth/login",
        json={"email": "demo@betting-system.local", "password": "demo1234"},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data.get("token_type", "").lower() == "bearer"


def test_login_invalid_password_returns_401(api_url, wait_for_api):
    resp = requests.post(
        f"{api_url}/api/v1/auth/login",
        json={"email": "demo@betting-system.local", "password": "wrongpassword"},
        timeout=10,
    )
    assert resp.status_code == 401


def test_login_unknown_user_returns_401(api_url, wait_for_api):
    resp = requests.post(
        f"{api_url}/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "whatever"},
        timeout=10,
    )
    assert resp.status_code == 401


def test_protected_endpoint_with_valid_token(api_url, auth_token, wait_for_api):
    resp = requests.get(
        f"{api_url}/api/v1/predictions",
        headers={"Authorization": f"Bearer {auth_token}"},
        timeout=10,
    )
    assert resp.status_code == 200


def test_protected_endpoint_with_invalid_token_returns_401(api_url, wait_for_api):
    resp = requests.get(
        f"{api_url}/api/v1/predictions",
        headers={"Authorization": "Bearer invalid.token.here"},
        timeout=10,
    )
    assert resp.status_code == 401


def test_token_refresh(api_url, auth_token, wait_for_api):
    resp = requests.post(
        f"{api_url}/api/v1/auth/refresh",
        headers={"Authorization": f"Bearer {auth_token}"},
        timeout=10,
    )
    # 200 or 501 if not implemented
    assert resp.status_code in (200, 501)
    if resp.status_code == 200:
        assert "access_token" in resp.json()
