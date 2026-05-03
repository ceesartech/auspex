"""E2E: login + auth-required endpoint behavior."""

import os


def test_login_with_correct_dob_returns_token(client):
    response = client.post(
        "/api/v1/user/login",
        json={
            "username": "owner",
            "password": "any-password-of-eight-or-more",
            "date_of_birth": os.environ["USER_DOB"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and body["access_token"]
    assert body["user"]["username"] == "owner"


def test_login_with_wrong_dob_is_unauthorized(client):
    response = client.post(
        "/api/v1/user/login",
        json={
            "username": "owner",
            "password": "any-password-of-eight-or-more",
            "date_of_birth": "1900-01-01",
        },
    )
    assert response.status_code == 401


def test_protected_endpoint_without_token_is_rejected(client):
    response = client.get("/api/v1/user/preferences")
    # FastAPI's HTTPBearer returns 403 when the Authorization header is missing.
    assert response.status_code in (401, 403)


def test_protected_endpoint_with_invalid_token_is_rejected(client):
    response = client.get(
        "/api/v1/user/preferences",
        headers={"Authorization": "Bearer not.a.real.token"},
    )
    assert response.status_code == 401


def test_protected_endpoint_with_valid_token_is_accepted(client, auth_headers, db):
    response = client.get("/api/v1/user/preferences", headers=auth_headers)
    assert response.status_code == 200
    # No preferences seeded → empty mapping.
    assert response.json() == {}
