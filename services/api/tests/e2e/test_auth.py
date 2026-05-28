"""E2E: login + auth-required endpoint behavior."""


def test_login_with_correct_password_returns_token(client, seeded_user):
    response = client.post(
        "/api/v1/user/login",
        json={"username": seeded_user["username"], "password": seeded_user["password"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and body["access_token"]
    assert body["user"]["username"] == seeded_user["username"]


def test_login_with_wrong_password_is_unauthorized(client, seeded_user):
    response = client.post(
        "/api/v1/user/login",
        json={"username": seeded_user["username"], "password": "wrong-password"},
    )
    assert response.status_code == 401


def test_login_with_unknown_user_is_unauthorized(client, _apply_migrations):
    response = client.post(
        "/api/v1/user/login",
        json={"username": "nobody-known", "password": "anything-password"},
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
