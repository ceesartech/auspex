"""E2E: liveness/health endpoints."""


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "name" in body or "status" in body or "version" in body


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    # Health endpoint should at least include a status field.
    assert body.get("status") in {"ok", "healthy", "up", "ready"} or "status" in body
