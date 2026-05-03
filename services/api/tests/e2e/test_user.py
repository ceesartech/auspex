"""E2E: user preferences PUT/GET round-trip and betting history list."""


def test_update_then_read_preferences(client, auth_headers, db):
    update = {"bankroll": 1500.0, "risk_tolerance": "MEDIUM"}
    put = client.put("/api/v1/user/preferences", json=update, headers=auth_headers)
    assert put.status_code in (200, 204), put.text

    get = client.get("/api/v1/user/preferences", headers=auth_headers)
    assert get.status_code == 200
    prefs = get.json()
    # Both keys we set should appear in the response.
    assert "bankroll" in prefs
    assert "risk_tolerance" in prefs
    assert prefs["bankroll"]["value"] == 1500.0
    assert prefs["risk_tolerance"]["value"] == "MEDIUM"


def test_partial_update_does_not_clobber_other_keys(client, auth_headers, db):
    client.put(
        "/api/v1/user/preferences",
        json={"bankroll": 2000.0, "risk_tolerance": "HIGH"},
        headers=auth_headers,
    )
    # Update only bankroll.
    client.put(
        "/api/v1/user/preferences",
        json={"bankroll": 2500.0},
        headers=auth_headers,
    )

    prefs = client.get("/api/v1/user/preferences", headers=auth_headers).json()
    assert prefs["bankroll"]["value"] == 2500.0
    # risk_tolerance should still be present from the first update.
    assert prefs["risk_tolerance"]["value"] == "HIGH"
