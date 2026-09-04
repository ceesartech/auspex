"""E2E: Matches endpoints."""

from datetime import date

import pytest


def test_upcoming_matches(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/matches/upcoming", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, (list, dict))


def test_match_schema(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/matches/upcoming?limit=1", timeout=10)
    assert resp.status_code == 200
    matches = resp.json()
    if isinstance(matches, dict):
        matches = matches.get("items", [])
    if not matches:
        pytest.skip("No upcoming matches in seed data")

    match = matches[0]
    required = {"match_id", "match_date"}
    missing = required - match.keys()
    assert not missing, f"Missing match fields: {missing}"


def test_match_detail(authed, api_url, db):
    cur = db.cursor()
    cur.execute("SELECT match_id FROM matches LIMIT 1")
    row = cur.fetchone()
    if not row:
        pytest.skip("No matches in DB")

    match_id = row[0]
    resp = authed.get(f"{api_url}/api/v1/matches/{match_id}", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["match_id"] == match_id


def test_match_not_found(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/matches/99999999", timeout=10)
    assert resp.status_code == 404


def test_matches_filter_by_date(authed, api_url):
    today = date.today().isoformat()
    resp = authed.get(f"{api_url}/api/v1/matches/upcoming?date={today}", timeout=10)
    assert resp.status_code == 200
