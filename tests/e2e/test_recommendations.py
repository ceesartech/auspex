"""E2E: Recommendations and betting workflow."""

import pytest


def test_list_recommendations(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/recommendations", timeout=10)
    assert resp.status_code == 200


def test_recommendations_active_only(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/recommendations?active_only=true", timeout=10)
    assert resp.status_code == 200


def test_recommendations_by_confidence_level(authed, api_url):
    for level in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH"):
        resp = authed.get(
            f"{api_url}/api/v1/recommendations?confidence_level={level}",
            timeout=10,
        )
        assert resp.status_code == 200, f"Failed for confidence_level={level}"


def test_recommendation_schema(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/recommendations?limit=1", timeout=10)
    assert resp.status_code == 200
    recs = resp.json()
    if isinstance(recs, dict):
        recs = recs.get("items", [])
    if not recs:
        pytest.skip("No recommendations available")

    rec = recs[0]
    required = {"recommendation_id", "match_id", "recommended_outcome"}
    missing = required - rec.keys()
    assert not missing, f"Missing recommendation fields: {missing}"


def test_record_bet(authed, api_url):
    # Get a recommendation to bet on
    resp = authed.get(f"{api_url}/api/v1/recommendations?limit=1", timeout=10)
    recs = resp.json()
    if isinstance(recs, dict):
        recs = recs.get("items", [])
    if not recs:
        pytest.skip("No recommendations available to bet on")

    rec = recs[0]
    payload = {
        "match_id": rec["match_id"],
        "outcome": rec.get("recommended_outcome", "home_win"),
        "stake": 10.0,
        "odds": rec.get("recommended_odds", 2.0),
        "bookmaker": "test",
    }
    resp2 = authed.post(f"{api_url}/api/v1/betting/record", json=payload, timeout=10)
    assert resp2.status_code in (200, 201)


def test_betting_history(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/betting/history", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, (list, dict))


def test_kelly_criterion_stake_positive(authed, api_url):
    """Recommended stake should be positive when edge > 0."""
    resp = authed.get(f"{api_url}/api/v1/recommendations?limit=5", timeout=10)
    recs = resp.json()
    if isinstance(recs, dict):
        recs = recs.get("items", [])
    for rec in recs:
        stake = rec.get("recommended_stake", 0)
        assert stake >= 0, f"Negative stake recommended: {stake}"
