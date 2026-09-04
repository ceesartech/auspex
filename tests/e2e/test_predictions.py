"""E2E: Predictions endpoint pipeline."""

import pytest


def test_list_predictions(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/predictions", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, (list, dict))


def test_list_predictions_pagination(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/predictions?limit=5&offset=0", timeout=10)
    assert resp.status_code == 200


def test_filter_predictions_by_sport(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/predictions?sport=soccer", timeout=10)
    assert resp.status_code == 200


def test_filter_predictions_by_confidence(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/predictions?min_confidence=0.6", timeout=10)
    assert resp.status_code == 200
    predictions = resp.json()
    if isinstance(predictions, list):
        for pred in predictions:
            assert pred.get("confidence", 1.0) >= 0.6


def test_prediction_response_schema(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/predictions?limit=1", timeout=10)
    assert resp.status_code == 200
    predictions = resp.json()
    if not predictions:
        pytest.skip("No predictions available")
    pred = predictions[0] if isinstance(predictions, list) else predictions.get("items", [{}])[0]

    required_fields = {"match_id", "predicted_outcome", "confidence"}
    missing = required_fields - pred.keys()
    assert not missing, f"Missing fields in prediction: {missing}"


def test_prediction_probabilities_sum_to_one(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/predictions?limit=5", timeout=10)
    assert resp.status_code == 200
    predictions = resp.json()
    if isinstance(predictions, dict):
        predictions = predictions.get("items", [])
    for pred in predictions:
        total = pred.get("probability_home", 0) + pred.get("probability_draw", 0) + pred.get("probability_away", 0)
        if total > 0:
            assert abs(total - 1.0) < 0.01, f"Probabilities don't sum to 1: {total}"


def test_get_prediction_by_id(authed, api_url):
    # Get any prediction ID first
    resp = authed.get(f"{api_url}/api/v1/predictions?limit=1", timeout=10)
    assert resp.status_code == 200
    predictions = resp.json()
    if isinstance(predictions, dict):
        predictions = predictions.get("items", [])
    if not predictions:
        pytest.skip("No predictions available")

    pred_id = predictions[0]["prediction_id"]
    resp2 = authed.get(f"{api_url}/api/v1/predictions/{pred_id}", timeout=10)
    assert resp2.status_code == 200
    assert resp2.json()["prediction_id"] == pred_id


def test_get_nonexistent_prediction_returns_404(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/predictions/99999999", timeout=10)
    assert resp.status_code == 404
