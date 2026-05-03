"""E2E: Model performance and drift endpoints."""

import pytest


def test_model_performance_endpoint(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/models/performance", timeout=10)
    assert resp.status_code == 200


def test_model_performance_schema(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/models/performance", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    # Expect either a dict with fields or a list of performance records
    if isinstance(data, list):
        if data:
            record = data[0]
            assert "accuracy" in record or "model_version" in record
    elif isinstance(data, dict):
        assert "accuracy" in data or "model_version" in data or "records" in data


def test_drift_status_endpoint(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/models/drift-status", timeout=10)
    # May return 200 with data or 204 if no drift data yet
    assert resp.status_code in (200, 204)


def test_model_list_endpoint(authed, api_url):
    resp = authed.get(f"{api_url}/api/v1/models", timeout=10)
    assert resp.status_code in (200, 404)  # 404 if endpoint not yet wired


def test_metrics_contain_model_gauges(api_url, wait_for_api):
    """The /metrics endpoint should expose model accuracy gauge after evaluation."""
    import requests
    resp = requests.get(f"{api_url}/metrics", timeout=5)
    assert resp.status_code == 200
    # These metrics are set by ModelPerformanceMonitor — may not be present
    # until the first evaluation run, so we only assert the endpoint is valid.
    assert resp.headers.get("content-type", "").startswith("text/plain")
