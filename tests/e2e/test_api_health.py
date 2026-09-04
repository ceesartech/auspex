"""E2E: API health and basic reachability."""

import pytest
import requests


def test_health_returns_200(api_url, wait_for_api):
    resp = requests.get(f"{api_url}/health", timeout=5)
    assert resp.status_code == 200


def test_health_body(api_url, wait_for_api):
    resp = requests.get(f"{api_url}/health", timeout=5)
    data = resp.json()
    assert data.get("status") in ("ok", "healthy")


def test_metrics_endpoint_reachable(api_url, wait_for_api):
    resp = requests.get(f"{api_url}/metrics", timeout=5)
    assert resp.status_code == 200
    # Prometheus text format starts with "# HELP"
    assert "# HELP" in resp.text or "# TYPE" in resp.text


def test_openapi_schema_available(api_url, wait_for_api):
    resp = requests.get(f"{api_url}/openapi.json", timeout=5)
    assert resp.status_code == 200
    schema = resp.json()
    assert "openapi" in schema
    assert "paths" in schema


def test_unauthenticated_predictions_returns_401(api_url, wait_for_api):
    resp = requests.get(f"{api_url}/api/v1/predictions", timeout=5)
    assert resp.status_code in (401, 403)
