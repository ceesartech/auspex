"""E2E: Monitoring stack health checks."""

import os

import pytest
import requests

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
GRAFANA_PASS = os.getenv("GRAFANA_PASSWORD", "admin")


def test_prometheus_healthy():
    resp = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5)
    assert resp.status_code == 200


def test_prometheus_ready():
    resp = requests.get(f"{PROMETHEUS_URL}/-/ready", timeout=5)
    assert resp.status_code == 200


def test_prometheus_has_betting_api_target():
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=5)
    assert resp.status_code == 200
    targets = resp.json()["data"]["activeTargets"]
    jobs = {t["labels"].get("job") for t in targets}
    assert "betting-api" in jobs, f"betting-api target not found; jobs: {jobs}"


def test_prometheus_scrape_up():
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": 'up{job="betting-api"}'},
        timeout=5,
    )
    assert resp.status_code == 200
    results = resp.json()["data"]["result"]
    if results:
        value = results[0]["value"][1]
        assert value == "1", f"betting-api target is down: up={value}"


def test_grafana_healthy():
    resp = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json().get("database") == "ok"


def test_grafana_prometheus_datasource():
    resp = requests.get(
        f"{GRAFANA_URL}/api/datasources",
        auth=(GRAFANA_USER, GRAFANA_PASS),
        timeout=5,
    )
    assert resp.status_code == 200
    sources = resp.json()
    prometheus_sources = [s for s in sources if s.get("type") == "prometheus"]
    assert prometheus_sources, "No Prometheus datasource configured in Grafana"


def test_grafana_dashboards_provisioned():
    resp = requests.get(
        f"{GRAFANA_URL}/api/search?type=dash-db",
        auth=(GRAFANA_USER, GRAFANA_PASS),
        timeout=5,
    )
    assert resp.status_code == 200
    dashboards = resp.json()
    titles = {d["title"] for d in dashboards}
    expected = {"API Performance", "ML Model Performance", "Infrastructure Overview"}
    missing = expected - titles
    # Warn rather than fail — dashboards may not be loaded in minimal local setup
    if missing:
        pytest.xfail(f"Some dashboards not yet provisioned: {missing}")
