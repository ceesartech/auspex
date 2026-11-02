"""Config pins for the disk guards (OPERATIONS.md "Disk maintenance").

There is no promtool/amtool in CI, so the YAML is parsed and the shape that
matters is pinned: DiskSpaceLow unchanged; DiskSpaceWarning at < 20% free for
30m as a host warning; Alertmanager mutes the warning while DiskSpaceLow
fires and routes it to Telegram like every other alert; Prometheus keeps its
30-day retention plus a 4GB size cap on the named volume, with no prod
overlay override; the api passes BACKUP_LOCAL_KEEP through. (The rules were
also checked and unit-tested with promtool 3.15.1 and amtool on 2026-11-02.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]


def load(rel: str):
    return yaml.safe_load((REPO / rel).read_text())


def host_rules() -> dict[str, dict]:
    groups = load("monitoring/alerts/infrastructure-alerts.yml")["groups"]
    rules = next(g for g in groups if g["name"] == "host_alerts")["rules"]
    return {r["alert"]: r for r in rules}


def squash(expr: str) -> str:
    return " ".join(expr.split())


def test_disk_space_low_is_unchanged():
    r = host_rules()["DiskSpaceLow"]
    assert squash(r["expr"]).endswith("< 0.10")
    assert r["for"] == "15m" and r["labels"] == {"severity": "critical", "component": "host"}


def test_disk_space_warning_fires_below_20_percent_for_30m():
    rules = host_rules()
    w = rules["DiskSpaceWarning"]
    assert w["for"] == "30m"
    assert w["labels"] == {"severity": "warning", "component": "host"}
    expr = squash(w["expr"])
    assert expr.endswith("< 0.20")
    # Same ratio (and the same filesystem selector) as the critical rule.
    assert expr.rsplit("<", 1)[0] == squash(rules["DiskSpaceLow"]["expr"]).rsplit("<", 1)[0]
    assert "OPERATIONS.md" in w["annotations"]["description"]


def test_alertmanager_routes_the_warning_and_mutes_it_under_the_critical():
    am = load("monitoring/alertmanager/alertmanager.yml")
    assert am["route"]["receiver"] == "telegram"  # warnings take the default route
    assert all("DiskSpace" not in str(r.get("matchers")) for r in am["route"].get("routes", []))
    inhibits = am["inhibit_rules"]
    assert {
        "source_matchers": ['alertname = "DiskSpaceLow"'],
        "target_matchers": ['alertname = "DiskSpaceWarning"'],
        "equal": ["component"],
    } in inhibits
    # The generic critical-mutes-warning rule is still there.
    assert any(i.get("equal") == ["alertname", "component"] for i in inhibits)


def test_prometheus_keeps_30d_and_caps_size_on_the_named_volume():
    base = load("docker-compose.yml")["services"]["prometheus"]
    cmd = base["command"]
    assert "--storage.tsdb.retention.time=30d" in cmd
    assert "--storage.tsdb.retention.size=4GB" in cmd
    assert "--storage.tsdb.path=/prometheus" in cmd
    assert "prometheus_data:/prometheus" in base["volumes"]
    prod = load("docker-compose.prod.yml")["services"]["prometheus"]
    assert "command" not in prod  # the overlay must not drop the caps


def test_api_passes_backup_local_keep():
    env = load("docker-compose.yml")["services"]["api"]["environment"]
    assert "BACKUP_LOCAL_KEEP=${BACKUP_LOCAL_KEEP:-3}" in env
    assert "BACKUP_LOCAL_RETENTION=${BACKUP_LOCAL_RETENTION:-7}" in env
