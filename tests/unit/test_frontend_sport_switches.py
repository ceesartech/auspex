"""The frontend's horse-racing display switch mirrors the DAG switch.

services/frontend/src/lib/sport-switches.ts holds its own HORSE_RACING_ENABLED
(the frontend cannot import Python). It hides the Race cards nav entry and
puts the /races pages into a paused state. If it drifted from
services/data-ingestion/dags/sport_switches.py, the UI would either advertise
a dormant sport or hide a live one. Plain file reads: no Airflow, no Node."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAG_SWITCHES = ROOT / "services" / "data-ingestion" / "dags" / "sport_switches.py"
FRONTEND_SWITCHES = ROOT / "services" / "frontend" / "src" / "lib" / "sport-switches.ts"
FRONTEND_SRC = ROOT / "services" / "frontend" / "src"


def _dag_value() -> bool:
    spec = importlib.util.spec_from_file_location("sport_switches_for_frontend_mirror", DAG_SWITCHES)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return bool(mod.HORSE_RACING_ENABLED)


def _frontend_value() -> bool:
    source = FRONTEND_SWITCHES.read_text()
    found = re.findall(r"^export const HORSE_RACING_ENABLED(?:\s*:\s*boolean)?\s*=\s*(true|false)\s*;", source, re.M)
    assert len(found) == 1, f"expected one HORSE_RACING_ENABLED literal in {FRONTEND_SWITCHES}, found {found}"
    return found[0] == "true"


def test_the_frontend_switch_equals_the_dag_switch():
    assert _frontend_value() is _dag_value()


def test_the_frontend_switch_points_at_the_dag_switch():
    assert "services/data-ingestion/dags/sport_switches.py" in FRONTEND_SWITCHES.read_text()


def test_every_races_nav_entry_and_page_reads_the_switch():
    """The two nav lists filter through isRouteEnabled and both /races pages
    branch on HORSE_RACING_ENABLED, so flipping the one constant is enough."""
    for nav in ("components/layout/sidebar.tsx", "components/layout/header.tsx"):
        text = (FRONTEND_SRC / nav).read_text()
        assert "href: '/races'" in text and "isRouteEnabled(item.href)" in text, nav
    for page in ("app/(authenticated)/races/page.tsx", "app/(authenticated)/races/[id]/page.tsx"):
        text = (FRONTEND_SRC / page).read_text()
        assert "if (!HORSE_RACING_ENABLED) return <RacesPaused />;" in text, page
    hrefs = [p for p in FRONTEND_SRC.rglob("*.tsx") if re.search(r"href[:=]\s*['\"{`]+/races", p.read_text())]
    assert sorted(p.relative_to(FRONTEND_SRC).as_posix() for p in hrefs) == [
        "components/layout/header.tsx",
        "components/layout/sidebar.tsx",
        "components/races/race-card.tsx",  # rendered only by the races list page
    ]
