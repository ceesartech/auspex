"""Team identity SQL against a real Postgres (skipped unless
AUSPEX_TEST_PG_URL names a throwaway database; nothing is left behind: every
test runs in its own schema inside a transaction that is rolled back).

The FakeDB in test_team_identity_soccer.py answers the country query by
re-implementing its rule, so these tests are what pin the SQL itself: the
first-league-OR-latest-league rule of COUNTRY_CANDIDATES_SQL, the guards of
STAMP_ESPN_SQL, and ensure_team's insert-then-stamp path."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor  # noqa: E402

pytestmark = pytest.mark.integration

PG_URL = os.environ.get("AUSPEX_TEST_PG_URL")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ti = _load("team_identity")
fu = _load("fetch_upcoming")

DDL = """
CREATE TABLE leagues (id text PRIMARY KEY, name text NOT NULL, country text, sport text NOT NULL DEFAULT 'soccer');
CREATE TABLE teams (
    id text PRIMARY KEY DEFAULT md5(random()::text),
    name text NOT NULL, normalized_name text NOT NULL, league_id text REFERENCES leagues(id),
    sport text NOT NULL DEFAULT 'soccer', external_ids jsonb DEFAULT '{}', metadata jsonb DEFAULT '{}',
    updated_at timestamptz DEFAULT now(), UNIQUE (normalized_name, sport));
CREATE TABLE matches (
    id serial PRIMARY KEY, league_id text REFERENCES leagues(id), home_team_id text REFERENCES teams(id),
    away_team_id text REFERENCES teams(id), match_date timestamptz NOT NULL, status text DEFAULT 'finished');
INSERT INTO leagues VALUES ('eng', 'Premier League', 'England'), ('esp', 'La Liga', 'Spain'),
    ('sco', 'Premiership', 'Scotland'), ('mex', 'Liga MX', 'Mexico'), ('bra', 'Brasileirao', 'Brazil');
"""


@pytest.fixture
def cur():
    if not PG_URL:
        pytest.skip("AUSPEX_TEST_PG_URL not set (a throwaway Postgres for the SQL tests)")
    conn = psycopg2.connect(PG_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("CREATE SCHEMA ti_test; SET LOCAL search_path TO ti_test;")
            c.execute(DDL)
            yield c
    finally:
        conn.rollback()
        conn.close()


def _team(cur, team_id, name, league_id=None, espn=None, sport="soccer"):
    cur.execute(
        "INSERT INTO teams (id, name, normalized_name, league_id, sport, external_ids) VALUES (%s, %s, %s, %s, %s, %s)",
        (
            team_id,
            name,
            fu.normalize_team(name),
            league_id,
            sport,
            psycopg2.extras.Json({"espn": espn} if espn else {}),
        ),
    )


def _match(cur, league_id, home, away, when):
    cur.execute(
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date) VALUES (%s, %s, %s, %s)",
        (league_id, home, away, when),
    )


def test_country_candidates_are_the_first_league_or_the_latest_match(cur):
    _team(cur, "a", "Promoted", league_id="esp")  # first league Spain, latest match England
    _team(cur, "b", "Relegated", league_id="eng")  # first league England, latest match Scotland
    _team(cur, "c", "Elsewhere", league_id="esp")  # Spain, then Scotland: never England
    _team(cur, "d", "Homeless")  # no first league, latest match England
    _team(cur, "x", "Opponent", league_id="mex")
    _match(cur, "esp", "a", "x", "2024-01-01")
    _match(cur, "eng", "x", "a", "2026-01-01")  # away side counts as much as home
    _match(cur, "eng", "b", "x", "2024-01-01")
    _match(cur, "sco", "b", "x", "2026-01-01")
    _match(cur, "sco", "x", "c", "2026-01-01")
    _match(cur, "eng", "d", "x", "2026-02-01")

    def ids(country):
        return sorted(r["id"] for r in ti.country_candidates(cur, "soccer", country) if r["id"] != "x")

    assert ids("England") == ["a", "b", "d"]
    assert ids("Spain") == ["a", "c"]
    assert ids("Scotland") == ["b", "c"]
    latest = {r["id"]: r["latest_date"].year for r in ti.country_candidates(cur, "soccer", "England")}
    assert latest["a"] == 2026 and latest["b"] == 2026  # the latest match, home or away


def test_the_stamp_guard(cur):
    _team(cur, "free", "Free")
    _team(cur, "held", "Held", espn="111")
    _team(cur, "nhl", "Nhl Twin", sport="nhl", espn="222")
    assert ti.stamp_espn_id(cur, "free", "222")  # the same id in another sport is no conflict
    assert not ti.stamp_espn_id(cur, "held", "333")  # never overwrites
    _team(cur, "other", "Other")
    assert not ti.stamp_espn_id(cur, "other", "111")  # never a second holder
    assert ti.stamp_espn_id(cur, "held", "111")  # already there: fine
    cur.execute("SELECT id, external_ids->>'espn' AS espn FROM teams ORDER BY id")
    assert {r["id"]: r["espn"] for r in cur.fetchall()} == {
        "free": "222",
        "held": "111",
        "nhl": "222",
        "other": None,
    }


def test_ensure_team_end_to_end_seed_and_namesake(cur, monkeypatch):
    monkeypatch.setattr(ti, "_SEED_CACHE", {})
    for key in list(fu.IDENTITY_STATS):
        monkeypatch.setitem(fu.IDENTITY_STATS, key, 0)
    soccer = fu.SPORT_CONFIGS["soccer"]
    _team(cur, "santos", "Santos", league_id="bra")
    # Santos Laguna (seeded 225) arrives as 'Santos' before Santos FC: no 'Santos Laguna' row exists here,
    # so the seed creates it rather than handing the 'Santos' row to the Mexican club.
    laguna = fu.ensure_team(cur, soccer, "Santos", "mex", espn_id="225")
    assert laguna != "santos"
    assert fu.ensure_team(cur, soccer, "Santos", "bra", espn_id="2674") == "santos"
    # An unseeded club whose name belongs to another ESPN team gets a distinct, stamped row.
    _team(cur, "pl", "Arsenal", league_id="eng", espn="359")
    sarandi = fu.ensure_team(cur, soccer, "Arsenal", "esp", espn_id="4242", alt_names=("Arsenal",))
    cur.execute("SELECT id, name, external_ids->>'espn' AS espn FROM teams WHERE id IN %s", ((laguna, sarandi),))
    rows = {r["id"]: (r["name"], r["espn"]) for r in cur.fetchall()}
    assert rows == {laguna: ("Santos Laguna", "225"), sarandi: ("Arsenal (ESPN 4242)", "4242")}
    assert fu.IDENTITY_STATS["team_namesakes"] == 1 and fu.IDENTITY_STATS["team_conflicts"] == 0
