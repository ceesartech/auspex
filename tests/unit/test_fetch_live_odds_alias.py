"""fetch_live_odds.find_match_id's alias tier: a merged club keeps the history
row's name ('Man City') and carries the odds/ESPN spelling ('Manchester City')
in teams.metadata.aliases (scripts/merge_soccer_teams.py), so the odds feed's
spelling must resolve through team_identity's alias index — the same
resolver fetch_upcoming and promote_raw use — or its fixtures stop matching.
No DB: a cursor that answers the two queries find_match_id makes."""

from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


flo = _load("fetch_live_odds", "fetch_live_odds.py")

T0 = datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc)
CITY, ARSENAL, UNITED = "t-city", "t-arsenal", "t-united"


def _fixture(mid, home, home_id, away, away_id, when=T0):
    return {
        "id": mid,
        "home_name": home,
        "away_name": away,
        "match_date": when,
        "home_team_id": home_id,
        "away_team_id": away_id,
    }


class _Cursor:
    def __init__(self, fixtures, aliases):
        self.fixtures, self.aliases, self.sql = fixtures, aliases, ""
        self.queries: list[str] = []

    def execute(self, sql, params=None):
        self.sql = sql
        self.queries.append(sql)

    def fetchall(self):
        if "metadata->'aliases'" in self.sql:
            return [{"id": tid, "aliases": names} for tid, names in self.aliases.items()]
        return list(self.fixtures)


def _find(fixtures, aliases, home="Manchester City", away="Arsenal", when=T0):
    return flo.find_match_id(_Cursor(fixtures, aliases), "soccer", home, away, when)


CITY_V_ARSENAL = _fixture("m1", "Man City", CITY, "Arsenal", ARSENAL)
WOLVES = "t-wolves"
WOLVES_V_CITY = _fixture("m3", "Wolves", WOLVES, "Man City", CITY)
BOTH = {WOLVES: ["wolverhampton wanderers"], CITY: ["manchester city"]}


def test_the_odds_spelling_resolves_through_the_kept_teams_alias():
    home, away = "Wolverhampton Wanderers", "Manchester City"
    assert _find([WOLVES_V_CITY], {}, home=home, away=away) is None  # the fuzzy ratio averages 0.55
    assert _find([WOLVES_V_CITY], BOTH, home=home, away=away) == "m3"
    assert _find([CITY_V_ARSENAL], {CITY: ["manchester city"]}) == "m1"


def test_alias_comparison_is_team_identitys_alias_form():
    aliases = {CITY: ["Manchester  CITY"]}  # case, whitespace and accents fold on both sides
    assert _find([CITY_V_ARSENAL], aliases, home="Manchester City ") == "m1"


def test_both_sides_may_be_aliases():
    fixture = _fixture("m1", "Man City", CITY, "Man United", UNITED)
    aliases = {CITY: ["manchester city"], UNITED: ["manchester united"]}
    assert _find([fixture], aliases, away="Manchester United") == "m1"


def test_an_alias_of_another_team_does_not_match():
    other = _fixture("m2", "Man United", UNITED, "Arsenal", ARSENAL)
    assert _find([other], {CITY: ["manchester city"]}) is None


def test_an_alias_two_teams_claim_resolves_to_neither():
    aliases = {**BOTH, UNITED: ["manchester city"]}
    assert _find([WOLVES_V_CITY], aliases, home="Wolverhampton Wanderers", away="Manchester City") is None


def test_the_exact_name_still_wins_and_skips_the_alias_query():
    fixture = _fixture("m1", "Manchester City", CITY, "Arsenal", ARSENAL)
    cur = _Cursor([fixture, CITY_V_ARSENAL], {CITY: ["manchester city"]})
    assert flo.find_match_id(cur, "soccer", "Manchester City", "Arsenal", T0) == "m1"
    assert len(cur.queries) == 1


def test_several_alias_hits_take_the_nearest_kickoff(caplog):
    far = _fixture("m0", "Man City", CITY, "Arsenal", ARSENAL, T0 - timedelta(days=1))
    near = _fixture("m1", "Man City", CITY, "Arsenal", ARSENAL, T0 + timedelta(minutes=30))
    with caplog.at_level(logging.WARNING, logger="fetch_live_odds"):
        assert _find([far, near], {CITY: ["manchester city"]}) == "m1"
    assert "match through aliases" in caplog.text


def test_no_aliases_leaves_the_fuzzy_tier_as_it_was():
    fixture = _fixture("m1", "Manchester Cty", CITY, "Arsenal", ARSENAL)
    assert _find([fixture], {}) == "m1"
