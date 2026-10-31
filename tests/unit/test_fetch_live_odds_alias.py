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

import pytest

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


SPURS, SPURS_W = "t-spurs", "t-spurs-women"
SPURS_V_WOLVES = _fixture("m4", "Spurs", SPURS, "Wolves", WOLVES)


def test_an_alias_two_teams_claim_neither_resolves_nor_scores():
    event = {"home": "Tottenham Hotspur", "away": "Wolverhampton Wanderers"}
    mine = {SPURS: ["tottenham hotspur"], WOLVES: ["wolverhampton wanderers"]}
    assert _find([SPURS_V_WOLVES], mine, **event) == "m4"
    shared = {**mine, SPURS_W: ["Tottenham  Hotspur"]}
    # the alias tier refuses the shared alias, and tier 3 scores 'spurs' (0.45) not the alias: 0.73 < 0.8
    assert _find([SPURS_V_WOLVES], shared, **event) is None


def test_two_aliases_one_odds_form_apart_are_shared(caplog):
    aliases = flo.TeamAliases("soccer", {"st. louis city": {"a"}, "st louis city": {"b"}})
    with caplog.at_level(logging.ERROR, logger="fetch_live_odds"):
        assert aliases.resolve("St Louis City") is None and aliases.by_team == {}
    assert "is claimed by 2 soccer teams (a, b)" in caplog.text


# Prod's fetch_live_odds unmatched-event logs (2026-10): the odds feed's spelling, ESPN's spelling (the twin the
# fixture carried before the merge, then the kept team's alias), the kept football-data name. Each matched its
# fixture before the merge, and none did after it under efeee9d's alias tier (alias_form, no fuzzy alias).
PROD_SPELLINGS = [
    ("Brighton and Hove Albion", "Brighton & Hove Albion", "Brighton"),
    ("Paris Saint Germain", "Paris Saint-Germain", "Paris SG"),
    ("FC Twente Enschede", "FC Twente", "Twente"),
    ("Kyoto Purple Sanga", "Kyoto Sanga", "Kyoto"),
    ("Deportivo La Coruña", "Deportivo", "La Coruna"),
    ("Real Racing Club de Santander", "Racing Santander", "Santander"),
    ("Vitória SC", "Vitória de Guimaraes", "Guimaraes"),
    ("Henan FC", "Henan", "Henan Songshan Longmen"),
    ("Atlético Huracán", "Huracán", "Huracan"),
    ("Belgrano de Cordoba", "Belgrano (Córdoba)", "Belgrano"),
    ("Leuven", "OH Leuven", "Oud-Heverlee Leuven"),
]


@pytest.mark.parametrize("side", ["home", "away"])
@pytest.mark.parametrize("odds, espn, kept", PROD_SPELLINGS)
def test_the_odds_feeds_spelling_still_matches_once_the_club_is_merged(odds, espn, kept, side):
    def fixture(name):
        if side == "home":
            return _fixture("m1", name, CITY, "Arsenal", ARSENAL)
        return _fixture("m1", "Arsenal", ARSENAL, name, CITY)

    event = {"home": odds, "away": "Arsenal"} if side == "home" else {"home": "Arsenal", "away": odds}
    assert _find([fixture(espn)], {}, **event) == "m1"  # before: the fixture named the twin
    merged = {CITY: [" ".join(espn.lower().split())]}  # the alias merge_soccer_teams stores
    assert _find([fixture(kept)], merged, **event) == "m1"
    assert _find([fixture(kept)], {}, **event) is None  # the alias is what carries it


def test_the_alias_tier_folds_ampersands_dots_and_accents():
    aliases = flo.TeamAliases("soccer", {"brighton & hove albion": {"b"}, "st. louis city sc": {"s"}})
    assert aliases.resolve("Brighton and Hove Albion") == "b"
    assert aliases.resolve("St Louis City SC") == "s"
    assert aliases.resolve("Brighton") is None
    # and on the incoming side: the feed's '&' and '.' meet aliases stored without them
    plain = flo.TeamAliases("soccer", {"brighton and hove albion": {"b"}, "st louis city sc": {"s"}, "malaga": {"m"}})
    assert plain.resolve("Brighton & Hove Albion") == "b" and plain.resolve("St. Louis City SC") == "s"
    assert plain.resolve("Málaga") == "m"


def test_tier_3_scores_a_side_against_its_aliases():
    fixture = _fixture("m1", "Paris SG", CITY, "Arsenal", ARSENAL)
    assert _find([fixture], {CITY: ["paris saint-germain"]}, home="Paris Saint Germain") == "m1"
    assert _find([fixture], {CITY: ["olympique lyonnais"]}, home="Paris Saint Germain") is None


def test_a_not_yet_merged_accent_twin_does_not_join_an_alias_hit():
    # Juarez carries 'fc juarez'; a fixture of the 'FC Juárez' twin (merged by the next pair) must not tie with it
    kept = _fixture("m1", "Juarez", CITY, "Club Tijuana", ARSENAL, T0 + timedelta(minutes=5))
    twin = _fixture("m2", "FC Juárez", "t-twin", "Club Tijuana", ARSENAL)  # nearer the event's kickoff
    assert _find([twin, kept], {CITY: ["fc juarez"]}, home="FC Juarez", away="Club Tijuana") == "m1"


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
