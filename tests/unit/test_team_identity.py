"""Accent-insensitive team identity (scripts/team_identity.py) and its use by
both fixture loaders: ESPN's "Montreal Canadiens" must reuse the NHL API's
"Montréal Canadiens" row instead of minting a history-less twin."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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

HISTORY = {"id": "a6b6fabf", "normalized_name": "montréal canadiens"}
OTHER = {"id": "b0b0b0b0", "normalized_name": "toronto maple leafs"}


class FakeCursor:
    """Answers the exact lookup, the accent-candidate query and the insert."""

    def __init__(self, teams):
        self.teams = list(teams)
        self.sql: list[str] = []
        self._result = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        if "INSERT INTO teams" in sql:
            self._result = [{"id": "NEW"}]
        elif "normalized_name !~" in sql:
            folded = params[1]
            self._result = [
                t for t in self.teams if t["normalized_name"] == folded or not t["normalized_name"].isascii()
            ]
        elif "WHERE normalized_name = %s" in sql:
            self._result = [t for t in self.teams if t["normalized_name"] == params[0]]
        else:
            self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


def test_strip_accents():
    assert ti.strip_accents("Montréal Canadiens") == "Montreal Canadiens"
    assert ti.strip_accents("plain") == "plain"


def test_finds_the_accented_row_for_an_unaccented_name():
    assert ti.find_team_accent_insensitive(FakeCursor([HISTORY, OTHER]), "nhl", "montreal canadiens") == "a6b6fabf"


def test_no_match_returns_none():
    assert ti.find_team_accent_insensitive(FakeCursor([OTHER]), "nhl", "montreal canadiens") is None


def test_an_ambiguous_fold_never_merges():
    twins = [HISTORY, {"id": "c1", "normalized_name": "montreal canadiens"}]
    assert ti.find_team_accent_insensitive(FakeCursor(twins), "nhl", "montréal canadiens") is None


def test_espn_resolver_reuses_the_accented_row_instead_of_inserting():
    fu = _load("fetch_upcoming")
    cfg = fu.SPORT_CONFIGS["nhl"]
    cur = FakeCursor([HISTORY, OTHER])
    assert fu.ensure_team(cur, cfg, "Montreal Canadiens", "league-1") == "a6b6fabf"
    assert not any("INSERT INTO teams" in s for s in cur.sql)


def test_nhl_loader_reuses_the_unaccented_row_and_merges_its_api_id():
    nl = _load("load_nhl_historical")
    espn_only = {"id": "b8284384", "normalized_name": "montreal canadiens"}
    cur = FakeCursor([espn_only])
    payload = {"placeName": {"default": "Montréal"}, "commonName": {"default": "Canadiens"}, "abbrev": "MTL", "id": 8}
    assert nl.ensure_nhl_team(cur, payload, "league-1") == "b8284384"
    assert any(s.startswith("UPDATE teams SET external_ids") for s in cur.sql)
    assert not any("INSERT INTO teams" in s for s in cur.sql)
