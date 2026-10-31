"""Soccer team identity: the ESPN team id, aliases and the club-name match
(scripts/team_identity.py) as fetch_upcoming.find_team / ensure_team and
promote_raw.upsert_teams use them.

ESPN's display names ('Manchester City', 'Tottenham Hotspur') missed the
football-data rows ('Man City', 'Tottenham'), so every first sight minted a
twin and later sweeps matched the twin: 320 soccer teams split from their
history from 2026-05-27. The examples below are prod's."""

from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
pr = _load("promote_raw")

SOCCER = fu.SPORT_CONFIGS["soccer"]
NOW = datetime.now(timezone.utc)
ACTIVE = NOW - timedelta(days=20)
EPL = "lg-epl"
UCL = "lg-ucl"
LEAGUES = {EPL: ("Premier League", "England"), UCL: ("UEFA Champions League", "International")}


def team(team_id, name, *, sport="soccer", espn=None, aliases=None, country="England", latest=ACTIVE):
    return {
        "id": team_id,
        "name": name,
        "normalized_name": fu.normalize_team(name),
        "sport": sport,
        "external_ids": {"espn": espn} if espn else {},
        "metadata": {"aliases": aliases} if aliases is not None else {},
        "league_id": None,
        "country": country,  # stands in for the first/latest league's country
        "latest_date": latest,
    }


class FakeDB:
    """A teams/leagues table answering exactly the SQL team identity issues.
    Unknown SQL fails the test, so a new query cannot slip in untested."""

    def __init__(self, teams=(), leagues=None, established=0):
        self.teams = [dict(t) for t in teams]
        self.leagues = dict(LEAGUES if leagues is None else leagues)
        self.established = established
        self.sql: list[str] = []
        self._result: list[dict] = []
        self._inserted = 0

    def _row(self, team_id):
        return next(t for t in self.teams if t["id"] == team_id)

    def _holder(self, row, espn):
        return next(
            (
                t["id"]
                for t in self.teams
                if t["id"] != row["id"] and t["sport"] == row["sport"] and t["external_ids"].get("espn") == espn
            ),
            None,
        )

    def execute(self, sql, params=None):
        q = " ".join(sql.split())
        self.sql.append(q)
        self._result = self._answer(q, params)

    def _answer(self, q, p):  # noqa: C901 — one branch per statement
        if q.startswith("UPDATE teams t SET external_ids"):
            row = self._row(p["id"])
            if "espn" in row["external_ids"] or self._holder(row, p["espn"]):
                return []
            row["external_ids"] = {**row["external_ids"], "espn": p["espn"]}
            return [{"id": row["id"]}]
        if "AS holder" in q:
            row = self._row(p["id"])
            return [{"espn": row["external_ids"].get("espn"), "holder": self._holder(row, p["espn"])}]
        if q.startswith("SELECT id FROM teams WHERE sport = %s AND external_ids @>"):
            sport, espn = p
            return [
                {"id": t["id"]} for t in self.teams if t["sport"] == sport and t["external_ids"].get("espn") == espn
            ]
        if q.startswith("SELECT external_ids->>'espn' AS espn FROM teams WHERE id"):
            return [{"espn": self._row(p[0])["external_ids"].get("espn")}]
        if "metadata->'aliases'" in q:
            return [
                {"id": t["id"], "aliases": t["metadata"]["aliases"]}
                for t in self.teams
                if t["sport"] == p[0] and "aliases" in t["metadata"]
            ]
        if "normalized_name !~" in q:
            sport, folded = p
            return [
                t
                for t in self.teams
                if t["sport"] == sport and (t["normalized_name"] == folded or not t["normalized_name"].isascii())
            ]
        if q.startswith("SELECT id FROM teams WHERE normalized_name = %s AND sport = %s"):
            return [{"id": t["id"]} for t in self.teams if t["normalized_name"] == p[0] and t["sport"] == p[1]]
        if q.startswith("SELECT name, country FROM leagues"):
            league = self.leagues.get(p[0])
            return [{"name": league[0], "country": league[1]}] if league else []
        if "LEFT JOIN LATERAL" in q:
            return [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "normalized_name": t["normalized_name"],
                    "espn": t["external_ids"].get("espn"),
                    "latest_date": t["latest_date"],
                }
                for t in self.teams
                if t["sport"] == p["sport"] and t["country"] == p["country"]
            ]
        if "count(*) AS n" in q:
            return [{"n": self.established}]
        if q.startswith("INSERT INTO teams"):
            name, norm, league_id, sport, ext = p
            ext = dict(getattr(ext, "adapted", ext))
            hit = next((t for t in self.teams if t["normalized_name"] == norm and t["sport"] == sport), None)
            if hit:
                return [{"id": hit["id"], "inserted": False, "espn": hit["external_ids"].get("espn")}]
            self._inserted += 1
            row = team(f"new-{self._inserted}", name, sport=sport, country=None, latest=None)
            row.update(league_id=league_id, external_ids=ext)
            self.teams.append(row)
            return [{"id": row["id"], "inserted": True, "espn": ext.get("espn")}]
        if q.startswith("SELECT id, normalized_name FROM teams WHERE sport = 'soccer'"):
            return [
                {"id": t["id"], "normalized_name": t["normalized_name"]} for t in self.teams if t["sport"] == "soccer"
            ]
        raise AssertionError(f"unexpected SQL: {q}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    def writes(self):
        return [q for q in self.sql if q.startswith(("INSERT", "UPDATE"))]


def _cands(*names, latest=ACTIVE):
    return [{"id": f"id-{n}", "name": n, "normalized_name": fu.normalize_team(n), "latest_date": latest} for n in names]


def _match(name, *existing, latest=ACTIVE):
    return ti.match_club_name(name, _cands(*existing, latest=latest))


# ── the club-name match ──────────────────────────────────────────────


class TestClubKey:
    def test_accents_generic_words_apostrophes_and_initials(self):
        assert ti.club_key("BK Häcken") == ("hacken",)
        assert ti.club_key("Nott'm Forest") == ("nottm", "forest")
        assert ti.club_key("D.C. United") == ("dc", "united")
        assert ti.club_key("Heart of Midlothian") == ("heart", "midlothian")

    def test_a_name_of_only_generic_words_keeps_them(self):
        assert ti.club_key("FC") == ("fc",)


class TestMatchClubName:
    def test_manchester_city_is_man_city_and_manchester_united_is_man_united(self):
        existing = ("Man City", "Man United", "Bristol City")
        assert _match("Manchester City", *existing).name == "Man City"
        assert _match("Manchester United", *existing).name == "Man United"

    def test_man_city_and_man_united_never_merge(self):
        assert _match("Manchester City", "Man United").outcome == "no-match"
        assert _match("Man City", "Man United").outcome == "no-match"
        assert _match("Man United", "Manchester City").outcome == "no-match"

    def test_kalamata_is_not_kalamaria(self):
        # 0.82 by whole-string ratio: no spelling tier exists any more.
        assert _match("Kalamata", "Kalamaria").outcome == "no-match"

    def test_espn_names_holding_the_history_name_match(self):
        assert _match("Tottenham Hotspur", "Tottenham", "Nott'm Forest").name == "Tottenham"
        assert _match("Borussia Dortmund", "Dortmund", "M'gladbach").name == "Dortmund"
        assert _match("1. FC Union Berlin", "Union Berlin", "Hertha").name == "Union Berlin"

    def test_initials_and_abbreviations(self):
        assert _match("Queens Park Rangers", "QPR").name == "QPR"
        assert _match("OH Leuven", "Oud-Heverlee Leuven").name == "Oud-Heverlee Leuven"
        assert _match("Atlanta United FC", "Atlanta Utd").name == "Atlanta Utd"
        assert _match("Paris Saint-Germain", "Paris SG", "Paris FC").name == "Paris SG"

    def test_a_shorter_incoming_name_never_matches_by_containment(self):
        # Extra words on OUR row tell clubs apart.
        assert _match("Yokohama FC", "Yokohama F. Marinos").outcome == "no-match"
        assert _match("Dundee", "Dundee United").outcome == "no-match"

    def test_containment_does_not_reach_a_defunct_club(self):
        defunct = NOW - timedelta(days=4 * 365)
        # Wuhan FC last played in 2022; Wuhan Three Towns is another club.
        assert _match("Wuhan Three Towns", "Wuhan FC", latest=defunct).outcome == "no-match"
        # The price: a club back after three idle years needs an alias.
        assert _match("Tottenham Hotspur", "Tottenham", latest=defunct).outcome == "no-match"
        assert _match("BK Häcken", "Hacken", latest=None).name == "Hacken"  # equal words need no recency

    def test_reserve_sides_never_merge_with_the_first_team(self):
        assert _match("Barcelona B", "Barcelona").outcome == "no-match"
        assert _match("FC Barcelona", "Barcelona B").outcome == "no-match"

    def test_a_candidate_inside_another_cannot_win_from_inside(self):
        # 'Paris' (Paris FC) sits inside 'Paris SG': no win for Paris FC from inside a longer name.
        verdict = _match("Paris Saint-Germain Football Club", "Paris FC", "Paris SG")
        assert verdict.name == "Paris SG"

    def test_full_abbreviation_outranks_bare_containment(self):
        verdict = _match("Independiente Rivadavia", "Independiente", "Ind. Rivadavia")
        assert (verdict.outcome, verdict.name) == ("matched", "Ind. Rivadavia")

    def test_a_tie_is_ambiguous_and_merges_nothing(self):
        verdict = _match("Tokyo Verdy 1969", "FC Tokyo", "Verdy")
        assert (verdict.outcome, verdict.team_id) == ("ambiguous", None)

    @pytest.mark.parametrize(
        "incoming,existing",
        [
            ("Legia", "Legnica"),  # one-word subsequence
            ("Reggina", "Reggiana"),  # 0.93 whole-string ratio
            ("Atletico GO", "Atletico-MG"),  # 0.89
            ("San Martin S.J.", "San Martin T."),  # 0.88
            ("Gaziantepspor", "Gaziantep"),  # one-word prefix
        ],
    )
    def test_different_clubs_with_similar_spellings_never_merge(self, incoming, existing):
        assert _match(incoming, existing).outcome == "no-match"


# ── aliases ──────────────────────────────────────────────────────────


class TestAliases:
    def test_an_alias_resolves_in_any_case_or_accent(self):
        db = FakeDB([team("t-mci", "Manchester City", aliases=["Man City", "Mánchester"])])
        index = ti.load_alias_index(db, "soccer")
        assert ti.resolve_alias(index, "soccer", "man city") == "t-mci"
        assert ti.resolve_alias(index, "soccer", "MANCHESTER") == "t-mci"
        assert ti.resolve_alias(index, "soccer", "man united") is None

    def test_an_alias_two_teams_claim_resolves_to_neither(self, caplog):
        db = FakeDB([team("a", "Nacional", aliases=["nacional"]), team("b", "C.D. Nacional", aliases=["nacional"])])
        with caplog.at_level(logging.ERROR):
            assert ti.find_team_by_alias(db, "soccer", "nacional") is None
        assert "claimed by 2 soccer teams" in caplog.text

    def test_malformed_aliases_are_ignored_loudly(self, caplog):
        db = FakeDB([team("a", "Wolves", aliases="wolverhampton wanderers")])
        with caplog.at_level(logging.ERROR):
            assert ti.load_alias_index(db, "soccer") == {}
        assert "not an array of names" in caplog.text


# ── fetch_upcoming: ESPN id first, stamped on every name decision ────


class TestEnsureTeamEspnId:
    def test_an_espn_id_hit_wins_over_the_name(self):
        db = FakeDB([team("twin", "Manchester City"), team("history", "Man City", espn="382")])
        assert fu.ensure_team(db, SOCCER, "Manchester City", EPL, espn_id="382") == "history"
        assert db.writes() == []

    def test_an_exact_name_hit_is_stamped(self):
        db = FakeDB([team("history", "Man City")])
        assert fu.ensure_team(db, SOCCER, "Man City", EPL, espn_id="382") == "history"
        assert db._row("history")["external_ids"] == {"espn": "382"}

    def test_an_alias_hit_is_stamped(self):
        db = FakeDB([team("history", "Wolves", aliases=["wolverhampton wanderers"])])
        assert fu.ensure_team(db, SOCCER, "Wolverhampton Wanderers", EPL, espn_id="380") == "history"
        assert db._row("history")["external_ids"] == {"espn": "380"}
        assert not any(q.startswith("INSERT") for q in db.sql)

    def test_a_club_name_match_is_stamped_and_the_next_sweep_follows_the_id(self):
        db = FakeDB([team("history", "Man City"), team("utd", "Man United")])
        assert fu.ensure_team(db, SOCCER, "Manchester City", EPL, espn_id="382") == "history"
        assert db._row("history")["external_ids"] == {"espn": "382"}
        db.sql.clear()
        assert fu.ensure_team(db, SOCCER, "Manchester City", EPL, espn_id="382") == "history"
        assert len(db.sql) == 1  # the ESPN id lookup alone

    def test_the_stamp_never_overwrites_a_different_espn_id(self, caplog):
        db = FakeDB([team("history", "Man City", espn="999")])
        with caplog.at_level(logging.ERROR):
            assert fu.ensure_team(db, SOCCER, "Man City", EPL, espn_id="382") is None
        assert db._row("history")["external_ids"] == {"espn": "999"}
        assert db.writes() == []
        assert "is ESPN team 999" in caplog.text

    def test_stamp_espn_id_refuses_an_id_another_row_holds(self, caplog):
        db = FakeDB([team("a", "Man City"), team("b", "Manchester City", espn="382")])
        with caplog.at_level(logging.ERROR):
            assert ti.stamp_espn_id(db, "a", "382", "soccer 'Man City'") is False
        assert db._row("a")["external_ids"] == {}
        assert "team b holds this one" in caplog.text

    def test_the_stamp_sql_is_guarded_on_the_observed_state(self):
        sql = " ".join(ti.STAMP_ESPN_SQL.split())
        assert "NOT (COALESCE(t.external_ids, '{}'::jsonb) ? 'espn')" in sql
        assert "NOT EXISTS ( SELECT 1 FROM teams o" in sql

    def test_an_inserted_row_carries_the_espn_id_and_trips_the_canary(self, caplog):
        db = FakeDB([team("ars", "Arsenal")], established=20)
        with caplog.at_level(logging.WARNING):
            new_id = fu.ensure_team(db, SOCCER, "Brand New Rovers", EPL, espn_id="777")
        assert db._row(new_id)["external_ids"] == {"espn": "777"}
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any(
            "new soccer team Brand New Rovers in Premier League (England) that already has 20 established teams" in m
            for m in errors
        )

    def test_a_new_team_in_a_league_without_history_is_only_a_warning(self, caplog):
        db = FakeDB([], established=0)
        with caplog.at_level(logging.WARNING):
            fu.ensure_team(db, SOCCER, "Brand New Rovers", EPL, espn_id="777")
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert "already has 0 established teams" in caplog.text

    def test_the_fuzzy_step_never_runs_in_an_international_competition(self):
        db = FakeDB([team("fcb", "Barcelona", country="Spain")])
        new_id = fu.ensure_team(db, SOCCER, "Barcelona SC", UCL, espn_id="2686")
        assert new_id != "fcb"
        assert not any("LEFT JOIN LATERAL" in q for q in db.sql)

    def test_the_fuzzy_step_skips_rows_of_another_espn_team(self):
        db = FakeDB([team("spurs", "Tottenham", espn="367")])
        assert fu.ensure_team(db, SOCCER, "Tottenham Hotspur", EPL, espn_id="999") != "spurs"

    def test_find_team_resolves_by_espn_id_and_never_writes(self):
        db = FakeDB([team("history", "Man City")])
        assert fu.find_team(db, SOCCER, "Manchester City", EPL, espn_id="382") == "history"
        assert db.writes() == []
        assert db._row("history")["external_ids"] == {}


class TestCompetitorEspnId:
    def test_team_sports_read_the_team_id(self):
        assert fu._competitor_espn_id({"id": "382", "team": {"id": "382"}}, False) == "382"
        assert fu._competitor_espn_id({"id": 359}, False) == "359"
        assert fu._competitor_espn_id({"team": {}}, False) is None

    def test_individual_sports_carry_none(self):
        assert fu._competitor_espn_id({"id": "3623", "athlete": {"id": "3623"}}, True) is None


def _comp(home, away, comp_id="401"):
    return {
        "id": comp_id,
        "date": (NOW + timedelta(days=2)).strftime("%Y-%m-%dT%H:%MZ"),
        "status": {"type": {"state": "pre"}},
        "competitors": [{"homeAway": "home", **home}, {"homeAway": "away", **away}],
    }


class TestCallSitesPassTheEspnId:
    @pytest.fixture
    def calls(self, monkeypatch):
        seen: list[tuple] = []

        def fake_ensure(cur, cfg, name, league_id, espn_id=None):
            seen.append((name, espn_id))
            return f"id-{name}"

        monkeypatch.setattr(fu, "ensure_team", fake_ensure)
        monkeypatch.setattr(fu, "_reconcile_match_row", lambda *a, **k: True)
        monkeypatch.setattr(fu, "insert_scheduled_match", lambda *a, **k: 1)
        monkeypatch.setattr(fu, "_stamp_identity", lambda *a, **k: True)
        return seen

    def test_soccer_passes_each_sides_team_id(self, calls):
        comp = _comp(
            {"id": "382", "team": {"id": "382", "displayName": "Manchester City"}},
            {"id": "359", "team": {"id": "359", "displayName": "Arsenal"}},
        )
        assert fu._process_competition(None, SOCCER, EPL, comp)
        assert calls == [("Manchester City", "382"), ("Arsenal", "359")]

    @pytest.mark.parametrize("sport", ["tennis", "mma"])
    def test_athlete_paths_pass_no_espn_id(self, calls, sport):
        comp = _comp(
            {"id": "3623", "athlete": {"id": "3623", "displayName": "Carlos Alcaraz"}},
            {"id": "3782", "athlete": {"id": "3782", "displayName": "Jannik Sinner"}},
        )
        assert fu._process_competition(None, fu.SPORT_CONFIGS[sport], "lg", comp)
        assert calls == [("Carlos Alcaraz", None), ("Jannik Sinner", None)]

    def test_an_athlete_resolution_issues_no_espn_or_fuzzy_sql(self):
        db = FakeDB([team("p1", "Carlos Alcaraz", sport="tennis")])
        assert fu.ensure_team(db, fu.SPORT_CONFIGS["tennis"], "Carlos Alcaraz", "lg") == "p1"
        assert not any("external_ids" in q or "LATERAL" in q for q in db.sql)


# ── promote_raw.upsert_teams: the same aliases ───────────────────────


class TestUpsertTeams:
    @pytest.fixture
    def inserted(self, monkeypatch):
        rows: list[tuple] = []

        def fake_execute_values(cur, sql, argslist):
            for name, norm, league_id, sport in argslist:
                rows.append((name, norm))
                if not any(t["normalized_name"] == norm for t in cur.teams):
                    cur.teams.append(team(f"new-{norm}", name, sport=sport, country=None))

        monkeypatch.setattr(pr, "execute_values", fake_execute_values)
        return rows

    def test_an_alias_maps_a_raw_name_onto_the_existing_row(self, inserted):
        db = FakeDB([team("mci", "Manchester City", aliases=["man city"]), team("ars", "Arsenal")])
        ids = pr.upsert_teams(db, {"Man City", "Arsenal"}, {"Man City": EPL, "Arsenal": EPL})
        assert ids["man city"] == "mci"
        assert ids["arsenal"] == "ars"
        assert inserted == [("Arsenal", "arsenal")]  # the league refresh for an exact name, nothing minted

    def test_an_accented_row_absorbs_the_unaccented_raw_name(self, inserted):
        db = FakeDB([team("ala", "Alavés")])
        assert pr.upsert_teams(db, {"Alaves"}, {"Alaves": EPL})["alaves"] == "ala"
        assert inserted == []

    def test_a_new_raw_name_is_inserted_and_trips_the_canary(self, inserted, caplog):
        db = FakeDB([team("ars", "Arsenal")], established=20)
        with caplog.at_level(logging.ERROR):
            ids = pr.upsert_teams(db, {"Leeds"}, {"Leeds": EPL})
        assert ids["leeds"] == "new-leeds"
        assert "new soccer team Leeds in Premier League (England) that already has 20 established teams" in caplog.text
