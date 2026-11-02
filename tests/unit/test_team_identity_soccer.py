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
LIGA = "lg-laliga"
GRE = "lg-gre"
MX = "lg-ligamx"
BR = "lg-bra"
LIB = "lg-lib"
LEAGUES = {
    EPL: ("Premier League", "England"),
    UCL: ("UEFA Champions League", "International"),
    LIGA: ("La Liga", "Spain"),
    GRE: ("Super League", "Greece"),
    MX: ("Liga MX", "Mexico"),
    BR: ("Brasileirão Série A", "Brazil"),
    LIB: ("Copa Libertadores", "International"),
}


def team(
    team_id,
    name,
    *,
    sport="soccer",
    espn=None,
    aliases=None,
    country="England",
    latest=ACTIVE,
    league_id=None,
    first_country=None,
    latest_country=None,
):
    """A teams row as the FakeDB serves it. `country` stands for both the
    first league's and the latest match's country unless first_country /
    latest_country say otherwise (COUNTRY_CANDIDATES_SQL matches either)."""
    return {
        "id": team_id,
        "name": name,
        "normalized_name": fu.normalize_team(name),
        "sport": sport,
        "external_ids": {"espn": espn} if espn else {},
        "metadata": {"aliases": aliases} if aliases is not None else {},
        "league_id": league_id,
        "first_country": country if first_country is None else first_country,
        "latest_country": country if latest_country is None else latest_country,
        "latest_date": latest,
    }


class FakeDB:
    """A teams/leagues table answering exactly the SQL team identity issues.
    Unknown SQL fails the test, so a new query cannot slip in untested.
    `race` maps a normalized name to a row that appears just before the
    INSERT (another sweep inserting the same name first)."""

    def __init__(self, teams=(), leagues=None, established=0, race=None):
        self.teams = [dict(t) for t in teams]
        self.leagues = dict(LEAGUES if leagues is None else leagues)
        self.established = established
        self.race = dict(race or {})
        self.sql: list[str] = []
        self._result: list[dict] = []
        self._inserted = 0

    def _row(self, team_id):
        return next(t for t in self.teams if t["id"] == team_id)

    def by_name(self, name):
        return next((t for t in self.teams if t["normalized_name"] == fu.normalize_team(name)), None)

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
        if q.startswith("SELECT normalized_name FROM teams WHERE id"):
            return [{"normalized_name": self._row(p[0])["normalized_name"]}]
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
        if q.startswith("SELECT id, name, normalized_name, external_ids->>'espn' AS espn FROM teams WHERE league_id"):
            league_id, sport = p
            return [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "normalized_name": t["normalized_name"],
                    "espn": t["external_ids"].get("espn"),
                }
                for t in self.teams
                if t["league_id"] == league_id and t["sport"] == sport
            ]
        if q.startswith("SELECT name, country FROM leagues"):
            league = self.leagues.get(p[0])
            return [{"name": league[0], "country": league[1]}] if league else []
        if "LEFT JOIN LATERAL" in q:
            # COUNTRY_CANDIDATES_SQL: the first league's OR the latest match's country.
            return [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "normalized_name": t["normalized_name"],
                    "espn": t["external_ids"].get("espn"),
                    "latest_date": t["latest_date"],
                }
                for t in self.teams
                if t["sport"] == p["sport"] and p["country"] in (t["first_country"], t["latest_country"])
            ]
        if "count(*) AS n" in q:
            return [{"n": self.established}]
        if q.startswith("INSERT INTO teams"):
            name, norm, league_id, sport = p  # never an external_ids: the stamp is guarded, separately
            if norm in self.race:
                self.teams.append(self.race.pop(norm))
            hit = next((t for t in self.teams if t["normalized_name"] == norm and t["sport"] == sport), None)
            if hit:
                return [{"id": hit["id"], "inserted": False}]
            self._inserted += 1
            row = team(f"new-{self._inserted}", name, sport=sport, country=None, latest=None, league_id=league_id)
            self.teams.append(row)
            return [{"id": row["id"], "inserted": True}]
        if q.startswith("SELECT id, normalized_name FROM teams WHERE sport = 'soccer'"):
            return [
                {"id": t["id"], "normalized_name": t["normalized_name"]} for t in self.teams if t["sport"] == "soccer"
            ]
        if q.startswith("SELECT id, normalized_name, league_id FROM teams WHERE sport = 'soccer'"):
            return [
                {"id": t["id"], "normalized_name": t["normalized_name"], "league_id": t["league_id"]}
                for t in self.teams
                if t["sport"] == "soccer"
            ]
        raise AssertionError(f"unexpected SQL: {q}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    def writes(self):
        return [q for q in self.sql if q.startswith(("INSERT", "UPDATE"))]


@pytest.fixture(autouse=True)
def identity_stats(monkeypatch):
    """Every counter starts at 0 and is restored after the test."""
    for key in list(fu.IDENTITY_STATS):
        monkeypatch.setitem(fu.IDENTITY_STATS, key, 0)
    return fu.IDENTITY_STATS


@pytest.fixture(autouse=True)
def fresh_seed_cache(monkeypatch):
    """The real seed file, re-read per test (a test may point SEED_PATH elsewhere)."""
    monkeypatch.setattr(ti, "_SEED_CACHE", {})


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    """Point the resolver at a seed written by the test: seed_file(rows)."""

    def write(rows=()):
        path = tmp_path / "seed.csv"
        lines = ["espn_id,team_name,league_slug,evidence"] + [",".join(r) for r in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(ti, "SEED_PATH", path)
        monkeypatch.setattr(ti, "_SEED_CACHE", {})
        return path

    return write


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

    @pytest.mark.parametrize(
        "incoming,first_team,side",
        [
            ("Real Sociedad", "Sociedad", "Sociedad B"),
            ("Paris Saint-Germain", "Paris SG", "Paris SG B"),
            ("Manchester City", "Man City", "Man City U21"),
            ("Athletic Club Bilbao", "Ath Bilbao", "Ath Bilbao B"),
        ],
    )
    def test_a_reserve_side_in_the_pool_does_not_block_its_first_team(self, incoming, first_team, side):
        # The B side holds the first team's words, but it can never be the incoming first team
        # (club_match_tier refuses across side markers), so it is not 'another club' for the guard.
        verdict = _match(incoming, first_team, side)
        assert (verdict.outcome, verdict.name) == ("matched", first_team)

    def test_a_same_side_candidate_still_blocks(self):
        # Paris FC and Paris SG are both first teams: the guard keeps working for them.
        assert _match("Paris Saint-Germain Football Club", "Paris FC", "Paris SG", "Paris SG B").name == "Paris SG"

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
            new_id = fu.ensure_team(db, SOCCER, "Man City", EPL, espn_id="382")
        assert db._row("history")["external_ids"] == {"espn": "999"}  # never overwritten
        assert new_id != "history"  # never merged
        assert db._row(new_id)["name"] == "Man City (ESPN 382)"  # never dropped either
        assert "(exact match, ESPN team 999)" in caplog.text

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

        def fake_ensure(cur, cfg, name, league_id, espn_id=None, alt_names=()):
            seen.append((name, espn_id))
            alts.append(alt_names)
            return f"id-{name}"

        alts: list[tuple] = []
        self.alts = alts

        monkeypatch.setattr(fu, "ensure_team", fake_ensure)
        monkeypatch.setattr(fu, "_listed_reversed", lambda *a, **k: False)
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

    def test_soccer_passes_each_sides_other_espn_names(self, calls):
        comp = _comp(
            {"id": "3458", "team": {"id": "3458", "displayName": "Athletico-PR", "name": "Athletico Paranaense"}},
            {"id": "2674", "team": {"id": "2674", "displayName": "Santos", "name": "Santos", "location": "Santos"}},
        )
        assert fu._process_competition(None, SOCCER, BR, comp)
        assert self.alts == [("Athletico Paranaense",), ("Santos",)]

    @pytest.mark.parametrize("sport, ids", [("tennis", (None, None)), ("mma", ("3623", "3782"))])
    def test_athlete_paths_pass_an_athlete_id_only_for_mma(self, calls, sport, ids):
        """MMA resolves fighters by ESPN athlete id since 2026-11-02 (test_fetch_upcoming_mma_athlete_ids.py)."""
        comp = _comp(
            {"id": "3623", "athlete": {"id": "3623", "displayName": "Carlos Alcaraz"}},
            {"id": "3782", "athlete": {"id": "3782", "displayName": "Jannik Sinner"}},
        )
        assert fu._process_competition(None, fu.SPORT_CONFIGS[sport], "lg", comp)
        assert calls == [("Carlos Alcaraz", ids[0]), ("Jannik Sinner", ids[1])]

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
        assert inserted == [("Arsenal", "arsenal")]  # an exact name with no league: its league filled, nothing minted

    def test_an_existing_team_keeps_its_league_and_is_not_sent(self, inserted):
        """football-data filed five Scottish Championship rows as SP1: the raw first-seen league must not move an
        existing club (Dunfermline under La Liga on every run), and an unchanged row is neither rewritten nor locked."""
        db = FakeDB([team("dun", "Dunfermline", league_id="sc1")])
        assert pr.upsert_teams(db, {"Dunfermline"}, {"Dunfermline": LIGA}) == {"dunfermline": "dun"}
        assert inserted == [] and db._row("dun")["league_id"] == "sc1"

    def test_a_missing_league_is_filled_and_the_conflict_branch_keeps_an_existing_one(self, monkeypatch):
        sent: list[str] = []
        monkeypatch.setattr(pr, "execute_values", lambda cur, sql, rows: sent.append(" ".join(sql.split())))
        pr.upsert_teams(FakeDB([team("ars", "Arsenal")]), {"Arsenal"}, {"Arsenal": EPL})
        assert sent == [
            "INSERT INTO teams (name, normalized_name, league_id, sport) VALUES %s "
            "ON CONFLICT (normalized_name, sport) DO UPDATE "
            "SET league_id = COALESCE(teams.league_id, EXCLUDED.league_id) "
            "WHERE teams.league_id IS NULL AND EXCLUDED.league_id IS NOT NULL"
        ]

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


# ── the in-league spelling ratio, kept as the last tier ──────────────


class TestLeagueRatioTier:
    """The pre-2026-10 fuzzy step (whole-string ratio >= 0.85 among the teams
    whose first league is the swept league) resolves names today that no
    structural rule pairs. Dropping it minted a twin for each, stamped with
    its ESPN id, and every existing ESPN-identified fixture of the club then
    raised ForkRefused (prod: Espanol 9 such rows, Levadeiakos 8)."""

    def test_espanyol_resolves_to_espanol(self):
        db = FakeDB(
            [
                team("esp", "Espanol", country="Spain", league_id=LIGA),
                team("ray", "Vallecano", country="Spain", league_id=LIGA),
                team("hue", "Huesca", country="Spain", league_id=LIGA),
            ]
        )
        assert fu.ensure_team(db, SOCCER, "Espanyol", LIGA, espn_id="88") == "esp"
        assert db._row("esp")["external_ids"] == {"espn": "88"}
        assert not any(q.startswith("INSERT") for q in db.sql)

    def test_levadiakos_resolves_to_levadeiakos(self):
        db = FakeDB(
            [
                team("lev", "Levadeiakos", country="Greece", league_id=GRE),
                team("oly", "Olympiakos", country="Greece", league_id=GRE),
                team("lam", "Lamia", country="Greece", league_id=GRE),
            ]
        )
        assert fu.find_team(db, SOCCER, "Levadiakos", GRE, espn_id="5276") == "lev"
        assert db.writes() == []
        assert fu.ensure_team(db, SOCCER, "Levadiakos", GRE, espn_id="5276") == "lev"
        assert db._row("lev")["external_ids"] == {"espn": "5276"}

    def test_the_ratio_only_looks_at_the_leagues_own_teams(self):
        # Same spelling, but Espanol's first league is another one: the old step never saw it either.
        db = FakeDB([team("esp", "Espanol", country="Spain", league_id="lg-other")])
        assert fu.find_team(db, SOCCER, "Espanyol", LIGA, espn_id="88") is None

    def test_the_ratio_never_picks_a_row_of_another_espn_team(self):
        db = FakeDB([team("esp", "Espanol", country="Spain", league_id=LIGA, espn="999")])
        assert fu.find_team(db, SOCCER, "Espanyol", LIGA, espn_id="88") is None

    def test_the_ratio_never_merges_a_reserve_side_into_its_first_team(self):
        # 'barcelona b' vs 'barcelona' is 0.90: the old step merged them.
        db = FakeDB([team("fcb", "Barcelona", country="Spain", league_id=LIGA)])
        assert fu.find_team(db, SOCCER, "Barcelona B", LIGA, espn_id="5555") is None

    def test_a_ratio_tie_merges_nothing(self):
        verdict = ti.match_league_ratio(
            "Oster", "oster", [({"id": "a", "name": "Osters"}, "osters"), ({"id": "b", "name": "Ostera"}, "ostera")]
        )
        assert (verdict.outcome, verdict.team_id) == ("ambiguous", None)

    def test_below_the_threshold_is_no_match(self):
        # Kalamata / Kalamaria is 0.82.
        verdict = ti.match_league_ratio("Kalamata", "kalamata", [({"id": "k", "name": "Kalamaria"}, "kalamaria")])
        assert verdict.outcome == "no-match"

    def test_kalamata_is_inserted_not_merged_into_kalamaria(self, caplog):
        db = FakeDB(
            [
                team("kal", "Kalamaria", country="Greece", league_id=GRE),
                team("oly", "Olympiakos", country="Greece", league_id=GRE),
            ],
            established=14,
        )
        with caplog.at_level(logging.ERROR):
            new_id = fu.ensure_team(db, SOCCER, "Kalamata", GRE, espn_id="21540")
        assert new_id not in ("kal", "oly")
        assert db._row(new_id)["name"] == "Kalamata"
        assert db._row(new_id)["external_ids"] == {"espn": "21540"}
        assert db._row("kal")["external_ids"] == {}
        assert "new soccer team Kalamata in Super League (Greece) that already has 14 established teams" in caplog.text


# ── a name held by another ESPN team: never merged, never dropped ─────


class TestNamesake:
    """Santos FC (bra.1, 2674) and Santos Laguna (mex.1, 225) both arrive as
    'Santos'. Once one id is on the 'Santos' row the other club must neither
    take that row nor vanish from every sweep with exit 0."""

    def test_the_other_club_gets_its_own_row_and_the_run_exits_3_once(self, seed_file, identity_stats, caplog):
        seed_file([])  # no seed: the namesake path is the last line of defence
        db = FakeDB([team("santos", "Santos", country="Brazil", espn="2674", league_id=BR)])
        with caplog.at_level(logging.ERROR):
            laguna = fu.ensure_team(db, SOCCER, "Santos", MX, espn_id="225", alt_names=("Santos",))
        assert laguna != "santos"
        assert db._row(laguna)["name"] == "Santos (ESPN 225)"
        assert db._row(laguna)["external_ids"] == {"espn": "225"}
        assert db._row("santos")["external_ids"] == {"espn": "2674"}
        assert identity_stats["team_namesakes"] == 1
        assert fu.identity_exit_code(SOCCER) == fu.IDENTITY_EXIT_CODE
        assert "a different club under the same name" in caplog.text
        assert "fix the row or its aliases" not in caplog.text  # the old, misleading advice
        # The next sweep resolves by id: no insert, nothing counted.
        db.sql.clear()
        assert fu.ensure_team(db, SOCCER, "Santos", MX, espn_id="225") == laguna
        assert db.writes() == []
        assert identity_stats["team_namesakes"] == 1

    def test_espns_full_name_is_preferred_when_it_is_free(self, seed_file):
        seed_file([])
        db = FakeDB([team("cap", "Athletico-PR", country="Brazil", espn="9999", league_id=BR)])
        new_id = fu.ensure_team(db, SOCCER, "Athletico-PR", BR, espn_id="3458", alt_names=("Athletico Paranaense",))
        assert db._row(new_id)["name"] == "Athletico Paranaense"

    def test_an_alternative_name_another_team_resolves_is_not_used(self, seed_file):
        seed_file([])
        db = FakeDB(
            [
                team("cap", "Athletico-PR", country="Brazil", espn="9999", league_id=BR),
                team("other", "Somebody", aliases=["athletico paranaense"], country="Brazil"),
            ]
        )
        new_id = fu.ensure_team(db, SOCCER, "Athletico-PR", BR, espn_id="3458", alt_names=("Athletico Paranaense",))
        assert db._row(new_id)["name"] == "Athletico-PR (ESPN 3458)"

    def test_an_alias_resolves_the_collision_before_any_insert(self, seed_file, identity_stats):
        seed_file([])
        db = FakeDB(
            [
                team("santos", "Santos", country="Brazil", espn="2674", league_id=BR),
                team("laguna", "Santos Laguna", country="Mexico", aliases=["Santos"], league_id=MX),
            ]
        )
        assert fu.ensure_team(db, SOCCER, "Santos", MX, espn_id="225") == "laguna"
        assert db._row("laguna")["external_ids"] == {"espn": "225"}
        assert identity_stats["team_namesakes"] == 0

    def test_the_club_name_tier_runs_after_a_blocked_exact_hit(self, seed_file):
        # 'Racing Club' is Racing Club de Montevideo's row (another ESPN id); Avellaneda's history is 'Racing'.
        seed_file([])
        db = FakeDB(
            [
                team("mvd", "Racing Club", espn="9001", country="Uruguay"),
                team("arg", "Racing", country="Argentina"),
            ],
            leagues={**LEAGUES, "lg-arg": ("Primera División", "Argentina")},
        )
        assert fu.ensure_team(db, SOCCER, "Racing Club", "lg-arg", espn_id="5") == "arg"
        assert db._row("arg")["external_ids"] == {"espn": "5"}

    def test_find_team_returns_none_for_a_namesake_and_writes_nothing(self, seed_file):
        seed_file([])
        db = FakeDB([team("santos", "Santos", country="Brazil", espn="2674")])
        assert fu.find_team(db, SOCCER, "Santos", MX, espn_id="225") is None
        assert db.writes() == []


# ── the seed: collisions pinned before the first sweep ───────────────


class TestSeed:
    def test_the_checked_in_seed_pins_every_known_collision(self):
        seed = ti.load_espn_seed()
        pins = {espn: entry.team_name for espn, entry in seed.items()}
        assert pins["2674"] == "Santos"  # Santos FC, bra.1
        assert pins["225"] == "Santos Laguna"  # mex.1, ESPN display name 'Santos'
        assert pins["364"] == "Liverpool"  # eng.1 / uefa.champions
        assert pins["5492"] == "Liverpool Montevideo"  # Libertadores, ESPN display name 'Liverpool'
        assert pins["219"] == "Guadalajara Chivas"  # mex.1 'Guadalajara'; the 'Guadalajara' row is CD Guadalajara
        assert pins["2684"] == "Nacional Montevideo"  # Libertadores 'Nacional'; the 'Nacional' row is Madeira's
        assert "3472" not in pins  # C.D. Nacional follows its ESPN twin through the team merge
        assert all(entry.evidence for entry in seed.values())

    @pytest.mark.parametrize("order", [("bra", "mex"), ("mex", "bra")])
    def test_sweep_order_cannot_hand_santos_to_the_wrong_club(self, order):
        db = FakeDB(
            [
                team("santos", "Santos", country="Brazil", league_id=BR),
                team("laguna", "Santos Laguna", country="Mexico", league_id=MX, latest=NOW - timedelta(days=90)),
            ]
        )
        sweep = {"bra": (BR, "2674"), "mex": (MX, "225")}
        got = {key: fu.ensure_team(db, SOCCER, "Santos", sweep[key][0], espn_id=sweep[key][1]) for key in order}
        assert got == {"bra": "santos", "mex": "laguna"}
        assert db._row("santos")["external_ids"] == {"espn": "2674"}
        assert db._row("laguna")["external_ids"] == {"espn": "225"}
        assert not any(q.startswith("INSERT") for q in db.sql)

    def test_liverpool_montevideo_gets_its_own_row_not_the_premier_leagues(self, identity_stats, caplog):
        db = FakeDB([team("pl", "Liverpool", country="England", league_id=EPL)], established=16)
        assert fu.find_team(db, SOCCER, "Liverpool", LIB, espn_id="5492") is None  # read-only: no row yet
        with caplog.at_level(logging.WARNING):
            uru = fu.ensure_team(db, SOCCER, "Liverpool", LIB, espn_id="5492")
        # A seeded row is a reviewed decision, not a split: the canary stays a WARNING.
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert "pinned by scripts/data/soccer_team_espn_seed.csv" in caplog.text
        assert uru != "pl"
        assert (db._row(uru)["name"], db._row(uru)["external_ids"]) == ("Liverpool Montevideo", {"espn": "5492"})
        assert fu.ensure_team(db, SOCCER, "Liverpool", EPL, espn_id="364") == "pl"
        assert db._row("pl")["external_ids"] == {"espn": "364"}
        assert identity_stats["team_namesakes"] == identity_stats["team_conflicts"] == 0

    def test_nacional_montevideo_never_takes_madeiras_nacional_row(self, identity_stats):
        # 'Nacional' is CD Nacional (Madeira); ESPN 3472 'C.D. Nacional' still sits on its ESPN twin.
        db = FakeDB(
            [
                team("mad", "Nacional", country="Portugal"),
                team("twin", "C.D. Nacional", country="Portugal", espn="3472"),
            ]
        )
        uru = fu.ensure_team(db, SOCCER, "Nacional", LIB, espn_id="2684")
        assert uru not in ("mad", "twin")
        assert (db._row(uru)["name"], db._row(uru)["external_ids"]) == ("Nacional Montevideo", {"espn": "2684"})
        assert db._row("mad")["external_ids"] == {}  # left for the merge to give it 3472
        assert fu.ensure_team(db, SOCCER, "C.D. Nacional", "lg-por", espn_id="3472") == "twin"

    def test_a_seeded_row_holding_another_id_is_refused_and_counted(self, identity_stats, caplog):
        db = FakeDB([team("laguna", "Santos Laguna", country="Mexico", espn="777", league_id=MX)])
        with caplog.at_level(logging.ERROR):
            assert fu.ensure_team(db, SOCCER, "Santos", MX, espn_id="225") is None
        assert db.writes() == []
        assert identity_stats["team_conflicts"] == 1
        assert fu.identity_exit_code(SOCCER) == 3
        assert "the seed pins it to team laguna" in caplog.text

    def test_an_id_stamped_against_the_seed_is_refused_and_counted(self, identity_stats, caplog):
        # 225 was stamped onto the Brazilian row (the order problem the seed exists for).
        db = FakeDB([team("santos", "Santos", espn="225"), team("laguna", "Santos Laguna", country="Mexico")])
        with caplog.at_level(logging.ERROR):
            assert fu.ensure_team(db, SOCCER, "Santos", MX, espn_id="225") is None
        assert identity_stats["team_conflicts"] == 1
        assert "but the seed pins it to 'Santos Laguna'" in caplog.text

    def test_the_seed_is_soccer_only(self):
        assert ti.seed_entry("nhl", "225") is None
        assert ti.seed_entry("soccer", "225").team_name == "Santos Laguna"

    @pytest.mark.parametrize(
        "rows,error",
        [
            ([("225", "Santos Laguna", "mex.1", "x"), ("225", "Santos", "bra.1", "y")], "seeded twice"),
            ([("225", "Santos", "mex.1", "x"), ("2674", "santos", "bra.1", "y")], "one team row is one ESPN team"),
            ([("SAN", "Santos", "mex.1", "x")], "must be digits"),
            ([("225", "", "mex.1", "x")], "must be digits"),
        ],
    )
    def test_a_malformed_seed_fails_loudly(self, seed_file, rows, error):
        path = seed_file(rows)
        with pytest.raises(ValueError, match=error):
            ti.load_espn_seed(path)

    def test_a_seed_missing_a_column_fails_loudly(self, tmp_path):
        path = tmp_path / "seed.csv"
        path.write_text("espn_id,team_name\n225,Santos Laguna\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            ti.load_espn_seed(path)


# ── ESPN id corruption and the insert path ───────────────────────────


class TestEspnIdConflicts:
    def test_an_id_on_two_rows_is_refused_never_resolved_by_name_or_inserted(self, identity_stats, caplog):
        db = FakeDB([team("a", "Man City", espn="382"), team("b", "Manchester City", espn="382")])
        with caplog.at_level(logging.ERROR):
            assert fu.ensure_team(db, SOCCER, "Manchester City", EPL, espn_id="382") is None
        assert db.writes() == []
        assert identity_stats["team_conflicts"] == 1
        assert "is on 2 teams" in caplog.text

    def test_the_insert_carries_no_espn_id_and_the_guarded_stamp_adds_it(self):
        db = FakeDB([])
        new_id = fu.ensure_team(db, SOCCER, "Brand New Rovers", EPL, espn_id="777")
        insert = next(q for q in db.sql if q.startswith("INSERT INTO teams"))
        assert "external_ids" not in insert
        assert db.sql.index(insert) < next(i for i, q in enumerate(db.sql) if q.startswith("UPDATE teams t"))
        assert db._row(new_id)["external_ids"] == {"espn": "777"}

    def test_race_a_row_inserted_first_by_another_sweep_for_another_espn_team_is_refused(self, identity_stats):
        rival = team("rival", "Brand New Rovers", espn="999")
        db = FakeDB([], race={"brand new rovers": rival})
        assert fu.ensure_team(db, SOCCER, "Brand New Rovers", EPL, espn_id="777") is None
        assert db._row("rival")["external_ids"] == {"espn": "999"}
        assert identity_stats["team_conflicts"] == 1
        assert fu.identity_exit_code(SOCCER) == 3

    def test_race_a_row_inserted_first_without_an_id_is_adopted_and_stamped(self, identity_stats):
        db = FakeDB([], race={"brand new rovers": team("other", "Brand New Rovers")})
        assert fu.ensure_team(db, SOCCER, "Brand New Rovers", EPL, espn_id="777") == "other"
        assert db._row("other")["external_ids"] == {"espn": "777"}
        assert identity_stats["team_conflicts"] == 0

    def test_a_stamp_refused_on_a_new_row_skips_the_side(self, identity_stats, monkeypatch, caplog):
        db = FakeDB([])
        monkeypatch.setattr(fu.team_identity, "stamp_espn_id", lambda *a, **k: False)
        with caplog.at_level(logging.ERROR):
            assert fu.ensure_team(db, SOCCER, "Brand New Rovers", EPL, espn_id="777") is None
        assert identity_stats["team_conflicts"] == 1
        assert "the guarded stamp refused it" in caplog.text


# ── the country rule of the club-name tier ───────────────────────────


class TestCountryCandidates:
    """A team belongs to a country through its first league OR its latest
    match's league (promoted, relegated and renamed clubs). The SQL itself is
    exercised against Postgres in test_team_identity_pg.py."""

    def test_the_sql_matches_first_or_latest_league(self):
        sql = " ".join(ti.COUNTRY_CANDIDATES_SQL.split())
        assert "AND (fl.country = %(country)s OR ll.country = %(country)s)" in sql
        assert "LEFT JOIN leagues fl ON fl.id = t.league_id" in sql
        assert sql.count("ORDER BY match_date DESC LIMIT 1") == 2  # latest home AND latest away match

    def test_a_team_whose_latest_league_is_in_the_country_is_a_candidate(self):
        # Tottenham's row came from another country's first league, but it plays in England now.
        db = FakeDB([team("spurs", "Tottenham", first_country="Spain", latest_country="England")])
        assert ti.find_team_fuzzy(db, "soccer", "Tottenham Hotspur", EPL) == "spurs"

    def test_a_team_whose_first_league_is_in_the_country_is_a_candidate(self):
        db = FakeDB([team("spurs", "Tottenham", first_country="England", latest_country="Scotland")])
        assert ti.find_team_fuzzy(db, "soccer", "Tottenham Hotspur", EPL) == "spurs"

    def test_a_team_of_neither_is_not(self):
        db = FakeDB([team("spurs", "Tottenham", first_country="Spain", latest_country="Scotland")])
        assert ti.find_team_fuzzy(db, "soccer", "Tottenham Hotspur", EPL) is None


class TestCompetitorAltNames:
    def test_team_sports_read_name_then_location(self):
        comp = {"team": {"displayName": "OH Leuven", "name": "Oud-Heverlee Leuven", "location": "Oud-Heverlee Leuven"}}
        assert fu._competitor_alt_names(comp, False) == ("Oud-Heverlee Leuven",)
        assert fu._competitor_alt_names({"team": {"abbreviation": "SAN", "shortDisplayName": "Santos"}}, False) == ()

    def test_individual_sports_carry_none(self):
        assert fu._competitor_alt_names({"athlete": {"displayName": "Jannik Sinner"}}, True) == ()


# ── the status path resolves sides as ensure_team does ───────────────


class StatusFakeDB(FakeDB):
    """FakeDB plus the matches rows fetch_upcoming's status path reads
    (STATUS_ROW_BY_ESPN_SQL, STATUS_ROW_BY_TRIPLE_SQL) and writes
    (STATUS_UPDATE_SQL, guarded on the status read)."""

    def __init__(self, teams=(), matches=(), **kwargs):
        super().__init__(teams, **kwargs)
        self.matches = [dict(m) for m in matches]

    def _answer(self, q, p):
        if q.startswith("SELECT m.id, m.status, m.home_team_id, m.away_team_id, m.match_date, m.metadata FROM matches"):
            espn, _sport = p
            return [m for m in self.matches if m.get("espn") == espn][:2]
        if q.startswith("SELECT id, status, home_team_id, away_team_id, match_date, metadata FROM matches"):
            home, away, when = p
            return [
                m for m in self.matches if (m["home_team_id"], m["away_team_id"], m["match_date"]) == (home, away, when)
            ]
        if q.startswith("UPDATE matches SET status"):
            status, _patch, _drop, rid, old = p
            row = next((m for m in self.matches if m["id"] == rid and m["status"] == old), None)
            if row is None:
                return []
            row["status"] = status
            return [{"id": rid}]
        return super()._answer(q, p)


def _postponed_liga_mx():
    """ESPN's Liga MX competition 401: Santos Laguna, whose ESPN display name
    is 'Santos' (mex.1 id 225), v Tijuana, postponed."""
    comp = _comp(
        {"id": "225", "team": {"id": "225", "displayName": "Santos"}},
        {"id": "10125", "team": {"id": "10125", "displayName": "Tijuana"}},
    )
    comp["status"] = {"type": {"name": "STATUS_POSTPONED", "state": "post", "completed": False}}
    return comp


def _liga_mx_db(*, laguna_espn=None, extra=()):
    comp = _postponed_liga_mx()
    kickoff = fu._competition_parts(comp, SOCCER)[4]
    teams = [
        team("santos", "Santos", country="Brazil", espn="2674", league_id=BR),
        team("laguna", "Santos Laguna", country="Mexico", espn=laguna_espn, league_id=MX),
        team("tijuana", "Tijuana", country="Mexico", league_id=MX),
        *extra,
    ]
    row = {
        "id": "fixture",
        "status": "scheduled",
        "espn": "401",
        "home_team_id": "laguna",
        "away_team_id": "tijuana",
        "match_date": kickoff,
        "metadata": {},
    }
    return StatusFakeDB(teams, [row]), comp


class TestStatusPathResolvesByEspnId:
    """ESPN's 'Santos' is two clubs. The fixture and results paths resolve a
    side by its ESPN team id (and the seed) first; the status path must too,
    or a Santos Laguna postponement resolves 'Santos' by name to Santos FC,
    finds its own fixture row 'drifted' and is refused every sweep (exit 3)."""

    @pytest.mark.parametrize("laguna_espn", [None, "225"], ids=["seeded", "stamped"])
    @pytest.mark.parametrize("path", ["results", "fixtures"])
    def test_a_santos_laguna_postponement_lands_on_its_row(self, identity_stats, laguna_espn, path):
        db, comp = _liga_mx_db(laguna_espn=laguna_espn)
        if path == "results":
            assert fu._record_result(db, SOCCER, MX, comp) is False  # not a result
        else:
            assert fu._process_competition(db, SOCCER, MX, comp) is False
        assert db.matches[0]["status"] == "postponed"
        assert identity_stats["status_refused"] == identity_stats["team_conflicts"] == 0
        assert fu.identity_exit_code(SOCCER) == 0
        assert [q for q in db.writes() if "teams" in q] == []  # never inserts or stamps a team

    def test_an_espn_team_id_conflict_writes_nothing_and_is_counted(self, identity_stats, caplog):
        # 225 on two rows: corruption, never resolved by name.
        dup = team("laguna-twin", "Club Santos Laguna", country="Mexico", espn="225", league_id=MX)
        db, comp = _liga_mx_db(laguna_espn="225", extra=(dup,))
        with caplog.at_level(logging.ERROR):
            assert fu._record_result(db, SOCCER, MX, comp) is False
        assert db.matches[0]["status"] == "scheduled" and db.writes() == []
        assert identity_stats["team_conflicts"] == 1 and identity_stats["status_refused"] == 0
        assert fu.identity_exit_code(SOCCER) == fu.IDENTITY_EXIT_CODE
        assert "ESPN team id 225 is on 2 teams" in caplog.text
