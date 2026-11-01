"""scripts/repair_soccer_labels.py: the plan (relabel the Scottish fixtures
football-data served at the SP1/SP2 URLs, their raw rows and their clubs'
leagues; rename the mis-decoded 'Preußen Münster'), every refusal, the
reviewed map, the reversal SQL — and, against a real Postgres when
AUSPEX_TEST_PG_URL names a throwaway database, an apply that a re-plan finds
done, --verify accepts and the reversal SQL undoes exactly."""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from datetime import datetime, timezone
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


rsl = _load("repair_soccer_labels")
GARBLED = "Preußen Münster".encode("utf-8").decode("latin-1")  # 'PreuÃ\x9fen MÃ¼nster', prod's team b6abfa7e
KICKOFF = datetime(2026, 8, 1, 15, 0, tzinfo=timezone.utc)  # the legacy kickoff; step 4 moves it to 14:00Z

LEAGUES = {
    "SP1": {"id": "l-sp1", "name": "La Liga", "country": "Spain", "code": "SP1"},
    "SP2": {"id": "l-sp2", "name": "La Liga 2", "country": "Spain", "code": "SP2"},
    "SC0": {"id": "l-sc0", "name": "Premiership", "country": "Scotland", "code": "SC0"},
    "D2": {"id": "l-d2", "name": "2. Bundesliga", "country": "Germany", "code": "D2"},
}


def _map_row(home, away, hg, ag, frm, to):
    return rsl.MapRow("2627", "01/08/2026", "15:00", home, away, hg, ag, frm, to, "test")


MAP = [_map_row("Ayr", "Arbroath", 2, 0, "SP1", "SC1"), _map_row("Montrose", "Hamilton", 0, 5, "SP2", "SC2")]


def _state(**over) -> "rsl.State":
    leagues = {k: dict(v) for k, v in over.pop("leagues", LEAGUES).items()}
    teams = {
        "t-ayr": {"name": "Ayr", "normalized_name": "ayr", "league_id": "l-sp1", "metadata_text": "{}"},
        "t-arb": {"name": "Arbroath", "normalized_name": "arbroath", "league_id": "l-sp1", "metadata_text": "{}"},
        "t-mon": {"name": "Montrose", "normalized_name": "montrose", "league_id": "l-sp2", "metadata_text": "{}"},
        "t-ham": {"name": "Hamilton", "normalized_name": "hamilton", "league_id": "l-sp2", "metadata_text": "{}"},
        "t-pre": {"name": GARBLED, "normalized_name": GARBLED.lower(), "league_id": "l-d2", "metadata_text": "{}"},
        "t-kie": {
            "name": "Holstein Kiel",
            "normalized_name": "holstein kiel",
            "league_id": "l-d2",
            "metadata_text": "{}",
        },
    }
    teams.update(over.pop("teams", {}))
    raw = {
        ("01/08/2026", "Ayr", "Arbroath"): {
            "Time": "15:00",
            "FTHG": "2.0",
            "FTAG": "0.0",
            "Div": None,
            "league": "SP1",
            "season": "2627",
            "HomeTeam": "Ayr",
            "AwayTeam": "Arbroath",
        },  # noqa: E501
        ("01/08/2026", "Montrose", "Hamilton"): {
            "Time": "15:00",
            "FTHG": "0.0",
            "FTAG": "5.0",
            "Div": None,
            "league": "SP2",
            "season": "2627",
            "HomeTeam": "Montrose",
            "AwayTeam": "Hamilton",
        },  # noqa: E501
        ("31/10/2025", GARBLED, "Holstein Kiel"): {
            "Time": "17:30",
            "FTHG": "1.0",
            "FTAG": "1.0",
            "Div": None,
            "league": "D2",
            "season": "2526",
            "HomeTeam": GARBLED,
            "AwayTeam": "Holstein Kiel",
        },  # noqa: E501
    }
    raw.update(over.pop("raw", {}))
    for k in [k for k, v in raw.items() if v is None]:
        del raw[k]
    state = rsl.State(
        leagues_by_code={c: lg for c, lg in leagues.items()},
        leagues_by_id={lg["id"]: lg for lg in leagues.values()},
        teams=teams,
        team_by_norm={t["normalized_name"]: tid for tid, t in teams.items()},
        aliases=over.pop("aliases", {}),
        raw=raw,
    )
    state.candidates = over.pop(
        "candidates",
        {
            MAP[0].raw_key: [{"id": "m-1", "league_id": "l-sp1", "match_date": KICKOFF}],
            MAP[1].raw_key: [{"id": "m-2", "league_id": "l-sp2", "match_date": KICKOFF}],
        },
    )
    state.cross_country_now = over.pop("cross_country_now", {"t-ham"})
    state.sides = over.pop(
        "sides",
        {
            "t-ayr": [("m-1", "Spain")],
            "t-arb": [("m-1", "Spain")],
            "t-mon": [("m-2", "Spain")],
            "t-ham": [("m-2", "Spain"), ("m-old", "Scotland")],
        },
    )
    assert not over, over
    return state


def _done_state() -> "rsl.State":
    leagues = {
        **LEAGUES,
        "SC1": {"id": "l-sc1", "name": "Championship", "country": "Scotland", "code": "SC1"},
        "SC2": {"id": "l-sc2", "name": "League One", "country": "Scotland", "code": "SC2"},
    }
    raw_done = {
        ("01/08/2026", "Ayr", "Arbroath"): {
            "Time": "15:00",
            "FTHG": "2.0",
            "FTAG": "0.0",
            "Div": "SC1",
            "league": "SC1",
            "season": "2627",
            "HomeTeam": "Ayr",
            "AwayTeam": "Arbroath",
        },  # noqa: E501
        ("01/08/2026", "Montrose", "Hamilton"): {
            "Time": "15:00",
            "FTHG": "0.0",
            "FTAG": "5.0",
            "Div": "SC2",
            "league": "SC2",
            "season": "2627",
            "HomeTeam": "Montrose",
            "AwayTeam": "Hamilton",
        },  # noqa: E501
        ("31/10/2025", GARBLED, "Holstein Kiel"): None,
        ("31/10/2025", "Preußen Münster", "Holstein Kiel"): {
            "Time": "17:30",
            "FTHG": "1.0",
            "FTAG": "1.0",
            "Div": None,
            "league": "D2",
            "season": "2526",
            "HomeTeam": "Preußen Münster",
            "AwayTeam": "Holstein Kiel",
        },  # noqa: E501
    }
    teams = {
        "t-ayr": {"name": "Ayr", "normalized_name": "ayr", "league_id": "l-sc1", "metadata_text": "{}"},
        "t-arb": {"name": "Arbroath", "normalized_name": "arbroath", "league_id": "l-sc1", "metadata_text": "{}"},
        "t-mon": {"name": "Montrose", "normalized_name": "montrose", "league_id": "l-sc2", "metadata_text": "{}"},
        "t-ham": {"name": "Hamilton", "normalized_name": "hamilton", "league_id": "l-sc0", "metadata_text": "{}"},
        "t-pre": {
            "name": "Preußen Münster",
            "normalized_name": "preußen münster",
            "league_id": "l-d2",
            "metadata_text": "{}",
        },  # noqa: E501
    }
    return _state(
        leagues=leagues,
        teams=teams,
        raw=raw_done,
        candidates={
            MAP[0].raw_key: [{"id": "m-1", "league_id": "l-sc1", "match_date": KICKOFF}],
            MAP[1].raw_key: [{"id": "m-2", "league_id": "l-sc2", "match_date": KICKOFF}],
        },
        cross_country_now=set(),
        sides={"t-ham": [("m-2", "Scotland"), ("m-old", "Scotland")]},
    )


def _writes(p) -> dict:
    return {k: p.counts[k] for k in rsl.WRITE_COUNTERS}


class TestPlan:
    def test_the_prod_shape_plans_every_kind(self):
        p = rsl.plan(_state(), MAP)
        assert _writes(p) == {
            "create_league": 2,
            "relabel_raw": 2,
            "relabel_match": 2,
            "relabel_team": 4,
            "rename_team": 1,
            "rename_raw": 1,
        }
        assert p.refusals == [] and p.counts["mojibake_left"] == 0 and p.counts["cross_country_after"] == 0
        assert p.creates == ["SC1", "SC2"]
        assert [(tid, code) for tid, _old, code in p.relabel_team] == [
            ("t-arb", "SC1"),
            ("t-ayr", "SC1"),
            ("t-ham", "SC2"),
            ("t-mon", "SC2"),
        ]
        assert p.rename_team == [("t-pre", GARBLED, GARBLED.lower(), "Preußen Münster", "preußen münster")]
        assert p.rename_raw == [(("31/10/2025", GARBLED, "Holstein Kiel"), ("Preußen Münster", "Holstein Kiel"))]

    def test_the_audit_entries_carry_old_and_new_per_column(self):
        p = rsl.plan(_state(), MAP)
        by = {(e["kind"], e["row_key"], e["column"]): (e["old_value"], e["new_value"]) for e in p.entries}
        assert by[("relabel_raw", "01/08/2026 | Ayr | Arbroath", "league")] == ("SP1", "SC1")
        assert by[("relabel_raw", "01/08/2026 | Ayr | Arbroath", "Div")] == ("", "SC1")
        assert by[("relabel_match", "m-1", "league_id")] == ("l-sp1", "(new SC1)")
        assert by[("create_league", "SC2", "row")] == ("", "League One | Scotland | soccer")

    def test_a_finished_repair_plans_nothing(self):
        p = rsl.plan(_done_state(), MAP)
        assert set(_writes(p).values()) == {0} and p.refusals == []
        assert (p.counts["already_raw"], p.counts["already_match"], p.counts["mojibake_left"]) == (2, 2, 0)

    def test_a_team_already_in_any_scottish_league_stays(self):
        # promote_raw, re-run after the repair, gives a club the league of its first raw row: SC0 for Hamilton.
        state = _state(
            teams={
                "t-ham": {
                    "name": "Hamilton",
                    "normalized_name": "hamilton",
                    "league_id": "l-sc0",
                    "metadata_text": "{}",
                }
            }
        )  # noqa: E501
        assert "t-ham" not in [tid for tid, _o, _c in rsl.plan(state, MAP).relabel_team]

    def test_the_match_is_found_at_either_kickoff(self):
        moved = KICKOFF.replace(hour=14)
        lo, hi = rsl.mapped_kickoff_window(MAP[0])
        assert lo <= moved <= hi and lo <= KICKOFF <= hi

    def test_cross_country_teams_outside_the_map_are_reported_not_refused(self):
        state = _state(
            cross_country_now={"t-ham", "t-santos"},
            sides={**_state().sides, "t-santos": [("x", "Brazil"), ("y", "Mexico")]},
        )  # noqa: E501
        state.teams["t-santos"] = {
            "name": "Santos",
            "normalized_name": "santos",
            "league_id": "l-sp1",
            "metadata_text": "{}",
        }  # noqa: E501
        p = rsl.plan(state, MAP)
        assert p.refusals == [] and p.counts["cross_country_after"] == 1 and p.cross_country_names == ["Santos"]


class TestRefusals:
    def _refused(self, state) -> str:
        p = rsl.plan(state, MAP)
        assert p.refusals, p.counts
        assert p.counts["refuse"] == len(p.refusals)
        return " | ".join(p.refusals)

    def test_a_missing_raw_row(self):
        assert "no raw_football_data row" in self._refused(_state(raw={("01/08/2026", "Ayr", "Arbroath"): None}))

    def test_a_score_that_disagrees_with_the_map(self):
        bad = dict(_state().raw[("01/08/2026", "Ayr", "Arbroath")], FTHG="3.0")
        assert "the map 15:00 2-0" in self._refused(_state(raw={("01/08/2026", "Ayr", "Arbroath"): bad}))

    def test_a_raw_row_in_another_season(self):
        bad = dict(_state().raw[("01/08/2026", "Ayr", "Arbroath")], season="2526")
        assert "not 2627" in self._refused(_state(raw={("01/08/2026", "Ayr", "Arbroath"): bad}))

    def test_no_or_two_match_rows(self):
        cands = {MAP[0].raw_key: [], MAP[1].raw_key: [{"id": "a", "league_id": "l-sp2", "match_date": KICKOFF}] * 2}
        text = self._refused(_state(candidates=cands))
        assert "0 matches row(s)" in text and "2 matches row(s)" in text

    def test_a_match_row_in_a_third_league(self):
        cands = {**_state().candidates, MAP[0].raw_key: [{"id": "m-1", "league_id": "l-d2", "match_date": KICKOFF}]}
        assert "is in D2 2. Bundesliga" in self._refused(_state(candidates=cands))

    def test_a_mapped_team_that_would_still_play_in_two_countries(self):
        sides = {**_state().sides, "t-ham": [("m-2", "Spain"), ("m-3", "Spain"), ("m-old", "Scotland")]}
        assert "would still play domestic matches in ['Scotland', 'Spain']" in self._refused(_state(sides=sides))

    def test_a_correctly_spelled_twin_is_a_merge_not_a_rename(self):
        twin = {
            "t-twin": {
                "name": "Preußen Münster",
                "normalized_name": "preußen münster",
                "league_id": None,
                "metadata_text": "{}",
            }
        }  # noqa: E501
        assert "that is a MERGE" in self._refused(_state(teams=twin))

    def test_an_alias_twin_is_a_merge_too(self):
        assert "that is a MERGE" in self._refused(_state(aliases={"preußen munster": {"t-other"}}))

    def test_a_mis_decoded_name_that_does_not_decode_cleanly(self):
        lost = {
            "t-lost": {
                "name": "M\ufffdnchen 1860",
                "normalized_name": "m\ufffdnchen 1860",
                "league_id": None,
                "metadata_text": "{}",
            }
        }  # noqa: E501
        state = _state(teams=lost)
        assert "does not decode cleanly" in self._refused(state)
        assert rsl.plan(state, MAP).counts["mojibake_left"] == 1

    def test_a_raw_rename_onto_an_existing_key(self):
        dup = {
            ("31/10/2025", "Preußen Münster", "Holstein Kiel"): dict(
                _state().raw[("31/10/2025", GARBLED, "Holstein Kiel")]
            )
        }  # noqa: E501
        assert "which already exists (a reload wrote it)" in self._refused(_state(raw=dup))

    def test_mojibake_in_a_league_name_is_left_and_counted(self):
        leagues = {
            **LEAGUES,
            "T1": {
                "id": "l-t1",
                "name": "Süper Lig".encode("utf-8").decode("latin-1"),
                "country": "Turkey",
                "code": "T1",
            },
        }  # noqa: E501
        assert rsl.plan(_state(leagues=leagues), MAP).counts["mojibake_left"] == 1


class TestMap:
    def test_the_reviewed_map(self):
        rows = rsl.load_map()
        assert len(rows) == 10
        assert {(r.from_league, r.to_league) for r in rows} == {("SP1", "SC1"), ("SP2", "SC2")}
        assert {(r.season, r.date, r.time) for r in rows} == {("2627", "01/08/2026", "15:00")}
        assert sum(r.from_league == "SP1" for r in rows) == 5

    @pytest.mark.parametrize(
        "line, error",
        [
            ("2627,01/08/2026,15:00,Ayr,Arbroath,2,0,SP1,ZZ9,x", "not in promote_raw.LEAGUE_MAP"),
            ("2627,01/08/2026,15:00,Ayr,Arbroath,2,0,SP1,SP1,x", "from_league equals to_league"),
            ("2627,01/08/2026,15:00,Ayr,Arbroath,two,0,SP1,SC1,x", "row 1"),
        ],
    )
    def test_a_malformed_map_is_refused(self, tmp_path, line, error):
        path = tmp_path / "map.csv"
        path.write_text("# comment\n" + ",".join(rsl.MAP_FIELDS) + "\n" + line + "\n")
        with pytest.raises(ValueError, match=error):
            rsl.load_map(path)

    def test_a_row_listed_twice_is_refused(self, tmp_path):
        line = "2627,01/08/2026,15:00,Ayr,Arbroath,2,0,SP1,SC1,x"
        path = tmp_path / "map.csv"
        path.write_text(",".join(rsl.MAP_FIELDS) + "\n" + line + "\n" + line + "\n")
        with pytest.raises(ValueError, match="listed twice"):
            rsl.load_map(path)


class _Mogrify:
    """Enough of a cursor for reversal_sql: inline literals the way psycopg2 does for these types."""

    def mogrify(self, sql, params):
        def lit(v):
            return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"

        return (sql % tuple(lit(v) for v in params)).encode()


class TestReversal:
    def test_every_change_is_undone_newest_first_and_each_statement_raises_on_a_moved_row(self):
        p = rsl.plan(_state(), MAP)
        sql = rsl.reversal_sql(_Mogrify(), p, {"SC1": "l-sc1", "SC2": "l-sc2"})
        lines = sql.splitlines()
        assert lines[0].startswith("-- Reversal") and lines[-1] == "COMMIT;" and "BEGIN;" in lines
        body = [ln for ln in lines if ln.startswith("DO $rev$")]
        assert len(body) == 1 + 1 + 4 + 2 + 2 + 2  # rename_raw, rename_team, teams, matches, raw, leagues
        assert all("IF NOT FOUND THEN RAISE EXCEPTION" in ln for ln in body)
        assert 'SET "HomeTeam" = \'' + GARBLED + "'" in body[0]
        assert "UPDATE matches SET league_id = 'l-sp1' WHERE id = 'm-1' AND league_id = 'l-sc1'" in sql
        assert body[-1].startswith("DO $rev$ BEGIN DELETE FROM leagues l WHERE l.id = 'l-sc1'")


class TestMain:
    def test_expect_plan_needs_apply(self):
        assert rsl.main(["--database-url", "postgres://x", "--expect-plan", "refuse=0"]) == 2

    def test_a_bad_expect_plan_key_is_refused(self):
        assert rsl.main(["--database-url", "postgres://x", "--apply", "--expect-plan", "relabel=1"]) == 2

    def test_the_plan_line_has_every_counter_in_order(self):
        line = rsl.plan_line(rsl.plan(_state(), MAP).counts)
        assert [kv.split("=")[0] for kv in line.split()[1:]] == list(rsl.PLAN_COUNTERS)


# --- against a real Postgres --------------------------------------------------------------------------------------

psycopg2 = pytest.importorskip("psycopg2")
PG_URL = os.environ.get("AUSPEX_TEST_PG_URL")

PG_DDL = """
CREATE TABLE leagues (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text NOT NULL, country text NOT NULL,
    sport text NOT NULL DEFAULT 'soccer', external_ids jsonb DEFAULT '{}', UNIQUE (name, country, sport));
CREATE TABLE teams (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text NOT NULL, normalized_name text NOT NULL,
    league_id uuid REFERENCES leagues(id), sport text NOT NULL DEFAULT 'soccer', metadata jsonb DEFAULT '{}',
    UNIQUE (normalized_name, sport));
CREATE TABLE matches (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), league_id uuid REFERENCES leagues(id),
    home_team_id uuid REFERENCES teams(id), away_team_id uuid REFERENCES teams(id), match_date timestamptz NOT NULL,
    status text DEFAULT 'finished', season text);
CREATE TABLE raw_football_data ("Div" text, "Date" text, "Time" text, "HomeTeam" text, "AwayTeam" text, "FTHG" text,
    "FTAG" text, league text, season text, UNIQUE ("Date", "HomeTeam", "AwayTeam"));
INSERT INTO leagues (name, country, external_ids) VALUES ('La Liga', 'Spain', '{"football_data": "SP1"}'),
    ('La Liga 2', 'Spain', '{"football_data": "SP2"}'), ('Premiership', 'Scotland', '{"football_data": "SC0"}'),
    ('2. Bundesliga', 'Germany', '{"football_data": "D2"}');
INSERT INTO teams (name, normalized_name, league_id)
SELECT n, lower(n), (SELECT id FROM leagues WHERE external_ids->>'football_data' = c)
  FROM (VALUES ('Ayr', 'SP1'), ('Arbroath', 'SP1'), ('Montrose', 'SP2'), ('Hamilton', 'SP2'), ('Holstein Kiel', 'D2'),
               (%(garbled)s, 'D2'), ('Celtic', 'SC0')) v(n, c);
INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, season)
SELECT (SELECT id FROM leagues WHERE external_ids->>'football_data' = c),
       (SELECT id FROM teams WHERE name = h), (SELECT id FROM teams WHERE name = a), k::timestamptz, s
  FROM (VALUES ('SP1', 'Ayr', 'Arbroath', '2026-08-01 14:00+00', '2026-2027'),
               ('SP2', 'Montrose', 'Hamilton', '2026-08-01 15:00+00', '2026-2027'),
               ('SC0', 'Hamilton', 'Celtic', '2024-05-01 18:45+00', '2023-2024'),
               ('D2', %(garbled)s, 'Holstein Kiel', '2025-10-31 17:30+00', '2025-2026')) v(c, h, a, k, s);
INSERT INTO raw_football_data VALUES (NULL, '01/08/2026', '15:00', 'Ayr', 'Arbroath', '2.0', '0.0', 'SP1', '2627'),
    (NULL, '01/08/2026', '15:00', 'Montrose', 'Hamilton', '0.0', '5.0', 'SP2', '2627'),
    ('SC0', '01/05/2024', '19:45', 'Hamilton', 'Celtic', '0.0', '2.0', 'SC0', '2324'),
    (NULL, '31/10/2025', '17:30', %(garbled)s, 'Holstein Kiel', '1.0', '1.0', 'D2', '2526');
"""
FINGERPRINT = """
SELECT (SELECT string_agg(name || country || external_ids::text, ',' ORDER BY name, country) FROM leagues),
       (SELECT string_agg(name || normalized_name || coalesce(league_id::text, '-'), ',' ORDER BY id) FROM teams),
       (SELECT string_agg(id::text || league_id::text, ',' ORDER BY id) FROM matches),
       (SELECT string_agg("Date" || "HomeTeam" || "AwayTeam" || league || coalesce("Div", '-'), ','
                          ORDER BY "Date", "HomeTeam") FROM raw_football_data)
"""


@pytest.fixture
def pg():
    if not PG_URL:
        pytest.skip("AUSPEX_TEST_PG_URL not set (a throwaway Postgres for the SQL tests)")
    conn = psycopg2.connect(PG_URL)
    schema = f"rsl_test_{os.getpid()}"
    try:
        with conn.cursor() as c:
            c.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}; SET TimeZone TO 'UTC'")
            c.execute(PG_DDL, {"garbled": GARBLED})
        conn.commit()
        yield conn
    finally:
        conn.rollback()
        with conn.cursor() as c:
            c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()
        conn.close()


@pytest.mark.integration
def test_apply_then_verify_then_the_reversal_restores_every_row(pg, tmp_path):
    with pg.cursor() as c:
        c.execute(FINGERPRINT)
        before = c.fetchone()
    pg.commit()
    out = str(tmp_path / "apply.csv")
    p = rsl.run(pg, True, out, {"create_league": 2, "relabel_raw": 2, "relabel_match": 2, "rename_team": 1}, MAP)
    assert _writes(p) == {
        "create_league": 2,
        "relabel_raw": 2,
        "relabel_match": 2,
        "relabel_team": 4,
        "rename_team": 1,
        "rename_raw": 1,
    }
    again = rsl.run(pg, False, str(tmp_path / "again.csv"), None, MAP)
    assert set(_writes(again).values()) == {0} and again.counts["already_match"] == 2 and again.refusals == []
    assert rsl.verify(pg, out)["now_new"] == len(list(csv.DictReader(open(out))))
    with pg.cursor() as c:
        c.execute(
            "SELECT t.name, l.external_ids->>'football_data' FROM teams t JOIN leagues l ON l.id = t.league_id "
            "ORDER BY t.name"
        )
        assert dict(c.fetchall())["Hamilton"] == "SC2"
        c.execute(open(rsl.reversal_path(out)).read())
        c.execute(FINGERPRINT)
        assert c.fetchone() == before
    pg.commit()


@pytest.mark.integration
def test_a_drifted_plan_writes_nothing(pg, tmp_path):
    with pytest.raises(rsl.PlanDrift):
        rsl.run(pg, True, str(tmp_path / "apply.csv"), {"relabel_match": 3}, MAP)
    with pg.cursor() as c:
        c.execute("SELECT count(*) FROM leagues")
        assert c.fetchone()[0] == 4
    assert not (tmp_path / "apply.csv").exists()
