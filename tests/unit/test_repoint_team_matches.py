"""scripts/repoint_team_matches.py: moving reviewed match rows between team
rows (Santos Laguna's Liga MX rows off Santos FC's 'Santos', Chivas' off CD
Guadalajara's 'Guadalajara'). Pure planning is tested on synthetic rows; the
write path through a recording cursor."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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
rtm = _load("repoint_team_matches")

T0 = datetime(2026, 9, 6, 23, 0, tzinfo=timezone.utc)
TEAMS = {
    "santos": {"id": "S", "name": "Santos", "normalized_name": "santos", "espn": None},
    "santos laguna": {"id": "L", "name": "Santos Laguna", "normalized_name": "santos laguna", "espn": None},
    "guadalajara": {"id": "G", "name": "Guadalajara", "normalized_name": "guadalajara", "espn": None},
    "guadalajara chivas": {
        "id": "C",
        "name": "Guadalajara Chivas",
        "normalized_name": "guadalajara chivas",
        "espn": None,
    },
    "cruz azul": {"id": "X", "name": "Cruz Azul", "normalized_name": "cruz azul", "espn": None},
    "atlas": {"id": "A", "name": "Atlas", "normalized_name": "atlas", "espn": None},
}
NAMES = {t["id"]: t["name"] for t in TEAMS.values()}
SEED = {
    "2674": ti.SeedEntry("2674", "Santos", "bra.1", "e"),
    "225": ti.SeedEntry("225", "Santos Laguna", "mex.1", "e"),
    "219": ti.SeedEntry("219", "Guadalajara Chivas", "mex.1", "e"),
}


def match(match_id, home, away, when=T0, *, status="finished", espn=None, league="Liga MX"):
    return {
        "id": match_id,
        "sport": "soccer",
        "league": league,
        "league_id": "lg-mx",
        "home_id": home,
        "away_id": away,
        "home": NAMES[home],
        "away": NAMES[away],
        "match_date": when,
        "status": status,
        "espn": espn,
    }


def row(match_id, side, frm="Santos", to="Santos Laguna", opponent="Cruz Azul", when=T0):
    return rtm.MapRow(match_id, side, frm, to, "Liga MX", when.strftime("%Y-%m-%d %H:%M"), opponent, "401", "ev")


def plan(rows, matches, neighbours=(), stats=(), teams=None, holders=None, seed=None):
    by_id = {m["id"]: m for m in matches}
    return rtm.plan_repoints(
        rows,
        by_id,
        teams or TEAMS,
        list(neighbours) + list(matches),
        list(stats),
        SEED if seed is None else seed,
        holders or {},
    )


def actions(p):
    return {(e["match_id"], e["side"]): (e["action"], e["reason"]) for e in p.entries}


# ── planning ─────────────────────────────────────────────────────────


class TestPlan:
    def test_a_reviewed_side_moves_with_a_guarded_update(self):
        p = plan([row("m1", "away")], [match("m1", "X", "S")])
        assert actions(p)[("m1", "away")] == ("repoint", "")
        assert p.updates["m1"] == {
            "id": "m1",
            "old_home": "X",
            "old_away": "S",
            "new_home": "X",
            "new_away": "L",
            "league_id": "lg-mx",
            "match_date": T0,
        }

    def test_a_row_that_drifted_is_refused(self):
        p = plan([row("m1", "away", opponent="Atlas")], [match("m1", "X", "S")])
        action, reason = actions(p)[("m1", "away")]
        assert action == "refuse" and "drifted since the review" in reason
        assert p.updates == {}

    def test_a_re_dated_row_is_refused(self):
        p = plan([row("m1", "away", when=T0 - timedelta(hours=3))], [match("m1", "X", "S")])
        assert "not the reviewed fixture" in actions(p)[("m1", "away")][1]

    def test_a_missing_row_or_target_is_refused(self):
        assert actions(plan([row("gone", "away")], []))[("gone", "away")] == ("refuse", "match row not found")
        teams = {k: v for k, v in TEAMS.items() if k != "santos laguna"}
        p = plan([row("m1", "away")], [match("m1", "X", "S")], teams=teams)
        assert "does not exist" in actions(p)[("m1", "away")][1]

    def test_a_moved_side_is_done_and_not_written_again(self):
        p = plan([row("m1", "away")], [match("m1", "X", "L")])
        assert actions(p)[("m1", "away")] == ("done", "already on the target team")
        assert p.updates == {}

    def test_a_side_that_would_play_itself_is_refused(self):
        p = plan([row("m1", "home", to="Cruz Azul")], [match("m1", "S", "X")])
        assert actions(p)[("m1", "home")] == ("refuse", "both sides would be one team")

    def test_a_unique_collision_is_refused(self):
        existing = match("fd", "X", "L")  # the same triple already on the target pair
        p = plan([row("m1", "away")], [match("m1", "X", "S")], neighbours=[existing])
        assert "UNIQUE(home, away, match_date)" in actions(p)[("m1", "away")][1]

    def test_a_same_pair_twin_within_36h_is_refused(self):
        copy = match("fd", "X", "L", T0 + timedelta(hours=1))  # football-data's copy, 1 h late
        p = plan([row("m1", "away")], [match("m1", "X", "S")], neighbours=[copy])
        assert "same-pair twin within 36 h: fd" in actions(p)[("m1", "away")][1]

    def test_a_same_pair_match_36h_away_is_another_fixture(self):
        later = match("next", "X", "L", T0 + timedelta(hours=36))
        assert actions(plan([row("m1", "away")], [match("m1", "X", "S")], neighbours=[later]))[("m1", "away")][0] == (
            "repoint"
        )

    def test_existing_twins_on_the_old_pair_move_together(self):
        stale = match("stale", "X", "S", T0 + timedelta(hours=3), status="scheduled")
        fin = match("fin", "X", "S", T0)
        p = plan([row("stale", "away", when=T0 + timedelta(hours=3)), row("fin", "away")], [stale, fin])
        assert {a for a, _ in actions(p).values()} == {"repoint"}
        assert set(p.updates) == {"stale", "fin"}

    def test_a_twin_left_behind_keeps_its_partner_too(self):
        # The stale row was re-dated since the review: refused. Its finished twin must not move alone.
        stale = match("stale", "X", "S", T0 + timedelta(hours=4), status="scheduled")
        fin = match("fin", "X", "S", T0)
        p = plan([row("stale", "away", when=T0 + timedelta(hours=3)), row("fin", "away")], [stale, fin])
        assert actions(p)[("stale", "away")][0] == "refuse"
        assert actions(p)[("fin", "away")] == ("refuse", "its twin stale on the same pair is refused")
        assert p.updates == {}

    def test_both_sides_of_one_match_move_in_one_update(self):
        rows = [
            row("sg", "home", opponent="Guadalajara"),
            row("sg", "away", frm="Guadalajara", to="Guadalajara Chivas", opponent="Santos"),
        ]
        p = plan(rows, [match("sg", "S", "G")])
        assert {a for a, _ in actions(p).values()} == {"repoint"}
        assert (p.updates["sg"]["new_home"], p.updates["sg"]["new_away"]) == ("L", "C")
        # After the apply both sides plan as done.
        assert {a for a, _ in actions(plan(rows, [match("sg", "L", "C")])).values()} == {"done"}

    def test_a_match_is_never_half_moved(self):
        rows = [
            row("sg", "home", opponent="Guadalajara"),
            row("sg", "away", frm="Guadalajara", to="Guadalajara Chivas", opponent="Somebody Else"),
        ]
        p = plan(rows, [match("sg", "S", "G")])
        assert actions(p)[("sg", "home")] == ("refuse", "another side of this match is refused")
        assert p.updates == {}

    def test_match_stats_of_the_moved_side_move_with_it(self):
        stats = [{"match_id": "m1", "team_id": "S"}, {"match_id": "m1", "team_id": "X"}]
        p = plan([row("m1", "away")], [match("m1", "X", "S")], stats=stats)
        assert p.stats_moves == [{"match_id": "m1", "from": "S", "to": "L"}]

    def test_match_stats_already_held_by_the_target_refuse_the_move(self):
        stats = [{"match_id": "m1", "team_id": "S"}, {"match_id": "m1", "team_id": "L"}]
        p = plan([row("m1", "away")], [match("m1", "X", "S")], stats=stats)
        assert "match_stats already has a row" in actions(p)[("m1", "away")][1]


class TestStamps:
    def test_every_seeded_team_the_map_names_is_stamped(self):
        p = plan([row("m1", "away")], [match("m1", "X", "S")])
        assert {(s["to_team"], s["espn_team_id"], s["action"]) for s in p.stamps} == {
            ("Santos", "2674", "stamp"),
            ("Santos Laguna", "225", "stamp"),
        }

    def test_an_existing_stamp_is_done_and_a_conflict_refused(self):
        teams = {
            **TEAMS,
            "santos": {**TEAMS["santos"], "espn": "2674"},
            "santos laguna": {**TEAMS["santos laguna"], "espn": "999"},
        }
        p = plan([row("m1", "away")], [match("m1", "X", "S")], teams=teams, holders={"2674": ["S"]})
        got = {s["to_team"]: (s["action"], s["reason"]) for s in p.stamps}
        assert got["Santos"] == ("done", "already stamped")
        assert got["Santos Laguna"] == ("refuse", "'Santos Laguna' carries ESPN id 999")
        assert p.counts["stamp_refuse"] == 1

    def test_an_id_another_row_holds_is_refused(self):
        p = plan([row("m1", "away")], [match("m1", "X", "S")], holders={"225": ["elsewhere"]})
        got = {s["to_team"]: s for s in p.stamps}
        assert got["Santos Laguna"]["action"] == "refuse"
        assert "elsewhere" in got["Santos Laguna"]["reason"]


# ── the map ──────────────────────────────────────────────────────────


class TestMap:
    HEADER = "match_id,side,from_team,to_team,league,kickoff_utc,opponent,espn_event,evidence\n"

    def _write(self, tmp_path, body):
        path = tmp_path / "map.csv"
        path.write_text(self.HEADER + body, encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        "body,error",
        [
            ("m1,middle,Santos,Santos Laguna,Liga MX,2026-09-06 23:00,Cruz Azul,401,e\n", "home or away"),
            ("m1,away,Santos,Santos Laguna,Liga MX,06/09/2026,Cruz Azul,401,e\n", "does not match format"),
            ("m1,away,Santos,santos,Liga MX,2026-09-06 23:00,Cruz Azul,401,e\n", "different to_team"),
            (
                "m1,away,Santos,Santos Laguna,Liga MX,2026-09-06 23:00,Cruz Azul,401,e\n"
                "m1,away,Santos,Santos Laguna,Liga MX,2026-09-06 23:00,Cruz Azul,401,e\n",
                "listed twice",
            ),
        ],
    )
    def test_a_malformed_map_fails_loudly(self, tmp_path, body, error):
        with pytest.raises(ValueError, match=error):
            rtm.load_map(self._write(tmp_path, body))

    def test_a_map_missing_a_column_fails_loudly(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text("match_id,side\nm1,away\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            rtm.load_map(path)

    def test_the_checked_in_map_moves_only_onto_seeded_clubs_with_espn_evidence(self):
        rows = rtm.load_map()
        seed = ti.load_espn_seed()
        seeded = {e.team_name for e in seed.values()}
        assert len(rows) == 39
        assert {r.to_team for r in rows} == {"Santos Laguna", "Guadalajara Chivas"} <= seeded
        assert {r.from_team for r in rows} == {"Santos", "Guadalajara"}
        assert all(r.league == "Liga MX" and r.espn_event.isdigit() for r in rows)
        assert all(f"ESPN mex.1 event {r.espn_event}" in r.evidence for r in rows)


# ── writing ──────────────────────────────────────────────────────────


class RecordingCursor:
    def __init__(self, rowcount=1):
        self.calls: list[tuple[str, object]] = []
        self.rowcount_value = rowcount

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))
        self.rowcount = self.rowcount_value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, cursor):
        self.cur = cursor
        self.committed = self.rolled_back = 0

    def cursor(self, cursor_factory=None):
        return self.cur

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


class TestApply:
    def _plan(self):
        stats = [{"match_id": "m1", "team_id": "S"}]
        return plan([row("m1", "away")], [match("m1", "X", "S")], stats=stats)

    def test_every_write_is_guarded_and_counted(self):
        cur = RecordingCursor()
        assert rtm.apply_plan(cur, self._plan()) == 4  # the match, its match_stats row, two stamps
        update = cur.calls[0][0]
        assert "AND home_team_id = %(old_home)s::uuid AND away_team_id = %(old_away)s::uuid" in update
        assert "AND league_id = %(league_id)s::uuid AND match_date = %(match_date)s" in update
        assert "WHERE match_id = %(match_id)s::uuid AND team_id = %(from)s::uuid" in cur.calls[1][0]
        assert "NOT (COALESCE(t.external_ids, '{}'::jsonb) ? 'espn')" in cur.calls[2][0]

    def test_a_guard_miss_raises(self):
        with pytest.raises(RuntimeError, match="changed between plan and write"):
            rtm.apply_plan(RecordingCursor(rowcount=0), self._plan())

    def _run(self, monkeypatch, tmp_path, *, apply_changes, expect=None, rows=None, matches=None):
        rows = rows or [row("m1", "away")]
        matches = {m["id"]: m for m in (matches or [match("m1", "X", "S")])}
        state = (matches, TEAMS, list(matches.values()), [], {}, {})
        monkeypatch.setattr(rtm, "read_state", lambda cur, r, s: state)
        monkeypatch.setattr(rtm, "features_command", lambda cur, p, t: "docker compose exec ... --match-ids m1")
        cur = RecordingCursor()
        conn = FakeConn(cur)
        out = tmp_path / "audit.csv"
        counts = rtm.run(conn, rows, SEED, apply_changes, str(out), expect or {})
        return counts, conn, cur, out

    def test_a_dry_run_writes_the_audit_csv_and_nothing_else(self, monkeypatch, tmp_path):
        counts, conn, cur, out = self._run(monkeypatch, tmp_path, apply_changes=False)
        assert out.exists() and "m1" in out.read_text()
        assert not any(sql.startswith("UPDATE") for sql, _ in cur.calls)
        assert conn.committed == 0 and counts["repoint"] == 1

    def test_the_apply_locks_writes_and_commits(self, monkeypatch, tmp_path):
        counts, conn, cur, out = self._run(monkeypatch, tmp_path, apply_changes=True, expect={"repoint": 1})
        assert cur.calls[0][0].startswith("SELECT id FROM matches WHERE id = ANY") and "FOR UPDATE" in cur.calls[0][0]
        assert [sql.split()[0] for sql, _ in cur.calls[1:]] == ["UPDATE", "UPDATE", "UPDATE"]
        assert conn.committed == 1 and counts["written"] == 3

    def test_a_pin_mismatch_writes_nothing(self, monkeypatch, tmp_path):
        counts, conn, cur, _ = self._run(monkeypatch, tmp_path, apply_changes=True, expect={"repoint": 2})
        assert counts["pin_mismatch"] == 1
        assert not any(sql.startswith("UPDATE") for sql, _ in cur.calls)
        assert conn.committed == 0

    def test_a_refused_stamp_writes_nothing(self, monkeypatch, tmp_path):
        teams = {**TEAMS, "santos laguna": {**TEAMS["santos laguna"], "espn": "999"}}
        matches = {"m1": match("m1", "X", "S")}
        monkeypatch.setattr(rtm, "read_state", lambda cur, r, s: (matches, teams, list(matches.values()), [], {}, {}))
        monkeypatch.setattr(rtm, "features_command", lambda *a: None)
        cur = RecordingCursor()
        counts = rtm.run(FakeConn(cur), [row("m1", "away")], SEED, True, str(tmp_path / "a.csv"), {})
        assert counts["stamp_refuse"] == 1
        assert not any(sql.startswith("UPDATE") for sql, _ in cur.calls)

    def test_the_audit_csv_is_write_once(self, monkeypatch, tmp_path):
        (tmp_path / "audit.csv").write_text("earlier run\n")
        with pytest.raises(FileExistsError):
            self._run(monkeypatch, tmp_path, apply_changes=True)

    def test_main_fails_on_unpinned_refusals_and_passes_pinned_ones(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rtm, "load_map", lambda path: [])
        monkeypatch.setattr(rtm.team_identity, "load_espn_seed", lambda path=None: {})
        monkeypatch.setattr(rtm, "run", lambda *a, **k: Counter(refuse=6, repoint=33))

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", lambda url: _Conn())
        base = ["--database-url", "postgresql://x", "--out-csv", str(tmp_path / "o.csv")]
        assert rtm.main(base) == 1
        assert rtm.main(base + ["--expect-refused", "6"]) == 0
