"""backfill_soccer_regulation_scores: which finished soccer ties get their
90-minute score stamped, and the write discipline (dry run by default,
write-once audit CSV before any write, guarded UPDATEs in one transaction).

ESPN payloads come from fixtures/espn_soccer_results.json (real
competitions, trimmed)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "espn_soccer_results.json").read_text())


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bf = _load("backfill_soccer_regulation_scores", "backfill_soccer_regulation_scores.py")
SOCCER = bf.SOCCER


def comp_of(fixture: str, **status) -> dict:
    comp = json.loads(json.dumps(FIXTURES[fixture]))
    comp["status"]["type"].update(status)
    return comp


def cand(
    cid="m-bel", league=("FIFA World Cup", "International"), espn="760493", score=(3, 2), detail="AET", reg=(None, None)
):
    home_id, away_id = f"{cid}-home", f"{cid}-away"
    kickoff = {"m-bel": "2026-07-01T20:00:00+00:00", "m-ldu": "2026-09-16T22:00:00+00:00"}.get(
        cid, "2026-07-19T19:00:00+00:00"
    )
    return {
        "id": cid,
        "league_name": league[0],
        "country": league[1],
        "match_date": datetime.fromisoformat(kickoff),
        "home_id": home_id,
        "away_id": away_id,
        "home": "Home",
        "away": "Away",
        "home_score": score[0],
        "away_score": score[1],
        "espn_id": espn,
        "result_detail": detail,
        "reg_home": reg[0],
        "reg_away": reg[1],
    }


def assess(c: dict, comp: dict):
    return bf.assess(c, comp, bf.fu._competition_parts(comp, SOCCER))


class TestAssess:
    def test_belgium_senegal_is_stamped_2_2(self):
        assert assess(cand(), comp_of("belgium_senegal_aet_3_2")) == (bf.STAMP, "", (2, 2))

    def test_a_shootout_without_extra_time_is_stamped_with_its_final(self):
        c = cand("m-ldu", ("Copa Libertadores", "International"), "401912525", detail="FT-PENS")
        assert assess(c, comp_of("ldu_palmeiras_ft_pens_3_2")) == (bf.STAMP, "", (3, 2))

    def test_a_different_final_is_refused(self):
        action, reason, _ = assess(cand(score=(2, 2)), comp_of("belgium_senegal_aet_3_2"))
        assert action == bf.REFUSE and "ESPN final 3-2" in reason

    def test_espn_no_longer_saying_extra_time_is_refused(self):
        comp = comp_of("belgium_senegal_aet_3_2", name="STATUS_FULL_TIME", detail="FT", shortDetail="FT")
        assert assess(cand(), comp)[0] == bf.REFUSE

    def test_plays_contradicting_the_header_are_refused(self):
        comp = comp_of("belgium_senegal_aet_3_2")
        comp["details"].pop()
        action, reason, _ = assess(cand(), comp)
        assert action == bf.REFUSE and "add up to 2-2" in reason

    def test_plays_without_a_readable_clock_are_refused(self):
        comp = comp_of("belgium_senegal_aet_3_2")
        comp["details"][0]["clock"]["displayValue"] = ""
        action, reason, _ = assess(cand(), comp)
        assert action == bf.REFUSE and "do not give the 90-minute score" in reason

    def test_an_incomplete_competition_is_refused(self):
        assert assess(cand(), comp_of("belgium_senegal_aet_3_2", completed=False))[0] == bf.REFUSE

    def test_the_same_keys_are_already_set(self):
        assert assess(cand(reg=("2", "2")), comp_of("belgium_senegal_aet_3_2"))[0] == bf.ALREADY_SET

    def test_different_keys_are_never_overwritten(self):
        action, reason, _ = assess(cand(reg=("3", "2")), comp_of("belgium_senegal_aet_3_2"))
        assert action == bf.REFUSE and "different 90-minute score" in reason


class FakeCursor:
    def __init__(self, candidate_rows=(), league_ids=None):
        self.candidate_rows = [dict(r) for r in candidate_rows]
        self.league_ids = league_ids or {
            ("FIFA World Cup", "International"): "lg-wc",
            ("Copa Libertadores", "International"): "lg-lib",
        }
        self.updates: list[dict] = []
        self.rowcount = 1
        self.update_rowcount = 1
        self._result: list = []

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        if "FROM matches m JOIN leagues l" in sql and "result_detail" in sql and sql.startswith("SELECT"):
            self._result = list(self.candidate_rows)
        elif sql.startswith("SELECT id::text AS id FROM leagues"):
            lid = self.league_ids.get(tuple(params))
            self._result = [{"id": lid}] if lid else []
        elif sql.startswith("UPDATE matches"):
            self.updates.append(dict(params))
            self.rowcount = self.update_rowcount
        else:
            raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, cur):
        self.cur = cur
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def espn(monkeypatch):
    """Scoreboards by (slug, day); find_team resolves 'Home'/'Away' style names."""
    boards: dict[tuple[str, date], list[dict]] = {}
    fetched: list[tuple[str, date]] = []

    def fetch_day(cfg, slug, day):
        fetched.append((slug, day))
        return boards.get((slug, day), [])

    teams: dict[str, str] = {}
    monkeypatch.setattr(bf.fu, "fetch_day", fetch_day)
    monkeypatch.setattr(bf.fu, "find_team", lambda cur, cfg, name, league_id, espn_id=None: teams.get(name))
    monkeypatch.setitem(bf.fu.FETCH_STATS, "ok", 1)
    monkeypatch.setitem(bf.fu.FETCH_STATS, "failed", 0)
    return boards, teams, fetched


def _event(comp):
    return {"id": comp["id"], "competitions": [comp]}


class TestPlan:
    def test_matched_by_espn_id_and_by_triple_only_among_candidates(self, espn):
        boards, teams, fetched = espn
        bel = cand()
        ldu = cand("m-ldu", ("Copa Libertadores", "International"), None, detail="FT-PENS")
        lib = comp_of("ldu_palmeiras_ft_pens_3_2")
        teams.update({"Liga de Quito": "m-ldu-home", "Palmeiras": "m-ldu-away"})
        boards[("fifa.world", date(2026, 7, 1))] = [_event(comp_of("belgium_senegal_aet_3_2"))]
        # a non-candidate on the same scoreboard (Spain v Argentina) must not match anything
        boards[("conmebol.libertadores", date(2026, 9, 16))] = [_event(lib), _event(comp_of("spain_argentina_aet_1_0"))]
        counts: Counter = Counter()

        entries = bf.plan(FakeCursor(), [bel, ldu], counts)

        assert [(e["match_id"], e["action"], e["regulation_home"], e["regulation_away"]) for e in entries] == [
            ("m-bel", bf.STAMP, 2, 2),
            ("m-ldu", bf.STAMP, 3, 2),
        ]
        assert entries[1]["espn_id"] == "401912525"  # learnt from ESPN for the row without one
        assert counts == Counter({bf.STAMP: 2})
        # each candidate day plus the day before
        assert ("fifa.world", date(2026, 6, 30)) in fetched and ("conmebol.libertadores", date(2026, 9, 15)) in fetched

    def test_the_triple_resolves_each_side_by_its_espn_team_id(self, espn, monkeypatch):
        # ESPN's 'Santos' is two clubs by name; the sweeps resolve a side by its ESPN team id first.
        boards, teams, _ = espn
        ldu = cand("m-ldu", ("Copa Libertadores", "International"), None, detail="FT-PENS")
        teams.update({"Liga de Quito": "m-ldu-home", "Palmeiras": "m-ldu-away"})
        boards[("conmebol.libertadores", date(2026, 9, 16))] = [_event(comp_of("ldu_palmeiras_ft_pens_3_2"))]
        lookups: list[tuple] = []

        def find_team(cur, cfg, name, league_id, espn_id=None):
            lookups.append((name, espn_id))
            return teams.get(name)

        monkeypatch.setattr(bf.fu, "find_team", find_team)
        entries = bf.plan(FakeCursor(), [ldu], Counter())
        assert entries[0]["action"] == bf.STAMP
        assert lookups == [("Liga de Quito", "4816"), ("Palmeiras", "2029")]

    def test_a_candidate_missing_from_espn_is_refused(self, espn):
        counts: Counter = Counter()
        entries = bf.plan(FakeCursor(), [cand()], counts)
        assert entries[0]["action"] == bf.REFUSE and "not found" in entries[0]["reason"]

    def test_a_league_without_an_espn_slug_is_refused(self, espn):
        entries = bf.plan(FakeCursor(), [cand(league=("Coupe de France", "France"))], Counter())
        assert entries[0]["action"] == bf.REFUSE and "no ESPN slug" in entries[0]["reason"]

    def test_candidates_keep_only_decided_after_90(self):
        rows = [cand(detail="AET"), cand("m-x", detail="FT-PENS"), cand("m-y", detail="SUSPENDED")]
        assert [c["id"] for c in bf.candidates(FakeCursor(rows), 365)] == ["m-bel", "m-x"]


class TestRun:
    def _setup(self, espn, *cands):
        boards, _teams, _ = espn
        boards[("fifa.world", date(2026, 7, 1))] = [_event(comp_of("belgium_senegal_aet_3_2"))]
        boards[("fifa.world", date(2026, 7, 19))] = [_event(comp_of("spain_argentina_aet_1_0"))]
        cur = FakeCursor(cands or [cand(), cand("m-esp", espn="760517", score=(1, 0))])
        return cur, FakeConn(cur)

    def test_dry_run_writes_the_audit_csv_and_nothing_else(self, espn, tmp_path):
        cur, conn = self._setup(espn)
        out = tmp_path / "audit.csv"
        counts = bf.run(conn, 365, apply_changes=False, out_csv=str(out))
        assert counts[bf.STAMP] == 2 and cur.updates == [] and conn.commits == 0
        rows = list(csv.DictReader(out.open()))
        assert [(r["match_id"], r["action"], r["regulation_home"], r["regulation_away"]) for r in rows] == [
            ("m-bel", "stamp", "2", "2"),
            ("m-esp", "stamp", "0", "0"),
        ]

    def test_the_csv_is_write_once(self, espn, tmp_path):
        cur, conn = self._setup(espn)
        out = tmp_path / "audit.csv"
        out.write_text("earlier run\n")
        with pytest.raises(FileExistsError):
            bf.run(conn, 365, apply_changes=True, out_csv=str(out))
        assert cur.updates == [] and out.read_text() == "earlier run\n"

    def test_apply_writes_guarded_stamps_in_one_transaction(self, espn, tmp_path):
        cur, conn = self._setup(espn)
        counts = bf.run(conn, 365, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert counts["written"] == 2 and conn.commits == 1
        first = cur.updates[0]
        assert json.loads(first["patch"]) == {
            "regulation_home_score": 2,
            "regulation_away_score": 2,
            "went_extra": True,
        }
        assert (first["id"], first["home_score"], first["away_score"], first["result_detail"]) == ("m-bel", 3, 2, "AET")

    def test_a_row_that_changed_rolls_everything_back(self, espn, tmp_path):
        cur, conn = self._setup(espn)
        cur.update_rowcount = 0
        with pytest.raises(RuntimeError, match="changed between plan and write"):
            bf.run(conn, 365, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert conn.commits == 0 and conn.rollbacks >= 2

    def test_a_refusal_writes_nothing(self, espn, tmp_path):
        cur, conn = self._setup(espn, cand(), cand("m-gone", espn="999"))
        counts = bf.run(conn, 365, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert counts[bf.REFUSE] == 1 and cur.updates == [] and conn.commits == 0

    def test_skip_refused_stamps_the_rest(self, espn, tmp_path):
        cur, conn = self._setup(espn, cand(), cand("m-gone", espn="999"))
        counts = bf.run(conn, 365, apply_changes=True, out_csv=str(tmp_path / "a.csv"), skip_refused=True)
        assert counts["written"] == 1 and [u["id"] for u in cur.updates] == ["m-bel"]

    def test_every_espn_request_failing_writes_nothing(self, espn, tmp_path, monkeypatch):
        cur, conn = self._setup(espn)
        monkeypatch.setitem(bf.fu.FETCH_STATS, "ok", 0)
        monkeypatch.setitem(bf.fu.FETCH_STATS, "failed", 4)
        with pytest.raises(RuntimeError, match="Every ESPN request failed"):
            bf.run(conn, 365, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert cur.updates == []

    def test_the_update_is_guarded_on_what_the_plan_read(self):
        sql = " ".join(bf.UPDATE_SQL.split())
        for guard in (
            "status = 'finished'",
            "home_score = %(home_score)s",
            "away_score = %(away_score)s",
            "metadata->>'result_detail' = %(result_detail)s",
            "NOT (metadata ? 'regulation_home_score')",
            "NOT (metadata ? 'regulation_away_score')",
        ):
            assert guard in sql
        assert "COALESCE(metadata, '{}'::jsonb) || %(patch)s::jsonb" in sql

    def test_main_needs_a_database_and_a_positive_window(self):
        assert bf.main(["--database-url", ""]) == 2
        assert bf.main(["--database-url", "postgresql://x/y", "--days-back", "0"]) == 2

    def test_main_exits_1_on_a_refusal(self, espn, tmp_path, monkeypatch):
        cur, conn = self._setup(espn, cand("m-gone", espn="999"))
        monkeypatch.setattr(bf.psycopg2, "connect", lambda url: _Ctx(conn))
        assert bf.main(["--database-url", "postgresql://x/y", "--out-csv", str(tmp_path / "a.csv")]) == 1


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *a):
        return False


def test_scoreboard_days_include_the_day_before():
    kickoffs = [datetime(2026, 10, 22, 0, 30, tzinfo=timezone.utc), datetime(2026, 10, 23, 0, 30, tzinfo=timezone.utc)]
    assert bf.scoreboard_days(kickoffs) == [date(2026, 10, 21), date(2026, 10, 22), date(2026, 10, 23)]
