"""fix_unplayed_soccer_rows: which of the four rows it may write, what it
writes, and that a refusal or a moved row writes nothing. No DB."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
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


fx = _load("fix_unplayed_soccer_rows", "fix_unplayed_soccer_rows.py")
BY_ID = {f.match_id: f for f in fx.FIXES}
BASTIA = BY_ID["ff74d919-d543-4729-a438-5f59e826b7b9"]
NIKI = BY_ID["681794ae-f1e5-4e10-a365-8e71f86ef4a1"]


def _observed(fix) -> dict:
    """The row as prod held it on 2026-10-31."""
    ht = fix.observed_ht or (None, None)
    return {
        "id": fix.match_id,
        "sport": "soccer",
        "league": fix.league,
        "home": fix.home,
        "away": fix.away,
        "match_day": fix.match_day,
        "status": "finished",
        "home_score": None,
        "away_score": None,
        "half_time_home": ht[0],
        "half_time_away": ht[1],
        "external_ids": {},
        "metadata": {},
    }


def _fixed(fix) -> dict:
    return {
        **_observed(fix),
        "status": "cancelled",
        "half_time_home": None,
        "half_time_away": None,
        "external_ids": {"espn": fix.espn_id} if fix.espn_id else {},
        "metadata": {"result_note": fix.note},
    }


class TestFixes:
    def test_the_four_verified_rows(self):
        assert {(f.league, f.home, f.away, f.match_day, f.observed_ht, f.espn_id) for f in fx.FIXES} == {
            ("G1", "Panathinaikos", "Olympiakos", "2019-03-17", (0, 3), "524152"),
            ("F2", "Bastia", "Red Star", "2025-12-05", (0, 3), "747597"),
            ("G1", "Niki Volos", "OFI", "2015-05-10", None, None),
            ("BR1", "Chapecoense-SC", "Atletico-MG", "2016-12-11", None, "445952"),
        }

    def test_every_note_cites_its_source(self):
        for f in fx.FIXES:
            assert "ESPN" in f.note and "football-data" in f.note, f.match_id
            if f.espn_id:
                assert f.espn_id in f.note


class TestAssess:
    def test_observed_row_is_fixed(self):
        assert fx.assess(BASTIA, _observed(BASTIA)) == ("fix", "")
        assert fx.assess(NIKI, _observed(NIKI)) == ("fix", "")

    def test_fixed_row_is_done(self):
        assert fx.assess(BASTIA, _fixed(BASTIA)) == ("done", "already fixed")
        assert fx.assess(NIKI, _fixed(NIKI)) == ("done", "already fixed")

    def test_same_espn_id_already_stamped_is_still_fixable(self):
        assert fx.assess(BASTIA, {**_observed(BASTIA), "external_ids": {"espn": "747597"}})[0] == "fix"

    @pytest.mark.parametrize(
        "change",
        [
            {"home_score": 0, "away_score": 0},  # someone filled a result
            {"half_time_home": 0, "half_time_away": 0},  # HT no longer the observed 0-3
            {"status": "scheduled"},
            {"external_ids": {"espn": "999"}},  # a different ESPN id
            {"status": "cancelled"},  # half-done by hand
        ],
    )
    def test_drifted_row_is_refused(self, change):
        action, reason = fx.assess(BASTIA, {**_observed(BASTIA), **change})
        assert action == "refuse" and "drifted" in reason

    def test_other_fixture_is_refused(self):
        action, reason = fx.assess(BASTIA, {**_observed(BASTIA), "home": "Ajaccio"})
        assert action == "refuse" and "not the expected fixture" in reason

    def test_missing_row_is_refused(self):
        assert fx.assess(BASTIA, None) == ("refuse", "row not found")


class TestUpdateParams:
    def test_bastia(self):
        p = fx.update_params(BASTIA)
        assert (p["id"], p["ht_home"], p["ht_away"], p["espn"]) == (BASTIA.match_id, 0, 3, "747597")
        assert json.loads(p["external"]) == {"espn": "747597"}
        assert json.loads(p["metadata"]) == {"result_note": BASTIA.note}

    def test_row_without_espn_id_adds_none(self):
        p = fx.update_params(NIKI)
        assert (p["ht_home"], p["ht_away"], p["espn"], json.loads(p["external"])) == (None, None, "", {})


class _Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if "SELECT" in sql and "UPDATE" not in sql:
            return
        self.conn.updates.append(params)
        self.rowcount = self.conn.rowcounts.pop(0) if self.conn.rowcounts else 1

    def fetchall(self):
        return self.conn.rows


class _Conn:
    def __init__(self, rows, rowcounts=None):
        self.rows, self.rowcounts = rows, list(rowcounts or [])
        self.updates: list[dict] = []
        self.committed = self.rolled_back = 0

    def cursor(self, cursor_factory=None):
        return _Cursor(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class TestRun:
    def test_dry_run_writes_the_csv_and_nothing_else(self, tmp_path):
        conn = _Conn([_observed(f) for f in fx.FIXES])
        counts = fx.run(conn, apply_changes=False, out_csv=str(tmp_path / "a.csv"))
        assert counts["fix"] == 4 and not conn.updates and not conn.committed
        assert [r["action"] for r in _csv(tmp_path / "a.csv")] == ["fix"] * 4

    def test_apply_writes_each_row_once_and_commits(self, tmp_path):
        conn = _Conn([_observed(f) for f in fx.FIXES])
        counts = fx.run(conn, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert counts["written"] == 4 and conn.committed == 1
        assert [u["id"] for u in conn.updates] == [f.match_id for f in fx.FIXES]

    def test_rerun_after_apply_writes_nothing(self, tmp_path):
        conn = _Conn([_fixed(f) for f in fx.FIXES])
        counts = fx.run(conn, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert counts["done"] == 4 and not conn.updates

    def test_one_refusal_writes_nothing(self, tmp_path):
        rows = [_observed(f) for f in fx.FIXES]
        rows[1]["home_score"] = 1
        conn = _Conn(rows)
        counts = fx.run(conn, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert counts["refuse"] == 1 and not conn.updates and not conn.committed
        assert (tmp_path / "a.csv").exists()  # the audit trail still lands

    def test_row_moved_between_plan_and_write_rolls_everything_back(self, tmp_path):
        conn = _Conn([_observed(f) for f in fx.FIXES], rowcounts=[1, 0])
        with pytest.raises(RuntimeError, match="changed between plan and write"):
            fx.run(conn, apply_changes=True, out_csv=str(tmp_path / "a.csv"))
        assert not conn.committed and conn.rolled_back >= 2  # the read snapshot + the failed write

    def test_audit_csv_is_write_once(self, tmp_path):
        out = tmp_path / "a.csv"
        out.write_text("x")
        conn = _Conn([_observed(f) for f in fx.FIXES])
        with pytest.raises(FileExistsError):
            fx.run(conn, apply_changes=True, out_csv=str(out))
        assert not conn.updates
