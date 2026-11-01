"""scripts/grade_matches.py: exactly the listed finished matches, graded in one
transaction by grade_completed_matches.grade_match, what changed read back by
diffing their predictions and recs, the dry run rolled back, the apply
committed only at --expect-plan."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
_spec = importlib.util.spec_from_file_location("grade_matches", SCRIPTS_DIR / "grade_matches.py")
gm = importlib.util.module_from_spec(_spec)
sys.modules["grade_matches"] = gm
_spec.loader.exec_module(gm)

M1 = "11111111-1111-1111-1111-111111111111"
M2 = "22222222-2222-2222-2222-222222222222"


def _pred(pid, mid=M1, correct=None, actual=None, ptype="double_chance"):
    return {
        "id": pid,
        "match_id": mid,
        "prediction_type": ptype,
        "model_name": "ensemble",
        "is_correct": correct,
        "actual_outcome": actual,
    }


def _rec(rid, mid=M1, status="pending", result=None, pl=None):
    return {
        "id": rid,
        "match_id": mid,
        "bet_type": "1x2",
        "selection": "home",
        "status": status,
        "actual_result": result,
        "profit_loss": pl,
    }


class TestReadIds:
    def test_one_id_per_line_blank_lines_ignored(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text(f"{M1}\n\n  {M2.upper()}  \n")
        assert gm.read_ids(str(f)) == [M1, M2]

    @pytest.mark.parametrize(
        "text, why", [("", "lists no match id"), ("nope\n", "not a match id"), (f"{M1}\n{M1}\n", "more than once")]
    )
    def test_refuses_an_empty_bad_or_repeated_list(self, tmp_path, text, why):
        f = tmp_path / "ids.txt"
        f.write_text(text)
        with pytest.raises(gm.Refused, match=why):
            gm.read_ids(str(f))


class TestParseExpectPlan:
    def test_every_key_once(self):
        assert gm.parse_expect_plan("matches=26,predictions_graded=504,predictions_regraded=0,recs_changed=0") == {
            "matches": 26,
            "predictions_graded": 504,
            "predictions_regraded": 0,
            "recs_changed": 0,
        }

    @pytest.mark.parametrize(
        "text",
        [
            "matches=26,predictions_graded=504,predictions_regraded=0",  # one missing
            "matches=26,predictions_graded=504,predictions_regraded=0,recs_changed=0,extra=1",
            "matches=26,predictions_graded=x,predictions_regraded=0,recs_changed=0",
            "matches=26,matches=26,predictions_graded=1,predictions_regraded=0,recs_changed=0",
            "",
        ],
    )
    def test_refuses_anything_else(self, text):
        with pytest.raises(gm.Refused):
            gm.parse_expect_plan(text)


class TestDiff:
    def test_graded_regraded_and_rec_changes(self):
        before = {
            ("predictions", "p1"): _pred("p1"),
            ("predictions", "p2"): _pred("p2", correct=True, actual="1X"),
            ("predictions", "p3"): _pred("p3", correct=False, actual="X2"),
            ("predictions", "p4"): _pred("p4"),
            ("betting_recommendations", "r1"): _rec("r1"),
            ("betting_recommendations", "r2"): _rec("r2", status="void", result="void: x", pl="0.00"),
        }
        after = dict(before)
        after[("predictions", "p1")] = _pred("p1", correct=True, actual="1X")
        after[("predictions", "p2")] = _pred("p2", correct=False, actual="X2")  # the 90-minute re-grade
        after[("predictions", "p4")] = _pred("p4", actual="push")  # a push: actual set, is_correct stays NULL
        after[("betting_recommendations", "r1")] = _rec("r1", status="won", result="won", pl="9.50")
        changes = gm.diff(before, after)
        assert [(c["id"], c["kind"]) for c in changes] == [
            ("r1", "rec"),
            ("p1", "graded"),
            ("p2", "regraded"),
            ("p4", "graded"),
        ]
        counts = gm.plan_counts(2, changes)
        assert gm.plan_line(counts) == "PLAN matches=2 predictions_graded=2 predictions_regraded=1 recs_changed=1"
        assert counts["rec:pending->won"] == 1
        assert changes[0]["old_status"] == "pending" and changes[0]["new_profit_loss"] == "9.50"

    def test_a_row_that_appears_or_vanishes_refuses(self):
        before = {("predictions", "p1"): _pred("p1")}
        with pytest.raises(gm.Refused, match="appeared or vanished"):
            gm.diff(before, {})
        with pytest.raises(gm.Refused, match="appeared or vanished"):
            gm.diff(before, {**before, ("predictions", "p2"): _pred("p2")})


class _Conn:
    def __init__(self):
        self.commits = self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def graded(monkeypatch):
    """run() with the DB faked: M1 finished; grading flips p1 graded and r1 won."""
    state = {"before": True}

    def snapshot(cur, ids):
        if state["before"]:
            return {("predictions", "p1"): _pred("p1"), ("betting_recommendations", "r1"): _rec("r1")}
        return {
            ("predictions", "p1"): _pred("p1", correct=True, actual="1X"),
            ("betting_recommendations", "r1"): _rec("r1", status="won", result="won", pl="9.50"),
        }

    def grade(cur, matches, stats=None):
        state["before"] = False

    monkeypatch.setattr(gm, "snapshot", snapshot)
    monkeypatch.setattr(gm, "grade_listed", grade)
    monkeypatch.setattr(
        gm.gcm, "list_finished_matches", lambda cur, days, mid: [{"match_id": mid}] if mid == M1 else []
    )
    return state


PLAN = {"matches": 1, "predictions_graded": 1, "predictions_regraded": 0, "recs_changed": 1}


def test_a_dry_run_grades_inside_the_transaction_and_rolls_back(graded, tmp_path):
    conn = _Conn()
    counts = gm.run(conn, [M1], False, None, str(tmp_path / "dry.csv"))
    assert {k: counts[k] for k in gm.PLAN_KEYS} == PLAN
    assert (conn.commits, conn.rollbacks) == (0, 1)
    rows = list(csv.DictReader((tmp_path / "dry.csv").open()))
    assert [(r["table"], r["kind"]) for r in rows] == [("betting_recommendations", "rec"), ("predictions", "graded")]


def test_an_apply_commits_only_at_its_pin(graded, tmp_path):
    conn = _Conn()
    gm.run(conn, [M1], True, PLAN, str(tmp_path / "apply.csv"))
    assert (conn.commits, conn.rollbacks) == (1, 0)


def test_an_apply_that_drifted_rolls_back(graded, tmp_path):
    conn = _Conn()
    with pytest.raises(gm.Refused, match="drifted"):
        gm.run(conn, [M1], True, {**PLAN, "predictions_graded": 2}, str(tmp_path / "apply.csv"))
    assert (conn.commits, conn.rollbacks) == (0, 1)


def test_a_listed_match_that_is_not_finished_refuses_before_grading(graded, tmp_path):
    with pytest.raises(gm.Refused, match="not finished"):
        gm.run(_Conn(), [M1, M2], False, None, str(tmp_path / "dry.csv"))
    assert graded["before"] is True and not (tmp_path / "dry.csv").exists()


def test_the_audit_csv_is_write_once(graded, tmp_path):
    (tmp_path / "dry.csv").write_text("an earlier run's record\n")
    with pytest.raises(gm.Refused, match="write-once"):
        gm.run(_Conn(), [M1], False, None, str(tmp_path / "dry.csv"))
    assert (tmp_path / "dry.csv").read_text() == "an earlier run's record\n"


def test_main_refuses_an_apply_without_a_pin(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text(M1 + "\n")
    argv = ["--match-ids-file", str(ids), "--apply", "--out-csv", str(tmp_path / "a.csv"), "--database-url", "x"]
    assert gm.main(argv) == 1
