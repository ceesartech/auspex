"""scripts/ops/repoint_team_matches.sh --pass 1|2, run for real against a
scripted docker: each pass's pins and after-state, pass 1's before-state gate
(exactly the map's 39 sides on santos/guadalajara), an after-state counted
over soccer rows created before T0, and the read-only 'already done' exit."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WRAPPER = HERE.parents[1] / "scripts" / "ops" / "repoint_team_matches.sh"
MAP = HERE.parents[1] / "scripts" / "data" / "repoint_team_matches.csv"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

STAMPS = "stamps=Guadalajara Chivas:219,Santos:2674,Santos Laguna:225"
PLAN = {
    1: ("PLAN repoint=33 done=0 refuse=6 not_found=0 stamp_stamp=3 stamp_done=0 stamp_refuse=0", 39, 6),
    2: ("PLAN repoint=6 done=32 refuse=1 not_found=1 stamp_stamp=0 stamp_done=3 stamp_refuse=0", 6, 0),
}


def _state(left, n=1000, stamps=STAMPS):
    return f"{stamps}\nleft={left}\nsoccer_before_t0={n}"


def _rules(pass_, pre=None, undone="33", states=None):
    plan, left0, left1 = PLAN[pass_]
    states = states or [_state(left0), _state(left0), _state(left1)]
    return [
        {"match": r"repoint_team_pre\.log 2>&1; echo", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN repoint=\.\*' /tmp/repoint_team_pre\.log", "out": pre or plan},
        {"match": r"api python -c import csv, sys; left", "out": undone},
        *({"match": r"postgres sh -c psql", "out": s, "times": 1} for s in states),
        {"match": r"repoint_team_dry\.log 2>&1; rc", "out": "tool exit 0"},
        {"match": r"repoint_team_apply\.log 2>&1; rc", "out": "tool exit 0"},
        {"match": r"repoint_team_after\.log 2>&1; rc", "out": "tool exit 0"},
        {"match": r"Refresh features", "out": "docker compose exec -T api python compute_features.py --match-ids x"},
        {"match": r"postgres rm -f", "out": ""},
    ]


def _run(tmp_path, pass_args, rules):
    env = fd.install(tmp_path, rules)
    (tmp_path / "auspex" / "scripts" / "data").mkdir(parents=True)
    shutil.copy(MAP, tmp_path / "auspex" / "scripts" / "data" / MAP.name)
    rc, out = fd.run(WRAPPER, env, *pass_args)
    return rc, out, fd.calls(tmp_path)


def _tool(calls, log):
    return [c for c in calls if f"/tmp/{log}.log" in c and "repoint_team_matches.py" in c]


@pytest.mark.parametrize("args", [[], ["--pass", "3"], ["--pass"], ["--bogus"]])
def test_a_pass_is_required(tmp_path, args):
    rc, out, calls = _run(tmp_path, args, _rules(1))
    assert rc == 2 and "--pass 1|2" in out and calls == []


@pytest.mark.parametrize(
    "pass_, pins, after",
    [
        (
            1,
            "--expect-repoint 33 --expect-done 0 --expect-refused 6",
            "--expect-repoint 0 --expect-done 33 --expect-refused 6",
        ),
        (
            2,
            "--expect-repoint 6 --expect-done 32 --expect-refused 1",
            "--expect-repoint 0 --expect-done 38 --expect-refused 1",
        ),
    ],
)
def test_each_pass_runs_with_its_own_pins_and_after_state(tmp_path, pass_, pins, after):
    rc, out, calls = _run(tmp_path, ["--pass", str(pass_)], _rules(pass_))
    assert rc == 0, out
    (dry,) = _tool(calls, "repoint_team_dry")
    (apply_,) = _tool(calls, "repoint_team_apply")
    (check,) = _tool(calls, "repoint_team_after")
    assert pins in dry and "--apply" not in dry
    assert "--apply" in apply_ and pins in apply_
    assert after in check
    assert ("0 Liga MX rows left" in out) == (pass_ == 2)


def test_pass_1_refuses_unless_exactly_the_maps_39_sides_are_there(tmp_path):
    states = [_state(40), _state(40), _state(6)]  # a fixture the map does not name
    rc, out, calls = _run(tmp_path, ["--pass", "1"], _rules(1, states=states))
    assert rc == 1 and "ABORT at step 2: 40 Liga MX side(s)" in out and "exactly 39" in out, out
    assert not _tool(calls, "repoint_team_dry") and not _tool(calls, "repoint_team_apply")


def test_pass_2_refuses_unless_exactly_its_6_sides_are_left(tmp_path):
    states = [_state(7), _state(7), _state(0)]
    rc, out, calls = _run(tmp_path, ["--pass", "2"], _rules(2, states=states))
    assert rc == 1 and "exactly 6" in out and not _tool(calls, "repoint_team_apply")


def test_the_after_state_counts_soccer_rows_created_before_t0(tmp_path):
    rc, out, calls = _run(tmp_path, ["--pass", "1"], _rules(1))
    assert rc == 0, out
    sql = (tmp_path / "fake" / "cp" / "repoint_team_q.sql").read_text()
    assert "m.created_at < TIMESTAMPTZ '" in sql  # a row the 00:00 UTC fixture insert adds after T0 never counts


def test_a_soccer_row_lost_before_t0_fails_the_after_state(tmp_path):
    states = [_state(39), _state(39, n=1000), _state(6, n=999)]
    rc, out, _ = _run(tmp_path, ["--pass", "1"], _rules(1, states=states))
    assert rc == 1 and "FAILED" in out and "1000 -> 999" in out, out


def test_pass_1_already_done_exits_0_without_a_backup(tmp_path):
    pre = "PLAN repoint=6 done=32 refuse=1 not_found=1 stamp_stamp=0 stamp_done=3 stamp_refuse=0"  # after the fd repair
    rc, out, calls = _run(tmp_path, ["--pass", "1"], _rules(1, pre=pre, undone="0", states=[_state(6)]))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0
    assert not _tool(calls, "repoint_team_dry")


def test_pass_2_is_not_done_while_a_side_still_plans_repoint(tmp_path):
    pre = "PLAN repoint=6 done=32 refuse=1 not_found=1 stamp_stamp=0 stamp_done=3 stamp_refuse=0"
    rc, out, calls = _run(
        tmp_path, ["--pass", "2"], _rules(2, pre=pre, undone="0", states=[_state(0), _state(6), _state(0)])
    )
    assert "ALREADY DONE" not in out and fd.airflow_state(tmp_path).get("triggers", 0) == 1


def test_pass_2_already_done(tmp_path):
    pre = "PLAN repoint=0 done=38 refuse=1 not_found=1 stamp_stamp=0 stamp_done=3 stamp_refuse=0"
    rc, out, calls = _run(tmp_path, ["--pass", "2"], _rules(2, pre=pre, undone="0", states=[_state(0)]))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0
