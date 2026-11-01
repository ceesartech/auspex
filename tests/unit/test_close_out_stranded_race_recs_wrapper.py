"""scripts/ops/close_out_stranded_race_recs.sh, run for real against a scripted
docker: the read-only 'already done' exit, the dry-run ceilings (49 candidates,
33 races, 17 settle, 39 void, nothing unmatched or errored, adding up, equal to
the before-state), an apply pinned to the dry run's exact counts (every key),
the exact after-state, and the aborts before any write. Stranded recs on races
at or after the pin (the tool's BEYOND_PIN line) are named, never touched, and
keep a re-run from reporting ALREADY DONE."""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WRAPPER = REPO / "scripts" / "ops" / "close_out_stranded_race_recs.sh"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN = "PLAN candidates=42 races=30 settle=10 void=32 unmatched=0 errors=0"
ZERO = "PLAN candidates=0 races=0 settle=0 void=0 unmatched=0 errors=0"
RID = "00000000-0000-0000-0000-{:012d}"
TRIPLES = ",".join(
    [f"{RID.format(i)}:lost:-7.66" for i in range(9)]
    + [f"{RID.format(9)}:won:75.00"]
    + [f"{RID.format(100 + i)}:void:0.00" for i in range(32)]
)
SETTLED_RACES = ",".join(f"11111111-0000-0000-0000-{i:012d}" for i in range(8))


def _rules(
    pre=PLAN,
    dry=PLAN,
    after=ZERO,
    as_planned="42",
    finished="8",
    states=("42|2605", "0|2605"),
    apply_rc=0,
    dry_rc=0,
    triples=TRIPLES,
    races=SETTLED_RACES,
    beyond_pre="BEYOND_PIN recs=0",
    beyond_after="BEYOND_PIN recs=0",
):
    return [
        {"match": r"/tmp/race_closeout_pre\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/race_closeout_pre\.log", "out": pre},
        {"match": r"grep -o 'BEYOND_PIN recs=\[0-9\]\*' /tmp/race_closeout_pre\.log", "out": beyond_pre},
        {"match": r"grep -o 'BEYOND_PIN recs=\[0-9\]\*' /tmp/race_closeout_verify\.log", "out": beyond_after},
        {"match": r"grep 'beyond pin:' /tmp/race_closeout_pre\.log", "out": "  beyond pin: rec r1  2026-11-01T13:23 X"},
        {"match": r"/tmp/race_closeout_dry\.log 2>&1", "out": f"tool exit {dry_rc}", "rc": dry_rc},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/race_closeout_dry\.log", "out": dry},
        {"match": r"/tmp/race_closeout_apply\.log 2>&1", "out": f"tool exit {apply_rc}", "rc": apply_rc},
        {"match": r"api python -c import csv, sys; rows = \[r for r", "out": triples},
        {"match": r"api python -c import csv, sys; print\(','\.join\(sorted", "out": races},
        {"match": r"/tmp/race_closeout_verify\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/race_closeout_verify\.log", "out": after},
        {"match": r"postgres sh -c psql", "out": states[0], "times": 1},
        {"match": r"postgres sh -c psql", "out": as_planned, "times": 1},
        {"match": r"postgres sh -c psql", "out": finished, "times": 1},
        {"match": r"postgres sh -c psql", "out": states[1], "times": 1},
        {"match": r"postgres rm -f", "out": ""},
    ]


def _run(tmp_path, rules, airflow=None):
    env = fd.install(tmp_path, rules, airflow=airflow)
    rc, out = fd.run(WRAPPER, env)
    return rc, out, fd.calls(tmp_path)


def _tool_runs(calls, flag="--apply"):
    return [c for c in calls if "close_out_stranded_race_recs.py" in c and flag in c]


def _sent_sql(tmp_path):
    sent = sorted(
        (tmp_path / "fake" / "cp").glob("[0-9]*_race_closeout_q.sql"), key=lambda f: int(f.name.split("_")[0])
    )
    return [f.read_text() for f in sent]


def test_a_clean_run_is_pinned_to_the_dry_run_and_checks_the_exact_after_state(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: 10 rec(s) settled" in out and "32 voided" in out, out
    assert not [c for c in fd.calls(tmp_path) if c.startswith("UNMATCHED")]
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=42,races=30,settle=10,void=32,unmatched=0,errors=0" in apply_
    assert "--raced-before 2026-11-01T00:00:00Z" in apply_
    assert re.search(r"--out-csv /app/backups/closed_out_race_recs_apply_\d{8}T\d{6}Z\.csv", apply_)
    # Every tool run is scoped to the reviewed set; only one of them applies.
    runs = [c for c in calls if "close_out_stranded_race_recs.py" in c]
    assert len(runs) == 4 and all("--raced-before 2026-11-01T00:00:00Z" in c for c in runs)
    assert fd.airflow_state(tmp_path)["triggers"] == 1
    state, as_planned, finished, state_after = _sent_sql(tmp_path)
    assert (
        state == state_after and "make_interval(days => 3)" in state and "TIMESTAMPTZ '2026-11-01T00:00:00Z'" in state
    )
    assert f"('{RID.format(9)}'::uuid,'won',75.00::numeric)" in as_planned
    assert f"('{RID.format(100)}'::uuid,'void',0.00::numeric)" in as_planned
    assert "rec.status = v.status AND rec.profit_loss = v.pl AND rec.settled_at IS NOT NULL" in as_planned
    assert finished.count("::uuid") == 8 and "status = 'finished'" in finished


def test_the_southwell_recs_joining_still_apply_pinned_to_their_counts(tmp_path):
    joined = "PLAN candidates=49 races=33 settle=10 void=39 unmatched=0 errors=0"
    triples = TRIPLES + "," + ",".join(f"{RID.format(200 + i)}:void:0.00" for i in range(7))
    rc, out, calls = _run(
        tmp_path, _rules(pre=joined, dry=joined, as_planned="49", states=("49|2605", "0|2605"), triples=triples)
    )
    assert rc == 0, out
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=49,races=33,settle=10,void=39,unmatched=0,errors=0" in apply_


@pytest.mark.parametrize(
    "dry",
    [
        "PLAN candidates=50 races=33 settle=10 void=40 unmatched=0 errors=0",  # more candidates than reviewed
        "PLAN candidates=42 races=34 settle=10 void=32 unmatched=0 errors=0",  # more races
        "PLAN candidates=42 races=30 settle=18 void=24 unmatched=0 errors=0",  # more settles
        "PLAN candidates=42 races=30 settle=2 void=40 unmatched=0 errors=0",  # more voids
        "PLAN candidates=42 races=30 settle=9 void=32 unmatched=1 errors=0",  # unmatched
        "PLAN candidates=42 races=30 settle=10 void=31 unmatched=0 errors=1",  # an API error
        "PLAN candidates=42 races=30 settle=10 void=31 unmatched=0 errors=0",  # does not add up
        "PLAN candidates=41 races=30 settle=10 void=31 unmatched=0 errors=0",  # not the before-state's 42
        "PLAN candidates=42 races=30 settle=10 void=32 unmatched=0",  # unreadable
    ],
)
def test_the_dry_run_gate_aborts_before_any_apply(tmp_path, dry):
    rc, out, calls = _run(tmp_path, _rules(dry=dry))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not _tool_runs(calls)


def test_a_failed_dry_run_aborts_before_any_apply(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(dry_rc=1))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not _tool_runs(calls)


@pytest.mark.parametrize(
    "kw",
    [
        {"as_planned": "41"},  # one rec not as planned
        {"finished": "7"},  # a settled race not 'finished'
        {"states": ("42|2605", "1|2605")},  # the stranded count fell by less than planned
        {"states": ("42|2605", "0|2604")},  # a race rec disappeared
        {"after": PLAN},  # something left to do
        {"triples": "not a csv;drop table"},  # unreadable apply CSV
        {"beyond_after": ""},  # the verify run's BEYOND_PIN line is missing
    ],
)
def test_a_wrong_after_state_fails(tmp_path, kw):
    rc, out, _ = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "FAILED after-state" in out, out


def test_a_failed_apply_aborts_at_step_3(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(apply_rc=1))
    assert rc == 1 and "ABORT at step 3" in out, out
    assert len(_tool_runs(calls)) == 1 and not [c for c in calls if "race_closeout_verify" in c]


def test_nothing_left_exits_0_without_a_backup_or_a_write(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=ZERO))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert "NOTE" not in out and "NEW PIN" not in out
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0 and not _tool_runs(calls)


def test_stranded_recs_beyond_the_pin_are_never_already_done(tmp_path):
    # An abandoned 1 Nov race strands its recs on 4 Nov: the drain sends the operator here, and the reviewed set
    # is empty — that is a new pin to review, not ALREADY DONE.
    rc, out, calls = _run(tmp_path, _rules(pre=ZERO, beyond_pre="BEYOND_PIN recs=3"))
    assert rc == 1 and "NEW PIN NEEDED" in out and "ALREADY DONE" not in out, out
    assert "3 stranded race rec(s) sit on races at or after 2026-11-01T00:00:00Z" in out
    assert "beyond pin: rec r1" in out  # the operator sees which, once
    assert not [c for c in calls if c.startswith("UNMATCHED")]
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0 and not _tool_runs(calls)


def test_the_reviewed_set_still_closes_with_recs_beyond_the_pin_named(tmp_path):
    rules = _rules(beyond_pre="BEYOND_PIN recs=3", beyond_after="BEYOND_PIN recs=3")
    rc, out, calls = _run(tmp_path, rules)
    assert rc == 0 and "DONE: 10 rec(s) settled" in out, out
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=42,races=30,settle=10,void=32,unmatched=0,errors=0" in apply_
    assert out.count("3 stranded race rec(s)") == 2 and "a re-run exits 1 (NEW PIN NEEDED)" in out


@pytest.mark.parametrize("beyond_pre", ["", "BEYOND_PIN recs=x"])
def test_an_unreadable_beyond_pin_line_aborts_before_the_backup(tmp_path, beyond_pre):
    rc, out, calls = _run(tmp_path, _rules(pre=ZERO, beyond_pre=beyond_pre))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0 and not _tool_runs(calls)


@pytest.mark.parametrize(
    "pre",
    [
        "PLAN candidates=42 races=30 settle=10 void=31 unmatched=0 errors=1",
        "PLAN candidates=42 races=30 settle=9 void=32 unmatched=1 errors=0",
        "",
    ],
)
def test_a_pre_plan_with_errors_or_unmatched_aborts_before_the_backup(tmp_path, pre):
    rc, out, calls = _run(tmp_path, _rules(pre=pre))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0 and not _tool_runs(calls)


def test_a_backup_that_does_not_land_aborts_before_any_write(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"backup_fails": True})
    assert rc == 1 and "backup did not land" in out, out
    assert not _tool_runs(calls) and not _tool_runs(calls, "race_closeout_dry")


def test_the_wrapper_and_the_tool_share_the_drain_window():
    lra = (REPO / "scripts" / "load_racing_api.py").read_text()
    wrapper = WRAPPER.read_text()
    (days,) = re.findall(r"^DRAIN_DAYS = (\d+)$", lra, re.M)
    assert f"DRAIN_DAYS={days}  # load_racing_api.DRAIN_DAYS" in wrapper
