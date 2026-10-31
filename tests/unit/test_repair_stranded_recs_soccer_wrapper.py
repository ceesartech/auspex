"""scripts/ops/repair_stranded_recs_soccer.sh, run for real against a scripted
docker: the dry-run ceilings (7 void_post_kickoff, 0 settle, 5 no-twin), an
apply pinned to the dry run's exact counts (every key), the exact after-state,
and the read-only 'already done' exit."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WRAPPER = HERE.parents[1] / "scripts" / "ops" / "repair_stranded_recs_soccer.sh"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN = "PLAN candidates=12 settle=0 void_post_kickoff=7 void_no_twin=0 no_twin=5 ambiguous=0 ungradable=0"
AFTER = "PLAN candidates=5 settle=0 void_post_kickoff=0 void_no_twin=0 no_twin=5 ambiguous=0 ungradable=0"
PAIRS = ",".join(f"{i:08d}-0000-0000-0000-000000000000:{i:08d}-1111-1111-1111-111111111111" for i in range(7))


def _rules(pre=PLAN, dry=PLAN, after=AFTER, voided="7", states=("12|1573", "5|1573")):
    return [
        {"match": r"/tmp/stranded_soccer_pre\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/stranded_soccer_pre\.log", "out": pre},
        {"match": r"/tmp/stranded_soccer_dry\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/stranded_soccer_dry\.log", "out": dry},
        {"match": r"/tmp/stranded_soccer_apply\.log 2>&1", "out": "tool exit 0"},
        {"match": r"api python -c import csv, sys; ids = \[r\['rec_id'\]", "out": PAIRS},
        {"match": r"/tmp/stranded_soccer_verify\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/stranded_soccer_verify\.log", "out": after},
        {"match": r"postgres sh -c psql", "out": states[0], "times": 1},
        {"match": r"postgres sh -c psql", "out": voided, "times": 1},
        {"match": r"postgres sh -c psql", "out": states[1], "times": 1},
        {"match": r"postgres rm -f", "out": ""},
    ]


def _run(tmp_path, rules):
    env = fd.install(tmp_path, rules)
    rc, out = fd.run(WRAPPER, env)
    return rc, out, fd.calls(tmp_path)


def _apply(calls):
    return [c for c in calls if "repair_stranded_recs.py" in c and "--apply" in c]


def test_a_clean_run_is_pinned_to_the_dry_run_and_checks_the_exact_after_state(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: 7 rec(s) voided" in out, out
    (apply_,) = _apply(calls)
    pin = "candidates=12,settle=0,void_post_kickoff=7,void_no_twin=0,no_twin=5,ambiguous=0,ungradable=0"
    assert f"--expect-plan {pin}" in apply_ and "--sport soccer" in apply_
    sent = sorted(
        (tmp_path / "fake" / "cp").glob("[0-9]*_stranded_soccer_q.sql"), key=lambda f: int(f.name.split("_")[0])
    )
    voided_sql = sent[1].read_text()  # before-state, voided check, after-state
    assert (
        "br.status = 'void' AND br.profit_loss = 0 AND br.actual_result = 'void: written after kickoff'" in voided_sql
    )
    assert "br.match_id = v.twin" in voided_sql


@pytest.mark.parametrize(
    "dry",
    [
        PLAN.replace("void_post_kickoff=7", "void_post_kickoff=8").replace("candidates=12", "candidates=13"),
        PLAN.replace("settle=0", "settle=1").replace("candidates=12", "candidates=13"),
        PLAN.replace("no_twin=5", "no_twin=6").replace("candidates=12", "candidates=13"),
        PLAN.replace("ambiguous=0", "ambiguous=1").replace("candidates=12", "candidates=13"),
        PLAN.replace("candidates=12", "candidates=14"),  # does not add up
        PLAN.replace(" ungradable=0", ""),  # unreadable
    ],
)
def test_the_dry_run_gate_aborts_before_any_apply(tmp_path, dry):
    rc, out, calls = _run(tmp_path, _rules(dry=dry))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not _apply(calls)


def test_fewer_post_kickoff_voids_than_reviewed_still_apply_pinned(tmp_path):
    dry = PLAN.replace("void_post_kickoff=7", "void_post_kickoff=6").replace("candidates=12", "candidates=11")
    rc, out, calls = _run(tmp_path, _rules(dry=dry, voided="6", states=("11|1573", "5|1573")))
    assert rc == 0, out
    (apply_,) = _apply(calls)
    assert "void_post_kickoff=6" in apply_ and "candidates=11" in apply_


@pytest.mark.parametrize(
    "rules",
    [
        _rules(voided="6"),  # one rec not voided onto its twin
        _rules(states=("12|1573", "6|1573")),  # the stranded count fell by less than planned
        _rules(states=("12|1573", "5|1572")),  # a rec disappeared
        _rules(after=AFTER.replace("void_post_kickoff=0", "void_post_kickoff=1")),  # something left to do
    ],
)
def test_a_wrong_after_state_fails(tmp_path, rules):
    rc, out, _ = _run(tmp_path, rules)
    assert rc == 1 and "FAILED after-state" in out, out


@pytest.mark.parametrize(
    "pre", [AFTER, AFTER.replace("candidates=5", "candidates=0").replace("no_twin=5", "no_twin=0")]
)
def test_nothing_left_to_write_exits_0_without_a_backup(tmp_path, pre):
    rc, out, calls = _run(tmp_path, _rules(pre=pre))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0 and not _apply(calls)
