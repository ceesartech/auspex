"""scripts/ops/merge_soccer_teams.sh, run for real against a scripted docker
whose Airflow behaves like 3.3: a paused DAG's runs never advance. The wrapper
must DRAIN before it pauses (a run in flight when the DAG is paused would stay
'running' for good), retry when a run slips in between, fail on unmatched
odds-feed spellings, record the plan and MERGED_AT for the runbook, and exit
0 at once when every pair is already merged."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WRAPPER = HERE.parents[1] / "scripts" / "ops" / "merge_soccer_teams.sh"
MAP = HERE.parents[1] / "scripts" / "data" / "soccer_team_merge_map.csv"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

NOT_DONE = "PLAN pairs_merge=171 pairs_done=0 pairs_refuse=0 kickoffs_adopted=0"
ALL_DONE = "PLAN pairs_merge=0 pairs_done=171 pairs_refuse=0 kickoffs_adopted=0"
WRITTEN = "WRITTEN pairs_merged=171 pairs_failed=0 odds_spelling_checks=47 odds_spellings_unmatched_before=0"


def _rules(pre=NOT_DONE, written=WRITTEN, after_rc=0):
    return [
        {"match": r"merge_teams_pre\.log 2>&1; echo", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN pairs_merge.*merge_teams_pre\.log", "out": pre},
        {"match": r"--check-after \S+precheck\S+ > /tmp/merge_teams_pre_after\.log", "out": "AFTER ok", "rc": after_rc},
        {"match": r"merge_soccer_teams_last_apply\.env 2>/dev/null", "out": "PLAN=/app/backups/x.plan.json"},
        {"match": r"merge_teams_dry\.log 2>&1; rc", "out": NOT_DONE},
        {"match": r"grep -o 'PLAN pairs_merge.*merge_teams_dry\.log", "out": NOT_DONE},
        {"match": r"GRADE_AFTER", "out": "11111111-1111-1111-1111-111111111111"},
        {"match": r"--apply --expect-plan", "out": "Archived 10 row(s)"},
        {"match": r"grep -o 'WRITTEN \.\*' /tmp/merge_teams\.log", "out": written},
        {"match": r"printf 'PLAN=%s", "out": ""},
        {"match": r"grade_completed_matches\.py --match-id", "out": "1"},
        {"match": r"--check-after \S+dryrun\S+ > /tmp/merge_teams_after\.log", "out": "AFTER ok"},
    ]


def _airflow(**dags):
    base = {"auspex_pipeline": {"paused": False, "runs": []}, "fetch_live_odds": {"paused": False, "runs": []}}
    base.update(dags)
    return {"dags": base}


def _run(tmp_path, rules=None, airflow=None):
    env = fd.install(tmp_path, rules or _rules(), airflow or _airflow())
    (tmp_path / "auspex" / "scripts" / "data").mkdir(parents=True)
    shutil.copy(MAP, tmp_path / "auspex" / "scripts" / "data" / MAP.name)
    rc, out = fd.run(WRAPPER, env)
    return rc, out, fd.calls(tmp_path), fd.airflow_state(tmp_path)


def _index(calls, needle):
    return next(i for i, c in enumerate(calls) if needle in c)


def test_a_run_in_flight_is_drained_before_the_pause(tmp_path):
    """auspex_pipeline is mid-run: pausing first would freeze it for good
    (Airflow 3.3), so the wrapper must wait for it while the DAG still runs."""
    inflight = {"paused": False, "runs": [{"run_id": "r1", "state": "running", "ticks": 3}]}
    rc, out, calls, af = _run(tmp_path, airflow=_airflow(auspex_pipeline=inflight))
    assert rc == 0, out
    assert "DONE: 171/171 pairs merged" in out
    assert af["dags"]["auspex_pipeline"]["runs"][0]["state"] == "success"
    pause = _index(calls, "dags pause -y auspex_pipeline")
    last_running_poll = max(
        i for i, c in enumerate(calls) if "list-runs auspex_pipeline --state running" in c and i < pause
    )
    assert last_running_poll < pause < _index(calls, "--apply --expect-plan")
    assert not af["dags"]["auspex_pipeline"]["paused"] and not af["dags"]["fetch_live_odds"]["paused"]


def test_a_run_that_slips_in_between_the_drain_and_the_pause_is_retried(
    tmp_path,
):
    airflow = _airflow()
    airflow["slip_in"] = {"auspex_pipeline": 1}  # the first pause meets a freshly queued run
    rc, out, calls, af = _run(tmp_path, airflow=airflow)
    assert rc == 0, out
    assert "slipped in between the drain and the pause" in out
    assert sum("dags pause -y auspex_pipeline" in c for c in calls) == 2
    assert _index(calls, "dags unpause -y auspex_pipeline") < _index(calls, "merge_teams_dry.log 2>&1; rc")
    assert not af["dags"]["auspex_pipeline"]["paused"]


def test_a_dag_paused_before_the_run_with_a_frozen_run_aborts_and_stays_paused(tmp_path):
    frozen = {"paused": True, "runs": [{"run_id": "r1", "state": "running", "ticks": 1}]}
    rc, out, calls, af = _run(tmp_path, airflow=_airflow(fetch_live_odds=frozen))
    assert rc == 1 and "ABORT: could not drain and pause" in out, out
    assert "already paused before this run" in out and "fetch_live_odds" in out
    assert not [c for c in calls if "--apply" in c]
    assert af["dags"]["fetch_live_odds"]["paused"]  # not ours to unpause
    assert not [c for c in calls if "unpause -y fetch_live_odds" in c]


def test_unmatched_odds_spellings_fail_the_run_after_it_unpauses(tmp_path):
    written = WRITTEN.replace("odds_spellings_unmatched_before=0", "odds_spellings_unmatched_before=2")
    rc, out, calls, af = _run(tmp_path, rules=_rules(written=written))
    assert rc == 1, out
    assert "FAILED: the apply reports odds_spellings_unmatched_before=2" in out and "DONE: 171" not in out
    assert not af["dags"]["auspex_pipeline"]["paused"] and not af["dags"]["fetch_live_odds"]["paused"]
    assert [c for c in calls if "printf 'PLAN=%s" in c]  # the odds check still gets its plan and MERGED_AT


def test_the_plan_and_merged_at_are_recorded_for_the_runbook(tmp_path):
    rc, out, calls, _ = _run(tmp_path)
    assert rc == 0, out
    (state,) = [c for c in calls if "printf 'PLAN=%s" in c]
    assert "merge_soccer_teams_last_apply.env" in state and "merge_soccer_teams_dryrun_" in state
    assert "after the next fetch_live_odds run has FINISHED" in out
    assert "airflow dags list-runs fetch_live_odds --state success -o json" in out


def test_all_pairs_done_exits_0_without_a_backup_or_a_pause(tmp_path):
    rc, out, calls, af = _run(tmp_path, rules=_rules(pre=ALL_DONE))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert af.get("triggers", 0) == 0
    assert not [c for c in calls if "dags pause" in c or "--apply" in c]


def test_all_pairs_done_but_a_broken_after_state_fails(tmp_path):
    rc, out, calls, af = _run(tmp_path, rules=_rules(pre=ALL_DONE, after_rc=1))
    assert rc == 1 and "after-state check failed" in out, out
    assert af.get("triggers", 0) == 0
