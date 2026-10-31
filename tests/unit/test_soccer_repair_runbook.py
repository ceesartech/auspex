"""scripts/ops/soccer_repair_runbook.sh, run for real in a throwaway git
checkout whose wrappers are stand-ins (each logs its call; FAIL_<NAME>,
ALREADY_<NAME> steer it) and against a scripted docker whose Airflow behaves
like 3.3. The order, --from / --only, the stop at the first failing step,
retrain_models paused and restored, the start refusals (VM HEAD, the two
prior repairs, a retrain in flight), the re-grade gate, step 10's skip and
step 11's wait. The real wrappers ran end to end through a docker shim
against a replica of prod (scripts/ops/soccer_repair_runbook.sh header)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
OPS = HERE.parents[1] / "scripts" / "ops"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None, reason="needs bash and git"
)

WRAPPERS = [
    "repoint_team_matches.sh",
    "migrate_027_outcome_trigger.sh",
    "merge_soccer_teams.sh",
    "repair_football_data_kickoffs.sh",
    "repair_stranded_recs_soccer.sh",
    "cleanup_soccer_orphans.sh",
    "backfill_soccer_regulation_scores.sh",
    "backfill_season_labels.sh",
]
STUB = r"""#!/bin/bash
name=$(basename "$0" .sh)
echo "$name $*" >> "$FAKE_DIR/steps.log"
var=$(echo "$name" | tr 'a-z' 'A-Z')
if [ "$name" = backfill_soccer_regulation_scores ]; then
  echo "REGRADE ids_file=/app/backups/ids.txt types=match_result,btts changes=${REGRADE_CHANGES:-135} min=132 max=138"
fi
fail="FAIL_$var"; already="ALREADY_$var"
[ -n "${!already:-}" ] && echo "[stub] ALREADY DONE: nothing to do"
exit "${!fail:-0}"
"""
EXPECTED_ORDER = [
    "repoint_team_matches --pass 1",
    "migrate_027_outcome_trigger",
    "merge_soccer_teams",
    "repair_football_data_kickoffs",
    "repoint_team_matches --pass 2",
    "repair_stranded_recs_soccer",
    "cleanup_soccer_orphans",
    "backfill_soccer_regulation_scores",
    "backfill_season_labels",
]


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path) -> Path:
    repo = tmp_path / "auspex"
    (repo / "scripts" / "ops").mkdir(parents=True, exist_ok=True)
    (repo / "services").mkdir(exist_ok=True)
    (repo / "services" / "README").write_text("services\n")
    shutil.copy(OPS / "soccer_repair_runbook.sh", repo / "scripts" / "ops")
    shutil.copy(OPS / "_airflow_lib.sh", repo / "scripts" / "ops")
    for w in WRAPPERS:
        (repo / "scripts" / "ops" / w).write_text(STUB)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "runbook")
    return repo


def _rules(pre="22|4", verify_left="0", check_rc=0):
    return [
        {"match": r"postgres sh -c psql", "out": f"BEGIN\n{pre}\nROLLBACK"},
        {"match": r"postgres rm -f", "out": ""},
        {"match": r"regrade_predictions\.py .*--apply --expect-changes", "out": "Rewrote"},
        {
            "match": r"grep -o '\[0-9\]\* change\(s\) planned' /tmp/runbook_regrade_verify\.log \| tail -1 \| cut",
            "out": verify_left,
        },
        {"match": r"regrade_regulation_verify", "out": f"{verify_left} change(s) planned"},
        {"match": r"compute_features\.py --backfill-finished soccer", "out": "features done"},
        {
            "match": r"merge_soccer_teams_last_apply\.env",
            "out": "PLAN=/app/backups/p.plan.json\nMERGED_AT=2026-10-31T10:00:00Z",
        },
        {
            "match": r"--check-after /app/backups/p\.plan\.json --odds-since 2026-10-31T10:00:00Z --after-later-steps",
            "out": "AFTER ok",
            "rc": check_rc,
        },
    ]


def _airflow(retrain=None, fetch_runs=None, fetch_paused=False):
    return {
        "dags": {
            "retrain_models": retrain or {"paused": False, "runs": []},
            "fetch_live_odds": {
                "paused": fetch_paused,
                "runs": (
                    fetch_runs
                    if fetch_runs is not None
                    else [
                        {"run_id": "r", "state": "success", "ticks": 0, "start_date": "2026-10-31T10:30:00.123+00:00"}
                    ]
                ),
            },
        }
    }


def _run(tmp_path, *args, rules=None, airflow=None, repo=None, **env):
    repo = repo or _repo(tmp_path)
    base = fd.install(tmp_path, rules or _rules(), airflow or _airflow())
    base.update({"AUSPEX_DIR": str(repo), "RUNBOOK_STATE": str(tmp_path / "state"), "ODDS_WAIT_POLLS": "3", **env})
    rc, out = fd.run(repo / "scripts" / "ops" / "soccer_repair_runbook.sh", base, *args)
    steps_log = tmp_path / "fake" / "steps.log"
    steps = [s.strip() for s in steps_log.read_text().splitlines()] if steps_log.exists() else []
    return rc, out, steps, fd.calls(tmp_path), fd.airflow_state(tmp_path)


def test_a_full_run_goes_through_every_step_in_order_and_restores_retrain(tmp_path):
    rc, out, steps, calls, af = _run(tmp_path)
    assert rc == 0 and "ALL STEPS DONE" in out, out
    assert steps == EXPECTED_ORDER
    assert [c for c in calls if "compute_features.py --backfill-finished soccer" in c]
    assert [c for c in calls if "--after-later-steps" in c]
    pause = next(i for i, c in enumerate(calls) if "dags pause -y retrain_models" in c)
    unpause = next(i for i, c in enumerate(calls) if "dags unpause -y retrain_models" in c)
    assert pause < unpause and not af["dags"]["retrain_models"]["paused"]
    for n in range(1, 12):
        assert f"STEP {n}/11 START" in out and f"STEP {n}/11 DONE" in out
    assert "re-grade: applying 135 change(s), pinned (--expect-changes 135)" in out


def test_from_and_only(tmp_path):
    rc, out, steps, calls, _ = _run(tmp_path, "--from", "6")
    assert rc == 0, out
    assert steps == EXPECTED_ORDER[5:]
    rc, out, steps, calls, _ = _run(tmp_path / "o", "--only", "9", repo=_repo(tmp_path / "o"))
    assert rc == 0 and steps == ["backfill_season_labels"], out
    assert not [c for c in calls if "compute_features" in c]


@pytest.mark.parametrize("args", [["--from"], ["--from", "0"], ["--from", "12"], ["--only", "x"], ["--bogus"]])
def test_bad_arguments_exit_2(tmp_path, args):
    rc, out, steps, _, _ = _run(tmp_path, *args)
    assert rc == 2 and "usage" in out and steps == []


def test_the_first_failing_step_stops_the_run_with_the_from_hint(tmp_path):
    rc, out, steps, _, af = _run(tmp_path, FAIL_REPAIR_FOOTBALL_DATA_KICKOFFS="1")
    assert rc == 1, out
    assert "STEP 4 FAILED (exit 1): repair_football_data_kickoffs — fix, then re-run with --from 4" in out
    assert steps == EXPECTED_ORDER[:4]
    assert not af["dags"]["retrain_models"]["paused"]


def test_a_cleanup_whose_follow_ups_failed_says_to_continue_from_8(tmp_path):
    rc, out, steps, _, _ = _run(tmp_path, FAIL_CLEANUP_SOCCER_ORPHANS="2")
    assert rc == 2 and "STEP 7 INCOMPLETE" in out and "--from 8" in out, out
    assert steps[-1] == "cleanup_soccer_orphans"


def test_a_runbook_that_is_not_the_one_committed_at_head_is_refused(tmp_path):
    repo = _repo(tmp_path)
    runbook = repo / "scripts" / "ops" / "soccer_repair_runbook.sh"
    runbook.write_text(runbook.read_text() + "\n# local edit\n")
    rc, out, steps, calls, _ = _run(tmp_path, repo=repo)
    assert rc == 1 and "REFUSED: this runbook" in out and "not the one committed at the VM HEAD" in out
    assert steps == [] and not [c for c in calls if "retrain_models" in c]


def test_local_edits_to_the_bind_mounted_code_are_refused(tmp_path):
    repo = _repo(tmp_path)
    (repo / "services" / "README").write_text("edited on the VM\n")
    rc, out, steps, _, _ = _run(tmp_path, repo=repo)
    assert rc == 1 and "REFUSED: scripts/ or services/ differ from HEAD" in out and steps == []


@pytest.mark.parametrize(
    "pre, why", [("21|4", "void_post_kickoff_recs.sh has not run"), ("22|3", "fix_unplayed_soccer_rows.sh has not run")]
)
def test_the_two_prior_repairs_must_have_run(tmp_path, pre, why):
    rc, out, steps, calls, _ = _run(tmp_path, rules=_rules(pre=pre))
    assert rc == 1 and why in out and steps == []
    assert not [c for c in calls if "dags pause" in c]


def test_a_retrain_in_flight_refuses_to_start(tmp_path):
    retrain = {"paused": False, "runs": [{"run_id": "weekly", "state": "running", "ticks": 50}]}
    rc, out, steps, _, af = _run(tmp_path, airflow=_airflow(retrain=retrain))
    assert rc == 1 and "REFUSED: retrain_models has a run in flight" in out and steps == []
    assert not af["dags"]["retrain_models"]["paused"]


def test_a_retrain_paused_before_the_run_stays_paused(tmp_path):
    rc, out, steps, calls, af = _run(tmp_path, airflow=_airflow(retrain={"paused": True, "runs": []}))
    assert rc == 0, out
    assert af["dags"]["retrain_models"]["paused"]
    assert not [c for c in calls if "unpause -y retrain_models" in c]


@pytest.mark.parametrize("changes, rc_want, applied", [("0", 0, False), ("140", 1, False), ("131", 1, False)])
def test_the_re_grade_is_applied_only_inside_its_reviewed_range(tmp_path, changes, rc_want, applied):
    rc, out, steps, calls, _ = _run(tmp_path, "--only", "8", REGRADE_CHANGES=changes)
    assert rc == rc_want, out
    assert bool([c for c in calls if "--apply --expect-changes" in c]) == applied
    if changes == "0":
        assert "re-grade: ALREADY DONE" in out


def test_a_re_grade_that_leaves_changes_behind_fails(tmp_path):
    rc, out, _, _, _ = _run(tmp_path, "--only", "8", rules=_rules(verify_left="3"))
    assert rc == 1 and "re-grade verify: '3' change(s) still planned" in out


def test_step_10_is_skipped_only_after_a_recompute_newer_than_every_data_change(tmp_path):
    state = tmp_path / "state"
    state.write_text("step=3 result=changed at=100 head=x\nstep=10 result=changed at=200 head=x\n")
    rc, out, _, calls, _ = _run(tmp_path, "--only", "10")
    assert rc == 0 and "ALREADY DONE: features were recomputed" in out, out
    assert not [c for c in calls if "compute_features" in c]
    state.write_text("step=10 result=changed at=200 head=x\nstep=7 result=changed at=300 head=x\n")
    rc, out, _, calls, _ = _run(tmp_path / "b", "--only", "10", repo=_repo(tmp_path / "b"), RUNBOOK_STATE=str(state))
    assert rc == 0 and [c for c in calls if "compute_features" in c], out


def test_step_11_waits_for_an_odds_run_started_after_the_merge(tmp_path):
    early = [{"run_id": "r", "state": "success", "ticks": 0, "start_date": "2026-10-31T09:30:00.000000+00:00"}]
    rc, out, _, calls, _ = _run(tmp_path, "--only", "11", airflow=_airflow(fetch_runs=early))
    assert rc == 1 and "no fetch_live_odds run has finished since 2026-10-31T10:00:00Z" in out, out
    assert "--from 11" in out and not [c for c in calls if "--check-after" in c]
    rc, out, _, _, _ = _run(
        tmp_path / "p", "--only", "11", repo=_repo(tmp_path / "p"), airflow=_airflow(fetch_paused=True)
    )
    assert rc == 1 and "fetch_live_odds is paused" in out


def test_step_11_fails_when_the_check_fails(tmp_path):
    rc, out, _, _, _ = _run(tmp_path, "--only", "11", rules=_rules(check_rc=1))
    assert rc == 1 and "STEP 11 FAILED" in out


def test_a_re_run_records_already_done_steps(tmp_path):
    env = {f"ALREADY_{w[:-3].upper()}": "1" for w in WRAPPERS}
    state = tmp_path / "state"
    state.write_text("step=10 result=changed at=1 head=x\n")
    rc, out, steps, calls, _ = _run(tmp_path, REGRADE_CHANGES="0", **env)
    assert rc == 0, out
    recorded = state.read_text()
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        assert f"step={n} result=already_done" in recorded, recorded
    assert "step=11 result=checked" in recorded
    assert not [c for c in calls if "compute_features" in c or "--apply" in c]
    assert os.path.exists(state)
