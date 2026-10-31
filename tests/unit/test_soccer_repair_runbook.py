"""scripts/ops/soccer_repair_runbook.sh, run for real in a throwaway git
checkout whose wrappers are stand-ins (each logs its call; FAIL_<NAME>,
ALREADY_<NAME>, HANG_<NAME> steer it) and against a scripted docker whose
Airflow behaves like 3.3. The order, --from / --only, the stop at the first
failing step, retrain_models and retrain_halftime_dc paused and restored only
once the corpus is consistent (kept paused, loudly, after a failed or
interrupted corpus step; adopted and restored by the --from N run that
finishes), Ctrl-C through a dead tee, the start refusals (VM HEAD, the two
prior repairs, a retrain in flight), the re-grade gate, step 10's skip and
step 11's wait. The real wrappers ran end to end through a docker shim
against a replica of prod (scripts/ops/soccer_repair_runbook.sh header).
A retrain DAG whose pause read-back fails or is interrupted still ends
unpaused.
A corpus step that failed or was cut short is recorded 'failed', so a later
ALREADY DONE re-run of it does not clear it.
An apply still running in the api container refuses the start."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
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
fail="FAIL_$var"; already="ALREADY_$var"; hang="HANG_$var"
[ -n "${!hang:-}" ] && /bin/sleep 60
[ -n "${!already:-}" ] && echo "[stub] ALREADY DONE: nothing to do"
exit "${!fail:-0}"
"""
RETRAIN = ("retrain_models", "retrain_halftime_dc")


def RUNBOOK_IN(tmp_path):  # noqa: N802 — the runbook inside a test's checkout
    return tmp_path / "auspex" / "scripts" / "ops" / "soccer_repair_runbook.sh"


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


def _airflow(retrain=None, fetch_runs=None, fetch_paused=False, halftime=None):
    return {
        "dags": {
            "retrain_models": retrain or {"paused": False, "runs": []},
            "retrain_halftime_dc": halftime or {"paused": False, "runs": []},
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


def _env(tmp_path, install=False):
    """The env of the runbook in tmp_path (install=True: a fresh fake docker first); a second run with it keeps the
    Airflow model, the state file and the markers."""
    base = (
        fd.install(tmp_path, _rules(), _airflow())
        if install
        else {
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
            "FAKE_DIR": str(tmp_path / "fake"),
            "TMPDIR": str(tmp_path / "tmp"),
            "AF_POLL_SECONDS": "0",
            "AF_STATE_DIR": str(tmp_path / "afstate"),
        }
    )
    return {
        **base,
        "AUSPEX_DIR": str(tmp_path / "auspex"),
        "RUNBOOK_STATE": str(tmp_path / "state"),
        "ODDS_WAIT_POLLS": "3",
    }


def _run(tmp_path, *args, rules=None, airflow=None, repo=None, **env):
    repo = repo or _repo(tmp_path)
    base = fd.install(tmp_path, rules or _rules(), airflow or _airflow())
    base.update({"AUSPEX_DIR": str(repo), "RUNBOOK_STATE": str(tmp_path / "state"), "ODDS_WAIT_POLLS": "3", **env})
    rc, out = fd.run(repo / "scripts" / "ops" / "soccer_repair_runbook.sh", base, *args)
    steps_log = tmp_path / "fake" / "steps.log"
    steps = [s.strip() for s in steps_log.read_text().splitlines()] if steps_log.exists() else []
    return rc, out, steps, fd.calls(tmp_path), fd.airflow_state(tmp_path)


def _paused(af) -> dict:
    return {d: af["dags"][d]["paused"] for d in RETRAIN}


def _markers(tmp_path) -> list[str]:
    return sorted(m.name for m in (tmp_path / "afstate").glob("paused_*"))


def test_a_full_run_goes_through_every_step_in_order_and_restores_retrain(tmp_path):
    rc, out, steps, calls, af = _run(tmp_path)
    assert rc == 0 and "ALL STEPS DONE" in out and "not complete" not in out, out
    assert steps == EXPECTED_ORDER
    assert [c for c in calls if "compute_features.py --backfill-finished soccer" in c]
    assert [c for c in calls if "--after-later-steps" in c]
    for dag in RETRAIN:
        pause = next(i for i, c in enumerate(calls) if f"dags pause -y {dag}" in c)
        unpause = next(i for i, c in enumerate(calls) if f"dags unpause -y {dag}" in c)
        first_step = next(i for i, c in enumerate(calls) if "compute_features" in c)
        assert pause < first_step < unpause
    assert _paused(af) == {"retrain_models": False, "retrain_halftime_dc": False} and _markers(tmp_path) == []
    for n in range(1, 12):
        assert f"STEP {n}/11 START" in out and f"STEP {n}/11 DONE" in out
    assert "re-grade: applying 135 change(s), pinned (--expect-changes 135)" in out


def test_from_and_only(tmp_path):
    rc, out, steps, calls, _ = _run(tmp_path, "--from", "6")
    assert rc == 0, out
    assert steps == EXPECTED_ORDER[5:]
    assert "ALL STEPS DONE (from step 6)" in out
    rc, out, steps, calls, _ = _run(tmp_path / "o", "--only", "9", repo=_repo(tmp_path / "o"))
    assert rc == 0 and steps == ["backfill_season_labels"], out
    assert "ALL STEPS DONE (step 9 only)" in out
    assert not [c for c in calls if "compute_features" in c]


@pytest.mark.parametrize("args", [["--from"], ["--from", "0"], ["--from", "12"], ["--only", "x"], ["--bogus"]])
def test_bad_arguments_exit_2(tmp_path, args):
    rc, out, steps, _, _ = _run(tmp_path, *args)
    assert rc == 2 and "usage" in out and steps == []


def test_a_failing_corpus_step_stops_keeps_the_retrains_paused_and_from_n_restores_them(tmp_path):
    """Step 4 fails after steps 1-3 changed the corpus: unpausing the retrain
    DAGs now would let a missed run fit a half-repaired corpus, so they stay
    paused with an ALERT (in the ledger too) and their markers. The --from 4
    run adopts them and unpauses them once step 10 has run."""
    rc, out, steps, _, af = _run(tmp_path, FAIL_REPAIR_FOOTBALL_DATA_KICKOFFS="1")
    assert rc == 1, out
    assert "STEP 4 FAILED (exit 1): repair_football_data_kickoffs — fix, then re-run with --from 4" in out
    assert steps == EXPECTED_ORDER[:4]
    assert _paused(af) == {"retrain_models": True, "retrain_halftime_dc": True}
    assert "ALERT: retrain_models stays PAUSED" in out and "ALERT: retrain_halftime_dc stays PAUSED" in out
    assert "airflow dags unpause -y retrain_models" in out
    assert "step=4 result=failed" in (tmp_path / "state").read_text()
    assert "stays PAUSED" in (tmp_path / "afstate" / "airflow_pauses.log").read_text()
    assert _markers(tmp_path) == ["paused_retrain_halftime_dc", "paused_retrain_models"]

    rc, out = fd.run(RUNBOOK_IN(tmp_path), _env(tmp_path), "--from", "4")
    af = fd.airflow_state(tmp_path)
    steps = [x.strip() for x in (tmp_path / "fake" / "steps.log").read_text().splitlines()]
    assert rc == 0 and "ALL STEPS DONE" in out and "not complete" not in out, out
    assert "adopted" in out and steps == EXPECTED_ORDER[:4] + EXPECTED_ORDER[3:]
    assert _paused(af) == {"retrain_models": False, "retrain_halftime_dc": False} and _markers(tmp_path) == []


def test_a_corpus_step_that_fails_after_a_complete_repair_keeps_the_retrains_paused(tmp_path):
    """The state file's last corpus record is step 10's, but this run's step 3
    failed: it may have committed part of its work, so the corpus is suspect."""
    (tmp_path / "state").write_text("step=3 result=changed at=1 head=x\nstep=10 result=changed at=2 head=x\n")
    rc, out, _, _, af = _run(tmp_path, "--only", "3", FAIL_MERGE_SOCCER_TEAMS="1")
    assert rc == 1 and "ALERT: retrain_models stays PAUSED" in out and "in step 3" in out, out
    assert _paused(af) == {"retrain_models": True, "retrain_halftime_dc": True}


def test_a_state_file_that_cannot_be_written_keeps_the_retrains_paused(tmp_path):
    (tmp_path / "state").mkdir()  # a directory: every record fails
    rc, out, steps, _, af = _run(tmp_path)
    assert rc == 0 and steps == EXPECTED_ORDER and "WARNING: could not write" in out, out
    assert "the repair is not complete" in out
    assert _paused(af) == {"retrain_models": True, "retrain_halftime_dc": True}


def test_a_failure_outside_the_corpus_steps_restores_the_retrains_when_step_10_is_recorded(tmp_path):
    rc, out, steps, _, af = _run(tmp_path, "--only", "11", rules=_rules(check_rc=1))
    assert rc == 1 and "STEP 11 FAILED" in out, out
    assert _paused(af) == {"retrain_models": False, "retrain_halftime_dc": False}


def test_a_single_corpus_step_leaves_the_retrains_paused_until_step_10(tmp_path):
    rc, out, steps, _, af = _run(tmp_path, "--only", "3")
    assert rc == 0 and "the repair is not complete" in out and "ALERT: retrain_models stays PAUSED" in out, out
    assert _paused(af) == {"retrain_models": True, "retrain_halftime_dc": True}
    rc, out = fd.run(RUNBOOK_IN(tmp_path), _env(tmp_path), "--only", "10")
    assert rc == 0 and "not complete" not in out, out
    assert _paused(fd.airflow_state(tmp_path)) == {"retrain_models": False, "retrain_halftime_dc": False}


def test_a_cleanup_whose_follow_ups_failed_says_to_continue_from_7(tmp_path):
    rc, out, steps, _, _ = _run(tmp_path, FAIL_CLEANUP_SOCCER_ORPHANS="2")
    assert rc == 2 and "STEP 7 INCOMPLETE" in out and "re-run with --from 7" in out, out
    assert "only its pending follow-ups" in out
    assert steps[-1] == "cleanup_soccer_orphans"


def _wait_for(pred, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize(
    "hang, step, args, kept",
    [
        ("HANG_MERGE_SOCCER_TEAMS", "merge_soccer_teams", [], True),
        ("HANG_MIGRATE_027_OUTCOME_TRIGGER", "migrate", ["--only", "2"], False),
    ],
    ids=["corpus-step-cut-short-kept-paused", "trigger-step-alone-restored"],
)
def test_ctrl_c_through_a_dead_tee_decides_the_retrains_and_says_so_in_the_ledger(tmp_path, hang, step, args, kept):
    """Ctrl-C while a step runs and the runbook's output goes through tee: tee
    dies with the process group, and the exit path must still reach its
    decision — kept paused (a corpus step was cut short) or unpaused — and
    record it in the ledger, which survives the dead terminal."""
    repo = _repo(tmp_path)
    env = {**_env(tmp_path, install=True), "AUSPEX_DIR": str(repo), hang: "1"}
    proc = subprocess.Popen(
        ["bash", "-c", f'bash "{RUNBOOK_IN(tmp_path)}" {" ".join(args)} 2>&1 | tee "{tmp_path / "o"}"'],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    steps_log = tmp_path / "fake" / "steps.log"
    try:
        assert _wait_for(lambda: steps_log.exists() and step in steps_log.read_text())
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
        ledger = tmp_path / "afstate" / "airflow_pauses.log"
        word = "stays PAUSED" if kept else "unpaused retrain_halftime_dc"
        assert _wait_for(lambda: ledger.exists() and word in ledger.read_text()), ledger.read_text()
        assert _wait_for(lambda: _paused(fd.airflow_state(tmp_path)) == {d: kept for d in RETRAIN})
        assert _markers(tmp_path) == (["paused_retrain_halftime_dc", "paused_retrain_models"] if kept else [])
        state = (tmp_path / "state").read_text() if (tmp_path / "state").exists() else ""
        assert ("step=3 result=failed" in state) == kept, state
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


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


@pytest.mark.parametrize("which", ["retrain", "halftime"])
def test_a_retrain_in_flight_refuses_to_start(tmp_path, which):
    busy = {"paused": False, "runs": [{"run_id": "r", "state": "running", "ticks": 50}]}
    rc, out, steps, _, af = _run(tmp_path, airflow=_airflow(**{which: busy}))
    assert rc == 1 and "REFUSED: retrain_models retrain_halftime_dc: a run in flight" in out and steps == []
    assert _paused(af) == {"retrain_models": False, "retrain_halftime_dc": False}


def test_a_retrain_paused_before_the_run_stays_paused(tmp_path):
    rc, out, steps, calls, af = _run(tmp_path, airflow=_airflow(retrain={"paused": True, "runs": []}))
    assert rc == 0, out
    assert af["dags"]["retrain_models"]["paused"] and not af["dags"]["retrain_halftime_dc"]["paused"]
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


# --- a retrain DAG whose pause read-back fails or is interrupted -----------------------------------------------------


def test_a_failed_read_right_after_pausing_a_retrain_dag_ends_with_it_unpaused(tmp_path):
    """Before, retrain_halftime_dc stayed paused for good: unlisted, no marker, no ALERT, and every later runbook
    'already paused — left as it is' (the monthly half-time retrain silently stopped)."""
    airflow = _airflow()
    airflow["details_fail_after_pause"] = ["retrain_halftime_dc"]
    rc, out, steps, _, af = _run(tmp_path, airflow=airflow)
    assert rc == 1 and "could not confirm that retrain_halftime_dc is paused" in out and steps == [], out
    assert _paused(af) == {"retrain_models": False, "retrain_halftime_dc": False} and _markers(tmp_path) == []
    assert "unpaused retrain_halftime_dc" in (tmp_path / "afstate" / "airflow_pauses.log").read_text()


def test_ctrl_c_during_the_read_right_after_pausing_a_retrain_dag_ends_with_it_unpaused(tmp_path):
    repo = _repo(tmp_path)
    env = {**_env(tmp_path, install=True), "AUSPEX_DIR": str(repo)}
    st = fd.airflow_state(tmp_path)
    st["details_delay_after_pause"] = {"retrain_halftime_dc": 60}
    (tmp_path / "fake" / "airflow.json").write_text(json.dumps(st))
    proc = subprocess.Popen(
        ["bash", "-c", f'bash "{RUNBOOK_IN(tmp_path)}" 2>&1 | tee "{tmp_path / "o"}"'],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    def in_the_read():
        c = fd.calls(tmp_path)
        pause = [i for i, x in enumerate(c) if "dags pause -y retrain_halftime_dc" in x]
        return bool(pause) and any("dags details retrain_halftime_dc" in x for x in c[pause[0] :])

    ledger = tmp_path / "afstate" / "airflow_pauses.log"
    try:
        assert _wait_for(in_the_read)
        time.sleep(0.3)
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
        assert _wait_for(lambda: ledger.exists() and "unpaused retrain_halftime_dc" in ledger.read_text())
        assert _paused(fd.airflow_state(tmp_path)) == {"retrain_models": False, "retrain_halftime_dc": False}
        assert _markers(tmp_path) == []
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


# --- a corpus step that failed, then reads ALREADY DONE --------------------------------------------------------------


def test_a_failed_corpus_step_is_not_forgotten_by_a_re_run_that_reads_it_already_done(tmp_path):
    """Step 1's apply committed, then its after-state failed (exit 1). Re-run alone, the wrapper re-plans and prints
    ALREADY DONE — no 'changed' record. Before, the committed change was then forgotten: no corpus record followed
    the last step-10 one, so --only 1 unpaused both retrain DAGs with no feature recompute and said ALL STEPS DONE.
    Now the failure is recorded 'failed' and counts as a corpus change until step 10 runs."""
    (tmp_path / "state").write_text("step=10 result=changed at=1 head=x\n")  # an earlier, complete repair
    rc, out, _, _, af = _run(tmp_path, FAIL_REPOINT_TEAM_MATCHES="1")
    assert rc == 1 and _paused(af) == {"retrain_models": True, "retrain_halftime_dc": True}, out
    assert "step=1 result=failed" in (tmp_path / "state").read_text()

    rc, out = fd.run(RUNBOOK_IN(tmp_path), {**_env(tmp_path), "ALREADY_REPOINT_TEAM_MATCHES": "1"}, "--only", "1")
    assert rc == 0 and "STEP 1/11 DONE (already_done)" in out, out
    assert "the repair is not complete" in out and "ALERT: retrain_models stays PAUSED" in out
    assert _paused(fd.airflow_state(tmp_path)) == {"retrain_models": True, "retrain_halftime_dc": True}

    calls_before = len(fd.calls(tmp_path))
    rc, out = fd.run(RUNBOOK_IN(tmp_path), _env(tmp_path), "--only", "10")
    assert rc == 0 and "not complete" not in out, out
    assert [c for c in fd.calls(tmp_path)[calls_before:] if "compute_features.py --backfill-finished soccer" in c]
    assert _paused(fd.airflow_state(tmp_path)) == {"retrain_models": False, "retrain_halftime_dc": False}

    # A full re-run of the finished sequence still only reads: step 10's record is the last corpus record.
    calls_before = len(fd.calls(tmp_path))
    env = {**_env(tmp_path), **{f"ALREADY_{w[:-3].upper()}": "1" for w in WRAPPERS}, "REGRADE_CHANGES": "0"}
    rc, out = fd.run(RUNBOOK_IN(tmp_path), env)
    assert rc == 0 and "not complete" not in out, out
    assert not [c for c in fd.calls(tmp_path)[calls_before:] if "compute_features" in c or "--apply" in c]


def test_a_failure_outside_the_corpus_steps_is_not_recorded(tmp_path):
    rc, out, _, _, _ = _run(tmp_path, "--only", "2", FAIL_MIGRATE_027_OUTCOME_TRIGGER="1")
    assert rc == 1, out
    state = tmp_path / "state"
    assert not state.exists() or "result=failed" not in state.read_text()


# --- an apply still running in the api container ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "proc",
    [
        "python /app/scripts/repair_fixture_orphans.py --sport soccer --delete-placeholders --apply --expect-plan x",
        "sh -c python /app/scripts/merge_soccer_teams.py --apply --expect-plan p > /tmp/merge_teams.log 2>&1; rc=$?",
        "python /app/scripts/regrade_predictions.py --match-ids-file ids --apply --expect-changes 135",
    ],
)
def test_an_apply_still_running_in_the_api_container_refuses_to_start(tmp_path, proc):
    """A run cut short during an apply leaves it running there (docker exec does not forward the interrupt)."""
    airflow = _airflow()
    airflow["procs"] = [proc, "python /app/scripts/merge_soccer_teams.py --check-after p"]
    rc, out, steps, calls, af = _run(tmp_path, airflow=airflow)
    assert rc == 1 and "REFUSED: a repair apply is still running in the api container" in out, out
    assert proc in out and "docker compose top api" in out and steps == []
    assert not [c for c in calls if "dags pause" in c]


def test_an_unreadable_process_list_refuses_to_start(tmp_path):
    airflow = _airflow()
    airflow["proc_scan_fails"] = True
    rc, out, steps, calls, _ = _run(tmp_path, airflow=airflow)
    assert rc == 1 and "could not read the api container's processes" in out and steps == [], out
    assert not [c for c in calls if "dags pause" in c]


def test_the_scan_covers_every_repair_tool_the_runbook_drives(tmp_path):
    rc, out, _, _, _ = _run(tmp_path, "--only", "9")
    assert rc == 0, out
    (scan,) = [c for c in fd.calls(tmp_path, scans=True) if fd.is_scan(c)]
    for tool in (
        "repoint_team_matches",
        "merge_soccer_teams",
        "repair_football_data_kickoffs",
        "repair_stranded_recs",
        "repair_fixture_orphans",
        "backfill_soccer_regulation_scores",
        "regrade_predictions",
        "backfill_season_labels",
    ):
        assert f"*/app/scripts/{tool}[.]py*--apply*" in scan, tool
