"""scripts/ops/merge_soccer_teams.sh, run for real against a scripted docker
whose Airflow behaves like 3.3: a paused DAG's runs never advance. The wrapper
must DRAIN before it pauses (a run in flight when the DAG is paused would stay
'running' for good), retry when a run slips in between, fail on unmatched
odds-feed spellings, record the plan, MERGED_AT, the rows to grade and the
run's STATUS for the runbook, exit 0 at once when every pair is merged and the
last apply finished, finish an apply whose grading failed, and unpause both
DAGs when Ctrl-C kills the tee its output goes through.
A DAG is on the run's list, with its marker, from before its pause: a read-back
that fails, or an interrupt during it, still ends with the DAG unpaused (or an
ALERT, marker kept)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
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
STATE = "/app/backups/merge_soccer_teams_last_apply.env"
KEPT = "11111111-1111-1111-1111-111111111111"
OLD = "22222222-2222-2222-2222-222222222222"
RECORDED = "PLAN=/app/backups/x.plan.json\nMERGED_AT=2026-10-31T12:00:00Z\nGRADE_IDS={ids}\nSTATUS={status}\n"


def _rules(pre=NOT_DONE, written=WRITTEN, after_rc=0, rec_rc=0, grade_rc=0, dry_delay=0):
    dry = {"match": r"merge_teams_dry\.log 2>&1; rc", "out": NOT_DONE}
    if dry_delay:
        dry["delay"] = dry_delay
    return [
        {"match": r"merge_teams_pre\.log 2>&1; echo", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN pairs_merge.*merge_teams_pre\.log", "out": pre},
        {"match": r"--check-after \S+precheck\S+ > /tmp/merge_teams_pre_after\.log", "out": "AFTER ok", "rc": after_rc},
        {
            "match": r"--check-after /app/backups/\S+\.plan\.json --after-later-steps > \S+merge_teams_rec_after\.log",
            "out": "AFTER recorded",
            "rc": rec_rc,
        },
        dry,
        {"match": r"grep -o 'PLAN pairs_merge.*merge_teams_dry\.log", "out": NOT_DONE},
        {"match": r"GRADE_AFTER", "out": KEPT},
        {"match": r"--apply --expect-plan", "out": "Archived 10 row(s)"},
        {"match": r"grep -o 'WRITTEN \.\*' /tmp/merge_teams\.log", "out": written},
        {"match": r"grade_completed_matches\.py --match-id", "out": "1", "rc": grade_rc},
        {"match": r"--check-after \S+dryrun\S+ > /tmp/merge_teams_after\.log", "out": "AFTER ok"},
    ]


def _record(tmp_path, status="done", ids=""):
    fd.put_file(tmp_path, "api", STATE, RECORDED.format(ids=ids, status=status))


def _recorded(tmp_path) -> dict:
    text = fd.get_file(tmp_path, "api", STATE) or ""
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _airflow(**dags):
    base = {"auspex_pipeline": {"paused": False, "runs": []}, "fetch_live_odds": {"paused": False, "runs": []}}
    base.update(dags)
    return {"dags": base}


def _install(tmp_path, rules=None, airflow=None):
    env = fd.install(tmp_path, rules or _rules(), airflow or _airflow())
    (tmp_path / "auspex" / "scripts" / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(MAP, tmp_path / "auspex" / "scripts" / "data" / MAP.name)
    return env


def _run(tmp_path, rules=None, airflow=None):
    env = _install(tmp_path, rules, airflow)
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
    rec = _recorded(tmp_path)  # the odds check still gets its plan and MERGED_AT
    assert "merge_soccer_teams_dryrun_" in rec["PLAN"] and rec["STATUS"] == "done"


def test_the_plan_merged_at_rows_and_status_are_recorded_for_the_runbook(tmp_path):
    rc, out, calls, _ = _run(tmp_path)
    assert rc == 0, out
    rec = _recorded(tmp_path)
    assert rec["PLAN"].startswith("/app/backups/merge_soccer_teams_dryrun_") and rec["PLAN"].endswith(".plan.json")
    assert rec["MERGED_AT"].endswith("Z") and rec["GRADE_IDS"].split() == [KEPT] and rec["STATUS"] == "done"
    assert "after the next fetch_live_odds run has FINISHED" in out
    assert "airflow dags list-runs fetch_live_odds --state success -o json" in out


def test_all_pairs_done_exits_0_without_a_backup_or_a_pause(tmp_path):
    env = _install(tmp_path, _rules(pre=ALL_DONE))
    _record(tmp_path, "done")
    rc, out = fd.run(WRAPPER, env)
    calls, af = fd.calls(tmp_path), fd.airflow_state(tmp_path)
    assert rc == 0 and "ALREADY DONE" in out, out
    assert af.get("triggers", 0) == 0
    assert not [c for c in calls if "dags pause" in c or "--apply" in c or "grade_completed_matches" in c]
    assert [c for c in calls if "--after-later-steps > /tmp/merge_teams_rec_after.log" in c]


def test_all_pairs_done_but_a_broken_after_state_fails(tmp_path):
    env = _install(tmp_path, _rules(pre=ALL_DONE, after_rc=1))
    _record(tmp_path, "done")
    rc, out = fd.run(WRAPPER, env)
    assert rc == 1 and "after-state check failed" in out, out
    assert fd.airflow_state(tmp_path).get("triggers", 0) == 0


def test_all_pairs_done_but_the_recorded_plans_after_state_fails(tmp_path):
    env = _install(tmp_path, _rules(pre=ALL_DONE, rec_rc=1))
    _record(tmp_path, "done")
    rc, out = fd.run(WRAPPER, env)
    assert rc == 1 and "ALREADY DONE" not in out and "the plan the last apply recorded" in out, out


def test_all_pairs_done_with_no_recorded_apply_fails_and_says_how_to_record_one(tmp_path):
    rc, out, calls, af = _run(tmp_path, rules=_rules(pre=ALL_DONE))
    assert rc == 1 and "no apply is recorded" in out and "STATUS=done" in out, out
    assert af.get("triggers", 0) == 0


def test_a_grading_failure_is_recorded_and_the_re_run_finishes_that_apply(tmp_path):
    """The apply commits but 4/5 grading fails (exit 1). The re-run reads every
    pair 'done': it must not say ALREADY DONE while the adopted rows' recs wait
    for a P&L — it grades the rows the apply listed, re-checks, records done."""
    rc, out, _, _ = _run(tmp_path, rules=_rules(grade_rc=1))
    assert rc == 1 and "FAILED" in out, out
    assert _recorded(tmp_path)["STATUS"] == "failed" and _recorded(tmp_path)["GRADE_IDS"].split() == [KEPT]

    fd.set_rules(tmp_path, _rules(pre=ALL_DONE))
    before = len(fd.calls(tmp_path))
    rc, out = fd.run(WRAPPER, _env(tmp_path))
    calls = fd.calls(tmp_path)[before:]
    assert rc == 0 and "COMPLETED" in out and "ALREADY DONE" not in out, out
    (grade,) = [c for c in calls if "grade_completed_matches.py --match-id" in c]
    assert KEPT in grade
    assert not [c for c in calls if "db_backup_daily" in c or "dags pause" in c or "--apply" in c]
    assert _recorded(tmp_path)["STATUS"] == "done"

    before = len(fd.calls(tmp_path))
    rc, out = fd.run(WRAPPER, _env(tmp_path))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert not [c for c in fd.calls(tmp_path)[before:] if "grade_completed_matches" in c]


def test_an_unfinished_finish_stays_failed(tmp_path):
    env = _install(tmp_path, _rules(pre=ALL_DONE, grade_rc=1))
    _record(tmp_path, "failed", KEPT)
    rc, out = fd.run(WRAPPER, env)
    assert rc == 1 and "finishing the last apply" in out and _recorded(tmp_path)["STATUS"] == "failed", out


def test_rows_an_unfinished_earlier_apply_listed_are_graded_with_this_runs(tmp_path):
    env = _install(tmp_path)
    _record(tmp_path, "grading", OLD)
    rc, out = fd.run(WRAPPER, env)
    assert rc == 0, out
    (grade,) = [c for c in fd.calls(tmp_path) if "grade_completed_matches.py --match-id" in c]
    assert KEPT in grade and OLD in grade
    assert sorted(_recorded(tmp_path)["GRADE_IDS"].split()) == sorted([KEPT, OLD])


def _env(tmp_path):
    """The env of an earlier _install, for a second run that keeps its state."""
    return {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_DIR": str(tmp_path / "fake"),
        "AUSPEX_DIR": str(tmp_path / "auspex"),
        "TMPDIR": str(tmp_path / "tmp"),
        "AF_POLL_SECONDS": "0",
        "AF_STATE_DIR": str(tmp_path / "afstate"),
    }


# --- a dead terminal ------------------------------------------------------------------------------------------------


def _wait_for(pred, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_ctrl_c_with_the_output_through_tee_still_unpauses_both_dags(tmp_path):
    """The operator runs `merge_soccer_teams.sh 2>&1 | tee -a log` (or the
    runbook pipes it through tee) and presses Ctrl-C while both DAGs are paused
    for the dry run: the terminal signals the whole process group, tee dies, and
    the EXIT trap's writes hit a dead pipe. Both DAGs must still end unpaused,
    and the ledger must say so (it survives the dead terminal)."""
    env = _install(tmp_path, _rules(dry_delay=60))
    proc = subprocess.Popen(
        ["bash", "-c", f'bash "{WRAPPER}" 2>&1 | tee "{tmp_path / "out.log"}" >/dev/null'],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert _wait_for(lambda: any("merge_teams_dry.log 2>&1; rc" in c for c in fd.calls(tmp_path)))
        af = fd.airflow_state(tmp_path)["dags"]
        assert af["auspex_pipeline"]["paused"] and af["fetch_live_odds"]["paused"]
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)

        def both_unpaused():
            d = fd.airflow_state(tmp_path)["dags"]
            return not d["auspex_pipeline"]["paused"] and not d["fetch_live_odds"]["paused"]

        assert _wait_for(both_unpaused), fd.airflow_state(tmp_path)
        ledger = tmp_path / "afstate" / "airflow_pauses.log"
        assert _wait_for(lambda: ledger.exists() and "unpaused fetch_live_odds" in ledger.read_text())
        text = ledger.read_text()
        assert "unpaused auspex_pipeline" in text and "unpaused fetch_live_odds" in text, text
        assert not list((tmp_path / "afstate").glob("paused_*"))
        assert not [c for c in fd.calls(tmp_path) if "--apply" in c]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


def test_a_dag_a_killed_run_left_paused_is_adopted_and_unpaused(tmp_path):
    """kill -9 (no trap) left fetch_live_odds paused with its marker: the next
    run takes it over and unpauses it at the end, instead of 'already paused —
    left as it is' for good."""
    env = _install(tmp_path, airflow=_airflow(fetch_live_odds={"paused": True, "runs": []}))
    (tmp_path / "afstate").mkdir()
    (tmp_path / "afstate" / "paused_fetch_live_odds").write_text("paused by merge_soccer_teams.sh pid 1 at x\n")
    rc, out = fd.run(WRAPPER, env)
    af = fd.airflow_state(tmp_path)["dags"]
    assert rc == 0, out
    assert "adopted" in out and not af["fetch_live_odds"]["paused"] and not af["auspex_pipeline"]["paused"]
    assert not list((tmp_path / "afstate").glob("paused_*"))


def test_an_unpause_that_does_not_take_is_an_alert_in_the_ledger_and_keeps_its_marker(tmp_path):
    airflow = _airflow()
    airflow["unpause_fails"] = ["fetch_live_odds"]
    rc, out, _, af = _run(tmp_path, airflow=airflow)
    assert "ALERT: fetch_live_odds is STILL PAUSED" in out
    assert "ALERT: fetch_live_odds is STILL PAUSED" in (tmp_path / "afstate" / "airflow_pauses.log").read_text()
    assert (tmp_path / "afstate" / "paused_fetch_live_odds").exists()
    assert (
        not (tmp_path / "afstate" / "paused_auspex_pipeline").exists() and not af["dags"]["auspex_pipeline"]["paused"]
    )


def _ledger(tmp_path) -> str:
    f = tmp_path / "afstate" / "airflow_pauses.log"
    return f.read_text() if f.exists() else ""


def _markers(tmp_path) -> list[str]:
    return sorted(m.name for m in (tmp_path / "afstate").glob("paused_*"))


# --- a pause whose read-back fails or is interrupted ------------------------------------------------------------------


def test_a_failed_read_right_after_a_pause_still_ends_with_the_dag_unpaused(tmp_path):
    """`dags pause` took, but the read that confirms it fails once (a transient CLI/DB error): the DAG is on this run's
    list from before the pause, so the exit unpauses it. Before, it was listed only after that read — the DAG stayed
    paused with no marker and no ALERT, and every later run said 'already paused — left as it is' for good."""
    airflow = _airflow()
    airflow["details_fail_after_pause"] = ["fetch_live_odds"]
    rc, out, calls, af = _run(tmp_path, airflow=airflow)
    assert rc == 1 and "ABORT: could not drain and pause" in out, out
    assert "could not confirm that fetch_live_odds is paused" in out
    assert not af["dags"]["fetch_live_odds"]["paused"] and not af["dags"]["auspex_pipeline"]["paused"]
    ledger = _ledger(tmp_path)
    assert "pausing fetch_live_odds" in ledger and "unpaused fetch_live_odds" in ledger, ledger
    assert _markers(tmp_path) == [] and not [c for c in calls if "--apply" in c]


def test_a_failed_read_after_a_pause_whose_unpause_fails_is_an_alert_with_its_marker_kept(tmp_path):
    airflow = _airflow()
    airflow["details_fail_after_pause"] = ["fetch_live_odds"]
    airflow["unpause_fails"] = ["fetch_live_odds"]
    rc, out, _, af = _run(tmp_path, airflow=airflow)
    assert rc == 1 and af["dags"]["fetch_live_odds"]["paused"], out
    assert "ALERT: fetch_live_odds is STILL PAUSED" in _ledger(tmp_path)
    assert _markers(tmp_path) == ["paused_fetch_live_odds"]
    assert not af["dags"]["auspex_pipeline"]["paused"]


def test_ctrl_c_during_the_read_right_after_a_pause_still_unpauses_it(tmp_path):
    """SIGINT to the process group (output through tee, as the operator runs it) while the read that confirms
    fetch_live_odds' pause is in flight: bash runs the trap as soon as that read returns, before any bookkeeping
    after it — so the bookkeeping is done before the pause."""
    airflow = _airflow()
    airflow["details_delay_after_pause"] = {"fetch_live_odds": 60}
    env = _install(tmp_path, airflow=airflow)
    proc = subprocess.Popen(
        ["bash", "-c", f'bash "{WRAPPER}" 2>&1 | tee "{tmp_path / "out.log"}" >/dev/null'],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    def in_the_read():
        c = fd.calls(tmp_path)
        pause = [i for i, x in enumerate(c) if "dags pause -y fetch_live_odds" in x]
        return bool(pause) and any("dags details fetch_live_odds" in x for x in c[pause[0] :])

    try:
        assert _wait_for(in_the_read)
        time.sleep(0.3)
        assert fd.airflow_state(tmp_path)["dags"]["fetch_live_odds"]["paused"]
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
        assert _wait_for(lambda: "unpaused fetch_live_odds" in _ledger(tmp_path)), _ledger(tmp_path)
        af = fd.airflow_state(tmp_path)["dags"]
        assert not af["fetch_live_odds"]["paused"] and not af["auspex_pipeline"]["paused"]
        assert _markers(tmp_path) == [] and not [c for c in fd.calls(tmp_path) if "--apply" in c]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
