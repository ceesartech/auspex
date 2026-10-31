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
ALERT, marker kept).
The plan, MERGED_AT, the rows to grade and STATUS=applying are recorded BEFORE
the apply; a re-run refuses while an apply still runs in the api container,
then finishes an apply whose run ended during it.
It refuses to start while another live run holds its lock, and leaves a DAG
whose marker names a live run to that run."""

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
    """The hint records the apply as unfinished with the rows its plan lists (grade_after), so the re-run grades
    them: never STATUS=done with an empty GRADE_IDS, which would skip grading the adopted rows for good."""
    rc, out, calls, af = _run(tmp_path, rules=_rules(pre=ALL_DONE))
    assert rc == 1 and "no apply is recorded" in out, out
    assert "STATUS=failed" in out and "['grade_after']" in out and "GRADE_IDS=%s" in out and '"$ids"' in out
    assert "STATUS=done" not in out and "GRADE_IDS=\\n" not in out
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


@pytest.mark.parametrize("status", ["grading", "applying"])
def test_rows_an_unfinished_earlier_apply_listed_are_graded_with_this_runs(tmp_path, status):
    """'applying': that run ended during its apply and the apply was cut short — this run merges the rest and grades
    the rows of both."""
    env = _install(tmp_path)
    _record(tmp_path, status, OLD)
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
    # Its command drops the marker too: a marker that outlived a hand unpause would adopt the operator's next pause.
    marker = tmp_path / "afstate" / "paused_fetch_live_odds"
    assert f"airflow dags unpause -y fetch_live_odds && rm -f {marker}" in out
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


# --- an apply cut short ------------------------------------------------------------------------------------------


def _set_airflow(tmp_path, **kv):
    """Change the fake's Airflow model / process list between runs."""
    path = tmp_path / "fake" / "airflow.json"
    st = json.loads(path.read_text())
    st.update(kv)
    path.write_text(json.dumps(st))


APPLY_CMD = "python /app/scripts/merge_soccer_teams.py --apply --expect-plan /app/backups/p.plan.json --out-csv x"


def test_an_interrupt_during_the_apply_leaves_its_record_and_the_re_run_waits_for_it_then_finishes_it(tmp_path):
    """The plan, MERGED_AT, the rows to grade and STATUS=applying are recorded BEFORE the apply. Ctrl-C during it
    (docker exec does not forward the interrupt: the apply may run on in the api container) still leaves them, and
    says so in the ledger. A re-run refuses while the apply runs; once it has ended, it finishes it: grades the rows
    and records done — never 'no apply is recorded' with a hint that skips the grading."""
    rules = _rules()
    next(r for r in rules if r["match"] == r"--apply --expect-plan")["delay"] = 60
    env = _install(tmp_path, rules)
    proc = subprocess.Popen(
        ["bash", "-c", f'bash "{WRAPPER}" 2>&1 | tee "{tmp_path / "out.log"}" >/dev/null'],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert _wait_for(lambda: any("--apply --expect-plan" in c for c in fd.calls(tmp_path)))
        rec = _recorded(tmp_path)
        assert rec["STATUS"] == "applying" and rec["GRADE_IDS"].split() == [KEPT], rec
        assert "merge_soccer_teams_dryrun_" in rec["PLAN"] and rec["MERGED_AT"].endswith("Z")
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
        assert _wait_for(lambda: "unpaused fetch_live_odds" in _ledger(tmp_path)), _ledger(tmp_path)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert "ALERT: this run ended during its 3/5 apply" in _ledger(tmp_path)
    assert "docker compose top api" in _ledger(tmp_path)
    assert _recorded(tmp_path) == rec  # nothing after the apply ran

    # The apply runs on in the api container: the re-run refuses before anything else.
    _set_airflow(tmp_path, procs=[APPLY_CMD])
    fd.set_rules(tmp_path, _rules(pre=ALL_DONE))
    before = len(fd.calls(tmp_path))
    rc, out = fd.run(WRAPPER, _env(tmp_path))
    assert rc == 1 and "REFUSED: a merge_soccer_teams.py --apply is still running" in out, out
    assert APPLY_CMD in out and "docker compose top api" in out
    assert fd.calls(tmp_path)[before:] == [] and _recorded(tmp_path) == rec

    # It has ended (every pair merged): step 0 finishes the 'applying' record.
    _set_airflow(tmp_path, procs=[])
    before = len(fd.calls(tmp_path))
    rc, out = fd.run(WRAPPER, _env(tmp_path))
    calls = fd.calls(tmp_path)[before:]
    assert rc == 0 and "STATUS=applying (that run ended during its apply" in out and "COMPLETED" in out, out
    (grade,) = [c for c in calls if "grade_completed_matches.py --match-id" in c]
    assert KEPT in grade and not [c for c in calls if "db_backup_daily" in c or "dags pause" in c or "--apply" in c]
    # The interrupted run unpaused the DAGs while its apply went on, so an odds run may have started mid-apply:
    # MERGED_AT moves to the time the re-run found the apply ended, and step 11 waits for an odds run after that.
    done = _recorded(tmp_path)
    assert done["STATUS"] == "done" and done["MERGED_AT"] >= rec["MERGED_AT"]
    assert f"MERGED_AT recorded as {done['MERGED_AT']}" in out


def test_the_applying_record_precedes_the_apply_and_merged_at_is_kept_to_the_end(tmp_path):
    rc, out, _, _ = _run(tmp_path)
    assert rc == 0, out
    calls = fd.calls(tmp_path)
    writes = [i for i, c in enumerate(calls) if c.startswith(f"compose exec -T api sh -c cat > {STATE}.tmp")]
    apply_ = next(i for i, c in enumerate(calls) if "--apply --expect-plan" in c)
    assert len(writes) == 3 and writes[0] < apply_ < writes[1]  # applying, grading, done
    assert "recorded STATUS=applying, MERGED_AT=" in out
    merged_at = out.split("recorded STATUS=applying, MERGED_AT=")[1].split(")")[0]
    assert _recorded(tmp_path)["MERGED_AT"] == merged_at and f"--odds-since {merged_at}" in out


def test_no_apply_runs_when_its_record_cannot_be_written_first(tmp_path):
    airflow = _airflow()
    airflow["write_fails"] = [STATE]
    rc, out, calls, af = _run(tmp_path, airflow=airflow)
    assert rc == 1 and "ABORT at step 3: could not record the apply (STATUS=applying)" in out, out
    assert not [c for c in calls if "--apply" in c]
    assert not af["dags"]["auspex_pipeline"]["paused"] and not af["dags"]["fetch_live_odds"]["paused"]


def test_an_unreadable_process_list_refuses_to_start(tmp_path):
    airflow = _airflow()
    airflow["proc_scan_fails"] = True
    rc, out, calls, af = _run(tmp_path, airflow=airflow)
    assert rc == 1 and "could not read the api container's processes" in out, out
    assert calls == [] and af.get("triggers", 0) == 0


def test_an_apply_of_another_tool_does_not_refuse_the_merge(tmp_path):
    airflow = _airflow()
    airflow["procs"] = ["python /app/scripts/repoint_team_matches.py --apply --expect-repoint 6"]
    rc, out, _, _ = _run(tmp_path, airflow=airflow)
    assert rc == 0 and "REFUSED" not in out, out


# --- the run lock and a live marker ---------------------------------------------------------------------------------


def test_an_own_marker_left_by_a_failed_unpause_in_the_retry_is_adopted_not_refused(tmp_path):
    """The first attempt's unpause (after a run slipped in) fails: the marker naming THIS run stays. The second
    attempt must take the DAG back, not refuse it as held by a live run (this run is live)."""
    airflow = _airflow()
    airflow["slip_in"] = {"fetch_live_odds": 1}  # its run advances once fetch_live_odds is unpaused again
    airflow["unpause_fails"] = ["auspex_pipeline"]
    rc, out, calls, _ = _run(tmp_path, airflow=airflow)
    assert "2/5 dry run" in out and "held by a run that is still going" not in out, out
    assert [c for c in calls if "--apply --expect-plan" in c]  # the DAG stayed on this run's list: no refusal
    # the retry's unpause, the one after the apply, then the exit's: it never unpauses, so the run is not DONE
    assert out.count("ALERT: auspex_pipeline is STILL PAUSED") == 3
    assert rc == 1 and "FAILED: the merge itself is through" in out and "DONE: 171/171" not in out


def _live(name):
    """A live process whose command line names `name`, like a wrapper still running."""
    return subprocess.Popen(["/bin/bash", "-c", "/bin/sleep 60; :", name])


def test_another_live_run_holding_the_lock_refuses_to_start(tmp_path):
    env = _install(tmp_path)
    holder = _live("merge_soccer_teams.sh")
    try:
        lock = tmp_path / "afstate" / "merge_soccer_teams.lock"
        lock.mkdir(parents=True)
        (lock / "holder").write_text(f"{holder.pid} merge_soccer_teams.sh\n")
        rc, out = fd.run(WRAPPER, env)
        assert rc == 1 and "REFUSED: another run holds" in out and f"pid {holder.pid}" in out, out
        assert fd.calls(tmp_path) == [] and (lock / "holder").exists()
    finally:
        holder.kill()
        holder.wait()


def test_a_lock_whose_run_is_gone_is_taken_over_and_released(tmp_path):
    env = _install(tmp_path)
    gone = subprocess.Popen(["/bin/bash", "-c", ":"])
    gone.wait()
    lock = tmp_path / "afstate" / "merge_soccer_teams.lock"
    lock.mkdir(parents=True)
    (lock / "holder").write_text(f"{gone.pid} merge_soccer_teams.sh\n")
    rc, out = fd.run(WRAPPER, env)
    assert rc == 0 and "taken over" in out and "taken over" in _ledger(tmp_path), out
    assert not lock.exists() and not list((tmp_path / "afstate").glob("*.stale.*"))


def test_a_marker_naming_a_run_still_going_is_left_to_it(tmp_path):
    """fetch_live_odds is paused by a merge run that is still going (another terminal): adopting it would unpause it
    under that run when this one ends. This run leaves it alone, says who holds it, and does not start."""
    holder = _live("merge_soccer_teams.sh")
    try:
        env = _install(tmp_path, airflow=_airflow(fetch_live_odds={"paused": True, "runs": []}))
        (tmp_path / "afstate").mkdir()
        marker = tmp_path / "afstate" / "paused_fetch_live_odds"
        marker.write_text(f"paused by merge_soccer_teams.sh pid {holder.pid} at 2026-10-31T13:00:00Z\n")
        rc, out = fd.run(WRAPPER, env)
        af = fd.airflow_state(tmp_path)["dags"]
        assert rc == 1 and "held by a run that is still going" in out and f"pid {holder.pid}" in out, out
        assert "held by a run that is still going" in _ledger(tmp_path)
        assert af["fetch_live_odds"]["paused"] and marker.exists() and "— adopted" not in out
        assert not af["auspex_pipeline"]["paused"]  # this run's own pause is undone at its exit
        assert not [c for c in fd.calls(tmp_path) if "unpause -y fetch_live_odds" in c or "--apply" in c]
    finally:
        holder.kill()
        holder.wait()


def test_a_marker_whose_run_is_gone_is_adopted(tmp_path):
    gone = subprocess.Popen(["/bin/bash", "-c", ":"])
    gone.wait()
    env = _install(tmp_path, airflow=_airflow(fetch_live_odds={"paused": True, "runs": []}))
    (tmp_path / "afstate").mkdir()
    (tmp_path / "afstate" / "paused_fetch_live_odds").write_text(
        f"paused by merge_soccer_teams.sh pid {gone.pid} at 2026-10-31T13:00:00Z\n"
    )
    rc, out = fd.run(WRAPPER, env)
    assert rc == 0 and "adopted" in out and not fd.airflow_state(tmp_path)["dags"]["fetch_live_odds"]["paused"], out
    assert _markers(tmp_path) == []


def test_dags_left_paused_by_a_run_killed_outright_during_its_apply_are_unpaused_by_the_re_run(tmp_path):
    """kill -9 (or OOM, a host crash) during the apply: no EXIT trap, both DAGs stay paused with their markers, and
    the orphaned apply finishes every pair. The re-run's step 0 finishes the 'applying' record — and before that
    unpauses what the dead run left paused, so COMPLETED never leaves the pipeline paused."""
    rules = _rules()
    next(r for r in rules if r["match"] == r"--apply --expect-plan")["delay"] = 60
    env = _install(tmp_path, rules)
    proc = subprocess.Popen(
        ["bash", "-c", f'bash "{WRAPPER}" 2>&1 | tee "{tmp_path / "out.log"}" >/dev/null'],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert _wait_for(lambda: any("--apply --expect-plan" in c for c in fd.calls(tmp_path)))
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    dags = fd.airflow_state(tmp_path)["dags"]
    assert dags["auspex_pipeline"]["paused"] and dags["fetch_live_odds"]["paused"]

    fd.set_rules(tmp_path, _rules(pre=ALL_DONE))
    rc, out = fd.run(WRAPPER, _env(tmp_path))
    dags = fd.airflow_state(tmp_path)["dags"]
    assert rc == 0 and "COMPLETED" in out, out
    assert not dags["auspex_pipeline"]["paused"] and not dags["fetch_live_odds"]["paused"]
    assert "was left paused by an earlier run that is gone" in _ledger(tmp_path)
    assert not list((tmp_path / "afstate").glob("paused_*"))


def test_an_unpause_that_does_not_take_after_the_apply_fails_the_run_and_the_re_run_releases_it(tmp_path):
    """A transient `dags unpause` failure after the apply: the ALERT alone was easy to miss under 'DONE' (and the
    runbook went on to ALL STEPS DONE with the pipeline paused). Now the run ends FAILED, so the runbook stops at
    step 3, and the re-run — ALREADY DONE — first unpauses the DAG the earlier run left paused."""
    airflow = _airflow()
    airflow["unpause_fails"] = ["auspex_pipeline"]
    rc, out, _, _ = _run(tmp_path, airflow=airflow)
    assert rc == 1 and "FAILED: the merge itself is through" in out and "DONE: 171/171" not in out, out
    assert fd.airflow_state(tmp_path)["dags"]["auspex_pipeline"]["paused"]

    path = tmp_path / "fake" / "airflow.json"
    state = json.loads(path.read_text())
    state["unpause_fails"] = []
    path.write_text(json.dumps(state))
    fd.set_rules(tmp_path, _rules(pre=ALL_DONE))
    rc, out = fd.run(WRAPPER, _env(tmp_path))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert not fd.airflow_state(tmp_path)["dags"]["auspex_pipeline"]["paused"]
    assert not list((tmp_path / "afstate").glob("paused_*"))
