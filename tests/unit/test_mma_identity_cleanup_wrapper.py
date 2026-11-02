"""scripts/ops/mma_identity_cleanup.sh, run for real against a scripted docker:
it refuses while another scripts/ops wrapper or the soccer runbook runs (but
never over its own parents, such as sudo, an ssh one-liner's `bash -c` or a
wrapper that called it, nor its own subshells), while
an apply still runs in the api container, while a UFC card is on, and unless
the fighter merge is verified (merge_mma_fighters.sh --verify, stubbed here);
the reviewed plan before the backup and again from the dry run, where
ids_stamp + ids_done is pinned and the split between them may drift; no write
before a backup that landed; the apply pinned to the dry run's plan file; the
exact after-state (--check-after and a done re-plan); the follow-ups (exit 2
when they fail, --followups-only); 'already done' only once the recorded
plan's after-state holds; --verify (read only, never a backup)."""

from __future__ import annotations

import importlib.util
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OPS = REPO / "scripts" / "ops"
DATA = REPO / "scripts" / "data"
STATE = "/app/backups/mma_identity_cleanup_last_apply.env"
LISTS = (
    "mma_fighter_split_20261102.csv",
    "mma_fighter_aliases_20261102.csv",
    "mma_orphan_fighters_20261102.csv",
    "mma_fighter_athlete_ids_20261102.csv",
)

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN_R = (
    "PLAN split=1 split_done=0 split_refuse=0 bouts_moved=7 predictions_on_moved=2 recs_on_moved=0 "
    "alias_add=1 alias_done=0 alias_refuse=0 orphan_delete=1 orphan_done=0 orphan_refuse=0 "
    "ids_stamp=1290 ids_done=9 ids_refuse=0"
)
DONE_P = (
    "PLAN split=0 split_done=1 split_refuse=0 bouts_moved=0 predictions_on_moved=0 recs_on_moved=0 "
    "alias_add=0 alias_done=1 alias_refuse=0 orphan_delete=0 orphan_done=1 orphan_refuse=0 "
    "ids_stamp=0 ids_done=1299 ids_refuse=0"
)


def _rules(
    pre=PLAN_R,
    dry=PLAN_R,
    verify=DONE_P,
    dry_rc=0,
    apply_rc=0,
    after_rc=0,
    features_rc=0,
    card="CARD_CHECK_OK",
    card_rc=0,
):
    rules = [
        {"match": r"> /tmp/mma_identity_apply\.log 2>&1", "out": f"tool exit {apply_rc}", "rc": apply_rc},
        {"match": r"> /tmp/mma_identity_after\.log 2>&1", "out": "AFTER-STATE (scripted)", "rc": after_rc},
        {"match": r"> /tmp/mma_identity_dry\.log 2>&1", "out": f"tool exit {dry_rc}", "rc": dry_rc},
        {"match": r"compute_features_mma\.py --all-finished --force", "out": "Wrote 1986", "rc": features_rc},
        {"match": r"> /tmp/mma_identity_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"postgres sh -c psql .*mma_identity_card\.sql", "out": card, "rc": card_rc},
        {"match": r"postgres rm -f", "out": ""},
    ]
    for name, plan in (("pre", pre), ("dry", dry), ("verify", verify)):
        rules.append({"match": rf"grep -oE 'PLAN .*' /tmp/mma_identity_{name}\.log", "out": plan})
    return rules


_ENVS: dict = {}


def _install(tmp_path, rules, state=None, airflow=None, merge_rc=0, ps="1 0 /sbin/init\n", real_ps=False):
    """`ps` is the process list ("pid ppid args" lines) the wrapper reads; with `real_ps` the host's real list,
    scoped to this test's processes (other sessions on a dev machine run their own wrappers), comes first, plus a
    fake subshell of the wrapper (a child of whatever started the list)."""
    env = fd.install(tmp_path, rules, airflow)
    ops = tmp_path / "auspex" / "scripts" / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    for name in ("mma_identity_cleanup.sh", "_airflow_lib.sh"):
        shutil.copy(OPS / name, ops / name)
    (ops / "merge_mma_fighters.sh").write_text(
        f'#!/bin/bash\necho "merge stub $*"\n[ "$1" = --verify ] || exit 9\nexit {merge_rc}\n'
    )
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in LISTS:
        shutil.copy(DATA / name, data / name)
    if state is not None:
        fd.put_file(tmp_path, "api", STATE, state)
    (tmp_path / "ps.txt").write_text(ps)
    lines = [f"cat {shlex.quote(str(tmp_path / 'ps.txt'))}"]
    if real_ps:
        lines = [
            f"ps -eo pid=,ppid=,args= | grep -F -- {shlex.quote(str(tmp_path))}",
            f'echo "999991 $PPID bash {ops / "mma_identity_cleanup.sh"}"',
        ] + lines
    (tmp_path / "ps.sh").write_text("\n".join(lines) + "\n")
    env["MMA_IDENTITY_PS"] = f"bash {tmp_path / 'ps.sh'}"
    _ENVS[tmp_path] = env
    return env


def _script(tmp_path):
    return tmp_path / "auspex" / "scripts" / "ops" / "mma_identity_cleanup.sh"


def _run(tmp_path, rules, *args, **kw):
    env = _install(tmp_path, rules, **kw)
    rc, out = fd.run(_script(tmp_path), env, *args)
    return rc, out, fd.calls(tmp_path)


def _rerun(tmp_path, rules, *args, state=None):
    """The next run on the same fake VM: the Airflow model, B2 and container files carry over."""
    (tmp_path / "fake" / "calls.log").unlink()
    fd.set_rules(tmp_path, rules)
    if state is not None:
        fd.put_file(tmp_path, "api", STATE, state)
    rc, out = fd.run(_script(tmp_path), _ENVS[tmp_path], *args)
    return rc, out, fd.calls(tmp_path)


def _tool(calls):
    return [c for c in calls if "mma_identity_cleanup.py" in c]


def _applies(calls):
    return [c for c in _tool(calls) if "--apply" in c]


def _backups(tmp_path):
    return fd.airflow_state(tmp_path).get("triggers", 0)


def _state(tmp_path):
    return fd.get_file(tmp_path, "api", STATE) or ""


def test_a_clean_run_pins_the_plan_applies_it_and_recomputes_features(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: the MMA fighter identities are cleaned up" in out, out
    assert "merge stub --verify" in out and _backups(tmp_path) == 1
    (apply,) = _applies(calls)
    (dry,) = [c for c in calls if "/tmp/mma_identity_dry.log" in c and "mma_identity_cleanup.py" in c]
    plan_file = dry.split("--plan-out ")[1].split()[0]
    assert re.fullmatch(r"/app/backups/mma_identity_cleanup_dryrun_\d{8}T\d{6}Z_\d+\.plan\.json", plan_file), plan_file
    assert f"--expect-plan {plan_file}" in apply and "--archive /app/backups/mma_identity_cleanup_apply_" in apply
    (check,) = [c for c in calls if "--check-after" in c]
    assert f"--check-after {plan_file}" in check
    assert any("compute_features_mma.py --all-finished --force" in c and "--days 14 --force" in c for c in calls)
    assert f"PLAN={plan_file}" in _state(tmp_path) and "STATUS=done" in _state(tmp_path)
    # the order: the checks, then the pre-plan, the backup, the dry run, the apply
    order = [i for i, c in enumerate(calls) if "mma_identity_card" in c or "mma_identity_cleanup.py" in c]
    assert "mma_identity_card" in calls[order[0]]


def test_ids_may_drift_from_stamp_to_done_before_the_run(tmp_path):
    drift = PLAN_R.replace("ids_stamp=1290 ids_done=9", "ids_stamp=738 ids_done=561")
    rc, out, calls = _run(tmp_path, _rules(pre=drift, dry=drift))
    assert rc == 0 and len(_applies(calls)) == 1, out


@pytest.mark.parametrize(
    "pre",
    [
        PLAN_R.replace("ids_stamp=1290 ids_done=9", "ids_stamp=1290 ids_done=8"),  # a row lost
        PLAN_R.replace("ids_refuse=0", "ids_refuse=1"),
        PLAN_R.replace("split=1 split_done=0", "split=0 split_done=0").replace("split_refuse=0", "split_refuse=1"),
        PLAN_R.replace("bouts_moved=7", "bouts_moved=8"),
        PLAN_R.replace("recs_on_moved=0", "recs_on_moved=1"),
        PLAN_R.replace("alias_add=1 alias_done=0", "alias_add=0 alias_done=1"),  # part done
        "",
    ],
)
def test_anything_but_the_reviewed_plan_aborts_before_the_backup(tmp_path, pre):
    rc, out, calls = _run(tmp_path, _rules(pre=pre))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


def test_another_ops_wrapper_running_refuses_before_anything(tmp_path):
    ps = "1 0 /sbin/init\n4242 1 bash /opt/auspex/scripts/ops/merge_soccer_teams.sh\n"
    rc, out, calls = _run(tmp_path, _rules(), ps=ps)
    assert rc == 1 and "pid 4242: bash /opt/auspex/scripts/ops/merge_soccer_teams.sh" in out, out
    assert "ABORT at step 0: the process(es) above run a scripts/ops wrapper" in out
    assert not _tool(calls) and _backups(tmp_path) == 0


@pytest.mark.parametrize(
    "line",
    [
        "77 1 bash ./soccer_repair_runbook.sh --resume",  # from inside scripts/ops
        "77 1 /bin/bash -x /opt/auspex/scripts/ops/soccer_repair_runbook.sh",
        "77 1 sh -o pipefail scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows",
        "77 1 bash ./merge_mma_fighters.sh --verify",  # a wrapper of this directory, by its name
    ],
)
def test_a_shell_running_a_wrapper_or_the_runbook_refuses(tmp_path, line):
    rc, out, calls = _run(tmp_path, _rules(), ps=f"1 0 /sbin/init\n{line}\n")
    assert rc == 1 and f"pid {line.split(' ', 2)[0]}: {line.split(' ', 2)[2]}" in out and not _tool(calls), out


def test_a_sudo_started_wrapper_is_caught_by_its_shell_and_a_second_run_of_this_one_too(tmp_path):
    script = _script(tmp_path)
    ps = (
        "1 0 /sbin/init\n"
        "500 1 sudo bash /opt/auspex/scripts/ops/close_out_stranded_race_recs.sh\n"
        "501 500 bash /opt/auspex/scripts/ops/close_out_stranded_race_recs.sh\n"
        f"502 1 bash {script}\n"
    )
    rc, out, calls = _run(tmp_path, _rules(), ps=ps)
    assert rc == 1 and "pid 501: bash /opt/auspex/scripts/ops/close_out_stranded_race_recs.sh" in out, out
    assert f"pid 502: bash {script}" in out and "pid 500:" not in out and not _tool(calls)


def test_editors_pagers_log_tails_and_multiplexers_are_not_runs(tmp_path):
    ps = (
        "1 0 /sbin/init\n"
        "123 1 tee -a /home/auspex/mma_identity_cleanup.log\n"
        "124 1 tail -f /home/auspex/soccer_repair_runbook.log\n"
        "125 1 python /app/scripts/fetch_upcoming.py --sport mma --results\n"
        "126 1 vim /opt/auspex/scripts/ops/merge_mma_fighters.sh\n"
        "127 1 less scripts/ops/soccer_repair_runbook.sh\n"
        "128 1 tmux new -s ops bash /opt/auspex/scripts/ops/mma_identity_cleanup.sh\n"
        "129 1 shellcheck scripts/ops/mma_identity_cleanup.sh\n"
        "130 1 bash -c vim scripts/ops/merge_soccer_teams.sh\n"
        "131 1 bash\n"
    )
    rc, out, calls = _run(tmp_path, _rules(), ps=ps)
    assert rc == 0 and "DONE" in out, out


@pytest.mark.parametrize("ps_cmd", ["false", "true"])  # fails / lists nothing
def test_an_unreadable_process_list_refuses(tmp_path, ps_cmd):
    env = _install(tmp_path, _rules())
    env["MMA_IDENTITY_PS"] = ps_cmd
    rc, out = fd.run(_script(tmp_path), env)
    assert rc == 1 and "could not list this host's processes" in out, out
    assert not _tool(fd.calls(tmp_path)) and _backups(tmp_path) == 0


def _launch(tmp_path, how):
    """Start the wrapper the way an operator might, with the host's real process list (scoped to this test)."""
    env = {**fd.os.environ, **_install(tmp_path, _rules(), real_ps=True)}
    script, log = _script(tmp_path), tmp_path / "run.log"
    if how == "plain":
        argv, exe = ["bash", str(script)], None
    elif how == "ssh_one_liner":  # what `ssh auspex 'bash ... | tee -a ...'` runs: a bash -c parent naming the wrapper
        argv, exe = ["bash", "-c", f"bash {shlex.quote(str(script))} 2>&1 | tee -a {shlex.quote(str(log))}"], None
    elif how == "sudo_parent":  # a parent whose command line names the wrapper (argv[0] 'sudo')
        argv, exe = ["sudo", "-c", 'bash "$1"; rc=$?; exit $rc', "sudo", str(script)], "/bin/bash"
    else:  # a scripts/ops wrapper that runs this one as its child
        outer = script.parent / "outer_stub.sh"
        outer.write_text('#!/bin/bash\nbash "$(dirname "$0")/mma_identity_cleanup.sh" "$@"\nrc=$?\nexit $rc\n')
        argv, exe = ["bash", str(outer)], None
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, executable=exe)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("how", ["plain", "ssh_one_liner", "sudo_parent", "parent_wrapper"])
def test_its_own_parents_and_subshells_are_not_another_run(tmp_path, how):
    rc, out = _launch(tmp_path, how)
    assert rc == 0 and "DONE: the MMA fighter identities are cleaned up" in out, out
    assert "ABORT" not in out and _backups(tmp_path) == 1


def test_an_apply_still_running_in_the_api_container_refuses(tmp_path):
    procs = ["python /app/scripts/merge_mma_fighters.py --apply --expect-plan /app/backups/x.plan.json"]
    rc, out, calls = _run(tmp_path, _rules(), airflow={"procs": procs})
    assert rc == 1 and "still runs in the api container" in out, out
    assert not _tool(calls)
    scan = [c for c in fd.calls(tmp_path, scans=True) if fd.is_scan(c)]
    assert scan and "*/app/scripts/*[.]py*--apply*" in scan[0]  # any scripts/ tool's apply, not only the named ones


def test_an_unreadable_process_scan_refuses(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"proc_scan_fails": True})
    assert rc == 1 and "could not scan the api container" in out and not _tool(calls)


def test_a_ufc_card_on_refuses(tmp_path):
    card = "0d6e...|2026-11-07 22:00:00+00|scheduled\nCARD_CHECK_OK"
    rc, out, calls = _run(tmp_path, _rules(card=card))
    assert rc == 1 and "ABORT at step 0: a UFC card is on" in out and "2026-11-07 22:00:00+00|scheduled" in out, out
    assert not _tool(calls) and _backups(tmp_path) == 0


@pytest.mark.parametrize("card, card_rc", [("", 0), ("ERROR: relation", 1)])
def test_an_unreadable_card_check_refuses(tmp_path, card, card_rc):
    rc, out, calls = _run(tmp_path, _rules(card=card, card_rc=card_rc))
    assert rc == 1 and "could not read whether a UFC card is on" in out and not _tool(calls)


def test_the_card_window_is_in_the_query(tmp_path):
    _run(tmp_path, _rules())
    sent = (tmp_path / "fake" / "cp" / "mma_identity_card.sql").read_text()
    assert "now() - interval '8 hours' AND now() + interval '3 hours'" in sent and "m.status = 'live'" in sent
    assert "l.sport = 'mma'" in sent


def test_an_unverified_merge_refuses_before_the_backup(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), merge_rc=1)
    assert rc == 1 and "the fighter merge is not done and verified" in out, out
    assert not _tool(calls) and _backups(tmp_path) == 0


def test_another_run_holding_the_lock_refuses(tmp_path):
    env = _install(tmp_path, _rules())
    lock = Path(env["AF_STATE_DIR"]) / "mma_identity_cleanup.lock"
    lock.mkdir(parents=True)
    holder = fd.subprocess.Popen(["bash", "-c", "sleep 30; :", "mma_identity_cleanup.sh"])
    try:
        (lock / "holder").write_text(f"{holder.pid} mma_identity_cleanup.sh\n")
        rc, out = fd.run(_script(tmp_path), env)
    finally:
        holder.kill()
    assert rc == 1 and "another run of this script holds its lock" in out, out
    assert not _tool(fd.calls(tmp_path)) and _backups(tmp_path) == 0


def test_a_list_with_other_counts_aborts(tmp_path):
    env = _install(tmp_path, _rules())
    ids = tmp_path / "auspex" / "scripts" / "data" / LISTS[3]
    ids.write_text("\n".join(ids.read_text().splitlines()[:-1]) + "\n")
    rc, out = fd.run(_script(tmp_path), env)
    assert rc == 1 and "this script expects 11/1/1/1299" in out, out
    assert not _tool(fd.calls(tmp_path))


def test_no_write_before_a_backup_that_landed(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"backup_fails": True})
    assert rc == 1 and "backup did not land" in out, out
    assert not _applies(calls) and not [c for c in calls if "/tmp/mma_identity_dry.log" in c]


@pytest.mark.parametrize("dry, dry_rc", [(PLAN_R, 1), (PLAN_R.replace("bouts_moved=7", "bouts_moved=6"), 0)])
def test_a_dry_run_off_the_review_aborts_after_the_backup_with_nothing_written(tmp_path, dry, dry_rc):
    rc, out, calls = _run(tmp_path, _rules(dry=dry, dry_rc=dry_rc))
    assert rc == 1 and "ABORT at step 2" in out and not _applies(calls)


@pytest.mark.parametrize("apply_rc, why", [(2, "refused before writing"), (1, "rolled back")])
def test_a_failed_apply_aborts_with_status_applying(tmp_path, apply_rc, why):
    rc, out, calls = _run(tmp_path, _rules(apply_rc=apply_rc))
    assert rc == 1 and why in out and "STATUS=applying" in _state(tmp_path)
    assert not [c for c in calls if "--check-after" in c]


def test_a_failed_after_state_exits_1_and_keeps_status_applying(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(after_rc=1))
    assert rc == 1 and "FAILED after-state at step 4" in out and "STATUS=applying" in _state(tmp_path)
    assert not [c for c in calls if "compute_features_mma" in c]


def test_a_re_plan_that_is_not_done_fails_the_after_state(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(verify=DONE_P.replace("ids_done=1299", "ids_done=1298")))
    assert rc == 1 and "FAILED after-state at step 4" in out


def test_a_failed_follow_up_exits_2_then_followups_only_finishes(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(features_rc=1))
    assert rc == 2 and "INCOMPLETE" in out and "STATUS=followups" in _state(tmp_path)
    rc, out, calls = _rerun(tmp_path, _rules(), "--followups-only")
    assert rc == 0 and "FOLLOW-UPS DONE" in out and "STATUS=done" in _state(tmp_path)
    assert _backups(tmp_path) == 1 and not _tool(calls)


def test_followups_only_refuses_an_apply_still_recorded_applying(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), "--followups-only", state="PLAN=/app/backups/p.json\nSTATUS=applying\n")
    assert rc == 1 and "STATUS=applying" in out and not [c for c in calls if "compute_features_mma" in c]


def test_a_re_run_is_already_done_without_a_backup(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0
    rc, out, calls = _rerun(tmp_path, _rules(pre=DONE_P))
    assert rc == 0 and "ALREADY DONE" in out and "nothing to write (no backup)" in out, out
    assert _backups(tmp_path) == 1 and not _applies(calls)
    assert any("--check-after /app/backups/mma_identity_cleanup_dryrun_" in c for c in calls)


def test_done_but_nothing_recorded_fails(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P))
    assert rc == 1 and "no apply is recorded" in out and _backups(tmp_path) == 0


def test_done_with_a_failing_after_state_fails(tmp_path):
    state = "PLAN=/app/backups/p.plan.json\nSTATUS=done\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P, after_rc=1), state=state)
    assert rc == 1 and "after-state of the recorded plan" in out and _backups(tmp_path) == 0


def test_done_but_recorded_applying_checks_then_runs_the_follow_ups(tmp_path):
    state = "PLAN=/app/backups/p.plan.json\nSTATUS=applying\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P), state=state)
    assert rc == 0 and "COMPLETED: follow-ups done" in out and "STATUS=done" in _state(tmp_path), out
    assert any("--check-after /app/backups/p.plan.json" in c for c in calls) and _backups(tmp_path) == 0


def test_verify_is_read_only_and_needs_a_recorded_after_state(tmp_path):
    state = "PLAN=/app/backups/p.plan.json\nSTATUS=done\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P), "--verify", state=state)
    assert rc == 0 and "IDENTITY CLEANUP VERIFIED" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls) and not [c for c in calls if "compute_features" in c]
    assert "merge stub" not in out  # --verify does not re-verify the merge
    rc, out, calls = _rerun(tmp_path, _rules(pre=PLAN_R), "--verify")
    assert rc == 1 and "NOT VERIFIED: the identity clean-up is not done" in out
    rc, out, calls = _rerun(
        tmp_path, _rules(pre=DONE_P), "--verify", state="PLAN=/app/backups/p.plan.json\nSTATUS=applying\n"
    )
    assert rc == 1 and "NOT VERIFIED" in out and "STATUS=applying" in out
