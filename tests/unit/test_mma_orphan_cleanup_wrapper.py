"""scripts/ops/mma_orphan_cleanup.sh, run for real against a scripted docker:
it refuses while another scripts/ops wrapper or the soccer runbook runs (never
over its own parents or subshells), while an apply still runs in the api
container, while a UFC card is on, unless the list holds exactly 8 rows and the
identity clean-up is verified (mma_identity_cleanup.sh --verify, stubbed here);
the reviewed plan before the backup and again from the dry run; a free-space
floor before the backup; no write before a backup that landed; the apply
pinned to the dry run's plan file; the exact after-state (--check-after and a
done re-plan); 'already done' only once the recorded plan's after-state holds;
--verify (read only, never a backup)."""

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
STATE = "/app/backups/mma_orphan_cleanup_last_apply.env"
LIST = "mma_orphan_fighters_20261103.csv"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN_R = "PLAN listed=8 delete=8 done=0 refuse=0"
DONE_P = "PLAN listed=8 delete=0 done=8 refuse=0"
DF_OK = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sda1 78643200 71303168 3984588 95% /"


def _rules(pre=PLAN_R, dry=PLAN_R, verify=DONE_P, dry_rc=0, apply_rc=0, after_rc=0, card="CARD_CHECK_OK", card_rc=0):
    rules = [
        {"match": r"> /tmp/mma_orphan_apply\.log 2>&1", "out": f"tool exit {apply_rc}", "rc": apply_rc},
        {"match": r"> /tmp/mma_orphan_after\.log 2>&1", "out": "AFTER-STATE (scripted)", "rc": after_rc},
        {"match": r"> /tmp/mma_orphan_dry\.log 2>&1", "out": f"tool exit {dry_rc}", "rc": dry_rc},
        {"match": r"> /tmp/mma_orphan_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"postgres sh -c psql .*mma_orphan_card\.sql", "out": card, "rc": card_rc},
        {"match": r"postgres rm -f", "out": ""},
    ]
    for name, plan in (("pre", pre), ("dry", dry), ("verify", verify)):
        rules.append({"match": rf"grep -oE 'PLAN .*' /tmp/mma_orphan_{name}\.log", "out": plan})
    return rules


_ENVS: dict = {}


def _install(tmp_path, rules, state=None, airflow=None, identity_rc=0, ps="1 0 /sbin/init\n", df=DF_OK, real_ps=False):
    env = fd.install(tmp_path, rules, airflow)
    ops = tmp_path / "auspex" / "scripts" / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    for name in ("mma_orphan_cleanup.sh", "_airflow_lib.sh"):
        shutil.copy(OPS / name, ops / name)
    (ops / "mma_identity_cleanup.sh").write_text(
        f'#!/bin/bash\necho "identity stub $*"\n[ "$1" = --verify ] || exit 9\nexit {identity_rc}\n'
    )
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / LIST, data / LIST)
    if state is not None:
        fd.put_file(tmp_path, "api", STATE, state)
    (tmp_path / "ps.txt").write_text(ps)
    lines = [f"cat {shlex.quote(str(tmp_path / 'ps.txt'))}"]
    if real_ps:
        lines = [
            f"ps -eo pid=,ppid=,args= | grep -F -- {shlex.quote(str(tmp_path))}",
            f'echo "999991 $PPID bash {ops / "mma_orphan_cleanup.sh"}"',
        ] + lines
    (tmp_path / "ps.sh").write_text("\n".join(lines) + "\n")
    env["MMA_ORPHAN_PS"] = f"bash {tmp_path / 'ps.sh'}"
    (tmp_path / "df.txt").write_text(df)
    env["MMA_ORPHAN_DF"] = f"cat {tmp_path / 'df.txt'}"
    _ENVS[tmp_path] = env
    return env


def _script(tmp_path):
    return tmp_path / "auspex" / "scripts" / "ops" / "mma_orphan_cleanup.sh"


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
    return [c for c in calls if "mma_orphan_cleanup.py" in c]


def _applies(calls):
    return [c for c in _tool(calls) if "--apply" in c]


def _backups(tmp_path):
    return fd.airflow_state(tmp_path).get("triggers", 0)


def _state(tmp_path):
    return fd.get_file(tmp_path, "api", STATE) or ""


def test_a_clean_run_pins_the_plan_and_applies_it(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: the 8 bout-less MMA fighter rows are deleted" in out, out
    assert "identity stub --verify" in out and _backups(tmp_path) == 1
    assert "backups filesystem: 3984588 KiB free" in out
    (apply,) = _applies(calls)
    (dry,) = [c for c in calls if "/tmp/mma_orphan_dry.log" in c and "mma_orphan_cleanup.py" in c]
    plan_file = dry.split("--plan-out ")[1].split()[0]
    assert re.fullmatch(r"/app/backups/mma_orphan_cleanup_dryrun_\d{8}T\d{6}Z_\d+\.plan\.json", plan_file), plan_file
    assert f"--expect-plan {plan_file}" in apply and "--archive /app/backups/mma_orphan_cleanup_apply_" in apply
    (check,) = [c for c in calls if "--check-after" in c]
    assert f"--check-after {plan_file}" in check
    assert f"PLAN={plan_file}" in _state(tmp_path) and "STATUS=done" in _state(tmp_path)
    order = [i for i, c in enumerate(calls) if "mma_orphan_card" in c or "mma_orphan_cleanup.py" in c]
    assert "mma_orphan_card" in calls[order[0]]  # the card check comes before any tool run


@pytest.mark.parametrize(
    "pre",
    [
        PLAN_R.replace("delete=8 done=0 refuse=0", "delete=7 done=0 refuse=1"),  # a row gained a bout / reference
        PLAN_R.replace("delete=8 done=0", "delete=7 done=1"),  # one deleted by hand
        PLAN_R.replace("listed=8", "listed=9"),
        "",
    ],
)
def test_anything_but_the_reviewed_plan_aborts_before_the_backup(tmp_path, pre):
    rc, out, calls = _run(tmp_path, _rules(pre=pre))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


def test_another_ops_wrapper_running_refuses_before_anything(tmp_path):
    ps = "1 0 /sbin/init\n4242 1 bash /opt/auspex/scripts/ops/settle_mma_held_rows.sh\n"
    rc, out, calls = _run(tmp_path, _rules(), ps=ps)
    assert rc == 1 and "pid 4242: bash /opt/auspex/scripts/ops/settle_mma_held_rows.sh" in out, out
    assert "ABORT at step 0: the process(es) above run a scripts/ops wrapper" in out
    assert not _tool(calls) and _backups(tmp_path) == 0


@pytest.mark.parametrize(
    "line",
    [
        "77 1 bash ./soccer_repair_runbook.sh --resume",
        "77 1 /bin/bash -x /opt/auspex/scripts/ops/soccer_repair_runbook.sh",
        "77 1 sh -o pipefail scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows",
        "77 1 bash ./mma_identity_cleanup.sh",
    ],
)
def test_a_shell_running_a_wrapper_or_the_runbook_refuses(tmp_path, line):
    rc, out, calls = _run(tmp_path, _rules(), ps=f"1 0 /sbin/init\n{line}\n")
    assert rc == 1 and f"pid {line.split(' ', 2)[0]}: {line.split(' ', 2)[2]}" in out and not _tool(calls), out


def test_editors_and_log_tails_are_not_runs(tmp_path):
    ps = (
        "1 0 /sbin/init\n"
        "123 1 tee -a /home/auspex/mma_orphan_cleanup.log\n"
        "124 1 tail -f /home/auspex/soccer_repair_runbook.log\n"
        "126 1 vim /opt/auspex/scripts/ops/mma_orphan_cleanup.sh\n"
        "128 1 tmux new -s ops bash /opt/auspex/scripts/ops/mma_orphan_cleanup.sh\n"
    )
    rc, out, calls = _run(tmp_path, _rules(), ps=ps)
    assert rc == 0 and "DONE" in out, out


@pytest.mark.parametrize("ps_cmd", ["false", "true"])  # fails / lists nothing
def test_an_unreadable_process_list_refuses(tmp_path, ps_cmd):
    env = _install(tmp_path, _rules())
    env["MMA_ORPHAN_PS"] = ps_cmd
    rc, out = fd.run(_script(tmp_path), env)
    assert rc == 1 and "could not list this host's processes" in out, out
    assert not _tool(fd.calls(tmp_path)) and _backups(tmp_path) == 0


@pytest.mark.parametrize("how", ["plain", "ssh_one_liner"])
def test_its_own_parents_and_subshells_are_not_another_run(tmp_path, how):
    env = {**fd.os.environ, **_install(tmp_path, _rules(), real_ps=True)}
    script, log = _script(tmp_path), tmp_path / "run.log"
    argv = (
        ["bash", str(script)]
        if how == "plain"
        else ["bash", "-c", f"bash {shlex.quote(str(script))} 2>&1 | tee -a {shlex.quote(str(log))}"]
    )
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "DONE: the 8 bout-less" in out and "ABORT" not in out, out


def test_an_apply_still_running_in_the_api_container_refuses(tmp_path):
    procs = ["python /app/scripts/mma_identity_cleanup.py --apply --expect-plan /app/backups/x.plan.json"]
    rc, out, calls = _run(tmp_path, _rules(), airflow={"procs": procs})
    assert rc == 1 and "still runs in the api container" in out, out
    assert not _tool(calls)
    scan = [c for c in fd.calls(tmp_path, scans=True) if fd.is_scan(c)]
    assert scan and "*/app/scripts/*[.]py*--apply*" in scan[0]  # any scripts/ tool's apply


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
    sent = (tmp_path / "fake" / "cp" / "mma_orphan_card.sql").read_text()
    assert "now() - interval '8 hours' AND now() + interval '3 hours'" in sent and "m.status = 'live'" in sent
    assert "l.sport = 'mma'" in sent


def test_an_unverified_identity_clean_up_refuses_before_the_backup(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), identity_rc=1)
    assert rc == 1 and "the MMA identity clean-up is not done and verified" in out, out
    assert not _tool(calls) and _backups(tmp_path) == 0


def test_another_run_holding_the_lock_refuses(tmp_path):
    env = _install(tmp_path, _rules())
    lock = Path(env["AF_STATE_DIR"]) / "mma_orphan_cleanup.lock"
    lock.mkdir(parents=True)
    holder = fd.subprocess.Popen(["bash", "-c", "sleep 30; :", "mma_orphan_cleanup.sh"])
    try:
        (lock / "holder").write_text(f"{holder.pid} mma_orphan_cleanup.sh\n")
        rc, out = fd.run(_script(tmp_path), env)
    finally:
        holder.kill()
    assert rc == 1 and "another run of this script holds its lock" in out, out
    assert not _tool(fd.calls(tmp_path)) and _backups(tmp_path) == 0


@pytest.mark.parametrize("cut", [1, -1])  # one row fewer / one row more
def test_a_list_with_another_count_aborts(tmp_path, cut):
    env = _install(tmp_path, _rules())
    f = tmp_path / "auspex" / "scripts" / "data" / LIST
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:-1] if cut == 1 else lines + [lines[-1]]) + "\n")
    rc, out = fd.run(_script(tmp_path), env)
    assert rc == 1 and "this script expects 8" in out, out
    assert not _tool(fd.calls(tmp_path))


@pytest.mark.parametrize(
    "df, why",
    [
        (
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sda1 78643200 77594624 1048576 99% /",
            "1048576 KiB free",
        ),
        ("", "an unreadable amount of KiB free"),
        ("df: backups: No such file or directory", "an unreadable amount of KiB free"),
    ],
)
def test_too_little_free_disk_or_an_unreadable_df_refuses_before_the_backup(tmp_path, df, why):
    rc, out, calls = _run(tmp_path, _rules(), df=df)
    assert rc == 1 and "ABORT at step 1" in out and why in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


def test_no_write_before_a_backup_that_landed(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"backup_fails": True})
    assert rc == 1 and "backup did not land" in out, out
    assert not _applies(calls) and not [c for c in calls if "/tmp/mma_orphan_dry.log" in c]


@pytest.mark.parametrize("dry, dry_rc", [(PLAN_R, 1), (PLAN_R.replace("delete=8 done=0", "delete=7 done=1"), 0)])
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


def test_a_re_plan_that_is_not_done_fails_the_after_state(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(verify=DONE_P.replace("delete=0 done=8", "delete=1 done=7")))
    assert rc == 1 and "FAILED after-state at step 4" in out and "STATUS=applying" in _state(tmp_path)


def test_a_re_run_is_already_done_without_a_backup(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0
    rc, out, calls = _rerun(tmp_path, _rules(pre=DONE_P))
    assert rc == 0 and "ALREADY DONE" in out and "nothing to write (no backup)" in out, out
    assert _backups(tmp_path) == 1 and not _applies(calls)
    assert any("--check-after /app/backups/mma_orphan_cleanup_dryrun_" in c for c in calls)


def test_done_but_nothing_recorded_fails(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P))
    assert rc == 1 and "no apply is recorded" in out and _backups(tmp_path) == 0


def test_done_with_a_failing_after_state_fails(tmp_path):
    state = "PLAN=/app/backups/p.plan.json\nSTATUS=done\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P, after_rc=1), state=state)
    assert rc == 1 and "after-state of the recorded plan" in out and _backups(tmp_path) == 0


def test_done_but_recorded_applying_checks_then_records_done(tmp_path):
    state = "PLAN=/app/backups/p.plan.json\nSTATUS=applying\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P), state=state)
    assert rc == 0 and "COMPLETED" in out and "STATUS=done" in _state(tmp_path), out
    assert any("--check-after /app/backups/p.plan.json" in c for c in calls) and _backups(tmp_path) == 0


def test_verify_is_read_only_and_needs_a_recorded_after_state(tmp_path):
    state = "PLAN=/app/backups/p.plan.json\nSTATUS=done\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_P), "--verify", state=state)
    assert rc == 0 and "ORPHAN CLEANUP VERIFIED" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)
    assert "identity stub" not in out  # --verify does not re-verify the identity clean-up
    assert not [c for c in calls if "psql" in c and "mma_orphan_card" in c]  # nor check the card
    rc, out, calls = _rerun(tmp_path, _rules(pre=PLAN_R), "--verify")
    assert rc == 1 and "NOT VERIFIED: the orphan clean-up is not done" in out
    rc, out, calls = _rerun(
        tmp_path, _rules(pre=DONE_P), "--verify", state="PLAN=/app/backups/p.plan.json\nSTATUS=applying\n"
    )
    assert rc == 1 and "NOT VERIFIED" in out and "STATUS=applying" in out
    rc, out, calls = _rerun(tmp_path, _rules(pre=DONE_P, after_rc=1), "--verify", state=state)
    assert rc == 1 and "after-state of the recorded plan" in out


def test_usage(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), "--apply")
    assert rc == 2 and "usage" in out and not calls
