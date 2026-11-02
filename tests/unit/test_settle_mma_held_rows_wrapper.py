"""scripts/ops/settle_mma_held_rows.sh, run for real against a scripted docker
(tests/unit/_ops_fake_docker.py).

The wrapper refuses, before anything else and with nothing changed, while
another run of it holds its lock, another scripts/ops/ wrapper or the soccer
runbook runs on the host, an apply that writes UFC rows runs in the api
container, or a UFC card is in progress; then unless run B's done check
(repair_stranded_recs_mma.sh --verify-b) passes. Its pre-check must plan
exactly the reviewed plan (or the done plan: ALREADY DONE after --check-after,
no backup); then the verified backup, the dry run with exactly the reviewed
plan, the apply pinned to every key, and the exact after-state (--check-after,
a re-plan of 0, B's done check again). --verify is read only.

B's done check is a stub here (its own tests are in
test_repair_stranded_recs_mma_wrapper.py), except in the last test, which
runs the real run-B and merge wrappers against the same scripted docker.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OPS = REPO / "scripts" / "ops"
DATA = REPO / "scripts" / "data"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN_S = (
    "PLAN record=1 record_done=0 record_predictions=2 record_recs_void=0 delete=1 delete_done=0 "
    "delete_predictions=2 delete_odds=0 delete_snapshots=0 delete_features=2 delete_other_rows=0 teams_delete=1 "
    "refuse=0"
)
DONE_S = (
    "PLAN record=0 record_done=1 record_predictions=0 record_recs_void=0 delete=0 delete_done=1 "
    "delete_predictions=0 delete_odds=0 delete_snapshots=0 delete_features=0 delete_other_rows=0 teams_delete=0 "
    "refuse=0"
)
PIN = ",".join(PLAN_S.split()[1:])
STUB_B = """#!/bin/bash
echo "$*" >> "$STUB_LOG"
rc=$(head -1 "$STUB_RC" 2>/dev/null); [ -n "$rc" ] && sed -i.bak '1d' "$STUB_RC"
if [ "${rc:-0}" = 0 ]; then echo "B VERIFIED (stub)"; else echo "B NOT VERIFIED (stub)"; fi
exit "${rc:-0}"
"""


def _rules(
    pre=PLAN_S,
    pre_rc=0,
    dry=PLAN_S,
    dry_rc=0,
    apply_rc=0,
    check_rc=(0,),
    replan=DONE_S,
    card=("0",),
    verify=None,
):
    rules = [
        {"match": r"--check-after > /tmp/settle_mma_(check|after)\.log", "out": f"tool exit {c}", "rc": c, "times": 1}
        for c in check_rc
    ]
    rules += [
        {"match": r"> /tmp/settle_mma_pre\.log 2>&1", "out": f"tool exit {pre_rc}", "rc": pre_rc},
        {"match": r"> /tmp/settle_mma_dry\.log 2>&1", "out": f"tool exit {dry_rc}", "rc": dry_rc},
        {"match": r"> /tmp/settle_mma_apply\.log 2>&1", "out": f"APPLIED\ntool exit {apply_rc}", "rc": apply_rc},
        {"match": r"> /tmp/settle_mma_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -oE 'PLAN .*' /tmp/settle_mma_pre\.log", "out": pre},
        {"match": r"grep -oE 'PLAN .*' /tmp/settle_mma_dry\.log", "out": dry},
        {"match": r"grep -oE 'PLAN .*' /tmp/settle_mma_replan\.log", "out": replan},
        {"match": r"grep -oE 'PLAN .*' /tmp/settle_mma_verify\.log", "out": verify or pre},
        {"match": r"postgres rm -f", "out": ""},
    ]
    rules += [{"match": r"postgres sh -c psql", "out": c, "times": 1} for c in card]
    return rules


def _vm(tmp_path, rules, airflow=None, b_rcs=(), ps_lines=(), stub_b=True):
    """A fake VM: the wrapper and its library copied to an ops dir beside a stub B (or the real B and merge
    wrappers), the reviewed list in AUSPEX_DIR, the scripted docker."""
    env = fd.install(tmp_path, rules, airflow)
    ops = tmp_path / "ops"
    ops.mkdir(exist_ok=True)
    for name in ("settle_mma_held_rows.sh", "_airflow_lib.sh"):
        shutil.copy(OPS / name, ops / name)
    if stub_b:
        (ops / "repair_stranded_recs_mma.sh").write_text(STUB_B)
    else:
        for name in ("repair_stranded_recs_mma.sh", "merge_mma_fighters.sh"):
            shutil.copy(OPS / name, ops / name)
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for f in DATA.glob("mma_*.csv"):
        shutil.copy(f, data / f.name)
    (tmp_path / "b_rc").write_text("".join(f"{rc}\n" for rc in b_rcs))
    (tmp_path / "ps.txt").write_text("".join(f"{line}\n" for line in ps_lines))
    env.update(
        STUB_LOG=str(tmp_path / "b_calls.log"),
        STUB_RC=str(tmp_path / "b_rc"),
        SETTLE_PS=f"cat {tmp_path / 'ps.txt'}",
    )
    return env, ops / "settle_mma_held_rows.sh"


def _run(tmp_path, rules, *args, **kw):
    env, script = _vm(tmp_path, rules, **{k: v for k, v in kw.items() if k != "extra_env"})
    env.update(kw.get("extra_env", {}))
    rc, out = fd.run(script, env, *args)
    return rc, out, fd.calls(tmp_path)


def _tool_calls(calls):
    return [c for c in calls if "settle_mma_held_rows.py" in c]


def _b_calls(tmp_path):
    log = tmp_path / "b_calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _backups(tmp_path):
    return fd.airflow_state(tmp_path).get("triggers", 0)


# ── the reviewed run ──────────────────────────────────────────────────


def test_a_clean_run_pins_the_plan_and_checks_the_after_state(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: Gatto v Kareckaite is recorded as a draw" in out, out
    assert _backups(tmp_path) == 1
    tools = _tool_calls(calls)
    logs = [t.split("> /tmp/")[1].split(".log")[0] for t in tools]
    assert logs == [
        "settle_mma_pre",
        "settle_mma_dry",
        "settle_mma_apply",
        "settle_mma_after",
        "settle_mma_replan",
    ]
    apply = tools[2]
    assert f"--apply --expect-plan {PIN} " in apply and "--archive /app/backups/settle_mma_held_rows_apply_" in apply
    assert "--check-after" in tools[3] and "--apply" not in tools[0] + tools[1] + tools[4]
    assert _b_calls(tmp_path) == ["--verify-b", "--verify-b"]
    # the backup lands before the dry run and the apply
    trigger = next(i for i, c in enumerate(calls) if "dags trigger db_backup_daily" in c)
    assert trigger < next(i for i, c in enumerate(calls) if "settle_mma_dry" in c)


def test_a_re_run_is_already_done_without_a_backup(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_S))
    assert rc == 0 and "ALREADY DONE" in out and "no backup" in out, out
    assert _backups(tmp_path) == 0
    assert [t for t in _tool_calls(calls) if "--apply" in t] == []
    assert any("--check-after" in t for t in _tool_calls(calls))


def test_a_done_plan_whose_after_state_fails_is_not_trusted(tmp_path):
    rc, out, _ = _run(tmp_path, _rules(pre=DONE_S, check_rc=(1,)))
    assert rc == 1 and "FAILED: the plan reads done" in out, out
    assert _backups(tmp_path) == 0


@pytest.mark.parametrize(
    "pre, pre_rc",
    [
        (PLAN_S.replace("record_predictions=2", "record_predictions=1"), 0),
        (PLAN_S.replace("refuse=0", "refuse=1"), 2),
        (PLAN_S, 1),  # the plan, but the run rolled back (ESPN disagreeing, a write outside the rows)
        ("", 2),
        (PLAN_S.replace("delete=1 delete_done=0", "delete=0 delete_done=1"), 0),  # the Machado row gone, Gatto not
    ],
)
def test_a_pre_check_that_is_neither_reviewed_nor_done_aborts_before_the_backup(tmp_path, pre, pre_rc):
    rc, out, calls = _run(tmp_path, _rules(pre=pre, pre_rc=pre_rc))
    assert rc == 1 and "ABORT at step 0" in out and "no backup" in out, out
    assert _backups(tmp_path) == 0 and not [t for t in _tool_calls(calls) if "--apply" in t]


RECORDED = PLAN_S.replace("record=1 record_done=0", "record=0 record_done=1")
GRADED = RECORDED.replace("record_predictions=2", "record_predictions=0")


@pytest.mark.parametrize("variant", [RECORDED, GRADED])
def test_the_draw_already_recorded_by_the_results_path_is_a_reviewed_variant_and_pins_the_run(tmp_path, variant):
    # A deep re-fetch with this commit's results path recorded the draw first (graded or not): same end state.
    rc, out, calls = _run(tmp_path, _rules(pre=variant, dry=variant))
    assert rc == 0 and "the reviewed variant" in out and "DONE: Gatto v Kareckaite" in out, out
    apply = [t for t in _tool_calls(calls) if "--apply" in t]
    assert len(apply) == 1 and f"--expect-plan {','.join(variant.split()[1:])} " in apply[0]


def test_a_variant_at_the_pre_check_pins_the_dry_run_to_it(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=GRADED, dry=PLAN_S))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not [t for t in _tool_calls(calls) if "--apply" in t]


def test_a_backup_that_does_not_land_aborts_before_the_dry_run(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"backup_fails": True})
    assert rc == 1 and "backup did not land" in out, out
    assert [t.split("> /tmp/")[1].split(".log")[0] for t in _tool_calls(calls)] == ["settle_mma_pre"]


@pytest.mark.parametrize("dry, dry_rc", [(PLAN_S.replace("teams_delete=1", "teams_delete=0"), 0), (PLAN_S, 1)])
def test_a_dry_run_other_than_the_reviewed_one_aborts_before_the_apply(tmp_path, dry, dry_rc):
    rc, out, calls = _run(tmp_path, _rules(dry=dry, dry_rc=dry_rc))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not [t for t in _tool_calls(calls) if "--apply" in t]


@pytest.mark.parametrize("apply_rc, why", [(2, "refused before writing"), (1, "it rolled back")])
def test_a_failed_apply_aborts(tmp_path, apply_rc, why):
    rc, out, calls = _run(tmp_path, _rules(apply_rc=apply_rc))
    assert rc == 1 and "ABORT at step 3" in out and why in out, out
    assert not [t for t in _tool_calls(calls) if "settle_mma_after" in t]


@pytest.mark.parametrize(
    "kw, b_rcs",
    [
        ({"check_rc": (1,)}, ()),
        ({"replan": PLAN_S}, ()),
        ({}, (0, 1)),  # B's done check fails afterwards
    ],
)
def test_a_wrong_after_state_fails(tmp_path, kw, b_rcs):
    rc, out, _ = _run(tmp_path, _rules(**kw), b_rcs=b_rcs)
    assert rc == 1 and "FAILED after-state at step 4" in out, out


# ── step 0: when it must not run ──────────────────────────────────────


def test_b_not_verified_aborts_before_any_tool_run(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), b_rcs=(1,))
    assert rc == 1 and "run B (and the fighter merge before it) is not done and verified" in out, out
    assert _tool_calls(calls) == [] and _backups(tmp_path) == 0


@pytest.mark.parametrize(
    "line",
    [
        "4242 bash /opt/auspex/scripts/ops/merge_mma_fighters.sh",
        "4243 bash /opt/auspex/scripts/ops/soccer_repair_runbook.sh",
        "4244 bash /opt/auspex/scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows",
    ],
)
def test_another_ops_run_on_the_host_refuses(tmp_path, line):
    ps = ["1 /sbin/init", "77 bash /opt/auspex/scripts/ops/settle_mma_held_rows.sh", line]
    rc, out, calls = _run(tmp_path, _rules(), ps_lines=ps)
    assert rc == 1 and "another ops run is in progress" in out and line in out, out
    assert _tool_calls(calls) == [] and _b_calls(tmp_path) == [] and _backups(tmp_path) == 0


def test_its_own_process_and_the_scan_itself_do_not_count(tmp_path):
    ps = [
        "77 bash /opt/auspex/scripts/ops/settle_mma_held_rows.sh",
        "78 tee -a /home/auspex/settle_mma_held_rows.log",
        "79 grep -E scripts/ops/|soccer_repair_runbook",
    ]
    rc, out, _ = _run(tmp_path, _rules(), ps_lines=ps)
    assert rc == 0, out


def test_an_apply_still_running_in_the_api_container_refuses(tmp_path):
    airflow = {"procs": ["python /app/scripts/backfill_results.py --sport mma --dates 2026-10-10 --apply"]}
    rc, out, calls = _run(tmp_path, _rules(), airflow=airflow)
    assert rc == 1 and "still running in the api container" in out and "backfill_results" in out, out
    assert _tool_calls(calls) == [] and _backups(tmp_path) == 0


def test_an_unreadable_process_scan_refuses(tmp_path):
    rc, out, _ = _run(tmp_path, _rules(), airflow={"proc_scan_fails": True})
    assert rc == 1 and "could not scan the api container" in out, out


@pytest.mark.parametrize("card", ["3", ""])
def test_a_ufc_card_in_progress_or_an_unreadable_check_refuses(tmp_path, card):
    rc, out, calls = _run(tmp_path, _rules(card=(card,)))
    assert rc == 1 and "a UFC card is in progress" in out, out
    assert _tool_calls(calls) == [] and _b_calls(tmp_path) == [] and _backups(tmp_path) == 0
    sql = (tmp_path / "fake" / "cp" / "settle_mma_q.sql").read_text()
    assert "NOW() - interval '8 hours' AND NOW() + interval '1 hour'" in sql and "l.sport = 'mma'" in sql


def test_another_run_holding_the_lock_refuses(tmp_path):
    env, script = _vm(tmp_path, _rules())
    holder = subprocess.Popen(["bash", "-c", "exec -a 'bash settle_mma_held_rows.sh' sleep 30"])
    try:
        time.sleep(0.2)
        lock = Path(env["AF_STATE_DIR"]) / "settle_mma_held_rows.lock"
        lock.mkdir(parents=True)
        (lock / "holder").write_text(f"{holder.pid} settle_mma_held_rows.sh\n")
        rc, out = fd.run(script, env)
    finally:
        holder.kill()
    assert rc == 1 and "another run holds" in out, out
    assert _tool_calls(fd.calls(tmp_path)) == [] and _backups(tmp_path) == 0


def test_the_lock_is_released_at_the_end(tmp_path):
    env, script = _vm(tmp_path, _rules())
    assert fd.run(script, env)[0] == 0
    assert not (Path(env["AF_STATE_DIR"]) / "settle_mma_held_rows.lock").exists()


def test_without_the_reviewed_list_nothing_runs(tmp_path):
    env, script = _vm(tmp_path, _rules())
    (tmp_path / "auspex" / "scripts" / "data" / "mma_held_rows_settle_20261102.csv").unlink()
    rc, out = fd.run(script, env)
    assert rc == 1 and "pull this commit first" in out, out
    assert [c for c in fd.calls(tmp_path) if "rm -f" not in c] == []  # only the exit trap's clean-up


def test_an_unknown_argument_is_refused(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), "--apply")
    assert rc == 2 and "usage" in out and calls == []


# ── --verify ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("plan, check_rc, rc_want", [(DONE_S, 0, 0), (PLAN_S, 0, 1), (DONE_S, 1, 1)])
def test_verify_is_read_only(tmp_path, plan, check_rc, rc_want):
    rc, out, calls = _run(tmp_path, _rules(verify=plan, check_rc=(check_rc,)), "--verify")
    assert rc == rc_want, out
    assert ("SETTLED VERIFIED" in out) == (rc_want == 0)
    assert _backups(tmp_path) == 0 and not [t for t in _tool_calls(calls) if "--apply" in t]
    assert _b_calls(tmp_path) == [] and not [c for c in calls if "psql" in c]


# ── with the real run-B and merge wrappers ────────────────────────────

MERGE_STATE = (
    "/app/backups/merge_mma_fighters_last_apply.env",
    "PLAN=/app/backups/m.plan.json\nGRADE_IDS=\nSTATUS=done\n",
)
DONE_M = (
    "PLAN pairs_merge=0 pairs_done=9 pairs_refuse=0 folds=0 folds_reversed=0 folds_done=1 folds_refuse=0 "
    "repoints=0 left_rec_held=0 odds_moved=0 snapshots_moved=0 predictions_moved=0 predictions_dropped=0 "
    "espn_ids_adopted=0 athlete_ids_stamped=0 nfc_renamed=0 nfc_refuse=0"
)
DONE_B = (
    "PLAN merge_delete=0 merge_cancel=0 delete_no_twin=0 delete_placeholder=0 mark_postponed=0 left=9 "
    "blocked_pending_recs=0 list_mismatch=0"
)


def test_with_the_real_b_and_merge_wrappers_before_and_after(tmp_path):
    rules = _rules(card=())
    rules[0:0] = [
        {"match": r"> /tmp/merge_mma_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -oE 'PLAN .*' /tmp/merge_mma_pre\.log", "out": DONE_M},
        {"match": r"> /tmp/stranded_mma_b_pre\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -oE 'PLAN .*' /tmp/stranded_mma_b_pre\.log", "out": DONE_B, "times": 1},
        {"match": r"grep -oE 'PLAN .*' /tmp/stranded_mma_b_pre\.log", "out": DONE_B.replace("left=9", "left=7")},
    ]
    # psql, in order: the card check; B at step 0 (A's reviewed recs, stranded|recs, stale|recs, the held rows);
    # B at step 4 (the same four, the held rows now settled).
    for out in ("0", "21", "0|49", "9|49", "held|held", "21", "0|49", "7|49", "settled|settled"):
        rules.append({"match": r"postgres sh -c psql", "out": out, "times": 1})
    env, script = _vm(tmp_path, rules, stub_b=False)
    fd.put_file(tmp_path, "api", *MERGE_STATE)
    rc, out = fd.run(script, env)
    assert rc == 0 and out.count("B VERIFIED") == 2 and "MERGE VERIFIED" in out, out
    assert "held rows (draw|replaced row) held|held" in out and "settled as decided 2026-11-02" in out, out
    assert "DONE: Gatto v Kareckaite" in out and _backups(tmp_path) == 1
