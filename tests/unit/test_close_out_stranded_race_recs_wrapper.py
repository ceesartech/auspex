"""scripts/ops/close_out_stranded_race_recs.sh (its second run: the 7 Southwell recs), run for real against a
scripted docker: the read-only preconditions (its lock, no other scripts/ops process, no apply still running in the
api container, no UFC card in progress), the read-only 'already done' / 'not yet' exits, the gate (7 candidates,
3 races, 7 settle, 7 void, nothing unmatched or errored, adding up, every race one of the three reviewed, equal to
the before-state), every tool run of a pass measured from one instant (--as-of), an apply pinned to the dry run's
exact counts (every key), the exact after-state, PARTIAL when recs are still inside the drain's window, and the
aborts before any write. Stranded recs on races at or after the pin (BEYOND_PIN) are named, never touched, and keep
a re-run from reporting ALREADY DONE; so do recs still in the window (IN_WINDOW)."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
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

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("ps") is None, reason="needs bash, ps")

PLAN = "PLAN candidates=7 races=3 settle=0 void=7 unmatched=0 errors=0"
ZERO = "PLAN candidates=0 races=0 settle=0 void=0 unmatched=0 errors=0"
NO_WINDOW = "IN_WINDOW recs=0 races=0 until=none"
REVIEWED = "rac_32302997831 rac_32303001705 rac_32303003642"
RID = "00000000-0000-0000-0000-{:012d}"
VOIDS7 = ",".join(f"{RID.format(i)}:void:0.00" for i in range(7))


def _rules(
    pre=PLAN,
    dry=PLAN,
    after=ZERO,
    as_planned="7",
    finished=None,
    states=("7|2667", "0|2667"),
    apply_rc=0,
    dry_rc=0,
    pre_rc=0,
    triples=VOIDS7,
    races="",
    beyond_pre="BEYOND_PIN recs=0",
    beyond_after="BEYOND_PIN recs=0",
    window_pre=NO_WINDOW,
    window_after=NO_WINDOW,
    ids_pre=REVIEWED,
    ids_dry=REVIEWED,
    ufc="0|none",
):
    psql = [ufc, states[0], as_planned] + ([finished] if races else []) + [states[1]]
    return [
        {"match": r"/tmp/race_closeout_pre\.log 2>&1", "out": f"tool exit {pre_rc}", "rc": pre_rc},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/race_closeout_pre\.log", "out": pre},
        {"match": r"grep -o 'BEYOND_PIN recs=\[0-9\]\*' /tmp/race_closeout_pre\.log", "out": beyond_pre},
        {"match": r"grep -o 'BEYOND_PIN recs=\[0-9\]\*' /tmp/race_closeout_verify\.log", "out": beyond_after},
        {"match": r"grep -o 'IN_WINDOW recs=.*/tmp/race_closeout_pre\.log", "out": window_pre},
        {"match": r"grep -o 'IN_WINDOW recs=.*/tmp/race_closeout_verify\.log", "out": window_after},
        {"match": r"grep 'beyond pin:' /tmp/race_closeout_pre\.log", "out": "  beyond pin: rec r1  2026-11-01T13:23 X"},
        {"match": r"grep 'in window:' /tmp/race_closeout_pre\.log", "out": "  in window: rec w1  2026-10-31T19:25 Y"},
        {"match": r"racing_api_race_id.*closed_out_race_recs_precheck_", "out": ids_pre},
        {"match": r"racing_api_race_id.*closed_out_race_recs_dryrun_", "out": ids_dry},
        {"match": r"/tmp/race_closeout_dry\.log 2>&1", "out": f"tool exit {dry_rc}", "rc": dry_rc},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/race_closeout_dry\.log", "out": dry},
        {"match": r"/tmp/race_closeout_apply\.log 2>&1", "out": f"tool exit {apply_rc}", "rc": apply_rc},
        {"match": r"api python -c import csv, sys; rows = \[r for r", "out": triples},
        {"match": r"api python -c import csv, sys; print\(','\.join\(sorted", "out": races},
        {"match": r"/tmp/race_closeout_verify\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/race_closeout_verify\.log", "out": after},
        *({"match": r"postgres sh -c psql", "out": out, "times": 1} for out in psql),
        {"match": r"postgres rm -f", "out": ""},
    ]


def _install_ps(tmp_path):
    """A `ps` whose process list is the real one minus every other scripts/ops process (a parallel test run's), plus
    the lines of fake/ps_extra; fake/ps_fail makes the list fail. `ps -p` (the lock's liveness check) is real."""
    real = shutil.which("ps")
    script = tmp_path / "bin" / "ps"
    script.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-eo" ]; then\n'
        '  [ -f "$FAKE_DIR/ps_fail" ] && exit 1\n'
        f"  {real} \"$@\" | awk '!/scripts\\/ops\\// || /close_out_stranded_race_recs\\.sh/'\n"
        '  [ -f "$FAKE_DIR/ps_extra" ] && cat "$FAKE_DIR/ps_extra"\n'
        "  exit 0\n"
        "fi\n"
        f'exec {real} "$@"\n'
    )
    script.chmod(0o755)


def _run(tmp_path, rules, airflow=None, ps_extra=None, ps_fail=False):
    env = fd.install(tmp_path, rules, airflow=airflow)
    _install_ps(tmp_path)
    if ps_extra:
        (tmp_path / "fake" / "ps_extra").write_text(ps_extra + "\n")
    if ps_fail:
        (tmp_path / "fake" / "ps_fail").write_text("")
    rc, out = fd.run(WRAPPER, env)
    return rc, out, fd.calls(tmp_path)


def _tool_runs(calls, flag="--apply"):
    return [c for c in calls if "close_out_stranded_race_recs.py" in c and flag in c]


def _no_write(tmp_path, calls):
    return fd.airflow_state(tmp_path).get("triggers", 0) == 0 and not _tool_runs(calls)


def _sent_sql(tmp_path):
    sent = sorted(
        (tmp_path / "fake" / "cp").glob("[0-9]*_race_closeout_q.sql"), key=lambda f: int(f.name.split("_")[0])
    )
    return [f.read_text() for f in sent]


def _as_of(call):
    (at,) = re.findall(r"--as-of (\S+)", call)
    return at


def test_the_second_run_voids_the_seven_pinned_to_one_instant_and_checks_the_exact_after_state(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: 0 rec(s) settled" in out and "7 voided" in out, out
    assert "PARTIAL" not in out and "NOTE" not in out
    assert not [c for c in fd.calls(tmp_path) if c.startswith("UNMATCHED")]
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=7,races=3,settle=0,void=7,unmatched=0,errors=0" in apply_
    assert re.search(r"--out-csv /app/backups/closed_out_race_recs_apply_\d{8}T\d{6}Z\.csv", apply_)
    # Every tool run is scoped to the reviewed set and measured from a given instant; the dry run, the apply and
    # the re-plan from the same T0, which the before-state query uses too.
    runs = [c for c in calls if "close_out_stranded_race_recs.py" in c]
    assert len(runs) == 4 and all("--raced-before 2026-11-01T00:00:00Z" in c for c in runs)
    pre, dry, apply_run, verify = runs
    assert "race_closeout_pre" in pre and "race_closeout_verify" in verify
    t0 = _as_of(dry)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", t0) and _as_of(apply_run) == _as_of(verify) == t0
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", _as_of(pre))
    assert fd.airflow_state(tmp_path)["triggers"] == 1
    ufc, state, as_planned, state_after = _sent_sql(tmp_path)  # no settled race: no 'finished' query
    assert "l.sport = 'mma'" in ufc and "INTERVAL '6 hours'" in ufc and "INTERVAL '1 hour'" in ufc
    assert state == state_after and "make_interval(days => 3)" in state
    assert f"TIMESTAMPTZ '{t0}' - make_interval" in state and "TIMESTAMPTZ '2026-11-01T00:00:00Z'" in state
    assert f"('{RID.format(6)}'::uuid,'void',0.00::numeric)" in as_planned
    assert "rec.status = v.status AND rec.profit_loss = v.pl AND rec.settled_at IS NOT NULL" in as_planned


def test_a_result_posted_late_settles_its_race_and_is_checked_finished(tmp_path):
    plan = "PLAN candidates=7 races=3 settle=4 void=3 unmatched=0 errors=0"
    triples = ",".join([f"{RID.format(i)}:lost:-25.00" for i in range(3)] + [f"{RID.format(3)}:won:26.68"])
    triples += "," + ",".join(f"{RID.format(10 + i)}:void:0.00" for i in range(3))
    race = "11111111-0000-0000-0000-000000000001"
    rc, out, calls = _run(tmp_path, _rules(pre=plan, dry=plan, triples=triples, races=race, finished="1"))
    assert rc == 0 and "DONE: 4 rec(s) settled" in out and "3 voided" in out, out
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=7,races=3,settle=4,void=3,unmatched=0,errors=0" in apply_
    finished = _sent_sql(tmp_path)[3]
    assert finished.count("::uuid") == 1 and "status = 'finished'" in finished


def test_an_early_run_closes_what_left_the_window_and_says_when_to_rerun(tmp_path):
    # 2026-11-03 18:00Z: the 17:40 race (4 recs) is out of the drain's window, 18:50 and 19:25 are not.
    early = "PLAN candidates=4 races=1 settle=0 void=4 unmatched=0 errors=0"
    window = "IN_WINDOW recs=3 races=2 until=2026-11-03T19:25:00+00:00"
    voids4 = ",".join(f"{RID.format(i)}:void:0.00" for i in range(4))
    rules = _rules(
        pre=early,
        dry=early,
        as_planned="4",
        states=("4|2667", "0|2667"),
        triples=voids4,
        window_pre=window,
        window_after=window,
        ids_pre="rac_32302997831",
        ids_dry="rac_32302997831",
    )
    rc, out, calls = _run(tmp_path, rules)
    assert rc == 3, out
    assert "DONE: 0 rec(s) settled from their race's result, 4 voided" in out
    assert "NOTE: 3 rec(s) on 2 race(s) before 2026-11-01T00:00:00Z are still inside the drain's 3-day window" in out
    assert "in window: rec w1" in out  # named before anything is written
    assert "PARTIAL: 3 rec(s) on 2 race(s)" in out and "re-run this script after 2026-11-03T19:25:00+00:00" in out
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=4,races=1,settle=0,void=4,unmatched=0,errors=0" in apply_


def test_a_later_run_finishes_the_rest(tmp_path):
    late = "PLAN candidates=3 races=2 settle=0 void=3 unmatched=0 errors=0"
    voids3 = ",".join(f"{RID.format(i)}:void:0.00" for i in range(3))
    two = "rac_32303001705 rac_32303003642"
    rules = _rules(
        pre=late, dry=late, as_planned="3", states=("3|2667", "0|2667"), triples=voids3, ids_pre=two, ids_dry=two
    )
    rc, out, calls = _run(tmp_path, rules)
    assert rc == 0 and "DONE: 0 rec(s) settled from their race's result, 3 voided" in out and "PARTIAL" not in out
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=3,races=2,settle=0,void=3,unmatched=0,errors=0" in apply_


def test_nothing_out_of_the_window_yet_is_not_yet_without_a_backup_or_a_write(tmp_path):
    window = "IN_WINDOW recs=7 races=3 until=2026-11-03T19:25:00+00:00"
    rc, out, calls = _run(tmp_path, _rules(pre=ZERO, window_pre=window))
    assert rc == 3 and "NOT YET" in out and "re-run after 2026-11-03T19:25:00+00:00" in out, out
    assert "ALREADY DONE" not in out and "in window: rec w1" in out
    assert _no_write(tmp_path, calls) and not _tool_runs(calls, "race_closeout_dry")


def test_nothing_left_exits_0_without_a_backup_or_a_write(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=ZERO))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert "NOTE" not in out and "NEW PIN" not in out and "NOT YET" not in out
    assert _no_write(tmp_path, calls)


def test_stranded_recs_beyond_the_pin_are_never_already_done(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=ZERO, beyond_pre="BEYOND_PIN recs=3"))
    assert rc == 1 and "NEW PIN NEEDED" in out and "ALREADY DONE" not in out, out
    assert "3 stranded race rec(s) sit on races at or after 2026-11-01T00:00:00Z" in out
    assert "beyond pin: rec r1" in out  # the operator sees which, once
    assert not [c for c in calls if c.startswith("UNMATCHED")]
    assert _no_write(tmp_path, calls)


def test_the_reviewed_set_still_closes_with_recs_beyond_the_pin_named(tmp_path):
    rules = _rules(beyond_pre="BEYOND_PIN recs=3", beyond_after="BEYOND_PIN recs=3")
    rc, out, calls = _run(tmp_path, rules)
    assert rc == 0 and "DONE: 0 rec(s) settled" in out, out
    (apply_,) = _tool_runs(calls)
    assert "--expect-plan candidates=7,races=3,settle=0,void=7,unmatched=0,errors=0" in apply_
    assert out.count("3 stranded race rec(s)") == 2 and "a re-run exits 1 (NEW PIN NEEDED)" in out


@pytest.mark.parametrize(
    "kw",
    [
        {"dry": "PLAN candidates=8 races=3 settle=0 void=8 unmatched=0 errors=0"},  # more candidates than reviewed
        {"dry": "PLAN candidates=8 races=3 settle=4 void=4 unmatched=0 errors=0"},  # ... with settle and void inside
        {"dry": "PLAN candidates=7 races=4 settle=0 void=7 unmatched=0 errors=0"},  # more races
        {"dry": "PLAN candidates=8 races=3 settle=8 void=0 unmatched=0 errors=0"},  # more settles
        {"dry": "PLAN candidates=7 races=3 settle=0 void=6 unmatched=1 errors=0"},  # unmatched
        {"dry": "PLAN candidates=7 races=3 settle=0 void=6 unmatched=0 errors=1"},  # an API error
        {"dry": "PLAN candidates=7 races=3 settle=0 void=6 unmatched=0 errors=0"},  # does not add up
        {"dry": "PLAN candidates=6 races=3 settle=0 void=6 unmatched=0 errors=0"},  # not the before-state's 7
        {"dry": "PLAN candidates=7 races=3 settle=0 void=7 unmatched=0"},  # unreadable
        {"ids_dry": "rac_32302997831 rac_32303001705 rac_99999999999"},  # a race nobody reviewed
        {"ids_dry": "rac_32302997831 rac_32303001705"},  # fewer races than the plan says
        {"ids_dry": ""},  # the dry-run CSV could not be read
        {"ids_dry": "rac_32302997831 rac_32303001705 rac_3230300364'"},  # not an id
        {"dry_rc": 1},  # the dry run failed
        {"states": ("x|2667", "0|2667")},  # the before-state query failed
    ],
)
def test_the_dry_run_gate_aborts_before_any_apply(tmp_path, kw):
    rc, out, calls = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not _tool_runs(calls)


def test_a_dry_run_that_plans_nothing_after_a_precheck_that_did_aborts(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(dry=ZERO, states=("0|2667", "0|2667")))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not _tool_runs(calls)


@pytest.mark.parametrize(
    "kw",
    [
        {"pre": "PLAN candidates=8 races=3 settle=0 void=8 unmatched=0 errors=0"},
        {"pre": "PLAN candidates=7 races=3 settle=0 void=6 unmatched=0 errors=0"},
        {"ids_pre": "rac_1 rac_2 rac_3"},
    ],
)
def test_a_precheck_outside_the_gate_aborts_before_the_backup(tmp_path, kw):
    rc, out, calls = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert _no_write(tmp_path, calls) and not _tool_runs(calls, "race_closeout_dry")


@pytest.mark.parametrize(
    "kw",
    [
        {"as_planned": "6"},  # one rec not as planned
        {"states": ("7|2667", "1|2667")},  # the stranded count fell by less than planned
        {"states": ("7|2667", "0|2666")},  # a race rec disappeared
        {"after": PLAN},  # something left to do
        {"triples": "not a csv;drop table"},  # unreadable apply CSV
        {"beyond_after": ""},  # the verify run's BEYOND_PIN line is missing
        {"window_after": ""},  # the verify run's IN_WINDOW line is missing
    ],
)
def test_a_wrong_after_state_fails(tmp_path, kw):
    rc, out, _ = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "FAILED after-state" in out, out


def test_a_settled_race_not_finished_fails_the_after_state(tmp_path):
    plan = "PLAN candidates=7 races=3 settle=1 void=6 unmatched=0 errors=0"
    triples = f"{RID.format(0)}:lost:-4.68," + ",".join(f"{RID.format(10 + i)}:void:0.00" for i in range(6))
    rules = _rules(pre=plan, dry=plan, triples=triples, races="11111111-0000-0000-0000-000000000001", finished="0")
    rc, out, _ = _run(tmp_path, rules)
    assert rc == 1 and "FAILED after-state" in out, out


def test_a_failed_apply_aborts_at_step_3(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(apply_rc=1))
    assert rc == 1 and "ABORT at step 3" in out, out
    assert len(_tool_runs(calls)) == 1 and not [c for c in calls if "race_closeout_verify" in c]


@pytest.mark.parametrize(
    "kw",
    [
        {"beyond_pre": ""},
        {"beyond_pre": "BEYOND_PIN recs=x"},
        {"window_pre": ""},
        {"window_pre": "IN_WINDOW recs=3 races=2"},  # no until
        {"pre": "PLAN candidates=7 races=3 settle=0 void=6 unmatched=0 errors=1"},
        {"pre": "PLAN candidates=7 races=3 settle=0 void=6 unmatched=1 errors=0"},
        {"pre": ""},
        {"pre_rc": 2},  # e.g. an --as-of the database's clock has not reached
    ],
)
def test_an_unreadable_or_failed_precheck_aborts_before_the_backup(tmp_path, kw):
    rc, out, calls = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert _no_write(tmp_path, calls)


def test_a_backup_that_does_not_land_aborts_before_any_write(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"backup_fails": True})
    assert rc == 1 and "backup did not land" in out, out
    assert not _tool_runs(calls) and not _tool_runs(calls, "race_closeout_dry")


# ── preconditions ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "line",
    [
        "4242 bash /opt/auspex/scripts/ops/soccer_repair_runbook.sh",
        "4243 /bin/bash scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows",
    ],
)
def test_another_ops_process_refuses_before_anything(tmp_path, line):
    rc, out, calls = _run(tmp_path, _rules(), ps_extra=line)
    assert rc == 1 and "REFUSED: another scripts/ops process" in out and line in out, out
    assert _no_write(tmp_path, calls) and not [c for c in calls if "close_out_stranded_race_recs.py" in c]


def test_its_own_processes_do_not_count_as_another_ops_run(tmp_path):
    extra = "4244 bash -c bash /opt/auspex/scripts/ops/close_out_stranded_race_recs.sh 2>&1 | tee -a x.log"
    rc, out, _ = _run(tmp_path, _rules(), ps_extra=extra)
    assert rc == 0 and "DONE" in out, out


def test_an_unreadable_process_list_refuses(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), ps_fail=True)
    assert rc == 1 and "REFUSED: cannot list this host's processes" in out, out
    assert _no_write(tmp_path, calls)


@pytest.mark.parametrize(
    "proc",
    [
        "python /app/scripts/close_out_stranded_race_recs.py --raced-before 2026-11-01T00:00:00Z --apply --x",
        "python /app/scripts/merge_mma_fighters.py --apply --expect-plan /app/backups/p.json",
    ],
)
def test_an_apply_still_running_in_the_api_container_refuses(tmp_path, proc):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"procs": [proc]})
    assert rc == 1 and "REFUSED: an apply is still running in the api container" in out and proc in out, out
    assert _no_write(tmp_path, calls)
    (scan,) = [c for c in fd.calls(tmp_path, scans=True) if fd.is_scan(c)]
    assert "*/app/scripts/close_out_stranded_race_recs[.]py*--apply*" in scan


def test_a_failed_process_scan_refuses(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), airflow={"proc_scan_fails": True})
    assert rc == 1 and "REFUSED: could not scan the api container's processes" in out, out
    assert _no_write(tmp_path, calls)


@pytest.mark.parametrize("ufc, why", [("2|2026-11-07T22:00Z", "a UFC card is in progress"), ("", "could not read")])
def test_a_ufc_card_in_progress_or_unreadable_refuses(tmp_path, ufc, why):
    rc, out, calls = _run(tmp_path, _rules(ufc=ufc))
    assert rc == 1 and f"REFUSED: {why}" in out, out
    assert _no_write(tmp_path, calls) and not [c for c in calls if "close_out_stranded_race_recs.py" in c]


def test_a_live_run_holding_the_lock_refuses(tmp_path):
    holder = subprocess.Popen(["bash", "-c", "sleep 30; :", "close_out_stranded_race_recs.sh"])  # "; :" keeps bash
    try:
        lock = tmp_path / "afstate" / "close_out_stranded_race_recs.lock"
        lock.mkdir(parents=True)
        (lock / "holder").write_text(f"{holder.pid} close_out_stranded_race_recs.sh\n")
        rc, out, calls = _run(tmp_path, _rules())
    finally:
        holder.kill()
        holder.wait()
    assert rc == 1 and "REFUSED: another run holds" in out, out
    assert not calls  # not one docker call
    assert (lock / "holder").read_text().startswith(str(holder.pid))  # the holder's lock is left alone


def test_the_lock_is_released_on_every_exit(tmp_path):
    rc, out, _ = _run(tmp_path, _rules(pre=ZERO))
    assert rc == 0, out
    assert not (tmp_path / "afstate" / "close_out_stranded_race_recs.lock").exists()


def test_the_wrapper_and_the_tool_share_the_drain_window():
    lra = (REPO / "scripts" / "load_racing_api.py").read_text()
    wrapper = WRAPPER.read_text()
    (days,) = re.findall(r"^DRAIN_DAYS = (\d+)$", lra, re.M)
    assert f"DRAIN_DAYS={days}  # load_racing_api.DRAIN_DAYS" in wrapper
