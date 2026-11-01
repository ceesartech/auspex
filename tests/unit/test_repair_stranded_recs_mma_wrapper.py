"""scripts/ops/repair_stranded_recs_mma.sh, run for real against a scripted
docker: the exact reviewed plans for the results backfill (A) and the orphan
cleanup (B), the reviewed rec outcomes checked in the dry run's CSV and in the
database after the apply, both applies pinned to every key, the exact
after-states, the follow-ups (exit 2 when one fails, --followups-only), and
the 'already done' exits — neither a backup nor a write."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WRAPPER = REPO / "scripts" / "ops" / "repair_stranded_recs_mma.sh"
REVIEWED = REPO / "scripts" / "data" / "mma_stranded_recs_20261101.csv"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN_A = (
    "PLAN results=56 flipped=44 inserted=0 cancelled=3 stamped=12 rescored=0 other=0 recs_settled=20 "
    "recs_voided=1 recs_other=0 recs_left=0 fetch_failed=0 identity_refused=0"
)
DONE_A = (
    "PLAN results=56 flipped=0 inserted=0 cancelled=0 stamped=0 rescored=0 other=0 recs_settled=0 "
    "recs_voided=0 recs_other=0 recs_left=0 fetch_failed=0 identity_refused=0"
)
PLAN_B = (
    "PLAN merge_delete=31 merge_cancel=0 delete_no_twin=42 delete_placeholder=4 mark_postponed=0 left=7 "
    "blocked_pending_recs=0 list_mismatch=0"
)
DONE_B = (
    "PLAN merge_delete=0 merge_cancel=0 delete_no_twin=0 delete_placeholder=0 mark_postponed=0 left=7 "
    "blocked_pending_recs=0 list_mismatch=0"
)


def _rules(
    pre_a=PLAN_A,
    dry_a=PLAN_A,
    verify_a=DONE_A,
    pre_b=PLAN_B,
    dry_b=PLAN_B,
    verify_b=DONE_B,
    check="OK 21",
    states=("21|49|21", "21", "0|49|0", "84|49", "7|49"),
    apply_a_rc=0,
    features_rc=0,
):
    rules = [
        {"match": r"> /tmp/stranded_mma_a_apply\.log 2>&1", "out": f"tool exit {apply_a_rc}", "rc": apply_a_rc},
        {
            "match": r"compute_features_mma\.py --all-finished --force",
            "out": "Wrote 1985 MMA feature rows",
            "rc": features_rc,
        },
        {"match": r"> /tmp/stranded_mma_\w+\.log 2>&1", "out": "tool exit 0"},
    ]
    for name, plan in (
        ("a_pre", pre_a),
        ("a_dry", dry_a),
        ("a_verify", verify_a),
        ("b_pre", pre_b),
        ("b_dry", dry_b),
        ("b_verify", verify_b),
    ):
        rules.append({"match": rf"grep -oE 'PLAN .*' /tmp/stranded_mma_{name}\.log", "out": plan})
    rules += [
        {"match": r"api python -c import csv, json, sys", "out": check},
        {"match": r"^compose exec -T api python -$", "out": "graded 22 predictions on 14 finished UFC rows"},
        {"match": r"postgres rm -f", "out": ""},
    ]
    rules += [{"match": r"postgres sh -c psql", "out": s, "times": 1} for s in states]
    return rules


def _run(tmp_path, rules, *args):
    env = fd.install(tmp_path, rules)
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy(REVIEWED, data / REVIEWED.name)
    rc, out = fd.run(WRAPPER, env, *args)
    return rc, out, fd.calls(tmp_path)


def _applies(calls, tool):
    return [c for c in calls if tool in c and "--apply" in c]


def _backups(tmp_path):
    return fd.airflow_state(tmp_path).get("triggers", 0)


def test_a_clean_run_pins_both_plans_checks_every_rec_and_runs_the_follow_ups(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: the 21 stranded UFC recs are closed out" in out, out
    assert _backups(tmp_path) == 1
    (apply_a,) = _applies(calls, "backfill_results.py")
    pin = PLAN_A.replace("PLAN ", "").replace(" ", ",")
    assert f"--expect-plan {pin}" in apply_a and "--sport mma" in apply_a
    assert "--dates 2026-06-06,2026-06-14,2026-06-20,2026-06-27,2026-07-18" in apply_a
    (apply_b,) = _applies(calls, "repair_fixture_orphans.py")
    assert "--as-of 2026-11-01T00:00:00Z" in apply_b
    assert "--expect-plan merge_delete=31,merge_cancel=0,delete_no_twin=42,delete_placeholder=4,mark_postponed=0" in (
        apply_b
    )
    assert calls.index(apply_a) < calls.index(apply_b)
    assert any("compute_features_mma.py --all-finished --force" in c for c in calls)
    sent = sorted((tmp_path / "fake" / "cp").glob("[0-9]*_stranded_mma_q.sql"), key=lambda f: int(f.name.split("_")[0]))
    reviewed_sql = sent[1].read_text()  # before-state, reviewed check, after-state, B before, B after
    assert reviewed_sql.count("::uuid, '") == 21
    assert "('3f56282f-7cd9-4f27-8d90-02460593c61c'::uuid, 'lost', -67.72::numeric, 'lost')" in reviewed_sql
    assert "('d9da951f-c6a9-47ce-bb26-e04112e0f443'::uuid, 'void', 0.00::numeric, 'void: match cancelled')" in (
        reviewed_sql
    )
    assert "m.status = CASE WHEN v.status = 'void' THEN 'cancelled' ELSE 'finished' END" in reviewed_sql
    assert "reviewed_pending" in sent[0].read_text() and "status = 'pending'" in sent[0].read_text()


def test_nothing_left_to_write_exits_0_without_a_backup(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, pre_b=DONE_B))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert _backups(tmp_path) == 0 and not [c for c in calls if "--apply" in c]


def test_with_the_backfill_done_only_the_cleanup_runs(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, states=("84|49", "7|49")))
    assert rc == 0 and "A is done" in out and "DONE" in out, out
    assert _backups(tmp_path) == 1
    assert not _applies(calls, "backfill_results.py") and len(_applies(calls, "repair_fixture_orphans.py")) == 1


@pytest.mark.parametrize(
    "dry_a",
    [
        PLAN_A.replace("cancelled=3", "cancelled=0"),  # code without cancel_dropped_bouts
        PLAN_A.replace("recs_settled=20", "recs_settled=19"),
        PLAN_A.replace("fetch_failed=0", "fetch_failed=1"),
        PLAN_A.replace("recs_left=0", "recs_left=1"),
        PLAN_A.replace(" identity_refused=0", ""),  # unreadable
    ],
)
def test_a_backfill_plan_that_is_not_the_reviewed_one_aborts_before_any_apply(tmp_path, dry_a):
    rc, out, calls = _run(tmp_path, _rules(dry_a=dry_a))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not [c for c in calls if "--apply" in c]


def test_a_reviewed_rec_settled_since_the_review_aborts_before_the_dry_run(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(states=("20|49|20",)))
    assert rc == 1 and "only 20 of the 21 reviewed recs are still pending" in out, out
    assert not [c for c in calls if "stranded_mma_a_dry" in c]


def test_rec_outcomes_other_than_the_reviewed_ones_abort(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(check="MISMATCH 3f56282f-7cd9-4f27-8d90-02460593c61c"))
    assert rc == 1 and "not the reviewed ones (MISMATCH" in out, out
    assert not [c for c in calls if "--apply" in c]


def test_a_failed_backfill_apply_aborts(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(apply_a_rc=1))
    assert rc == 1 and "ABORT at step 3" in out, out
    assert not _applies(calls, "repair_fixture_orphans.py")


@pytest.mark.parametrize(
    "kw",
    [
        {"states": ("21|49|21", "20", "0|49|0")},  # one rec not as reviewed
        {"states": ("21|49|21", "21", "1|49|0")},  # a UFC rec still stranded
        {"states": ("21|49|21", "21", "0|48|0")},  # a rec disappeared
        {"verify_a": DONE_A.replace("flipped=0", "flipped=1")},  # something left to write
    ],
)
def test_a_wrong_backfill_after_state_fails_before_the_cleanup(tmp_path, kw):
    rc, out, calls = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "FAILED after-state at step 4" in out, out
    assert not [c for c in calls if "stranded_mma_b_dry" in c]


@pytest.mark.parametrize(
    "dry_b",
    [
        PLAN_B.replace("delete_no_twin=42", "delete_no_twin=43"),
        PLAN_B.replace("left=7", "left=8"),
        PLAN_B.replace("blocked_pending_recs=0", "blocked_pending_recs=1"),
    ],
)
def test_a_cleanup_plan_that_is_not_the_reviewed_one_aborts_before_its_apply(tmp_path, dry_b):
    rc, out, calls = _run(tmp_path, _rules(dry_b=dry_b))
    assert rc == 1 and "ABORT at step 5" in out, out
    assert not _applies(calls, "repair_fixture_orphans.py")


@pytest.mark.parametrize(
    "kw",
    [
        {"states": ("21|49|21", "21", "0|49|0", "84|49", "8|49")},
        {"states": ("21|49|21", "21", "0|49|0", "84|49", "7|48")},
        {"verify_b": DONE_B.replace("merge_delete=0", "merge_delete=1")},
    ],
)
def test_a_wrong_cleanup_after_state_fails(tmp_path, kw):
    rc, out, _ = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "FAILED after-state at step 7" in out, out


def test_a_failed_follow_up_exits_2_and_followups_only_reruns_just_them(tmp_path):
    rc, out, _ = _run(tmp_path, _rules(features_rc=1))
    assert rc == 2 and "INCOMPLETE" in out and "--followups-only" in out, out
    fd.set_rules(tmp_path, _rules())
    (tmp_path / "fake" / "calls.log").unlink()
    rc, out = fd.run(WRAPPER, {**fd.install(tmp_path, _rules())}, "--followups-only")
    calls = fd.calls(tmp_path)
    assert rc == 0 and "FOLLOW-UPS DONE" in out, out
    assert not [c for c in calls if "--apply" in c or "backfill_results" in c or "airflow" in c]


def test_without_the_reviewed_file_nothing_runs(tmp_path):
    env = fd.install(tmp_path, _rules())
    rc, out = fd.run(WRAPPER, env)
    assert rc == 1 and "is missing" in out, out
    assert not [c for c in fd.calls(tmp_path) if "/app/scripts/" in c]
