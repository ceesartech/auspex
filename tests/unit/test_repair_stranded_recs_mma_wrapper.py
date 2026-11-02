"""scripts/ops/repair_stranded_recs_mma.sh, run for real against a scripted
docker. The default run (A, the results backfill): the exact reviewed plan,
every rec's outcome pinned (--expect-recs on the dry run and the apply, the
dry run's CSV, the database after the apply), the exact after-state, the
follow-ups (exit 2 when one fails, --followups-only), and the 'already done'
exit — only after A's committed state is verified in the database, never on a
0-write plan alone (a re-run after a failed step 4 refuses). The opt-in
--retire-stale-rows run (B, the orphan cleanup): only once A is done and
verified, the exact reviewed plan, the pinned apply, the exact after-state,
its own 'already done'; B also refuses unless the fighter merge
(merge_mma_fighters.sh --verify) is done and verified, and unless its dry
run plans exactly the reviewed rows of mma_stale_rows_retire_20261101.csv
and holds back exactly the rows of mma_stale_rows_hold_20261101.csv
(re-pinned to the post-merge state: 31 no-twin rows, not 42; 2 held for the
user's decision); its apply carries --expect-rows and its own audit CSV is
checked against both lists at step 7. --verify-a
checks A's committed state read only. Anything at step 0 that is neither the
reviewed nor the done plan aborts before the backup."""

from __future__ import annotations

import csv
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WRAPPER = REPO / "scripts" / "ops" / "repair_stranded_recs_mma.sh"
REVIEWED = REPO / "scripts" / "data" / "mma_stranded_recs_20261101.csv"
DATA_FILES = (
    "mma_stranded_recs_20261101.csv",
    "mma_stale_rows_retire_20261101.csv",
    "mma_stale_rows_hold_20261101.csv",
    "mma_fighter_merge_map.csv",
    "mma_reversed_fixture_folds.csv",
)
MERGE_STATE = (
    "/app/backups/merge_mma_fighters_last_apply.env",
    "PLAN=/app/backups/m.plan.json\nGRADE_IDS=\nSTATUS=done\n",
)

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
    "PLAN merge_delete=31 merge_cancel=0 delete_no_twin=31 delete_placeholder=4 mark_postponed=0 left=9 "
    "blocked_pending_recs=0 list_mismatch=0"
)
DONE_B = (
    "PLAN merge_delete=0 merge_cancel=0 delete_no_twin=0 delete_placeholder=0 mark_postponed=0 left=9 "
    "blocked_pending_recs=0 list_mismatch=0"
)
REVIEWED_IN_APP = "/app/scripts/data/mma_stranded_recs_20261101.csv"
# psql answers, in the order each flow asks: A's before-state, the reviewed check, A's after-state.
A_STATES = ("21|49|21", "21", "0|49|0")
# A verified at step 0 (reviewed check, stranded|recs before AS_OF), then B's before- and after-state.
B_STATES = ("21", "0|49", "75|49", "9|49")
ROWS_OK = "OK 66 held 2"
DONE_M = (
    "PLAN pairs_merge=0 pairs_done=9 pairs_refuse=0 folds=0 folds_reversed=0 folds_done=1 folds_refuse=0 "
    "repoints=0 left_rec_held=0 odds_moved=0 snapshots_moved=0 predictions_moved=0 predictions_dropped=0 "
    "espn_ids_adopted=0 athlete_ids_stamped=0 nfc_renamed=0 nfc_refuse=0"
)


def _rules(
    pre_a=PLAN_A,
    dry_a=PLAN_A,
    verify_a=DONE_A,
    pre_b=PLAN_B,
    dry_b=PLAN_B,
    verify_b=DONE_B,
    check="OK 21",
    states=A_STATES,
    apply_a_rc=0,
    dry_a_rc=0,
    features_rc=0,
    merge_pre=DONE_M,
    merge_after_rc=0,
    check_b=(ROWS_OK, ROWS_OK, ROWS_OK),
):
    rules = [
        {"match": r"> /tmp/merge_mma_after\.log 2>&1", "out": "AFTER (scripted)", "rc": merge_after_rc},
        {"match": r"> /tmp/merge_mma_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -oE 'PLAN .*' /tmp/merge_mma_pre\.log", "out": merge_pre},
        {"match": r"> /tmp/stranded_mma_a_apply\.log 2>&1", "out": f"tool exit {apply_a_rc}", "rc": apply_a_rc},
        {"match": r"> /tmp/stranded_mma_a_dry\.log 2>&1", "out": f"tool exit {dry_a_rc}", "rc": dry_a_rc},
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
        {"match": r"^compose exec -T api python -$", "out": "graded 0 predictions on 0 finished UFC rows"},
        {"match": r"postgres rm -f", "out": ""},
    ]
    rules += [{"match": r"postgres sh -c psql", "out": s, "times": 1} for s in states]
    rules += [{"match": r"api python -c import csv, sys\s+writes", "out": c, "times": 1} for c in check_b]
    return rules


def _run(tmp_path, rules, *args, merge_state=True):
    env = fd.install(tmp_path, rules)
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        shutil.copy(REVIEWED.parent / name, data / name)
    if merge_state:
        fd.put_file(tmp_path, "api", *MERGE_STATE)
    rc, out = fd.run(WRAPPER, env, *args)
    return rc, out, fd.calls(tmp_path)


def _rerun(tmp_path, rules, *args):
    """A second run in the same fake VM (the first run's calls dropped)."""
    (tmp_path / "fake" / "calls.log").unlink()
    rc, out = fd.run(WRAPPER, fd.install(tmp_path, rules), *args)
    return rc, out, fd.calls(tmp_path)


def _applies(calls, tool=""):
    return [c for c in calls if tool in c and "--apply" in c]


def _backups(tmp_path):
    return fd.airflow_state(tmp_path).get("triggers", 0)


def _sent_sql(tmp_path):
    return [
        f.read_text()
        for f in sorted(
            (tmp_path / "fake" / "cp").glob("[0-9]*_stranded_mma_q.sql"), key=lambda f: int(f.name.split("_")[0])
        )
    ]


# ── A: the default run ────────────────────────────────────────────────


def test_a_clean_run_pins_the_plan_and_every_rec_and_leaves_the_cleanup_alone(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: the 21 stranded UFC recs are closed out" in out, out
    assert "--retire-stale-rows" in out
    assert _backups(tmp_path) == 1
    (apply_a,) = _applies(calls)
    assert "backfill_results.py" in apply_a and "--sport mma" in apply_a
    pin = PLAN_A.replace("PLAN ", "").replace(" ", ",")
    assert f"--expect-plan {pin}" in apply_a and f"--expect-recs {REVIEWED_IN_APP}" in apply_a
    assert "--dates 2026-06-06,2026-06-14,2026-06-20,2026-06-27,2026-07-18" in apply_a
    (dry_a,) = [c for c in calls if "stranded_mma_a_dry.log" in c and "backfill_results.py" in c]
    assert f"--expect-recs {REVIEWED_IN_APP}" in dry_a
    assert not [c for c in calls if "repair_fixture_orphans.py" in c]
    assert any("compute_features_mma.py --all-finished --force" in c for c in calls)
    sent = _sent_sql(tmp_path)  # A's before-state, the reviewed check, A's after-state
    assert len(sent) == 3
    reviewed_sql = sent[1]
    assert reviewed_sql.count("::uuid, '") == 21
    assert "('3f56282f-7cd9-4f27-8d90-02460593c61c'::uuid, 'lost', -67.72::numeric, 'lost')" in reviewed_sql
    assert "('d9da951f-c6a9-47ce-bb26-e04112e0f443'::uuid, 'void', 0.00::numeric, 'void: match cancelled')" in (
        reviewed_sql
    )
    assert "m.status = CASE WHEN v.status = 'void' THEN 'cancelled' ELSE 'finished' END" in reviewed_sql
    assert "reviewed_pending" in sent[0] and "status = 'pending'" in sent[0]


def test_a_committed_and_verified_is_already_done_without_a_backup(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, states=("21", "0|49")))
    assert rc == 0 and "ALREADY DONE" in out and "--followups-only" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)
    reviewed_sql, done_sql = _sent_sql(tmp_path)
    assert reviewed_sql.count("::uuid, '") == 21
    assert "br.created_at < TIMESTAMPTZ '2026-11-01T00:00:00Z'" in done_sql and "m.status <> 'finished'" in done_sql


@pytest.mark.parametrize(
    "states",
    [
        ("20", "0|49"),  # one rec not as reviewed
        ("21", "1|49"),  # a UFC rec still stranded
        ("21", "0|48"),  # a rec disappeared
        ("", ""),  # the checks could not be read
    ],
)
def test_a_done_plan_is_not_trusted_without_the_database(tmp_path, states):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, states=states))
    assert rc == 1 and "A looks committed" in out and "not verified" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


def test_a_failed_step_4_is_not_bypassed_by_a_rerun(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(states=("21|49|21", "20", "0|49|0")))
    assert rc == 1 and "FAILED after-state at step 4" in out, out
    assert len([c for c in calls if "dags trigger db_backup_daily" in c]) == 1
    for args in ((), ("--retire-stale-rows",)):
        rc, out, calls = _rerun(tmp_path, _rules(pre_a=DONE_A, states=("20", "0|49")), *args)
        assert rc == 1 and "not verified" in out, out
        assert not _applies(calls) and not [c for c in calls if "repair_fixture_orphans.py" in c]
        assert not [c for c in calls if "dags trigger db_backup_daily" in c]  # no backup either


@pytest.mark.parametrize(
    "pre_a",
    [
        DONE_A.replace("results=56", "results=0").replace("fetch_failed=0", "fetch_failed=5"),  # ESPN down
        DONE_A.replace("results=56", "results=55"),
        DONE_A.replace("identity_refused=0", "identity_refused=1"),
        DONE_A.replace("recs_left=0", "recs_left=1"),
        PLAN_A.replace("recs_settled=20", "recs_settled=19"),
        PLAN_A.replace("fetch_failed=0", "fetch_failed=1"),
        "",  # unreadable
    ],
)
def test_a_pre_check_plan_that_is_neither_reviewed_nor_done_aborts_before_the_backup(tmp_path, pre_a):
    rc, out, calls = _run(tmp_path, _rules(pre_a=pre_a))
    assert rc == 1 and "ABORT at step 0" in out and "neither the reviewed plan" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)
    assert not [c for c in calls if "postgres sh -c psql" in c]


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
    assert not _applies(calls)


def test_a_dry_run_refusing_the_rec_outcomes_aborts(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(dry_a_rc=1))
    assert rc == 1 and "ABORT at step 2: the dry run failed (exit 1" in out, out
    assert not _applies(calls)


def test_a_reviewed_rec_settled_since_the_review_aborts_before_the_dry_run(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(states=("20|49|20",)))
    assert rc == 1 and "only 20 of the 21 reviewed recs are still pending" in out, out
    assert not [c for c in calls if "stranded_mma_a_dry" in c]


def test_rec_outcomes_other_than_the_reviewed_ones_abort(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(check="MISMATCH 3f56282f-7cd9-4f27-8d90-02460593c61c"))
    assert rc == 1 and "not the reviewed ones (MISMATCH" in out, out
    assert not _applies(calls)


def test_a_failed_backfill_apply_aborts(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(apply_a_rc=1))
    assert rc == 1 and "ABORT at step 3" in out, out
    assert not [c for c in calls if "compute_features_mma" in c]


@pytest.mark.parametrize(
    "kw",
    [
        {"states": ("21|49|21", "20", "0|49|0")},  # one rec not as reviewed
        {"states": ("21|49|21", "21", "1|49|0")},  # a UFC rec still stranded
        {"states": ("21|49|21", "21", "0|48|0")},  # a rec disappeared
        {"verify_a": DONE_A.replace("flipped=0", "flipped=1")},  # something left to write
        {"verify_a": DONE_A.replace("fetch_failed=0", "fetch_failed=1")},  # a partial re-plan proves nothing
    ],
)
def test_a_wrong_backfill_after_state_fails_before_the_follow_ups(tmp_path, kw):
    rc, out, calls = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "FAILED after-state at step 4" in out, out
    assert not [c for c in calls if "compute_features_mma" in c]


def test_a_failed_follow_up_exits_2_and_followups_only_reruns_just_them(tmp_path):
    rc, out, _ = _run(tmp_path, _rules(features_rc=1))
    assert rc == 2 and "INCOMPLETE" in out and "--followups-only" in out, out
    rc, out, calls = _rerun(tmp_path, _rules(), "--followups-only")
    assert rc == 0 and "FOLLOW-UPS DONE" in out, out
    assert not [c for c in calls if "--apply" in c or "backfill_results" in c or "airflow" in c]


def test_without_the_reviewed_file_nothing_runs(tmp_path):
    env = fd.install(tmp_path, _rules())
    rc, out = fd.run(WRAPPER, env)
    assert rc == 1 and "is missing" in out, out
    assert not [c for c in fd.calls(tmp_path) if "/app/scripts/" in c]


def test_an_unknown_argument_is_refused(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), "--apply")
    assert rc == 2 and "usage" in out, out
    assert not calls


# ── B: --retire-stale-rows ────────────────────────────────────────────


def test_b_runs_only_after_a_is_verified_and_pins_its_plan(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, states=B_STATES), "--retire-stale-rows")
    assert rc == 0 and "DONE: the stale UFC fixture rows are retired" in out, out
    assert "A's committed state: reviewed recs as reviewed 21" in out
    assert _backups(tmp_path) == 1
    assert not _applies(calls, "backfill_results.py")
    (apply_b,) = _applies(calls, "repair_fixture_orphans.py")
    assert "--as-of 2026-11-01T00:00:00Z" in apply_b
    assert "--expect-plan merge_delete=31,merge_cancel=0,delete_no_twin=31,delete_placeholder=4,mark_postponed=0" in (
        apply_b
    )
    assert "--expect-rows /app/scripts/data/mma_stale_rows_retire_20261101.csv" in apply_b
    # every run of the cleanup tool (pre-check, dry run, apply, re-plan) holds the same rows back
    orphan_runs = [c for c in calls if "repair_fixture_orphans.py" in c]
    assert len(orphan_runs) == 4
    assert all("--hold-list /app/scripts/data/mma_stale_rows_hold_20261101.csv" in c for c in orphan_runs)
    assert any("compute_features_mma.py --all-finished --force" in c for c in calls)
    assert "MERGE VERIFIED" in out and f"rows as reviewed: {ROWS_OK}" in out and f"rows as reviewed ({ROWS_OK})" in out
    checks = [c for c in calls if "import csv, sys" in c]
    assert len(checks) == 3
    assert all(
        "/app/scripts/data/mma_stale_rows_retire_20261101.csv /app/scripts/data/mma_stale_rows_hold_20261101.csv" in c
        for c in checks
    )
    assert "fixture_orphans_mma_precheck_" in checks[0] and "fixture_orphans_mma_dryrun_" in checks[1]
    assert "fixture_orphans_mma_apply_" in checks[2] and f"the apply's rows '{ROWS_OK}'" in out
    (verify,) = [c for c in calls if "--check-after" in c]
    assert "--check-after /app/backups/m.plan.json --after-later-steps" in verify


def test_b_before_a_is_refused(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), "--retire-stale-rows")
    assert rc == 1 and "A is not done" in out and "without --retire-stale-rows first" in out, out
    assert _backups(tmp_path) == 0 and not [c for c in calls if "repair_fixture_orphans.py" in c]


def test_b_done_and_verified_is_already_done_without_a_backup(tmp_path):
    rc, out, calls = _run(
        tmp_path, _rules(pre_a=DONE_A, pre_b=DONE_B, states=("21", "0|49", "9|49", "held|held")), "--retire-stale-rows"
    )
    assert rc == 0 and "ALREADY DONE" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


def test_b_done_plan_with_the_wrong_database_state_aborts(tmp_path):
    rc, out, calls = _run(
        tmp_path, _rules(pre_a=DONE_A, pre_b=DONE_B, states=("21", "0|49", "8|49")), "--retire-stale-rows"
    )
    assert rc == 1 and "B looks done" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


@pytest.mark.parametrize(
    "pre_b",
    [
        PLAN_B.replace("delete_no_twin=31", "delete_no_twin=42").replace("left=9", "left=7"),  # pre-merge, no hold
        PLAN_B.replace("delete_no_twin=31", "delete_no_twin=33").replace("left=9", "left=7"),  # no hold list
        DONE_B.replace("merge_delete=0", "merge_delete=1"),  # part-applied
        "",
    ],
)
def test_b_pre_check_plan_that_is_neither_reviewed_nor_done_aborts_before_the_backup(tmp_path, pre_b):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, pre_b=pre_b, states=("21", "0|49")), "--retire-stale-rows")
    assert rc == 1 and "B's pre-check plan" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


@pytest.mark.parametrize(
    "dry_b",
    [
        PLAN_B.replace("delete_no_twin=31", "delete_no_twin=32"),
        PLAN_B.replace("left=9", "left=8"),
        PLAN_B.replace("blocked_pending_recs=0", "blocked_pending_recs=1"),
    ],
)
def test_a_cleanup_plan_that_is_not_the_reviewed_one_aborts_before_its_apply(tmp_path, dry_b):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, dry_b=dry_b, states=B_STATES), "--retire-stale-rows")
    assert rc == 1 and "ABORT at step 5" in out, out
    assert not _applies(calls, "repair_fixture_orphans.py")


@pytest.mark.parametrize(
    "kw",
    [
        {"states": ("21", "0|49", "75|49", "10|49")},
        {"states": ("21", "0|49", "75|49", "9|48")},
        {"verify_b": DONE_B.replace("merge_delete=0", "merge_delete=1")},
        {"verify_b": DONE_B.replace("left=9", "left=8")},
        # the apply's own audit CSV planned other rows than the reviewed ones (or deleted a held row)
        {"check_b": (ROWS_OK, ROWS_OK, "OK 66 held 1")},
        {
            "check_b": (
                ROWS_OK,
                ROWS_OK,
                "MISMATCH 1 not reviewed [x], 0 reviewed not planned [], held [], hold list []",
            )
        },
    ],
)
def test_a_wrong_cleanup_after_state_fails(tmp_path, kw):
    rc, out, calls = _run(tmp_path, _rules(**{"pre_a": DONE_A, "states": B_STATES, **kw}), "--retire-stale-rows")
    assert rc == 1 and "FAILED after-state at step 7" in out, out
    assert not [c for c in calls if "compute_features_mma" in c]


def test_b_before_the_fighter_merge_is_refused_before_the_backup(tmp_path):
    rc, out, calls = _run(
        tmp_path,
        _rules(pre_a=DONE_A, merge_pre=PLAN_B.replace("PLAN ", "PLAN x"), states=("21", "0|49")),
        "--retire-stale-rows",
    )
    assert rc == 1 and "the fighter merge is not done and verified" in out, out
    assert _backups(tmp_path) == 0 and not [c for c in calls if "repair_fixture_orphans.py" in c]


def test_b_with_a_merge_whose_after_state_fails_is_refused(tmp_path):
    rc, out, calls = _run(
        tmp_path, _rules(pre_a=DONE_A, merge_after_rc=1, states=("21", "0|49")), "--retire-stale-rows"
    )
    assert rc == 1 and "the fighter merge is not done and verified" in out, out
    assert _backups(tmp_path) == 0


def test_b_without_a_recorded_merge_is_refused(tmp_path):
    rc, out, calls = _run(
        tmp_path, _rules(pre_a=DONE_A, states=("21", "0|49")), "--retire-stale-rows", merge_state=False
    )
    assert rc == 1 and "no apply is recorded" in out and "the fighter merge is not done" in out, out
    assert _backups(tmp_path) == 0


def test_b_pre_check_rows_other_than_the_reviewed_ones_abort_before_the_backup(tmp_path):
    rc, out, calls = _run(
        tmp_path,
        _rules(
            pre_a=DONE_A, states=("21", "0|49"), check_b=("MISMATCH 1 not reviewed [x], 1 reviewed not planned [y]",)
        ),
        "--retire-stale-rows",
    )
    assert rc == 1 and "ABORT at step 0: B's pre-check plans other rows" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


def test_b_dry_run_rows_other_than_the_reviewed_ones_abort_before_the_apply(tmp_path):
    rc, out, calls = _run(
        tmp_path, _rules(pre_a=DONE_A, states=B_STATES, check_b=(ROWS_OK, "OK 66 held 1")), "--retire-stale-rows"
    )
    assert rc == 1 and "ABORT at step 5: the dry run plans other rows" in out, out
    assert _backups(tmp_path) == 1 and not _applies(calls, "repair_fixture_orphans.py")


def test_b_without_the_reviewed_row_list_nothing_runs(tmp_path):
    env = fd.install(tmp_path, _rules(pre_a=DONE_A, states=("21", "0|49")))
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy(REVIEWED, data / REVIEWED.name)
    rc, out = fd.run(WRAPPER, env, "--retire-stale-rows")
    assert rc == 1 and "mma_stale_rows_retire_20261101.csv holds 0 row(s)" in out and "pinned to 66 and 2" in out, out
    assert not [c for c in fd.calls(tmp_path) if "/app/scripts/" in c]


def test_b_without_the_hold_list_nothing_runs(tmp_path):
    env = fd.install(tmp_path, _rules(pre_a=DONE_A, states=("21", "0|49")))
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        if name != "mma_stale_rows_hold_20261101.csv":
            shutil.copy(REVIEWED.parent / name, data / name)
    rc, out = fd.run(WRAPPER, env, "--retire-stale-rows")
    assert rc == 1 and "mma_stale_rows_hold_20261101.csv 0; B is pinned to 66 and 2" in out, out
    assert not [c for c in fd.calls(tmp_path) if "/app/scripts/" in c]


def test_the_committed_b_lists_are_disjoint_and_hold_the_two_undecided_rows():
    def rows(name):
        with (REVIEWED.parent / name).open(encoding="utf-8") as fh:
            return list(csv.DictReader(line for line in fh if not line.startswith("#")))

    retire, hold = rows("mma_stale_rows_retire_20261101.csv"), rows("mma_stale_rows_hold_20261101.csv")
    assert len(retire) == 66 and len(hold) == 2
    assert {r["match_id"] for r in retire}.isdisjoint({r["match_id"] for r in hold})
    assert {r["reason"] for r in hold} == {"fought_no_winner", "competitor_replaced_in_competition"}
    assert {r["action"] for r in retire} == {"merge_delete", "delete_no_twin", "delete_placeholder"}
    assert not [r for r in retire if r["class"] in ("fought_no_winner", "competitor_replaced_in_competition")]


@pytest.mark.parametrize("states, rc_want", [(("21", "0|49"), 0), (("20", "0|49"), 1), (("21", "1|49"), 1)])
def test_verify_a_reads_the_database_only(tmp_path, states, rc_want):
    rc, out, calls = _run(tmp_path, _rules(states=states), "--verify-a")
    assert rc == rc_want, out
    assert ("A VERIFIED" in out) == (rc_want == 0) and ("A NOT VERIFIED" in out) == (rc_want == 1)
    assert not [c for c in calls if "/app/scripts/" in c or "airflow" in c]
    assert len([c for c in calls if "postgres sh -c psql" in c]) == 2


# ── B's done state with its held rows held or settled, and --verify-b ──

DONE_B_LEFT = {n: DONE_B.replace("left=9", f"left={n}") for n in (7, 8, 9)}
HELD_IDS = ("b7f16d27-42a1-41b0-a6f6-6e8e9837da47", "787bd68e-2ce8-4beb-b346-c9f422c42574")


@pytest.mark.parametrize(
    "left, held",
    [
        (7, "settled|settled"),  # after scripts/ops/settle_mma_held_rows.sh
        (8, "settled|held"),  # a re-fetch recorded the draw first; the Machado row still held
        (8, "held|settled"),
        (9, "held|held"),  # B as it ran (also the existing done test above)
    ],
)
def test_b_done_with_its_held_rows_held_or_settled_is_already_done(tmp_path, left, held):
    rc, out, calls = _run(
        tmp_path,
        _rules(pre_a=DONE_A, pre_b=DONE_B_LEFT[left], states=("21", "0|49", f"{left}|49", held)),
        "--retire-stale-rows",
    )
    assert rc == 0 and "ALREADY DONE" in out and f"held rows (draw|replaced row) {held}" in out, out
    assert ("settled as decided 2026-11-02" in out) == (left != 9)
    assert _backups(tmp_path) == 0 and not _applies(calls)
    held_sql = _sent_sql(tmp_path)[-1]
    assert all(i in held_sql for i in HELD_IDS) and "metadata->>'no_winner' = 'draw'" in held_sql


@pytest.mark.parametrize(
    "left, states",
    [
        (7, ("21", "0|49", "9|49", "settled|settled")),  # the stale count of the unsettled state
        (7, ("21", "0|49", "7|49", "held|settled")),  # one still held: B would leave 8
        (9, ("21", "0|49", "7|49", "settled|settled")),  # settled, but B's plan leaves 9
        (7, ("21", "0|49", "7|49", "other|settled")),  # the draw row in some other state
        (7, ("21", "0|49", "7|49", "")),  # unreadable
        (7, ("21", "0|49", "7|48", "settled|settled")),  # a rec gone
    ],
)
def test_b_done_plan_with_a_database_that_does_not_match_aborts(tmp_path, left, states):
    rc, out, calls = _run(tmp_path, _rules(pre_a=DONE_A, pre_b=DONE_B_LEFT[left], states=states), "--retire-stale-rows")
    assert rc == 1 and "B looks done" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


@pytest.mark.parametrize(
    "pre_b, states, rc_want",
    [
        (DONE_B, ("21", "0|49", "9|49", "held|held"), 0),
        (DONE_B_LEFT[7], ("21", "0|49", "7|49", "settled|settled"), 0),
        (DONE_B_LEFT[8], ("21", "0|49", "8|49", "settled|held"), 0),
        (DONE_B, ("21", "0|49", "8|49", "held|held"), 1),
        (DONE_B_LEFT[7], ("21", "0|49", "7|49", "settled|other"), 1),
        (PLAN_B, ("21", "0|49"), 1),  # B not run
        (DONE_B, ("20", "0|49"), 1),  # A not verified
    ],
)
def test_verify_b_is_bs_done_check_read_only(tmp_path, pre_b, states, rc_want):
    rc, out, calls = _run(tmp_path, _rules(pre_b=pre_b, states=states), "--verify-b")
    assert rc == rc_want, out
    assert ("B VERIFIED" in out) == (rc_want == 0) and ("B NOT VERIFIED" in out) == (rc_want == 1)
    assert _backups(tmp_path) == 0 and not _applies(calls)
    assert not [c for c in calls if "backfill_results" in c or "airflow" in c]  # no ESPN, no backup


def test_bs_done_check_counts_only_the_ufc_recs_created_before_as_of(tmp_path):
    # generate_recommendations_mma prices every upcoming UFC card every 15 minutes: a rec written after AS_OF
    # must not turn B (or settle_mma_held_rows.sh, which calls --verify-b at its steps 0 and 4) into "not done".
    # test_settle_mma_held_rows_pg.py runs this SQL on a real database with such a rec.
    rc, out, _ = _run(tmp_path, _rules(pre_b=DONE_B, states=("21", "0|49", "9|49", "held|held")), "--verify-b")
    assert rc == 0 and "UFC recs created before 2026-11-01T00:00:00Z 9|49" in out, out
    state_b = [q for q in _sent_sql(tmp_path) if " AS stale" in q]
    assert len(state_b) == 1
    recs = state_b[0].split(") AS stale,")[1]
    assert "br.created_at < TIMESTAMPTZ '2026-11-01T00:00:00Z'" in recs, recs


def test_verify_b_needs_the_fighter_merge(tmp_path):
    rc, out, _ = _run(
        tmp_path, _rules(pre_b=DONE_B, states=("21", "0|49", "9|49", "held|held"), merge_after_rc=1), "--verify-b"
    )
    assert rc == 1 and "the fighter merge is not done and verified" in out, out


# ── the row check itself (CHECK_B_PY, run for real on audit CSVs) ─────


def _check_b_source() -> str:
    m = re.search(r'^CHECK_B_PY="(.*?)"\ncheck_b_rows\(\)', WRAPPER.read_text(), re.S | re.M)
    assert m, "CHECK_B_PY not found in the wrapper"
    return m.group(1)


def _audit_csv(path, rows):
    fields = ("match_id", "sport", "action", "reason", "twin_id")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({"sport": "mma", "reason": "", "twin_id": "", **r} for r in rows)
    return path


def _reviewed_audit_rows():
    with (REVIEWED.parent / "mma_stale_rows_retire_20261101.csv").open(encoding="utf-8") as fh:
        writes = [
            {"match_id": r["match_id"], "action": r["action"], "twin_id": r["twin_id"]}
            for r in csv.DictReader(line for line in fh if not line.startswith("#"))
        ]
    with (REVIEWED.parent / "mma_stale_rows_hold_20261101.csv").open(encoding="utf-8") as fh:
        held = [
            {"match_id": r["match_id"], "action": "leave", "reason": "held_for_decision"}
            for r in csv.DictReader(line for line in fh if not line.startswith("#"))
        ]
    left = [{"match_id": f"rec-held-{i}", "action": "leave", "reason": "has_recs"} for i in range(7)]
    return writes, held, left


def _check(tmp_path, rows):
    audit = _audit_csv(tmp_path / "audit.csv", rows)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            _check_b_source(),
            str(audit),
            str(REVIEWED.parent / "mma_stale_rows_retire_20261101.csv"),
            str(REVIEWED.parent / "mma_stale_rows_hold_20261101.csv"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_the_row_check_passes_exactly_the_reviewed_writes_and_the_held_rows(tmp_path):
    writes, held, left = _reviewed_audit_rows()
    assert _check(tmp_path, writes + held + left) == ROWS_OK


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w, h: (w[1:], h),  # a reviewed write the plan leaves
        lambda w, h: (w + [{"match_id": "x", "action": "delete_no_twin", "twin_id": ""}], h),  # an unreviewed write
        lambda w, h: ([dict(w[0], twin_id="elsewhere")] + w[1:], h),  # another twin
        lambda w, h: (w + [{"match_id": h[0]["match_id"], "action": "delete_no_twin", "twin_id": ""}], h[1:]),  # held
        lambda w, h: (w, h[1:]),  # a held row the plan does not hold back
    ],
)
def test_the_row_check_refuses_anything_else(tmp_path, mutate):
    writes, held, left = _reviewed_audit_rows()
    w, h = mutate(writes, held)
    assert _check(tmp_path, w + h + left).startswith("MISMATCH")
