"""scripts/ops/merge_mma_fighters.sh, run for real against a scripted docker:
it refuses unless run A is done and verified (repair_stranded_recs_mma.sh
--verify-a, read only); the exact reviewed plan before the backup and again
from the dry run; the apply pinned to the dry run's plan file; the exact
after-state (--check-after and a done re-plan); the follow-ups (exit 2 when
one fails, --followups-only); 'already done' only once the recorded plan's
after-state holds, finishing an apply whose follow-ups did not; --verify for
run B (read only, never a backup)."""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WRAPPER = REPO / "scripts" / "ops" / "merge_mma_fighters.sh"
DATA = REPO / "scripts" / "data"
STATE = "/app/backups/merge_mma_fighters_last_apply.env"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN_M = (
    "PLAN pairs_merge=9 pairs_done=0 pairs_refuse=0 folds=9 folds_reversed=2 folds_done=0 folds_refuse=0 "
    "repoints=11 left_rec_held=0 odds_moved=32 snapshots_moved=51 predictions_moved=15 predictions_dropped=8 "
    "espn_ids_adopted=2 athlete_ids_stamped=9 nfc_renamed=1 nfc_refuse=0"
)
DONE_M = (
    "PLAN pairs_merge=0 pairs_done=9 pairs_refuse=0 folds=0 folds_reversed=0 folds_done=1 folds_refuse=0 "
    "repoints=0 left_rec_held=0 odds_moved=0 snapshots_moved=0 predictions_moved=0 predictions_dropped=0 "
    "espn_ids_adopted=0 athlete_ids_stamped=0 nfc_renamed=0 nfc_refuse=0"
)
GRADE_IDS = "020b3420-4f21-483e-8f60-bc5239a57992\n4131d1c6-ba06-4a75-a2a4-102d3a1984d7"


def _rules(
    pre=PLAN_M,
    dry=PLAN_M,
    verify=DONE_M,
    a_states=("21", "0|49"),
    dry_rc=0,
    apply_rc=0,
    after_rc=0,
    grade_rc=0,
    features_rc=0,
):
    rules = [
        {"match": r"> /tmp/merge_mma_apply\.log 2>&1", "out": f"tool exit {apply_rc}", "rc": apply_rc},
        {"match": r"> /tmp/merge_mma_after\.log 2>&1", "out": "AFTER-STATE (scripted)", "rc": after_rc},
        {"match": r"> /tmp/merge_mma_dry\.log 2>&1", "out": f"tool exit {dry_rc}", "rc": dry_rc},
        {"match": r"grade_completed_matches\.py --match-id", "out": "graded rows: 2", "rc": grade_rc},
        {"match": r"compute_features_mma\.py --all-finished --force", "out": "Wrote 1986", "rc": features_rc},
        {"match": r"grep -o 'GRADE_AFTER", "out": GRADE_IDS},
        {"match": r"> /tmp/merge_mma_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"postgres rm -f", "out": ""},
    ]
    for name, plan in (("pre", pre), ("dry", dry), ("verify", verify)):
        rules.append({"match": rf"grep -oE 'PLAN .*' /tmp/merge_mma_{name}\.log", "out": plan})
    rules += [{"match": r"postgres sh -c psql", "out": s, "times": 1} for s in a_states]
    return rules


def _install(tmp_path, rules, state=None):
    env = fd.install(tmp_path, rules)
    data = tmp_path / "auspex" / "scripts" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("mma_fighter_merge_map.csv", "mma_reversed_fixture_folds.csv", "mma_stranded_recs_20261101.csv"):
        shutil.copy(DATA / name, data / name)
    if state is not None:
        fd.put_file(tmp_path, "api", STATE, state)
    return env


def _run(tmp_path, rules, *args, state=None):
    env = _install(tmp_path, rules, state)
    rc, out = fd.run(WRAPPER, env, *args)
    return rc, out, fd.calls(tmp_path)


def _rerun(tmp_path, rules, *args):
    (tmp_path / "fake" / "calls.log").unlink()
    fd.set_rules(tmp_path, rules)
    rc, out = fd.run(WRAPPER, {**_install(tmp_path, rules)}, *args)
    return rc, out, fd.calls(tmp_path)


def _applies(calls):
    return [c for c in calls if "merge_mma_fighters.py" in c and "--apply" in c]


def _backups(tmp_path):
    return fd.airflow_state(tmp_path).get("triggers", 0)


def _state(tmp_path):
    return fd.get_file(tmp_path, "api", STATE) or ""


def test_a_clean_run_pins_the_plan_applies_it_and_runs_the_follow_ups(tmp_path):
    rc, out, calls = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: the 9 UFC fighters are merged and resolving" in out, out
    assert "A VERIFIED" in out and "--retire-stale-rows" in out
    assert _backups(tmp_path) == 1
    (apply,) = _applies(calls)
    (dry,) = [c for c in calls if "/tmp/merge_mma_dry.log" in c and "merge_mma_fighters.py" in c]
    plan_file = dry.split("--plan-out ")[1].split()[0]
    # stamped to the second and the process id: two runs started within one second write different files
    assert re.fullmatch(r"/app/backups/merge_mma_fighters_dryrun_\d{8}T\d{6}Z_\d+\.plan\.json", plan_file), plan_file
    assert f"--expect-plan {plan_file}" in apply and "--archive /app/backups/merge_mma_fighters_apply_" in apply
    (check,) = [c for c in calls if "--check-after" in c]
    assert f"--check-after {plan_file}" in check and "--after-later-steps" not in check  # strict right after
    assert any("grade_completed_matches.py --match-id" in c and "4131d1c6" in c for c in calls)
    assert any("compute_features_mma.py --all-finished --force" in c and "--days 14 --force" in c for c in calls)
    assert f"PLAN={plan_file}" in _state(tmp_path) and "STATUS=done" in _state(tmp_path)
    # the A check ran read only: two queries, no backfill
    assert not [c for c in calls if "backfill_results.py" in c]


def test_it_refuses_before_run_a_is_verified(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(a_states=("0", "21|49")))
    assert rc == 1 and "A NOT VERIFIED" in out and "run A is not done and verified" in out, out
    assert _backups(tmp_path) == 0 and not [c for c in calls if "merge_mma_fighters.py" in c]


@pytest.mark.parametrize(
    "pre",
    [
        PLAN_M.replace("pairs_merge=9 pairs_done=0", "pairs_merge=8 pairs_done=1"),  # part done
        PLAN_M.replace("pairs_refuse=0", "pairs_refuse=1"),
        PLAN_M.replace("left_rec_held=0", "left_rec_held=1"),
        PLAN_M.replace("odds_moved=32", "odds_moved=33"),  # a price landed on a copy since the review
        DONE_M.replace("nfc_renamed=0", "nfc_renamed=1"),
        "",  # unreadable
    ],
)
def test_a_pre_plan_that_is_neither_reviewed_nor_done_aborts_before_the_backup(tmp_path, pre):
    rc, out, calls = _run(tmp_path, _rules(pre=pre))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)


def test_a_dry_run_plan_other_than_the_reviewed_one_aborts_before_the_apply(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(dry=PLAN_M.replace("repoints=11", "repoints=12")))
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not _applies(calls) and "STATUS" not in _state(tmp_path)


def test_a_refusing_dry_run_aborts(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(dry_rc=1))
    assert rc == 1 and "ABORT at step 2: the dry run failed (exit 1" in out, out
    assert not _applies(calls)


@pytest.mark.parametrize("apply_rc, why", [(2, "refused before writing"), (1, "rolled back")])
def test_a_failed_apply_aborts_and_says_nothing_committed(tmp_path, apply_rc, why):
    rc, out, calls = _run(tmp_path, _rules(apply_rc=apply_rc))
    assert rc == 1 and "ABORT at step 3" in out and why in out and "nothing committed" in out, out
    assert "STATUS=applying" in _state(tmp_path)
    assert not [c for c in calls if "compute_features_mma" in c or "--check-after" in c]


@pytest.mark.parametrize(
    "kw", [{"after_rc": 1}, {"verify": DONE_M.replace("pairs_done=9", "pairs_done=8")}, {"verify": PLAN_M}]
)
def test_a_wrong_after_state_fails_before_the_follow_ups(tmp_path, kw):
    rc, out, calls = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "FAILED after-state at step 4" in out, out
    assert not [c for c in calls if "compute_features_mma" in c] and "STATUS=applying" in _state(tmp_path)


def test_a_failed_follow_up_exits_2_and_followups_only_finishes_it(tmp_path):
    rc, out, _ = _run(tmp_path, _rules(features_rc=1))
    assert rc == 2 and "INCOMPLETE" in out and "--followups-only" in out, out
    assert "STATUS=followups" in _state(tmp_path)
    rc, out, calls = _rerun(tmp_path, _rules(), "--followups-only")
    assert rc == 0 and "FOLLOW-UPS DONE" in out, out
    assert "STATUS=done" in _state(tmp_path)
    assert not [c for c in calls if "--apply" in c or "airflow" in c or "merge_mma_fighters.py" in c]
    assert any("grade_completed_matches.py --match-id" in c for c in calls)


def test_a_rerun_after_success_is_already_done_without_a_backup(tmp_path):
    rc, _, _ = _run(tmp_path, _rules())
    assert rc == 0
    rc, out, calls = _rerun(tmp_path, _rules(pre=DONE_M))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert not [c for c in calls if "dags trigger db_backup_daily" in c] and not _applies(calls)
    (check,) = [c for c in calls if "--check-after" in c]
    assert "--check-after /app/backups/merge_mma_fighters_dryrun_" in check and "--after-later-steps" in check


def test_done_with_its_follow_ups_unfinished_runs_them_without_a_backup(tmp_path):
    rec = "PLAN=/app/backups/merge_mma_fighters_dryrun_X.plan.json\nGRADE_IDS=abc def\nSTATUS=followups\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_M), state=rec)
    assert rc == 0 and "COMPLETED" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)
    assert any("for mid in abc def" in c for c in calls)
    assert "STATUS=done" in _state(tmp_path)


def test_done_without_a_recorded_apply_fails(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_M))
    assert rc == 1 and "no apply is recorded" in out, out
    assert _backups(tmp_path) == 0


def test_done_with_a_failing_recorded_after_state_fails(tmp_path):
    rec = "PLAN=/app/backups/merge_mma_fighters_dryrun_X.plan.json\nGRADE_IDS=\nSTATUS=done\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_M, after_rc=1), state=rec)
    assert rc == 1 and "after-state of the recorded plan" in out, out
    assert _backups(tmp_path) == 0


def test_verify_passes_only_on_the_done_plan_and_never_backs_up(tmp_path):
    rec = "PLAN=/app/backups/merge_mma_fighters_dryrun_X.plan.json\nGRADE_IDS=\nSTATUS=done\n"
    rc, out, calls = _run(tmp_path, _rules(pre=DONE_M, a_states=()), "--verify", state=rec)
    assert rc == 0 and "MERGE VERIFIED" in out, out
    assert _backups(tmp_path) == 0 and not _applies(calls)
    assert not [c for c in calls if "postgres" in c]  # no A check: run B verified A already
    rc, out, calls = _rerun(tmp_path, _rules(pre=PLAN_M, a_states=()), "--verify")
    assert rc == 1 and "NOT VERIFIED" in out, out
    assert not [c for c in calls if "dags trigger db_backup_daily" in c] and not _applies(calls)


def test_lists_other_than_the_reviewed_ones_refuse(tmp_path):
    env = _install(tmp_path, _rules())
    p = tmp_path / "auspex" / "scripts" / "data" / "mma_fighter_merge_map.csv"
    p.write_text(p.read_text() + p.read_text().splitlines()[-1] + "\n")
    rc, out = fd.run(WRAPPER, env)
    assert rc == 1 and "this script expects 9" in out, out
    assert not fd.calls(tmp_path)


def test_an_unknown_argument_is_refused(tmp_path):
    rc, out, calls = _run(tmp_path, _rules(), "--apply")
    assert rc == 2 and "usage" in out and not calls
