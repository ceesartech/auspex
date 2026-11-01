"""The soccer runbook's steps 12-14, run for real against a scripted docker
(tests/unit/_ops_fake_docker.py): scripts/ops/backfill_soccer_never_resulted.sh,
promote_raw_after_repair.sh and grade_folded_soccer_rows.sh. Each re-plans
read-only first and exits 0 ('ALREADY DONE') without a backup or a write when
nothing is left; otherwise backup, dry run inside the reviewed bounds, apply
pinned to that dry run's counts, and an after-state that must re-plan to
nothing. The tools ran end to end through a docker shim against a replica of
prod (their wrappers' headers give the counts)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
OPS = HERE.parents[1] / "scripts" / "ops"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(tmp_path, wrapper, rules):
    env = fd.install(tmp_path, rules)
    rc, out = fd.run(OPS / wrapper, env)
    return rc, out, fd.calls(tmp_path), fd.airflow_state(tmp_path).get("triggers", 0)


def _applies(calls):
    return [c for c in calls if "--apply" in c]


# --- backfill_soccer_never_resulted.sh (step 12) -----------------------------------------------------------------

ORPHANS_DONE = (
    "PLAN merge_delete=0 merge_cancel=0 delete_no_twin=0 delete_placeholder=0 mark_postponed=0 left=104 "
    "blocked_pending_recs=0 list_mismatch=0"
)
ORPHANS_AFTER = ORPHANS_DONE.replace("merge_cancel=0", "merge_cancel=1").replace("left=104", "left=65")
NR_PLAN = (
    "PLAN candidates=42 heal=38 done=0 refuse=4 predictions_graded=3173 predictions_regraded=0 recs_changed=0 "
    "teams_stamped=3 espn_failed=0"
)
NR_ZERO = NR_PLAN.replace("heal=38", "heal=0").replace("predictions_graded=3173", "predictions_graded=0")
NR_VERIFY = (
    "PLAN candidates=42 heal=0 done=38 refuse=4 predictions_graded=0 predictions_regraded=0 recs_changed=0 "
    "teams_stamped=0 espn_failed=0"
)
HEALED = "\n".join(f"{i:08d}-0000-0000-0000-000000000000" for i in range(38))


def _nr_rules(
    pre_ids="42",
    pre_plan=NR_PLAN,
    orphans=ORPHANS_DONE,
    dry_plan=NR_PLAN,
    apply_rc=0,
    verify_plan=NR_VERIFY,
    after_ids="3",
    state="38|0",
    new_rows="0",
):
    return [
        {"match": r"/tmp/never_resulted_orphans_\w+\.log 2>&1; echo", "out": "orphans dry run exit 0"},
        {"match": r"grep -o 'PLAN \.\*' /tmp/never_resulted_orphans_precheck\.log", "out": ORPHANS_DONE},
        {"match": r"grep -o 'PLAN \.\*' /tmp/never_resulted_orphans_dryrun\.log", "out": orphans},
        {"match": r"grep -o 'PLAN \.\*' /tmp/never_resulted_orphans_verify\.log", "out": ORPHANS_AFTER},
        {"match": r"python -c import csv, sys; ids = \[r\['match_id'\].*_precheck_", "out": pre_ids},
        {"match": r"python -c import csv, sys; ids = \[r\['match_id'\].*_dryrun_", "out": "42"},
        {"match": r"python -c import csv, sys; ids = \[r\['match_id'\].*_verify_", "out": after_ids},
        {"match": r"python -c import csv, sys; ids = \[r\['match_id'\] .*r\['action'\] == 'heal'", "out": "38"},
        {"match": r"/tmp/nr_apply\.log 2>&1", "out": "tool exit", "rc": apply_rc},
        {"match": r"backfill_never_resulted\.py .* > /tmp/nr_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN \.\*' /tmp/nr_pre\.log", "out": pre_plan},
        {"match": r"grep -o 'PLAN \.\*' /tmp/nr_dry\.log", "out": dry_plan},
        {"match": r"grep -o 'PLAN \.\*' /tmp/nr_verify\.log", "out": verify_plan},
        {"match": r"sh -c cat /app/backups/never_resulted_healed_\S+ 2>/dev/null", "out": HEALED},
        {"match": r"postgres sh -c psql", "out": state},
        {"match": r"python -c import sys; before = set", "out": new_rows},
        {"match": r"postgres rm -f", "out": ""},
    ]


def test_never_resulted_with_no_row_left_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(pre_ids="0"))
    assert rc == 0 and "ALREADY DONE: no never_resulted soccer row left" in out, out
    assert triggers == 0 and not [c for c in calls if "backfill_never_resulted.py" in c]


def test_never_resulted_with_nothing_to_heal_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(pre_plan=NR_ZERO))
    assert rc == 0 and "ALREADY DONE: nothing to heal" in out, out
    assert triggers == 0 and not _applies(calls)
    (pre,) = [c for c in calls if "backfill_never_resulted.py" in c]
    assert "--espn-cache /app/backups/never_resulted_espn_precheck_" in pre


def test_never_resulted_full_run_pins_every_key_and_reuses_the_dry_runs_espn_cache(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules())
    assert rc == 0 and "DONE: 38 never_resulted row(s) healed and graded (3173 predictions)" in out, out
    assert triggers == 1
    tool = [c for c in calls if "backfill_never_resulted.py" in c]
    dry, apply_, verify = tool[1], tool[2], tool[3]
    assert "--apply" not in dry and "--apply" not in verify
    cache = dry.split("--espn-cache ")[1].split()[0]
    assert "_precheck_" not in cache and f"--espn-cache {cache}" in apply_ and f"--espn-cache {cache}" in verify
    ids = dry.split("--match-ids-file ")[1].split()[0]
    assert f"--match-ids-file {ids}" in apply_ and f"--match-ids-file {ids}" in verify
    assert (
        "--expect-plan candidates=42,heal=38,done=0,refuse=4,predictions_graded=3173,predictions_regraded=0,"
        "recs_changed=0,teams_stamped=3,espn_failed=0" in apply_
    )
    backup = next(i for i, c in enumerate(calls) if "db_backup_daily" in c)
    assert backup < calls.index(dry) < calls.index(apply_)
    assert "never_resulted rows now 3 (at most the 4 refused), new ones 0" in out


@pytest.mark.parametrize(
    "plan",
    [
        NR_PLAN.replace("heal=38", "heal=39"),
        NR_PLAN.replace("candidates=42", "candidates=43"),
        NR_PLAN.replace("recs_changed=0", "recs_changed=1"),
        NR_PLAN.replace("predictions_regraded=0", "predictions_regraded=2"),
        NR_PLAN.replace("espn_failed=0", "espn_failed=1"),
        NR_PLAN.replace("teams_stamped=3", "teams_stamped=4"),
        NR_PLAN.replace("predictions_graded=3173", "predictions_graded=3174"),
        NR_PLAN.replace("done=0", "done=1"),
        "",
    ],
    ids=["heal", "candidates", "recs", "regraded", "espn", "stamped", "graded", "done", "unreadable"],
)
def test_never_resulted_plan_outside_the_reviewed_set_aborts_before_any_write(tmp_path, plan):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(dry_plan=plan))
    assert rc == 1 and "ABORT at step 2" in out and "nothing changed" in out, out
    assert not _applies(calls)


def test_never_resulted_refuses_while_the_cleanup_is_not_done(tmp_path):
    rules = _nr_rules(orphans=ORPHANS_DONE.replace("merge_delete=0", "merge_delete=2"))
    rc, out, calls, _ = _run(tmp_path, "backfill_soccer_never_resulted.sh", rules)
    assert rc == 1 and "the soccer orphan cleanup is not done" in out and "runbook step 7" in out, out
    assert not [c for c in calls if "backfill_never_resulted.py" in c and "dryrun" in c]


def test_never_resulted_apply_refusal_says_nothing_changed(tmp_path):
    rc, out, _, _ = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(apply_rc=1))
    assert rc == 1 and "ABORT at step 3" in out and "rolled it back, nothing changed" in out, out


@pytest.mark.parametrize(
    "kw",
    [
        {"verify_plan": NR_VERIFY.replace("heal=0", "heal=1")},
        {"state": "37|0"},
        {"state": "38|2"},
        {"new_rows": "1"},
        {"after_ids": "5"},
    ],
    ids=["re-plan-heals", "not-finished", "open-recs", "new-never-resulted", "more-left-than-refused"],
)
def test_never_resulted_after_state_failures_exit_1(tmp_path, kw):
    rc, out, _, _ = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(**kw))
    assert rc == 1 and "FAILED after-state" in out, out


# --- promote_raw_after_repair.sh (step 13) ------------------------------------------------------------------------

KEYS = (
    "matches_inserted matches_changed match_stats_inserted match_stats_changed odds_inserted teams_inserted "
    "teams_changed leagues_inserted leagues_changed predictions_changed recs_changed refused duplicates"
).split()


def _promote_plan(**over):
    vals = {k: 0 for k in KEYS}
    vals.update(over)
    return "PLAN " + " ".join(f"{k}={vals[k]}" for k in KEYS)


PR_PLAN = _promote_plan(match_stats_inserted=4, odds_inserted=13)
PR_ZERO = _promote_plan()
ALIASED = " ".join(f"{i:08d}-1111-1111-1111-111111111111" for i in range(4))


def _pr_rules(pre_plan=PR_PLAN, pre_rc=0, dry_plan=PR_PLAN, verify_plan=PR_ZERO, sides="4/4", aliased=ALIASED):
    return [
        {"match": r"/tmp/promote_after_pre\.log 2>&1", "out": "tool exit", "rc": pre_rc},
        {"match": r"promote_raw\.py .* > /tmp/promote_after_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN \.\*' /tmp/promote_after_pre\.log", "out": pre_plan},
        {"match": r"grep -o 'PLAN \.\*' /tmp/promote_after_dry\.log", "out": dry_plan},
        {"match": r"grep -o 'PLAN \.\*' /tmp/promote_after_verify\.log", "out": verify_plan},
        {"match": r"python -c import csv, glob, sys", "out": aliased},
        {"match": r"postgres sh -c psql", "out": sides},
        {"match": r"postgres rm -f", "out": ""},
    ]


def test_promote_with_nothing_to_write_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(pre_plan=PR_ZERO))
    assert rc == 0 and "ALREADY DONE" in out, out
    assert triggers == 0 and not _applies(calls)


def test_promote_refusing_to_start_aborts(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(pre_rc=1))
    assert rc == 1 and "a legacy kickoff is left" in out and triggers == 0, out


def test_promote_full_run_pins_every_key(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules())
    assert rc == 0 and "DONE: promote_raw ran checked" in out and "both sides' match_stats: 4/4" in out, out
    assert triggers == 1
    (apply_,) = _applies(calls)
    pin = ",".join(f"{k}={4 if k == 'match_stats_inserted' else 13 if k == 'odds_inserted' else 0}" for k in KEYS)
    assert f"--apply --expect-plan {pin}" in apply_
    tool = [c for c in calls if "promote_raw.py" in c]
    assert [("--dry-run" in c, "--apply" in c) for c in tool] == [
        (True, False),
        (True, False),
        (False, True),
        (True, False),
    ]


@pytest.mark.parametrize(
    "plan",
    [
        _promote_plan(match_stats_inserted=5, odds_inserted=13),
        _promote_plan(match_stats_inserted=4, odds_inserted=14),
        _promote_plan(match_stats_inserted=4, matches_inserted=1),
        _promote_plan(match_stats_inserted=4, recs_changed=1),
        _promote_plan(match_stats_inserted=4, teams_inserted=1),
        _promote_plan(match_stats_inserted=4, refused=1),
        "",
    ],
    ids=["stats", "odds", "matches", "recs", "teams", "refused", "unreadable"],
)
def test_promote_plan_outside_the_reviewed_set_aborts_before_any_write(tmp_path, plan):
    rc, out, calls, _ = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(dry_plan=plan))
    assert rc == 1 and "ABORT at step 2" in out and not _applies(calls), out


@pytest.mark.parametrize(
    "kw, why",
    [
        ({"verify_plan": _promote_plan(odds_inserted=1)}, "a fresh dry run still plans changes"),
        ({"sides": "3/4"}, "only 3/4 alias-merged fixtures"),
        ({"sides": ""}, "could not count"),
    ],
)
def test_promote_after_state_failures_exit_1(tmp_path, kw, why):
    rc, out, _, _ = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(**kw))
    assert rc == 1 and "FAILED after-state" in out and why in out, out


def test_promote_without_an_alias_merge_listed_skips_that_check(tmp_path):
    rc, out, _, _ = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(aliased=""))
    assert rc == 0 and "no alias-merged fixture listed" in out, out


# --- grade_folded_soccer_rows.sh (step 14) ------------------------------------------------------------------------

FOLDED = "\n".join(f"{i:08d}-2222-2222-2222-222222222222" for i in range(26))
GF_PLAN = "PLAN matches=26 predictions_graded=504 predictions_regraded=0 recs_changed=0"
GF_ZERO = "PLAN matches=26 predictions_graded=0 predictions_regraded=0 recs_changed=0"


def _gf_rules(ids=FOLDED, pre_plan=GF_PLAN, dry_plan=GF_PLAN, verify_plan=GF_ZERO, apply_rc=0):
    return [
        {"match": r"postgres sh -c psql", "out": ids},
        {"match": r"/tmp/grade_folded_apply\.log 2>&1", "out": "tool exit", "rc": apply_rc},
        {"match": r"grade_matches\.py .* > /tmp/grade_folded_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN \.\*' /tmp/grade_folded_pre\.log", "out": pre_plan},
        {"match": r"grep -o 'PLAN \.\*' /tmp/grade_folded_dry\.log", "out": dry_plan},
        {"match": r"grep -o 'PLAN \.\*' /tmp/grade_folded_verify\.log", "out": verify_plan},
        {"match": r"postgres rm -f", "out": ""},
    ]


def test_grade_folded_with_no_folded_row_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "grade_folded_soccer_rows.sh", _gf_rules(ids=""))
    assert rc == 0 and "ALREADY DONE: no folded soccer row" in out and triggers == 0, out
    assert not [c for c in calls if "grade_matches.py" in c]


def test_grade_folded_with_nothing_left_to_grade_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "grade_folded_soccer_rows.sh", _gf_rules(pre_plan=GF_ZERO))
    assert rc == 0 and "ALREADY DONE" in out and triggers == 0 and not _applies(calls), out


def test_grade_folded_full_run_writes_the_ids_and_pins_every_key(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "grade_folded_soccer_rows.sh", _gf_rules())
    assert rc == 0 and "DONE: 504 prediction(s) graded on the 26 folded row(s)" in out, out
    assert triggers == 1
    (apply_,) = _applies(calls)
    assert "--expect-plan matches=26,predictions_graded=504,predictions_regraded=0,recs_changed=0" in apply_
    ids_path = apply_.split("--match-ids-file ")[1].split()[0]
    assert fd.get_file(tmp_path, "api", ids_path).split() == FOLDED.split()


@pytest.mark.parametrize(
    "kw",
    [
        {"dry_plan": GF_PLAN.replace("recs_changed=0", "recs_changed=1")},
        {"dry_plan": GF_PLAN.replace("predictions_graded=504", "predictions_graded=505")},
        {"dry_plan": GF_PLAN.replace("predictions_regraded=0", "predictions_regraded=1")},
        {"dry_plan": GF_PLAN.replace("matches=26", "matches=25")},
        {"dry_plan": ""},
    ],
    ids=["recs", "graded", "regraded", "matches", "unreadable"],
)
def test_grade_folded_plan_outside_the_reviewed_set_aborts_before_any_write(tmp_path, kw):
    rc, out, calls, _ = _run(tmp_path, "grade_folded_soccer_rows.sh", _gf_rules(**kw))
    assert rc == 1 and "ABORT at step 2" in out and not _applies(calls), out


def test_grade_folded_more_rows_than_the_fd_repair_folded_aborts(tmp_path):
    many = FOLDED + "\n" + "99999999-2222-2222-2222-222222222222"
    rc, out, calls, triggers = _run(tmp_path, "grade_folded_soccer_rows.sh", _gf_rules(ids=many))
    assert rc == 1 and "27 folded rows, more than the fd repair's 26" in out and triggers == 0, out


def test_grade_folded_after_state_failure_exits_1(tmp_path):
    rc, out, _, _ = _run(tmp_path, "grade_folded_soccer_rows.sh", _gf_rules(verify_plan=GF_PLAN))
    assert rc == 1 and "FAILED after-state" in out, out
