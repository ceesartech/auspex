"""The soccer runbook's steps 12, 14 and 15, run for real against a scripted docker
(tests/unit/_ops_fake_docker.py): scripts/ops/backfill_soccer_never_resulted.sh,
promote_raw_after_repair.sh and grade_folded_soccer_rows.sh. Each re-plans
read-only first and exits 0 ('ALREADY DONE') without a backup or a write when
nothing is left; otherwise backup, dry run inside the reviewed bounds, apply
pinned to that dry run's counts, and an after-state that must re-plan to
nothing. The tools ran end to end through a docker shim against a replica of
prod (their wrappers' headers give the counts)."""

from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import sys
from collections import Counter
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
    "PLAN candidates=38 heal=38 done=0 refuse=0 predictions_graded=3173 predictions_regraded=0 recs_changed=0 "
    "teams_stamped=3 espn_failed=0"
)
NR_ZERO = (
    "PLAN candidates=38 heal=0 done=0 refuse=38 predictions_graded=0 predictions_regraded=0 recs_changed=0 "
    "teams_stamped=0 espn_failed=0"
)
NR_VERIFY = (
    "PLAN candidates=38 heal=0 done=38 refuse=0 predictions_graded=0 predictions_regraded=0 recs_changed=0 "
    "teams_stamped=0 espn_failed=0"
)
HEALED = "\n".join(f"{i:08d}-0000-0000-0000-000000000000" for i in range(38))
# The reviewed list's 4 'leave' rows (the wrapper's split prints each one it leaves alone).
LEFT_REVIEWED = [
    "fd640b32-3ab4-4f15-9f48-20113d099564 2026-06-15T00:00 Coquimbo Unido v O'Higgins - reviewed, left alone: a fork",
    "0e1a768b-ec23-4a2c-b4c7-5f6d7be807e0 2026-09-05T23:00 Queretaro v Monterrey - reviewed, left alone: not on ESPN",
    "add92a04-78db-4d90-89c8-5edf329be034 2026-10-10T12:00 Vallecano v Ath Bilbao - reviewed, left alone: re-dated",
    "8a9a1d8d-95f6-4785-85d8-265ed2d377c0 2026-10-17T16:30 Atromitos v AEK - reviewed, left alone: postponed",
]
UNREVIEWED = (
    "5b94347b-daec-4117-aa09-710cd860a2b6 2026-11-01T23:00 Santos Laguna v Necaxa - NOT in the reviewed list: "
    "left as it is, for a separate review"
)
# Guadalajara Chivas v Atlante (ESPN 401876915, final 4-2) was reviewed ahead: a 'heal' row, never_resulted only when
# the runbook starts late. Its dry run alone (replica after steps 1-11, as of 2026-11-06): 76 graded, nothing stamped.
GVA = "6b2133b1-e702-4705-80f9-0218913bbdeb"
NR_PLAN_LATE = (
    "PLAN candidates=39 heal=39 done=0 refuse=0 predictions_graded=3249 predictions_regraded=0 recs_changed=0 "
    "teams_stamped=3 espn_failed=0"
)


def _split(n_ids, n_all, left=tuple(LEFT_REVIEWED)):
    """What the wrapper's split of the orphans CSV prints: a LEFT line per row it leaves alone, then the counts."""
    return "\n".join([f"LEFT {line}" for line in left] + [f"COUNTS {n_ids} {n_all}"])


def _nr_rules(
    pre_split=_split(38, 42),
    pre_plan=NR_PLAN,
    pre_rc=0,
    orphans=ORPHANS_DONE,
    dry_split=_split(38, 42),
    dry_plan=NR_PLAN,
    apply_rc=0,
    verify_plan=NR_VERIFY,
    verify_split=_split(0, 3, LEFT_REVIEWED[1:]),
    healed="38",
    state="38|0",
    new_rows="0",
):
    split = r"orphans, reviewed, ids_out, all_out = sys\.argv.*never_resulted_orphans_"
    return [
        {"match": r"/tmp/never_resulted_orphans_\w+\.log 2>&1; echo", "out": "orphans dry run exit 0"},
        {"match": r"grep -o 'PLAN \.\*' /tmp/never_resulted_orphans_precheck\.log", "out": ORPHANS_DONE},
        {"match": r"grep -o 'PLAN \.\*' /tmp/never_resulted_orphans_dryrun\.log", "out": orphans},
        {"match": r"grep -o 'PLAN \.\*' /tmp/never_resulted_orphans_verify\.log", "out": ORPHANS_AFTER},
        {"match": split + "precheck_", "out": pre_split},
        {"match": split + "dryrun_", "out": dry_split},
        {"match": split + "verify_", "out": verify_split},
        {"match": r"python -c import csv, sys; ids = \[r\['match_id'\] .*r\['action'\] == 'heal'", "out": healed},
        {"match": r"/tmp/nr_pre\.log 2>&1", "out": "tool exit", "rc": pre_rc},
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


def _tool_calls(calls):
    return [c for c in calls if "backfill_never_resulted.py" in c]


def test_never_resulted_with_no_reviewed_row_left_is_already_done(tmp_path):
    rules = _nr_rules(pre_split=_split(0, 4))
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", rules)
    assert rc == 0 and "ALREADY DONE: none of the reviewed never_resulted rows is left to heal" in out, out
    assert "4 never_resulted row(s) left alone" in out and "Atromitos v AEK - reviewed, left alone" in out
    assert triggers == 0 and not _tool_calls(calls)


def test_never_resulted_with_nothing_to_heal_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(pre_plan=NR_ZERO))
    assert rc == 0 and "ALREADY DONE: nothing to heal" in out, out
    assert triggers == 0 and not _applies(calls)
    (pre,) = _tool_calls(calls)
    assert "--espn-cache /app/backups/never_resulted_espn_precheck_" in pre
    assert "--match-ids-file /app/backups/never_resulted_ids_precheck_" in pre


@pytest.mark.parametrize(
    "kw",
    [
        {"pre_plan": NR_ZERO.replace("espn_failed=0", "espn_failed=6")},  # partly failed: every row near them refused
        {"pre_plan": NR_PLAN.replace("espn_failed=0", "espn_failed=1")},
        {"pre_rc": 1},
        {"pre_plan": ""},
    ],
    ids=["espn-failed-nothing-healable", "espn-failed-some-healable", "tool-failed", "unreadable"],
)
def test_never_resulted_precheck_that_failed_aborts_before_the_backup(tmp_path, kw):
    """A partial ESPN outage refuses the rows near the failed days, so 'heal=0' is not the real plan: it must not
    read ALREADY DONE (the runbook would record step 12 done with the rows unhealed), and no backup is spent."""
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(**kw))
    assert rc == 1 and "ABORT at step 0" in out and "nothing changed" in out, out
    assert "ALREADY DONE" not in out and triggers == 0 and not _applies(calls)


def test_never_resulted_full_run_pins_every_key_and_reuses_the_dry_runs_espn_cache(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules())
    assert rc == 0 and "DONE: 38 never_resulted row(s) healed and graded (3173 predictions)" in out, out
    assert triggers == 1
    tool = _tool_calls(calls)
    dry, apply_, verify = tool[1], tool[2], tool[3]
    assert "--apply" not in dry and "--apply" not in verify
    cache = dry.split("--espn-cache ")[1].split()[0]
    assert "_precheck_" not in cache and f"--espn-cache {cache}" in apply_ and f"--espn-cache {cache}" in verify
    ids = dry.split("--match-ids-file ")[1].split()[0]
    assert ids.startswith("/app/backups/never_resulted_ids_dryrun_")  # the reviewed 'heal' rows, not _all_
    assert f"--match-ids-file {ids}" in apply_ and f"--match-ids-file {ids}" in verify
    assert (
        "--expect-plan candidates=38,heal=38,done=0,refuse=0,predictions_graded=3173,predictions_regraded=0,"
        "recs_changed=0,teams_stamped=3,espn_failed=0" in apply_
    )
    backup = next(i for i, c in enumerate(calls) if "db_backup_daily" in c)
    assert backup < calls.index(dry) < calls.index(apply_)
    assert "reviewed rows still never_resulted 0 (at most the 0 refused), never_resulted rows now 3, new ones 0" in out
    splits = [c for c in calls if "orphans, reviewed, ids_out, all_out" in c]
    assert len(splits) == 3 and all("/app/scripts/data/soccer_never_resulted_reviewed.csv" in c for c in splits)
    (new_rows,) = [c for c in calls if "before = set" in c]  # every never_resulted row, not only the reviewed ones
    assert "never_resulted_all_dryrun_" in new_rows and "never_resulted_all_verify_" in new_rows


def test_a_never_resulted_row_outside_the_reviewed_list_is_listed_and_left_alone(tmp_path):
    """The class grows (a fixture whose result never arrives joins it 96 h after kickoff): such a row neither aborts
    the step nor reaches the tool — it is listed, and the after-state allows it because it was there before."""
    left = LEFT_REVIEWED + [UNREVIEWED]
    rules = _nr_rules(
        pre_split=_split(38, 43, left), dry_split=_split(38, 43, left), verify_split=_split(0, 4, left[1:])
    )
    rc, out, calls, _ = _run(tmp_path, "backfill_soccer_never_resulted.sh", rules)
    assert rc == 0 and "DONE: 38 never_resulted row(s) healed" in out, out
    assert "Santos Laguna v Necaxa - NOT in the reviewed list" in out
    assert "5 left alone" in out and "never_resulted rows now 4, new ones 0" in out
    assert "candidates=38," in _applies(calls)[0]


def test_a_late_start_heals_the_reviewed_liga_mx_fixture_within_the_bounds(tmp_path):
    """Started after the sweep's look-back passed Guadalajara v Atlante and 96 h after its kickoff: it is a
    never_resulted row with a reviewed 'heal' row, so it is passed with the other 38 and the plan 39 / 39 / 3,249 / 3
    is inside the bounds and applied, pinned."""
    verify = NR_VERIFY.replace("candidates=38", "candidates=39").replace("done=38", "done=39")
    rules = _nr_rules(
        pre_split=_split(39, 43),
        dry_split=_split(39, 43),
        pre_plan=NR_PLAN_LATE,
        dry_plan=NR_PLAN_LATE,
        verify_plan=verify,
        healed="39",
        state="39|0",
    )
    rc, out, calls, _ = _run(tmp_path, "backfill_soccer_never_resulted.sh", rules)
    assert rc == 0 and "DONE: 39 never_resulted row(s) healed and graded (3249 predictions)" in out, out
    assert (
        "--expect-plan candidates=39,heal=39,done=0,refuse=0,predictions_graded=3249,predictions_regraded=0,"
        "recs_changed=0,teams_stamped=3,espn_failed=0" in _applies(calls)[0]
    )


@pytest.mark.parametrize(
    "plan",
    [
        NR_PLAN.replace("heal=38", "heal=40"),
        NR_PLAN.replace("candidates=38", "candidates=40"),
        NR_PLAN.replace("candidates=38", "candidates=36"),  # fewer than the 38 passed: not the list it was given
        NR_PLAN.replace("recs_changed=0", "recs_changed=1"),
        NR_PLAN.replace("predictions_regraded=0", "predictions_regraded=2"),
        NR_PLAN.replace("espn_failed=0", "espn_failed=1"),
        NR_PLAN.replace("teams_stamped=3", "teams_stamped=4"),
        NR_PLAN.replace("predictions_graded=3173", "predictions_graded=3250"),
        NR_PLAN.replace("done=0", "done=1"),
        "",
    ],
    ids=["heal", "candidates", "candidates-not-passed", "recs", "regraded", "espn", "stamped", "graded", "done", "-"],
)
def test_never_resulted_plan_outside_the_reviewed_set_aborts_before_any_write(tmp_path, plan):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(dry_plan=plan))
    assert rc == 1 and "ABORT at step 2" in out and "nothing changed" in out, out
    assert not _applies(calls)


def test_never_resulted_refuses_while_the_cleanup_is_not_done(tmp_path):
    rules = _nr_rules(orphans=ORPHANS_DONE.replace("merge_delete=0", "merge_delete=2"))
    rc, out, calls, _ = _run(tmp_path, "backfill_soccer_never_resulted.sh", rules)
    assert rc == 1 and "the soccer orphan cleanup is not done" in out and "runbook step 7" in out, out
    assert not [c for c in _tool_calls(calls) if "dryrun" in c]


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
        {"verify_split": _split(1, 4)},
        {"healed": "37"},
    ],
    ids=["re-plan-heals", "not-finished", "open-recs", "new-never-resulted", "reviewed-left-unrefused", "csv-heals"],
)
def test_never_resulted_after_state_failures_exit_1(tmp_path, kw):
    rc, out, _, _ = _run(tmp_path, "backfill_soccer_never_resulted.sh", _nr_rules(**kw))
    assert rc == 1 and "FAILED after-state" in out, out


def _split_program():
    """The wrapper's SPLIT_PY, exactly as committed."""
    text = (OPS / "backfill_soccer_never_resulted.sh").read_text()
    return text.split('SPLIT_PY="', 1)[1].split('"\nnever_resulted_ids() {', 1)[0]


def _orphans_csv(path, rows):
    fields = ["match_id", "action", "reason", "match_date", "home", "away"]
    lines = [",".join(fields)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_the_split_passes_only_the_reviewed_heal_rows_and_lists_the_rest(tmp_path):
    reviewed = HERE.parents[1] / "scripts" / "data" / "soccer_never_resulted_reviewed.csv"
    heal = "b06808bf-385e-48cc-aefd-de83092a90f7"  # a 'heal' row of the committed list
    leave = "8a9a1d8d-95f6-4785-85d8-265ed2d377c0"  # Atromitos v AEK: 'leave'
    other = "5b94347b-daec-4117-aa09-710cd860a2b6"  # Santos Laguna v Necaxa: not in the list
    merged = "11111111-1111-1111-1111-111111111111"  # another class: not a never_resulted row
    _orphans_csv(
        tmp_path / "orphans.csv",
        [
            [heal, "leave", "never_resulted", "2026-05-28 22:00:00+00", "Cerro Porteño", "Sporting Cristal"],
            [leave, "leave", "never_resulted", "2026-10-17 16:30:00+00", "Atromitos", "AEK"],
            [other, "leave", "never_resulted", "2026-11-01 23:00:00+00", "Santos Laguna", "Necaxa"],
            [GVA, "leave", "never_resulted", "2026-11-01 01:07:00+00", "Guadalajara Chivas", "Atlante"],
            [merged, "merge_cancel", "twin", "2026-06-15 00:00:00+00", "Coquimbo Unido", "O'Higgins"],
        ],
    )
    args = [str(tmp_path / n) for n in ("orphans.csv",)] + [str(reviewed)]
    args += [str(tmp_path / "ids.txt"), str(tmp_path / "all.txt")]
    proc = subprocess.run([sys.executable, "-c", _split_program(), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "ids.txt").read_text().split() == [heal, GVA]  # the reviewed 'heal' rows, in orphans order
    assert (tmp_path / "all.txt").read_text().split() == [heal, leave, other, GVA]
    lines = proc.stdout.splitlines()
    assert lines[-1] == "COUNTS 2 4" and len(lines) == 3
    assert lines[0].startswith(f"LEFT {leave} 2026-10-17 16:30 Atromitos v AEK - reviewed, left alone: ESPN 401896749")
    assert lines[1].startswith(f"LEFT {other} 2026-11-01 23:00 Santos Laguna v Necaxa - NOT in the reviewed")
    again = subprocess.run([sys.executable, "-c", _split_program(), *args], capture_output=True, text=True)
    assert again.returncode != 0 and "FileExistsError" in again.stderr  # write-once


def test_the_split_refuses_a_reviewed_list_with_an_unknown_decision(tmp_path):
    (tmp_path / "reviewed.csv").write_text("match_id,decision,evidence\nx,maybe,?\n")
    _orphans_csv(tmp_path / "orphans.csv", [])
    args = [str(tmp_path / n) for n in ("orphans.csv", "reviewed.csv", "ids.txt", "all.txt")]
    proc = subprocess.run([sys.executable, "-c", _split_program(), *args], capture_output=True, text=True)
    assert proc.returncode != 0 and "a decision other than heal/leave" in proc.stderr
    assert not (tmp_path / "ids.txt").exists()


def test_the_reviewed_list_is_the_rehearsals_38_heals_4_left_alone_and_the_liga_mx_fixture_reviewed_ahead():
    rows = list(csv.DictReader((HERE.parents[1] / "scripts" / "data" / "soccer_never_resulted_reviewed.csv").open()))
    assert len(rows) == 43 and len({r["match_id"] for r in rows}) == 43
    assert sorted(Counter(r["decision"] for r in rows).items()) == [("heal", 39), ("leave", 4)]
    assert all(r["espn_id"] and r["espn_final"] for r in rows if r["decision"] == "heal")
    (gva,) = [r for r in rows if r["match_id"] == GVA]
    assert (gva["decision"], gva["espn_id"], gva["espn_final"], gva["kickoff"]) == (
        "heal",
        "401876915",
        "4-2",
        "2026-11-01T01:07:00+00:00",
    )
    assert "5b94347b-daec-4117-aa09-710cd860a2b6" not in {r["match_id"] for r in rows}  # not played when reviewed


# --- promote_raw_after_repair.sh (step 14) ------------------------------------------------------------------------

KEYS = (
    "matches_inserted matches_changed match_stats_inserted match_stats_changed odds_inserted teams_inserted "
    "teams_changed leagues_inserted leagues_changed predictions_changed recs_changed refused duplicates refused_raw"
).split()


def _promote_plan(**over):
    vals = {k: 0 for k in KEYS}
    vals.update(over)
    return "PLAN " + " ".join(f"{k}={vals[k]}" for k in KEYS)


PR_PLAN = _promote_plan(match_stats_inserted=4, odds_inserted=13)
PR_ZERO = _promote_plan()
ALIASED = " ".join(f"{i:08d}-1111-1111-1111-111111111111" for i in range(4))


def _pr_rules(
    pre_plan=PR_PLAN, pre_rc=0, dry_plan=PR_PLAN, verify_plan=PR_ZERO, sides="4/4", aliased=ALIASED, apply_rc=0
):
    return [
        {"match": r"/tmp/promote_after_pre\.log 2>&1", "out": "tool exit", "rc": pre_rc},
        {"match": r"/tmp/promote_after_apply\.log 2>&1", "out": "tool exit", "rc": apply_rc},
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


@pytest.mark.parametrize(
    "kw, why",
    [
        ({"pre_rc": 3, "pre_plan": _promote_plan(match_stats_inserted=4, refused=1)}, "refused, found duplicated"),
        # the 68 mis-decoded Preussen Muenster rows before the labels repair (runbook step 13) has run
        ({"pre_rc": 3, "pre_plan": _promote_plan(match_stats_inserted=4, refused_raw=68)}, "repair_soccer_labels.sh"),
        ({"pre_rc": 2}, "the checked dry run failed (exit 2"),
        ({"pre_plan": ""}, "no readable PLAN line"),
        # a PLAN line without refused_raw (a promote_raw older than the wrapper) is not read as 0
        ({"pre_plan": PR_PLAN.replace(" refused_raw=0", "")}, "no readable PLAN line"),
    ],
    ids=["refused-row", "screened-rows", "failed", "unreadable", "no-refused-raw-key"],
)
def test_promote_precheck_that_cannot_be_applied_aborts_before_the_backup(tmp_path, kw, why):
    """A refused or duplicated raw row (exit 3) is outside the reviewed set anyway: no 20-minute backup for it."""
    rc, out, calls, triggers = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(**kw))
    assert rc == 1 and "ABORT at step 0" in out and why in out and "nothing changed" in out, out
    assert triggers == 0 and not _applies(calls)


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
        _promote_plan(match_stats_inserted=4, refused_raw=1),
        PR_PLAN.replace(" refused_raw=0", ""),
        "",
    ],
    ids=["stats", "odds", "matches", "recs", "teams", "refused", "refused-raw", "no-refused-raw-key", "unreadable"],
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


def test_promote_apply_refused_says_rolled_back_nothing_changed(tmp_path):
    rc, out, calls, _ = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(apply_rc=1))
    assert rc == 1 and "ABORT at step 3" in out and "rolled it back, nothing changed" in out, out
    assert not [c for c in calls if "promote_after_verify" in c]


def test_promote_apply_exit_3_is_reported_committed_not_rolled_back(tmp_path):
    """A checked --apply exits 3 only after its commit (promote_raw's EXIT STATUS): step 2's bounds make it
    unreachable, but if it happens the wrapper must not claim a rollback."""
    rc, out, calls, _ = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(apply_rc=3))
    assert rc == 1 and "FAILED at step 3" in out and "AFTER COMMITTING" in out and "IS committed" in out, out
    assert "rolled it back" not in out and "nothing changed" not in out.split("3/4 apply", 1)[1]
    assert not [c for c in calls if "promote_after_verify" in c]


def test_promote_without_an_alias_merge_listed_skips_that_check(tmp_path):
    rc, out, _, _ = _run(tmp_path, "promote_raw_after_repair.sh", _pr_rules(aliased=""))
    assert rc == 0 and "no alias-merged fixture listed" in out, out


# --- grade_folded_soccer_rows.sh (step 15) ------------------------------------------------------------------------

FOLDED = "\n".join(f"{i:08d}-2222-2222-2222-222222222222" for i in range(26))
GF_PLAN = "PLAN matches=26 predictions_graded=504 predictions_regraded=0 recs_changed=0"
GF_ZERO = "PLAN matches=26 predictions_graded=0 predictions_regraded=0 recs_changed=0"


def _gf_rules(ids=FOLDED, pre_plan=GF_PLAN, dry_plan=GF_PLAN, verify_plan=GF_ZERO, apply_rc=0, pre_rc=0):
    return [
        {"match": r"postgres sh -c psql", "out": ids},
        {"match": r"/tmp/grade_folded_pre\.log 2>&1", "out": "tool exit", "rc": pre_rc},
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


@pytest.mark.parametrize("kw", [{"pre_rc": 1}, {"pre_plan": ""}], ids=["failed", "unreadable"])
def test_grade_folded_precheck_that_failed_aborts_before_the_backup(tmp_path, kw):
    rc, out, calls, triggers = _run(tmp_path, "grade_folded_soccer_rows.sh", _gf_rules(**kw))
    assert rc == 1 and "ABORT at step 0" in out and "nothing changed" in out, out
    assert triggers == 0 and not _applies(calls)


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
