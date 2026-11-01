"""The runbook's re-run contract, per wrapper: each re-plans read-only first
and exits 0 ('ALREADY DONE') without a backup or a write when nothing is left,
and goes the full way otherwise. Run for real against a scripted docker
(repair_football_data_kickoffs, backfill_soccer_regulation_scores,
backfill_season_labels; the merge, repoint, stranded-recs, cleanup and 027
wrappers have their own test files). Also the fd repair's full pin and its
DONE wording after alias merges."""

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

ZERO_DEDUPE = "PLAN mode=dedupe merge=0 alias_merge=0 fold=0 refuse=0 fold_unsettled_recs=0"
ZERO_SHIFT = "PLAN mode=shift shift=0 refuse=0 dedupe_pending=0"
DEDUPE = "PLAN mode=dedupe merge=68 alias_merge=4 fold=26 refuse=0 fold_unsettled_recs=0"
SHIFT = "PLAN mode=shift shift=63532 refuse=0 dedupe_pending=98"
SHIFT2 = "PLAN mode=shift shift=63532 refuse=0 dedupe_pending=0"


def _run(tmp_path, wrapper, rules):
    env = fd.install(tmp_path, rules)
    rc, out = fd.run(OPS / wrapper, env)
    return rc, out, fd.calls(tmp_path), fd.airflow_state(tmp_path).get("triggers", 0)


# --- repair_football_data_kickoffs.sh ----------------------------------------------------------------------------


def _fd_rules(pre_dedupe=ZERO_DEDUPE, pre_shift=ZERO_SHIFT, pre_check_rc=0, aliased="a1 a2 a3 a4"):
    return [
        {"match": r"/tmp/fd_kickoffs_check_pre\.log 2>&1", "out": "tool exit", "rc": pre_check_rc},
        {"match": r"grep -o 'PLAN \.\*' /tmp/fd_kickoffs_dedupe_pre\.log", "out": pre_dedupe},
        {"match": r"grep -o 'PLAN \.\*' /tmp/fd_kickoffs_shift_pre\.log", "out": pre_shift},
        {"match": r"grep -o 'PLAN \.\*' /tmp/fd_kickoffs_dedupe_dry\.log", "out": DEDUPE},
        {"match": r"grep -o 'PLAN \.\*' /tmp/fd_kickoffs_shift_dry\.log", "out": SHIFT},
        {"match": r"grep -o 'PLAN \.\*' /tmp/fd_kickoffs_shift_dry2\.log", "out": SHIFT2},
        {"match": r"grep -o 'PLAN \.\*' /tmp/fd_kickoffs_dedupe_verify\.log", "out": ZERO_DEDUPE},
        {"match": r"grep -o 'PLAN \.\*' /tmp/fd_kickoffs_shift_verify\.log", "out": ZERO_SHIFT},
        {"match": r"python -c import csv, sys; print.*football_data_dedupe_apply_\S+ fold$", "out": "f1 f2"},
        {"match": r"python -c import csv, sys; print.*football_data_dedupe_apply_\S+ alias_merge$", "out": aliased},
        {"match": r"/tmp/fd_kickoffs_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"postgres sh -c psql", "out": "1000|300", "times": 1},
        {"match": r"postgres sh -c psql", "out": "902|274", "times": 1},
        {"match": r"postgres sh -c psql", "out": "0"},
        {"match": r"postgres rm -f", "out": ""},
    ]


def test_fd_nothing_left_and_check_passing_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "repair_football_data_kickoffs.sh", _fd_rules())
    assert rc == 0 and "ALREADY DONE" in out, out
    assert triggers == 0 and not [c for c in calls if "--apply" in c]


@pytest.mark.parametrize(
    "kw", [{"pre_check_rc": 1}, {"pre_shift": ZERO_SHIFT.replace("shift=0", "shift=5")}, {"pre_dedupe": DEDUPE}]
)
def test_fd_anything_left_goes_the_full_way(tmp_path, kw):
    rc, out, calls, triggers = _run(tmp_path, "repair_football_data_kickoffs.sh", _fd_rules(**kw))
    assert "ALREADY DONE" not in out and triggers == 1, out


def test_fd_full_run_pins_every_key_and_words_done_for_the_alias_merges(tmp_path):
    rc, out, calls, _ = _run(tmp_path, "repair_football_data_kickoffs.sh", _fd_rules(pre_dedupe=DEDUPE))
    assert rc == 0, out
    (apply_,) = [c for c in calls if "--dedupe --apply" in c]
    assert "--expect-plan merge=68,alias_merge=4,fold=26,refuse=0,fold_unsettled_recs=0" in apply_
    assert "promote_raw runs and refuses the 4 alias-merged fixture(s) (exit 3)" in out
    assert "the 68 mis-decoded raw rows until scripts/ops/repair_soccer_labels.sh, runbook step 13" in out
    assert "promote_raw may run again" not in out
    assert "repoint_team_matches.sh --pass 2" in out


def test_fd_full_run_without_alias_merges_says_promote_raw_may_start_but_exits_3_until_the_labels_repair(tmp_path):
    """Until scripts/ops/repair_soccer_labels.sh (runbook step 13) has run, promote_raw commits everything else but
    exits 3 on the 68 mis-decoded raw rows its screen refuses: 'promote_raw may run again' was no longer true."""
    rc, out, _, _ = _run(tmp_path, "repair_football_data_kickoffs.sh", _fd_rules(pre_dedupe=DEDUPE, aliased=""))
    assert rc == 0 and "check_legacy_kickoffs passes, so promote_raw no longer refuses to start" in out, out
    assert "exits 3 on the 68 mis-decoded raw rows" in out and "repair_soccer_labels.sh (runbook step 13)" in out
    assert "promote_raw may run again" not in out and "--pass 2 —" not in out
    assert "runbook step 15 grades the rest of their predictions" in out


# --- backfill_soccer_regulation_scores.sh -------------------------------------------------------------------------


def _reg_rules(changes, pre="PLAN candidates=15 stamp=0 already_set=15 refuse=0", left="0"):
    return [
        {"match": r"/tmp/regulation_scores_pre\.log 2>&1; echo", "out": "tool exit 0"},
        {"match": r"grep -o 'PLAN candidates=\.\*' /tmp/regulation_scores_pre\.log", "out": pre},
        {"match": r"postgres sh -c psql", "out": left},
        {"match": r"api python -c import csv, sys; ids = sorted", "out": "15"},
        {
            "match": r"api python -c import sys; sys\.path\.insert\(0, '/app/scripts'\); import grading_outcomes",
            "out": "btts,match_result",
        },
        {"match": r"/tmp/regulation_regrade_dry\.log 2>&1; rc", "out": "Scanned"},
        {"match": r"grep -o '\[0-9\]\* change\(s\) planned' /tmp/regulation_regrade_dry\.log", "out": changes},
        {"match": r"postgres rm -f", "out": ""},
    ]


def test_regulation_stamped_and_regraded_is_already_done(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_regulation_scores.sh", _reg_rules("0"))
    assert rc == 0 and "ALREADY DONE" in out and "changes=0" in out, out
    assert triggers == 0 and not [c for c in calls if "--apply" in c]


def test_regulation_stamped_with_the_regrade_still_to_apply_exits_0_and_prints_it(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, "backfill_soccer_regulation_scores.sh", _reg_rules("135"))
    assert rc == 0 and "the re-grade above is still to apply" in out, out
    assert "REGRADE ids_file=/app/backups/soccer_regulation_regrade_ids_precheck_" in out and "changes=135" in out
    assert "--apply --expect-changes 135" in out and triggers == 0


def test_regulation_regrade_outside_the_range_fails(tmp_path):
    rc, out, _, triggers = _run(tmp_path, "backfill_soccer_regulation_scores.sh", _reg_rules("150"))
    assert rc == 1 and "CHECK" in out and triggers == 0, out


def test_regulation_rows_left_to_stamp_go_the_full_way(tmp_path):
    rules = _reg_rules("0", pre="PLAN candidates=15 stamp=15 already_set=0 refuse=0", left="15")
    rc, out, _, triggers = _run(tmp_path, "backfill_soccer_regulation_scores.sh", rules)
    assert "ALREADY DONE" not in out and triggers == 1, out


# --- backfill_season_labels.sh -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan, done",
    [
        ("PLAN relabel=0 relabel_promote_raw=0 relabel_espn=0 relabel_bad_shape=0 already_correct=171112", True),
        ("PLAN relabel=5 relabel_promote_raw=0 relabel_espn=5 relabel_bad_shape=0 already_correct=171107", False),
        ("", False),
    ],
)
def test_season_labels_already_done_only_with_nothing_to_relabel(tmp_path, plan, done):
    rules = [
        {"match": r"/tmp/season_labels_pre\.log 2>&1; echo", "out": "tool exit 0"},
        {"match": r"grep '\^PLAN ' /tmp/season_labels_pre\.log", "out": plan},
    ]
    rc, out, calls, triggers = _run(tmp_path, "backfill_season_labels.sh", rules)
    assert ("ALREADY DONE" in out) == done, out
    assert (rc == 0) if done else (triggers == 1)
