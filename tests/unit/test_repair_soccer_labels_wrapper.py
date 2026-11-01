"""scripts/ops/repair_soccer_labels.sh, run for real against a scripted docker:
the read-only ALREADY DONE exit (no backup), a refusal or an unrepairable
value aborting before the backup, the dry-run ceilings and map accounting, an
apply pinned to the dry run's exact counts (every key), and the exact
after-state read independently of the tool."""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WRAPPER = HERE.parents[1] / "scripts" / "ops" / "repair_soccer_labels.sh"

_spec = importlib.util.spec_from_file_location("_ops_fake_docker", HERE / "_ops_fake_docker.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["_ops_fake_docker"] = fd
_spec.loader.exec_module(fd)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PLAN = (
    "PLAN create_league=2 relabel_raw=10 relabel_match=10 relabel_team=20 rename_team=1 rename_raw=68 "
    "already_raw=0 already_match=0 refuse=0 mojibake_left=0 cross_country_after=0"
)
DONE = (
    "PLAN create_league=0 relabel_raw=0 relabel_match=0 relabel_team=0 rename_team=0 rename_raw=0 "
    "already_raw=10 already_match=10 refuse=0 mojibake_left=0 cross_country_after=0"
)
VERIFY = "VERIFY rows=122 now_new=122 still_old=0 other=0 missing=0"
BEFORE = "10|1|68|1|1369|171581|168077"
AFTER = "0|0|0|3|1369|171581|168077"


def _rules(pre=PLAN, dry=PLAN, replan=DONE, verify=VERIFY, states=(BEFORE, AFTER), cross="-", apply_rc=0):
    return [
        {"match": r"/tmp/soccer_labels_apply\.log 2>&1", "out": f"tool exit {apply_rc}", "rc": apply_rc},
        {"match": r"/tmp/soccer_labels_\w+\.log 2>&1", "out": "tool exit 0"},
        {"match": r"grep '\^PLAN ' /tmp/soccer_labels_pre\.log", "out": pre},
        {"match": r"grep '\^PLAN ' /tmp/soccer_labels_dry\.log", "out": dry},
        {"match": r"grep '\^PLAN ' /tmp/soccer_labels_replan\.log", "out": replan},
        {
            "match": r"grep '\^PLAN_CROSS_COUNTRY ' /tmp/soccer_labels_\w+\.log",
            "out": f"PLAN_CROSS_COUNTRY names={cross}",
        },
        {"match": r"grep '\^VERIFY ' /tmp/soccer_labels_verify\.log", "out": verify},
        {"match": r"postgres sh -c psql", "out": states[0], "times": 1},
        {"match": r"postgres sh -c psql", "out": states[1], "times": 1},
        {"match": r"postgres rm -f", "out": ""},
    ]


def _run(tmp_path, rules, airflow=None):
    env = fd.install(tmp_path, rules, airflow)
    rc, out = fd.run(WRAPPER, env)
    return rc, out, fd.calls(tmp_path), fd.airflow_state(tmp_path).get("triggers", 0)


def _applies(calls):
    return [c for c in calls if "repair_soccer_labels.py" in c and "--apply" in c]


def test_a_clean_run_is_pinned_to_the_dry_run_and_checks_the_exact_after_state(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, _rules())
    assert rc == 0 and "DONE: 10 fixture(s) and 10 raw row(s) relabelled, 20 team(s)" in out, out
    assert triggers == 1
    (apply_,) = _applies(calls)
    pin = (
        "create_league=2,relabel_raw=10,relabel_match=10,relabel_team=20,rename_team=1,rename_raw=68,"
        "already_raw=0,already_match=0,refuse=0,mojibake_left=0,cross_country_after=0"
    )
    assert f"--expect-plan {pin}" in apply_ and "--out-csv /app/backups/soccer_labels_apply_" in apply_
    assert "--verify /app/backups/soccer_labels_apply_" in "\n".join(calls)
    assert "compute_features.py --backfill-finished soccer" in out
    sent = (tmp_path / "fake" / "cp" / "soccer_labels_q.sql").read_text()
    assert "external_ids->>'football_data' IN ('SP1', 'SP2')" in sent and "pat(p) AS (VALUES (" in sent


def test_fewer_left_to_do_than_reviewed_still_applies_pinned(tmp_path):
    dry = PLAN.replace("relabel_team=20", "relabel_team=14").replace("create_league=2", "create_league=1")
    rc, out, calls, _ = _run(tmp_path, _rules(dry=dry, states=(BEFORE.replace("|1|1369", "|2|1369"), AFTER)))
    assert rc == 0, out
    (apply_,) = _applies(calls)
    assert "relabel_team=14" in apply_ and "create_league=1" in apply_


@pytest.mark.parametrize(
    "pre",
    [
        PLAN.replace("refuse=0", "refuse=1"),
        PLAN.replace("mojibake_left=0", "mojibake_left=2"),
        PLAN.replace(" rename_raw=68", ""),  # unreadable
        "",
    ],
)
def test_a_refusal_or_an_unreadable_plan_aborts_before_the_backup(tmp_path, pre):
    rc, out, calls, triggers = _run(tmp_path, _rules(pre=pre))
    assert rc == 1 and "ABORT at step 0" in out, out
    assert triggers == 0 and not _applies(calls)


def test_nothing_left_is_already_done_without_a_backup(tmp_path):
    rc, out, calls, triggers = _run(tmp_path, _rules(pre=DONE))
    assert rc == 0 and "ALREADY DONE" in out and "WARNING" not in out, out
    assert triggers == 0 and not _applies(calls)
    assert "compute_features.py --backfill-finished soccer" in out


def test_already_done_with_unexpected_cross_country_teams_warns(tmp_path):
    rc, out, _, triggers = _run(tmp_path, _rules(pre=DONE, cross="Partick"))
    assert rc == 0 and "ALREADY DONE" in out and "WARNING: soccer teams whose domestic matches span" in out, out
    assert triggers == 0


@pytest.mark.parametrize(
    "dry, cross",
    [
        (PLAN.replace("relabel_team=20", "relabel_team=21"), "-"),
        (PLAN.replace("rename_raw=68", "rename_raw=69"), "-"),
        (PLAN.replace("create_league=2", "create_league=3"), "-"),
        (PLAN.replace("refuse=0", "refuse=1"), "-"),
        (PLAN.replace("mojibake_left=0", "mojibake_left=1"), "-"),
        (PLAN.replace("relabel_match=10", "relabel_match=9"), "-"),  # a map row unaccounted for
        (PLAN.replace("relabel_raw=10", "relabel_raw=9").replace("already_raw=0", "already_raw=2"), "-"),
        (PLAN, "Guadalajara,Partick"),
        (PLAN.replace(" cross_country_after=0", ""), "-"),  # unreadable
    ],
)
def test_the_dry_run_gate_aborts_before_any_apply(tmp_path, dry, cross):
    rules = _rules(dry=dry, cross=cross)
    rc, out, calls, _ = _run(tmp_path, rules)
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not _applies(calls)


def test_the_repoint_passes_cross_country_teams_are_allowed_before_the_runbook(tmp_path):
    dry = PLAN.replace("cross_country_after=0", "cross_country_after=2")
    rc, out, calls, _ = _run(tmp_path, _rules(dry=dry, cross="Guadalajara,Santos"))
    assert rc == 0, out
    assert "cross_country_after=2" in _applies(calls)[0]


def test_a_backup_that_never_lands_aborts(tmp_path):
    rc, out, calls, _ = _run(tmp_path, _rules(), airflow={"backup_fails": True})
    assert rc == 1 and "backup did not land" in out, out
    assert not _applies(calls)


def test_an_apply_that_fails_is_reported_as_nothing_written(tmp_path):
    rc, out, _, _ = _run(tmp_path, _rules(apply_rc=1))
    assert rc == 1 and "ABORT at step 3" in out and "nothing was written" in out, out


@pytest.mark.parametrize(
    "kw",
    [
        {"verify": VERIFY.replace("now_new=122", "now_new=121").replace("still_old=0", "still_old=1")},
        {"replan": DONE.replace("relabel_team=0", "relabel_team=1")},
        {"replan": DONE.replace("already_match=10", "already_match=9")},
        {"states": (BEFORE, AFTER.replace("0|0|0|3", "1|0|0|3"))},  # a mapped fixture still in La Liga
        {"states": (BEFORE, AFTER.replace("0|0|0|3", "0|1|0|3"))},  # a mis-decoded team name left
        {"states": (BEFORE, AFTER.replace("|3|", "|4|"))},  # a league too many
        {"states": (BEFORE, AFTER.replace("|1369|", "|1370|"))},  # a team inserted
        {"states": (BEFORE, AFTER.replace("|168077", "|168076"))},  # a raw row lost
        {"states": (BEFORE, "")},  # unreadable
    ],
)
def test_a_wrong_after_state_fails(tmp_path, kw):
    rc, out, _, _ = _run(tmp_path, _rules(**kw))
    assert rc == 1 and "FAILED after-state" in out, out


def test_an_apply_still_running_in_the_container_refuses_before_anything(tmp_path):
    procs = ["python /app/scripts/repair_soccer_labels.py --apply --expect-plan create_league=2"]
    rc, out, calls, triggers = _run(tmp_path, _rules(), airflow={"procs": procs})
    assert rc == 1 and "REFUSED: a repair_soccer_labels.py --apply is still running" in out, out
    assert triggers == 0 and not [c for c in calls if "repair_soccer_labels.py" in c]


def test_the_after_state_sql_carries_text_encodings_pattern():
    # The wrapper reads mis-decoded names with Postgres `~`, independently of the tool: the same pattern, verbatim.
    spec = importlib.util.spec_from_file_location("text_encoding", HERE.parents[1] / "scripts" / "text_encoding.py")
    te = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(te)
    (sql_pattern,) = re.findall(r"pat\(p\) AS \(VALUES \('([^']+)'\)\)", WRAPPER.read_text())
    assert sql_pattern == te.MOJIBAKE_PATTERN
