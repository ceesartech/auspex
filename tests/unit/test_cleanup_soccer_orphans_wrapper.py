"""scripts/ops/cleanup_soccer_orphans.sh, run for real through a canned
`docker` (no database, no containers): every tool run and both state queries
share one --as-of instant, a failed step 5 or 6 exits 2, never DONE, and its
apply CSV stays on the pending list until a re-run (or --followups-only) has
run steps 5-6 for it — a re-run never says ALREADY DONE over them."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "cleanup_soccer_orphans.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# Answers the wrapper's docker calls from the environment; logs every call.
FAKE_DOCKER = r"""#!/bin/bash
echo "$*" >> "$FAKE_DIR/calls.log"
args="$*"
case "$args" in
  *"sh -c cat /app/backups/cleanup_soccer_orphans_followups_pending.txt 2>/dev/null")
    [ -f "$FAKE_DIR/pending" ] && cat "$FAKE_DIR/pending"; exit 0 ;;
  *"sh -c cat > /app/backups/cleanup_soccer_orphans_followups_pending.txt.tmp && mv "*)
    cat > "$FAKE_DIR/pending"; exit 0 ;;
  "compose exec -T airflow-scheduler"*) exit 0 ;;
  *"import b2_io"*) echo "$(( $(date +%s) + 60 )) postgres/fake.dump"; exit 0 ;;
  "compose cp "*)
    n=$(ls "$FAKE_DIR" | grep -c '^state_sql\.')
    cp "$3" "$FAKE_DIR/state_sql.$n"; exit 0 ;;
  "compose exec -T postgres sh -c psql"*)
    n=$(ls "$FAKE_DIR" | grep -c '^psql_call\.')
    : > "$FAKE_DIR/psql_call.$n"
    if [ "$n" = 0 ]; then echo "$BEFORE_STATE"; else echo "$AFTER_STATE"; fi
    exit 0 ;;
  "compose exec -T postgres rm"*) exit 0 ;;
  *"grep -o 'PLAN .*' /tmp/rfo_soccer_pre.log"*) echo "${PRE_PLAN:-$DRY_PLAN}"; exit 0 ;;
  *"grep -o 'PLAN .*' /tmp/rfo_soccer_dry.log"*) echo "$DRY_PLAN"; exit 0 ;;
  *"grep -o 'PLAN .*' /tmp/rfo_soccer_verify.log"*) echo "$VERIFY_PLAN"; exit 0 ;;
  *"repair_fixture_orphans.py"*) echo "tool exit 0"; exit 0 ;;
  *"python -c import csv"*) echo "11111111-1111-1111-1111-111111111111,22222222-2222-2222-2222-222222222222"; exit 0 ;;
  *"compute_features.py"*) exit "${FAIL_FEATURES:-0}" ;;
  *" api python -") cat > /dev/null; exit "${FAIL_GRADE:-0}" ;;
esac
echo "unexpected docker call: $args" >&2
exit 99
"""

DRY_PLAN = (
    "PLAN merge_delete=3 merge_cancel=1 delete_no_twin=0 delete_placeholder=2 mark_postponed=1 left=5 "
    "blocked_pending_recs=0 list_mismatch=0"
)
ZERO_PLAN = (
    "PLAN merge_delete=0 merge_cancel=0 delete_no_twin=0 delete_placeholder=0 mark_postponed=0 left=5 "
    "blocked_pending_recs=0 list_mismatch=0"
)
BEFORE = "100|2|500|0"  # stale | orphan_of | recs on past fixtures | postponed by the tool
AFTER = "93|3|500|1"  # 100 - 3 - 1 - 2 - 1, 2 + 1, unchanged, 0 + 1


def _run(tmp_path: Path, *args: str, **env: str) -> tuple[int, str, list[str]]:
    """Runs the wrapper; a second call on the same tmp_path keeps the fake's
    files (the pending list) and appends to its call log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "sleep").write_text("#!/bin/bash\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (tmp_path / "auspex").mkdir(exist_ok=True)
    calls_log = fake / "calls.log"
    before = len(calls_log.read_text().splitlines()) if calls_log.exists() else 0
    full_env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "AUSPEX_DIR": str(tmp_path / "auspex"),
        "FAKE_DIR": str(fake),
        "BEFORE_STATE": BEFORE,
        "AFTER_STATE": AFTER,
        "DRY_PLAN": DRY_PLAN,
        "VERIFY_PLAN": ZERO_PLAN,
        **env,
    }
    proc = subprocess.run(
        ["bash", str(WRAPPER), *args], env=full_env, capture_output=True, text=True, timeout=60, cwd=tmp_path
    )
    calls = calls_log.read_text().splitlines()[before:] if calls_log.exists() else []
    return proc.returncode, proc.stdout + proc.stderr, calls


def _tool_calls(calls: list[str]) -> list[str]:
    """The tool runs after the read-only pre-check (step 0)."""
    return [c for c in calls if "repair_fixture_orphans.py" in c and "precheck" not in c]


def test_a_clean_run_exits_0_and_says_done(tmp_path):
    rc, out, calls = _run(tmp_path)
    assert rc == 0, out
    assert "after-state OK" in out and "DONE." in out
    assert len(_tool_calls(calls)) == 3  # dry run, apply, re-plan
    precheck = [c for c in calls if "precheck" in c]
    assert len(precheck) == 1 and "--apply" not in precheck[0]
    assert calls.index(precheck[0]) < next(i for i, c in enumerate(calls) if "import b2_io" in c)


def test_a_re_run_with_nothing_to_do_exits_0_before_the_backup(tmp_path):
    rc, out, calls = _run(tmp_path, PRE_PLAN=ZERO_PLAN)
    assert rc == 0 and "ALREADY DONE" in out, out
    assert not [c for c in calls if "import b2_io" in c or "db_backup_daily" in c]
    assert not _tool_calls(calls)


@pytest.mark.parametrize("pre", [ZERO_PLAN.replace("blocked_pending_recs=0", "blocked_pending_recs=1"), ""])
def test_a_pre_check_with_anything_left_or_unreadable_goes_the_full_way(tmp_path, pre):
    rc, out, calls = _run(tmp_path, PRE_PLAN=pre or "garbage")
    assert "ALREADY DONE" not in out and [c for c in calls if "import b2_io" in c]


def test_every_run_and_both_state_queries_share_one_as_of_instant(tmp_path):
    rc, out, calls = _run(tmp_path)
    assert rc == 0, out
    tools = _tool_calls(calls)
    as_of = {m for c in tools for m in re.findall(r"--as-of (\S+)", c)}
    assert len(as_of) == 1 and all("--as-of" in c for c in tools), tools
    (instant,) = as_of
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", instant)
    sqls = sorted((tmp_path / "fake").glob("state_sql.*"))
    assert len(sqls) == 2  # before and after
    for sql in sqls:
        text = sql.read_text()
        assert f"m.match_date < TIMESTAMPTZ '{instant}' - interval '96 hours'" in text
        assert "NOW()" not in text
    # the instant is taken after the backup landed, right before the first read
    first_b2 = next(i for i, c in enumerate(calls) if "import b2_io" in c)
    first_cp = next(i for i, c in enumerate(calls) if c.startswith("compose cp "))
    assert first_b2 < first_cp


def test_the_apply_is_pinned_to_the_dry_runs_counts_including_the_postponed_marks(tmp_path):
    rc, out, calls = _run(tmp_path)
    (apply,) = [c for c in _tool_calls(calls) if "--apply" in c]
    assert "--expect-plan merge_delete=3,merge_cancel=1,delete_placeholder=2,mark_postponed=1,delete_no_twin=0" in apply
    assert "--postponed-list /app/scripts/data/soccer_postponed_stale_rows.csv" in apply


@pytest.mark.parametrize("failing", ["FAIL_FEATURES", "FAIL_GRADE"])
def test_a_failed_step_5_or_6_exits_2_and_never_says_done(tmp_path, failing):
    rc, out, _ = _run(tmp_path, **{failing: "1"})
    assert rc == 2, out
    assert "DONE" not in out
    assert "the cleanup is verified" in out and "steps 5/6 need a manual re-run" in out
    assert "cleanup_soccer_orphans.sh --followups-only" in out


def _pending(tmp_path: Path) -> list[str]:
    f = tmp_path / "fake" / "pending"
    return [line for line in f.read_text().splitlines() if line] if f.exists() else []


def _followups(calls: list[str]) -> list[str]:
    return [c for c in calls if "compute_features.py" in c or c.endswith(" api python -")]


@pytest.mark.parametrize("failing", ["FAIL_FEATURES", "FAIL_GRADE"])
def test_a_re_run_after_incomplete_runs_the_pending_follow_ups_not_already_done(tmp_path, failing):
    """The cleanup applied and verified, then step 5 or 6 failed (exit 2). The
    re-run plans nothing: it must run steps 5-6 for that apply's twins, not
    print ALREADY DONE over them (no other step grades those predictions)."""
    rc, out, _ = _run(tmp_path, **{failing: "1"})
    assert rc == 2, out
    (apply_csv,) = _pending(tmp_path)
    assert re.fullmatch(r"/app/backups/fixture_orphans_soccer_apply_\d{8}T\d{6}Z\.csv", apply_csv)

    rc, out, calls = _run(tmp_path, PRE_PLAN=ZERO_PLAN)
    assert rc == 0 and "FOLLOW-UPS DONE" in out and "ALREADY DONE" not in out, out
    (twins,) = [c for c in calls if "python -c import csv" in c]
    assert apply_csv in twins
    assert len(_followups(calls)) == 2
    assert not [c for c in calls if "import b2_io" in c or "db_backup_daily" in c or "--apply" in c]
    assert _pending(tmp_path) == []

    rc, out, calls = _run(tmp_path, PRE_PLAN=ZERO_PLAN)
    assert rc == 0 and "ALREADY DONE" in out and not _followups(calls), out


def test_a_re_run_whose_follow_ups_fail_again_stays_incomplete_and_pending(tmp_path):
    _run(tmp_path, FAIL_GRADE="1")
    rc, out, _ = _run(tmp_path, PRE_PLAN=ZERO_PLAN, FAIL_GRADE="1")
    assert rc == 2 and "INCOMPLETE" in out and len(_pending(tmp_path)) == 1, out


def test_a_failed_after_state_keeps_the_follow_ups_pending(tmp_path):
    rc, out, calls = _run(tmp_path, AFTER_STATE="92|3|500|1")
    assert rc == 1 and "FAILED after-state" in out and not _followups(calls), out
    assert len(_pending(tmp_path)) == 1
    rc, out, calls = _run(tmp_path, PRE_PLAN=ZERO_PLAN)
    assert rc == 0 and "FOLLOW-UPS DONE" in out and len(_followups(calls)) == 2, out


def test_an_earlier_apply_still_pending_is_followed_up_with_this_runs(tmp_path):
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "pending").write_text("/app/backups/fixture_orphans_soccer_apply_20261030T000000Z.csv\n")
    rc, out, calls = _run(tmp_path)
    assert rc == 0 and "DONE." in out, out
    (twins,) = [c for c in calls if "python -c import csv" in c]
    assert "fixture_orphans_soccer_apply_20261030T000000Z.csv" in twins
    assert re.search(r"fixture_orphans_soccer_apply_2026\d{4}T\d{6}Z\.csv", twins.replace("20261030T000000Z", ""))
    assert _pending(tmp_path) == []


def test_followups_only_runs_steps_5_and_6_alone(tmp_path):
    rc, out, calls = _run(tmp_path, "--followups-only")
    assert rc == 0 and "nothing pending" in out and not _followups(calls), out
    (tmp_path / "fake" / "pending").write_text("/app/backups/fixture_orphans_soccer_apply_20261031T120000Z.csv\n")
    rc, out, calls = _run(tmp_path, "--followups-only")
    assert rc == 0 and "FOLLOW-UPS DONE" in out and len(_followups(calls)) == 2, out
    assert not [c for c in calls if "repair_fixture_orphans.py" in c or "import b2_io" in c]
    assert _pending(tmp_path) == []


def test_an_unknown_argument_exits_2(tmp_path):
    rc, out, calls = _run(tmp_path, "--bogus")
    assert rc == 2 and "usage" in out and calls == []


def test_a_wrong_after_state_exits_1(tmp_path):
    rc, out, _ = _run(tmp_path, AFTER_STATE="94|3|500|1")
    assert rc == 1 and "FAILED after-state" in out and "DONE" not in out


@pytest.mark.parametrize(
    "plan",
    [
        DRY_PLAN.replace("list_mismatch=0", "list_mismatch=1"),
        DRY_PLAN.replace("blocked_pending_recs=0", "blocked_pending_recs=2"),
        DRY_PLAN.replace("mark_postponed=1", "mark_postponed=7"),
        DRY_PLAN.replace("delete_no_twin=0", "delete_no_twin=1"),
        DRY_PLAN.replace("merge_delete=3", "merge_delete=204"),
        DRY_PLAN.replace(" list_mismatch=0", ""),
    ],
)
def test_the_dry_run_gate_aborts_before_any_apply(tmp_path, plan):
    rc, out, calls = _run(tmp_path, DRY_PLAN=plan)
    assert rc == 1 and "ABORT at step 2" in out, out
    assert not [c for c in _tool_calls(calls) if "--apply" in c]
