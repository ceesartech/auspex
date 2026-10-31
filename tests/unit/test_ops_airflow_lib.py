"""scripts/ops/_airflow_lib.sh helpers run directly. The api container's scan
for an apply still running (af_applies_running): the in-container script it
sends, run by dash and sh against a fake /proc tree (the api image has no ps or
pgrep). Pausing as the wrappers use it is run for real in
test_merge_soccer_teams_wrapper.py and test_soccer_repair_runbook.py.
The run lock's edge cases (af_lock / af_unlock)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "_airflow_lib.sh"
SHELLS = [s for s in ("dash", "sh") if shutil.which(s)]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or not SHELLS, reason="needs bash and dash or sh")

RUNBOOK_TOOLS = (
    "repoint_team_matches merge_soccer_teams repair_football_data_kickoffs repair_stranded_recs "
    "repair_fixture_orphans backfill_soccer_regulation_scores regrade_predictions backfill_season_labels"
)
# A stand-in for `docker compose`: runs the `exec -T api sh -c <script>` it is given with $SCAN_SHELL, /proc/ pointing
# at $FAKE_PROC. Its own pid (exec keeps it) gets an entry that WOULD match: the scan must skip its own shell.
FAKE_DC = r"""#!/bin/bash
[ -n "${DC_FAILS:-}" ] && exit 1
script="${@: -1}"
[ -n "${CAPTURE:-}" ] && { printf '%s' "$script" > "$CAPTURE"; exit 1; }
mkdir -p "$FAKE_PROC/$$"
printf 'python\0/app/scripts/merge_soccer_teams.py\0--apply\0--self\0' > "$FAKE_PROC/$$/cmdline"
exec "$SCAN_SHELL" -c "${script//\/proc\//$FAKE_PROC/}"
"""


def _proc(root: Path, pid: int, *argv: str) -> None:
    (root / str(pid)).mkdir(parents=True)
    (root / str(pid) / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))


def _tree(root: Path, scan: str) -> None:
    _proc(root, 101, "python", "/app/scripts/merge_soccer_teams.py", "--apply", "--expect-plan", "/app/backups/p.json")
    _proc(root, 102, "python", "/app/scripts/merge_soccer_teams.py", "--check-after", "p")  # no --apply
    _proc(root, 103, "python", "/app/scripts/repoint_team_matches.py", "--apply", "--expect-repoint", "6")
    _proc(root, 105, "sh", "-c", "python /app/scripts/merge_soccer_teams.py --apply --expect-plan p > /tmp/m.log 2>&1")
    _proc(root, 106, "python", "/app/scripts/merge_soccer_teams_extra.py", "--apply")  # another tool
    _proc(root, 107)  # a kernel thread: an empty command line
    # The same scan, run by another wrapper at the same moment: its text names every tool with --apply (as '[.]py',
    # which its own pattern does not match). Its own shell is skipped by pid; the glob is taken before any subshell.
    _proc(root, 108, "sh", "-c", scan)


def _lib(tmp_path: Path, body: str, **env: str) -> tuple[int, str]:
    dc = tmp_path / "fake_dc"
    dc.write_text(FAKE_DC)
    dc.chmod(0o755)
    script = f'log() {{ echo "[log] $*"; }}\nDC="{dc} compose"\n. "{LIB}"\n{body}\n'
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "AF_STATE_DIR": str(tmp_path / "afstate"), **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    "tools, pids",
    [("merge_soccer_teams", ["101", "105"]), (RUNBOOK_TOOLS, ["101", "103", "105"])],
    ids=["merge-wrapper", "runbook"],
)
def test_the_scan_finds_exactly_the_running_applies_of_its_tools(tmp_path, shell, tools, pids):
    root = tmp_path / "proc"
    _lib(tmp_path, f"af_applies_running {RUNBOOK_TOOLS}", CAPTURE=str(tmp_path / "scan.sh"))
    _tree(root, (tmp_path / "scan.sh").read_text())
    rc, out = _lib(
        tmp_path, f'af_applies_running {tools}; echo "rc=$?"', FAKE_PROC=str(root), SCAN_SHELL=shutil.which(shell)
    )
    assert out.strip().endswith("rc=0"), out
    found = [line.split()[1].rstrip(":") for line in out.splitlines() if line.startswith("pid ")]
    assert found == pids, out
    assert "--self" not in out  # its own shell's entry is skipped


@pytest.mark.parametrize("shell", SHELLS)
def test_no_running_apply_is_1(tmp_path, shell):
    root = tmp_path / "proc"
    _proc(root, 102, "python", "/app/scripts/merge_soccer_teams.py", "--check-after", "p")
    rc, out = _lib(
        tmp_path,
        'af_applies_running merge_soccer_teams; echo "rc=$?"',
        FAKE_PROC=str(root),
        SCAN_SHELL=shutil.which(shell),
    )
    assert out.strip() == "rc=1", out


def test_a_scan_that_does_not_run_is_2(tmp_path):
    rc, out = _lib(tmp_path, 'af_applies_running merge_soccer_teams; echo "rc=$?"', DC_FAILS="1")
    assert out.strip() == "rc=2", out


def test_a_lock_dir_without_a_holder_is_taken_over(tmp_path):
    """A run killed between its mkdir and its holder write leaves an empty lock: it names no live run."""
    (tmp_path / "afstate" / "x.lock").mkdir(parents=True)
    rc, out = _lib(tmp_path, 'af_lock x; echo "rc=$?"; cat "$AF_STATE_DIR/x.lock/holder"')
    assert "its holder (none recorded) is gone — taken over" in out and "rc=0" in out, out
    assert (tmp_path / "afstate" / "x.lock" / "holder").exists()


def test_unlock_releases_only_this_runs_lock(tmp_path):
    rc, out = _lib(
        tmp_path,
        'af_lock x; echo "999999 someone_else.sh" > "$AF_STATE_DIR/x.lock/holder"; af_unlock; '
        'test -d "$AF_STATE_DIR/x.lock" && echo kept; '
        'af_lock y; af_unlock; test -d "$AF_STATE_DIR/y.lock" || echo gone',
    )
    assert "kept" in out and "gone" in out, out
