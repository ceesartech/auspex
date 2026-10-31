"""scripts/ops/migrate_027_outcome_trigger.sh, run for real against a stand-in
`docker` that plays Postgres: its step 3 must refuse a migration whose body
does not match the pinned md5 BEFORE step 4 commits it.

The stand-in answers each psql file the wrapper sends with what that SQL would
print on prod, from FAKE_* variables; the dry run prints the after-state line
only if the wrapper put the after-state query into it. `sleep` is stubbed so
the backup wait is instant."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "ops" / "migrate_027_outcome_trigger.sh"
MIG_NAME = "027_outcome_trigger_skips_post_kickoff.sql"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _pinned(name: str) -> str:
    m = re.search(rf'^{name}=(?:"([^"]*)"|(\S+))', WRAPPER.read_text(), re.M)
    assert m, name
    return m.group(1) if m.group(1) is not None else m.group(2)


MD5_011 = _pinned("MD5_011")
MD5_027 = _pinned("MD5_027")
EXPECT_STATE = f"{MD5_027}|1|1|1"
EXPECT_PROOF = _pinned("EXPECT_PROOF")

FAKE_DOCKER = r"""#!/bin/bash
# Stand-in for `docker compose ...` (tests/unit/test_migrate_027_wrapper.py).
echo "$*" >> "$FAKE_DIR/calls.log"
[ "$1" = compose ] || exit 97
case "$2" in
  cp) cp "$3" "$FAKE_DIR/ctr_$(basename "$3")"; exit 0 ;;
  exec) ;;
  *) exit 98 ;;
esac
case "$4" in
  airflow-scheduler) exit 0 ;;
  api) echo "$(( $(date -u +%s) + 60 )) postgres/fake.dump"; exit 0 ;;
  postgres) ;;
  *) exit 99 ;;
esac
cmd=$7
case "$cmd" in rm\ -f*) exit 0 ;; esac
f=${cmd##*-f /tmp/}
sql="$FAKE_DIR/ctr_$f"
[ -f "$sql" ] || { echo "psql: $f: No such file" >&2; exit 3; }
case "$f" in
  m027_fn.sql) echo "$FAKE_LIVE_MD5" ;;
  m027_verify.sql) echo "$FAKE_AFTER_STATE" ;;
  m027_prove.sql) echo "$FAKE_PROOF" ;;
  m027_dry.sql)
    grep -q "pg_get_functiondef" "$sql" && echo "$FAKE_DRY_STATE"
    grep -q "'m027 pre_pred='" "$sql" && echo "$FAKE_DRY_PROOF"
    ;;
  MIGNAME) touch "$FAKE_DIR/APPLIED" ;;
  *) echo "unexpected file $f" >&2; exit 4 ;;
esac
exit 0
""".replace(
    "MIGNAME", MIG_NAME
)


def _run(tmp_path: Path, **fake: str) -> tuple[int, str, Path]:
    bin_dir, fake_dir, tmp_dir = tmp_path / "bin", tmp_path / "fake", tmp_path / "tmp"
    for d in (bin_dir, fake_dir, tmp_dir):
        d.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for exe in ("docker", "sleep"):
        (bin_dir / exe).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "AUSPEX_DIR": str(ROOT),
        "TMPDIR": str(tmp_dir),
        "FAKE_DIR": str(fake_dir),
        "FAKE_LIVE_MD5": MD5_011,
        "FAKE_DRY_STATE": EXPECT_STATE,
        "FAKE_DRY_PROOF": EXPECT_PROOF,
        "FAKE_AFTER_STATE": EXPECT_STATE,
        "FAKE_PROOF": EXPECT_PROOF,
        **fake,
    }
    proc = subprocess.run(["bash", str(WRAPPER)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    return proc.returncode, proc.stdout + proc.stderr, fake_dir


def _calls(fake_dir: Path) -> str:
    return (fake_dir / "calls.log").read_text()


def test_a_dry_run_matching_the_pinned_state_goes_on_to_apply(tmp_path):
    rc, out, fake_dir = _run(tmp_path)
    assert rc == 0, out
    assert "DONE" in out
    assert (fake_dir / "APPLIED").exists()
    # Step 3's file: 027 without its COMMIT, then the after-state query, inside
    # the same transaction, ended by ROLLBACK.
    dry = (fake_dir / "ctr_m027_dry.sql").read_text()
    assert not re.search(r"^\s*COMMIT", dry, re.M | re.I)
    create = dry.index("CREATE OR REPLACE FUNCTION calculate_match_outcome()")
    state = dry.index("SELECT md5(d)")
    assert dry.rstrip().endswith("ROLLBACK;")
    assert create < state < dry.rindex("ROLLBACK;")
    # host-side SQL removed on exit
    assert not list((tmp_path / "tmp").glob("m027.*/*.sql"))


@pytest.mark.parametrize(
    "dry_state",
    [
        pytest.param("0123456789abcdef0123456789abcdef|1|1|1", id="drifted-body-md5"),
        pytest.param(f"{MD5_027}|0|1|1", id="prediction-guard-missing"),
        pytest.param(f"{MD5_027}|1|1|0", id="trigger-not-firing-it"),
        pytest.param("", id="no-state-line"),
    ],
)
def test_a_dry_run_off_the_pinned_state_is_refused_before_anything_is_committed(tmp_path, dry_state):
    rc, out, fake_dir = _run(tmp_path, FAKE_DRY_STATE=dry_state)
    assert rc == 1, out
    assert "ABORT at step 3" in out and EXPECT_STATE in out, out
    assert not (fake_dir / "APPLIED").exists(), "step 4 committed 027"
    calls = _calls(fake_dir)
    assert MIG_NAME not in calls.replace(f"rm -f /tmp/m027_*.sql /tmp/{MIG_NAME}", "")
    assert "m027_verify.sql" not in calls  # step 5 never ran
    # the backup did run first (step 2)
    assert "airflow dags trigger db_backup_daily" in calls


def test_a_dry_run_failing_the_proof_is_still_refused(tmp_path):
    rc, out, fake_dir = _run(tmp_path, FAKE_DRY_PROOF="m027 pre_pred=true post_pred=true pre_rec=won post_rec=won")
    assert rc == 1, out
    assert "ABORT at step 3" in out
    assert not (fake_dir / "APPLIED").exists()


def test_an_unknown_live_body_aborts_before_the_backup(tmp_path):
    rc, out, fake_dir = _run(tmp_path, FAKE_LIVE_MD5="ffffffffffffffffffffffffffffffff")
    assert rc == 1, out
    assert "ABORT at step 1" in out
    calls = _calls(fake_dir)
    assert "db_backup_daily" not in calls and not (fake_dir / "APPLIED").exists()


def test_a_failed_after_state_still_fails_loudly(tmp_path):
    # step 5 stays as a re-confirmation of the committed body
    rc, out, fake_dir = _run(tmp_path, FAKE_AFTER_STATE=f"{MD5_011}|0|0|1")
    assert rc == 1, out
    assert "FAILED at step 5" in out
    assert (fake_dir / "APPLIED").exists()
