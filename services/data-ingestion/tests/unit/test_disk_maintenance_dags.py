"""Tests for the disk guards: dags/disk_maintenance.py (daily) and
dags/docker_maintenance.py (weekly).

Airflow is not installed in the test environment, so the names the DAG
files import are stubbed and each file is executed as the dag-processor
would. Pinned: the daily schedule, live-on-creation, two independent tasks;
the image task runs scripts/prune_docker_images.py with --apply and flags
the script accepts; the weekly DAG no longer prunes images or logs (the old
`until=168h` prune is gone everywhere). The task-log prune command is then
RUN against a temp tree: old *.log and rotated *.log.N files go, fresh ones,
other files and symlinks stay, empty directories go only once empty for an
hour, and a missing log directory or a find error fails the task.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

DAGS = Path(__file__).resolve().parents[2] / "dags"
REPO = Path(__file__).resolve().parents[4]


class FakeDAG:
    current = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tasks = []

    def __enter__(self):
        FakeDAG.current = self
        return self

    def __exit__(self, *exc):
        FakeDAG.current = None
        return False


class FakeBashOperator:
    def __init__(self, task_id, bash_command, **kwargs):
        self.task_id = task_id
        self.bash_command = bash_command
        self.kwargs = kwargs
        self.downstream = []
        FakeDAG.current.tasks.append(self)

    def __rshift__(self, other):
        self.downstream.append(other)
        return other


def load(monkeypatch, name):
    names = [
        "airflow",
        "airflow.sdk",
        "airflow.providers",
        "airflow.providers.standard",
        "airflow.providers.standard.operators",
        "airflow.providers.standard.operators.bash",
        "alerting",
    ]
    mods = {n: types.ModuleType(n) for n in names}
    mods["airflow.sdk"].DAG = FakeDAG
    mods["airflow.providers.standard.operators.bash"].BashOperator = FakeBashOperator
    mods["alerting"].notify_failure = lambda context: None
    for n, mod in mods.items():
        monkeypatch.setitem(sys.modules, n, mod)
    spec = importlib.util.spec_from_file_location(name, DAGS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def disk(monkeypatch):
    return load(monkeypatch, "disk_maintenance")


@pytest.fixture
def weekly(monkeypatch):
    return load(monkeypatch, "docker_maintenance")


def tasks(mod):
    return {t.task_id: t for t in mod.dag.tasks}


# ---------------------------------------------------------------- DAG shape


def test_disk_maintenance_is_daily_live_and_independent(disk):
    kw = disk.dag.kwargs
    assert kw["dag_id"] == "disk_maintenance"
    assert kw["schedule"] == "15 5 * * *"
    assert kw["catchup"] is False and kw["max_active_runs"] == 1
    assert kw["is_paused_upon_creation"] is False  # a paused guard is a silent failure
    assert kw["default_args"]["on_failure_callback"] is sys.modules["alerting"].notify_failure
    t = tasks(disk)
    assert set(t) == {"prune_airflow_task_logs", "prune_docker_images"}
    assert all(task.downstream == [] for task in t.values())  # one failing never skips the other
    assert disk.TASK_LOG_RETENTION_DAYS == 14


def test_image_task_runs_the_script_with_flags_it_accepts(disk):
    cmd = tasks(disk)["prune_docker_images"].bash_command
    assert cmd.startswith("python /opt/auspex/scripts/prune_docker_images.py --apply ")
    assert "--keep-deploys 3" in cmd and "--grace-hours 24" in cmd
    assert " -f" not in cmd and "prune -af" not in cmd
    source = (REPO / "scripts" / "prune_docker_images.py").read_text()
    for flag in {tok for tok in cmd.split() if tok.startswith("--")}:
        assert f'"{flag}"' in source, flag
    # The script must parse those flags (dry-run parse only: argparse, no docker).
    spec = importlib.util.spec_from_file_location("pdi_for_dag_test", REPO / "scripts" / "prune_docker_images.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        calls = []

        def docker(args):
            calls.append(args)
            raise mod.DockerError("stub: no docker here")

        argv = cmd.split("prune_docker_images.py", 1)[1].split()
        assert mod.main(argv, docker=docker) == 1  # parsed fine, then failed closed on the stub
        assert calls and calls[0][0] == "ps"
    finally:
        sys.modules.pop(spec.name, None)


def test_weekly_dag_keeps_only_build_cache_and_b2(weekly):
    kw = weekly.dag.kwargs
    assert kw["dag_id"] == "docker_maintenance" and kw["schedule"] == "0 5 * * 0"
    t = tasks(weekly)
    assert list(t) == ["prune_build_cache", "prune_b2_modal_artifacts"]
    assert t["prune_build_cache"].downstream == [t["prune_b2_modal_artifacts"]]
    assert t["prune_build_cache"].bash_command == "docker builder prune -af"
    b2 = t["prune_b2_modal_artifacts"].bash_command
    assert b2.endswith("python /app/scripts/prune_modal_artifacts.py --days 14")
    assert '"--days"' in (REPO / "scripts" / "prune_modal_artifacts.py").read_text()


def test_no_task_prunes_images_by_age_any_more(disk, weekly):
    # `until` compares the VM pull time under the containerd store: it kept
    # untagged leftovers for a week and could drop the rollback target.
    for mod in (disk, weekly):
        for task in tasks(mod).values():
            assert "image prune" not in task.bash_command and "until=" not in task.bash_command, task.task_id


def test_log_prune_command_is_valid_bash(disk):
    subprocess.run(["bash", "-n", "-c", tasks(disk)["prune_airflow_task_logs"].bash_command], check=True)
    assert "{{" not in tasks(disk)["prune_airflow_task_logs"].bash_command  # BashOperator templates it


# ---------------------------------------------------------------- the log prune, run for real


DAY = 86400


def touch(p: Path, age_days: float, content: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    t = time.time() - age_days * DAY
    os.utime(p, (t, t))
    return p


def age_dir(p: Path, minutes: float) -> None:
    t = time.time() - minutes * 60
    os.utime(p, (t, t))


def run_prune(disk, log_dir: Path, days: int = 14) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", disk.task_log_prune_command(str(log_dir), days)], capture_output=True, text=True
    )


def test_log_prune_on_a_temp_tree(disk, tmp_path):
    logs = tmp_path / "logs"
    run_old = logs / "dag_id=auspex_pipeline" / "run_id=scheduled__2026-10-01T00:00:00+00:00"
    run_new = logs / "dag_id=auspex_pipeline" / "run_id=scheduled__2026-11-02T00:00:00+00:00"
    old_task = touch(run_old / "task_id=fetch_upcoming" / "attempt=1.log", 20)
    old_task2 = touch(run_old / "task_id=precompute" / "attempt=2.log", 15.2)
    edge = touch(run_new / "task_id=grade" / "attempt=1.log", 13.9)  # inside 14 days
    new_task = touch(run_new / "task_id=fetch_upcoming" / "attempt=1.log", 0.1)
    rot1 = touch(logs / "dag_processor_manager" / "dag_processor_manager.log.1", 120)
    rot2 = touch(logs / "dag_processor_manager" / "dag_processor_manager.log.2", 150)
    dp_old = touch(logs / "dag_processor" / "2026-10-01" / "auspex_pipeline.py.log", 32)
    not_a_log = touch(logs / "dag_id=auspex_pipeline" / "notes.txt", 90)
    json_meta = touch(logs / "dag_id=x" / "attempt=1.log.json", 90)  # not a rotated log: kept
    target = logs / "scheduler" / "2026-08-05"
    target.mkdir(parents=True)
    (logs / "scheduler" / "latest").symlink_to(target)
    fresh_empty = logs / "dag_id=auspex_pipeline" / "run_id=just_created" / "task_id=t"
    fresh_empty.mkdir(parents=True)
    stale_empty = logs / "dag_id=gone" / "run_id=r"
    stale_empty.mkdir(parents=True)
    age_dir(stale_empty, 120)
    age_dir(stale_empty.parent, 120)
    age_dir(target, 120)

    proc = run_prune(disk, logs)
    assert proc.returncode == 0, proc.stderr
    for gone in (old_task, old_task2, rot1, rot2, dp_old):
        assert not gone.exists(), gone
    for kept in (edge, new_task, not_a_log, json_meta):
        assert kept.exists(), kept
    assert (logs / "scheduler" / "latest").is_symlink()
    assert fresh_empty.is_dir()  # Airflow may be about to write into it
    assert not stale_empty.exists()
    # Directories emptied by THIS run have a fresh mtime: they go next day.
    assert (run_old / "task_id=fetch_upcoming").is_dir()
    assert "pruned 5 log file(s) older than 14 days" in proc.stdout
    assert logs.is_dir()


def test_log_prune_second_run_removes_the_dirs_it_emptied(disk, tmp_path):
    logs = tmp_path / "logs"
    old = touch(logs / "dag_id=a" / "run_id=r" / "task_id=t" / "attempt=1.log", 30)
    assert run_prune(disk, logs).returncode == 0
    for d in (old.parent, old.parent.parent, old.parent.parent.parent):
        age_dir(d, 90)
    proc = run_prune(disk, logs)
    assert proc.returncode == 0 and not (logs / "dag_id=a").exists() and logs.is_dir()


def test_log_prune_fails_loudly_on_a_missing_dir(disk, tmp_path):
    proc = run_prune(disk, tmp_path / "nope")
    assert proc.returncode != 0 and "missing" in proc.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_log_prune_fails_loudly_when_a_delete_is_refused(disk, tmp_path):
    logs = tmp_path / "logs"
    locked = logs / "dag_id=a" / "run_id=r"
    touch(locked / "attempt=1.log", 30)
    locked.chmod(0o500)  # files inside cannot be unlinked
    try:
        proc = run_prune(disk, logs)
    finally:
        locked.chmod(0o700)
    assert proc.returncode != 0
    assert "pruned" not in proc.stdout  # never reports success after a refused delete
