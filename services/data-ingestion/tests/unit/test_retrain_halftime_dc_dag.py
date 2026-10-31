"""Parse test for the retrain_halftime_dc DAG (dags/retrain_halftime_dc.py).

Airflow is not installed in the test environment, so the three names the DAG
imports (airflow.sdk.DAG, BashOperator, alerting.notify_failure) are stubbed
and the file is executed as the dag-processor would. What this pins: the
monthly schedule, one run at a time, train >> promote under the default
ALL_SUCCESS rule (a refused fit never reaches promote), and both tasks
rendering the SAME run id — which is what lets promote refuse a staged
manifest written by any other run."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

DAG_FILE = Path(__file__).resolve().parents[2] / "dags" / "retrain_halftime_dc.py"


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


@pytest.fixture
def dag_module(monkeypatch):
    names = [
        "airflow",
        "airflow.sdk",
        "airflow.providers",
        "airflow.providers.standard",
        "airflow.providers.standard.operators",
        "airflow.providers.standard.operators.bash",
        "alerting",
    ]
    mods = {name: types.ModuleType(name) for name in names}
    mods["airflow.sdk"].DAG = FakeDAG
    mods["airflow.providers.standard.operators.bash"].BashOperator = FakeBashOperator
    mods["alerting"].notify_failure = lambda context: None
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("retrain_halftime_dc", DAG_FILE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tasks(mod):
    return {t.task_id: t for t in mod.dag.tasks}


def test_dag_shape(dag_module):
    kw = dag_module.dag.kwargs
    assert kw["dag_id"] == "retrain_halftime_dc"
    assert kw["schedule"] == "30 5 1 * *"
    assert kw["catchup"] is False and kw["max_active_runs"] == 1
    assert kw["default_args"]["on_failure_callback"] is sys.modules["alerting"].notify_failure

    t = tasks(dag_module)
    assert list(t) == ["train_ht", "promote_ht"]
    assert t["train_ht"].downstream == [t["promote_ht"]] and t["promote_ht"].downstream == []
    # Default ALL_SUCCESS: a failed/refused train must skip the promote.
    assert "trigger_rule" not in t["train_ht"].kwargs and "trigger_rule" not in t["promote_ht"].kwargs
    # A retried promote would see staged == production and go green.
    assert t["promote_ht"].kwargs["retries"] == 0


def test_both_tasks_carry_the_same_run_id(dag_module):
    t = tasks(dag_module)
    tmpl = dag_module.RUN_ID_TMPL
    train, promote = t["train_ht"].bash_command, t["promote_ht"].bash_command
    assert "timeout 45m python /app/scripts/train_halftime_dixon_coles.py --run-id '" + tmpl + "'" in train
    assert "python /app/scripts/promote_dc_model.py --model ht --expect-run-id '" + tmpl + "'" in promote
    assert train.count("{{") == promote.count("{{") == 1
    # The api container gets the deployed commit for the manifest.
    assert '-e GIT_SHA="$GIT_SHA" api ' in train
    assert all(
        cmd.startswith(("docker compose -f /opt/auspex/docker-compose.yml", 'GIT_SHA="$(')) for cmd in (train, promote)
    )
    # Nothing here may write where retrain_models' swap auto-promotes from.
    assert "/app/models/staging" not in train + promote


def test_rendered_commands_are_valid_bash(dag_module):
    jinja2 = pytest.importorskip("jinja2")
    for task in tasks(dag_module).values():
        for run_id in ("scheduled__2026-11-01T05:30:00+00:00", "manual__2026-11-03T09:12:44.123456+00:00"):
            rendered = jinja2.Template(task.bash_command).render(run_id=run_id)
            safe = rendered.split("-id '")[1].split("'")[0]
            assert safe and not set(safe) & set(":+. '\"$")
            subprocess.run(["bash", "-n", "-c", rendered], check=True)
