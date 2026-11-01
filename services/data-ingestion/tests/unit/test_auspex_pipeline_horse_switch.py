"""auspex_pipeline under sport_switches.HORSE_RACING_ENABLED, both ways.

Airflow is not installed in the test environment, so the names the DAG
imports (airflow.sdk.DAG, BashOperator, TriggerRule, alerting, and the
switch) are stubbed and the file is executed as the dag-processor would,
once per switch state. What this pins:
  * dormant: none of the seven horse tasks exist, one drain_horse_racing
    task does (drain, then grade, failing if either failed), and nothing
    else changes — every other task, command and edge is identical to the
    enabled DAG, and send_pipeline_digest still fans in the six other
    sports' recommendation tasks;
  * enabled: the original branch, chained, and its recs feed the digest;
  * the committed switch is a plain bool literal (its VALUE is not pinned:
    re-enabling is the switch plus its frontend mirror, which
    tests/unit/test_frontend_sport_switches.py keeps equal, and both DAG
    shapes are tested here with the switch forced), and every flag the DAG
    passes a racing script is one that script accepts."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

DAGS = Path(__file__).resolve().parents[2] / "dags"
DAG_FILE = DAGS / "auspex_pipeline.py"
SCRIPTS = DAGS.parents[2] / "scripts"

HORSE_TASKS = [
    "fetch_horse_racing_upcoming",
    "fetch_horse_racing_results",
    "compute_features_horse_racing",
    "precompute_predictions_horse_racing",
    "precompute_predictions_horse_racing_ranker",
    "generate_recommendations_horse_racing",
]
OTHER_REC_TASKS = {
    "generate_recommendations",
    "generate_recommendations_nhl",
    "generate_recommendations_nba",
    "generate_recommendations_nfl",
    "generate_recommendations_tennis",
    "generate_recommendations_mma",
}


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
        self.upstream = []
        FakeDAG.current.tasks.append(self)

    def _link(self, other):
        self.downstream.append(other)
        other.upstream.append(self)

    def __rshift__(self, other):
        for o in other if isinstance(other, list) else [other]:
            self._link(o)
        return other

    def __rrshift__(self, others):  # [a, b] >> self
        for o in others:
            o._link(self)
        return self


def _load(monkeypatch, enabled: bool):
    names = [
        "airflow",
        "airflow.sdk",
        "airflow.utils",
        "airflow.utils.trigger_rule",
        "airflow.providers",
        "airflow.providers.standard",
        "airflow.providers.standard.operators",
        "airflow.providers.standard.operators.bash",
        "alerting",
        "sport_switches",
    ]
    mods = {name: types.ModuleType(name) for name in names}
    mods["airflow.sdk"].DAG = FakeDAG
    mods["airflow.providers.standard.operators.bash"].BashOperator = FakeBashOperator
    mods["airflow.utils.trigger_rule"].TriggerRule = types.SimpleNamespace(ALL_DONE="all_done")
    mods["alerting"].notify_failure = lambda context: None
    mods["sport_switches"].HORSE_RACING_ENABLED = enabled
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location(f"auspex_pipeline_{enabled}", DAG_FILE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {t.task_id: t for t in mod.dag.tasks}


@pytest.fixture
def dormant(monkeypatch):
    return _load(monkeypatch, enabled=False)


@pytest.fixture
def enabled(monkeypatch):
    return _load(monkeypatch, enabled=True)


def _shape(tasks: dict, drop: set) -> dict:
    """Every task outside `drop` with its command, kwargs and edges inside the kept set."""
    return {
        tid: (
            t.bash_command,
            sorted(t.kwargs.items()),
            sorted(d.task_id for d in t.downstream if d.task_id not in drop),
        )
        for tid, t in tasks.items()
        if tid not in drop
    }


def test_the_committed_switch_is_a_plain_bool():
    # Not pinned to False: flipping the switch (and its frontend mirror) to
    # re-enable horse racing must not turn CI red, or nothing deploys. The
    # dormant and enabled DAG shapes are each pinned above and below.
    spec = importlib.util.spec_from_file_location("sport_switches_real", DAGS / "sport_switches.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.HORSE_RACING_ENABLED is True or mod.HORSE_RACING_ENABLED is False


def test_dormant_creates_no_horse_task_but_the_drain(dormant):
    horse = [tid for tid in dormant if "horse" in tid or "race" in tid]
    assert horse == ["drain_horse_racing"]
    assert "grade_completed_races" not in dormant
    drain = dormant["drain_horse_racing"]
    assert drain.upstream == [] and drain.downstream == []
    cmd = drain.bash_command
    # Drain then grade; both always run; the task fails when either did.
    assert cmd.startswith("rc=0; ") and cmd.rstrip().endswith("exit $rc")
    assert "python /app/scripts/load_racing_api.py --drain || rc=1;" in cmd
    assert "python /app/scripts/grade_completed_races.py --days 14 || rc=1;" in cmd
    assert cmd.index("--drain") < cmd.index("grade_completed_races.py")
    for flag in ("--upcoming", "--results", "compute_features_horse_racing", "precompute_predictions_horse"):
        assert flag not in cmd


def test_dormant_leaves_every_other_task_as_it_was(dormant, enabled):
    drop = set(HORSE_TASKS) | {"grade_completed_races", "drain_horse_racing"}
    assert set(enabled) - set(dormant) == set(HORSE_TASKS) | {"grade_completed_races"}
    assert set(dormant) - set(enabled) == {"drain_horse_racing"}
    assert _shape(dormant, drop) == _shape(enabled, drop)


def test_the_digest_still_fans_in_every_other_sport(dormant, enabled):
    digest = dormant["send_pipeline_digest"]
    assert {t.task_id for t in digest.upstream} == OTHER_REC_TASKS
    assert digest.kwargs["trigger_rule"] == "all_done"
    assert {t.task_id for t in enabled["send_pipeline_digest"].upstream} == OTHER_REC_TASKS | {
        "generate_recommendations_horse_racing"
    }


def test_enabled_is_the_original_branch(enabled):
    for upstream, downstream in zip(HORSE_TASKS, HORSE_TASKS[1:]):
        assert [t.task_id for t in enabled[upstream].downstream] == [downstream], upstream
    assert enabled["fetch_horse_racing_upcoming"].upstream == []
    assert "load_racing_api.py --upcoming 7" in enabled["fetch_horse_racing_upcoming"].bash_command
    assert "grade_completed_races.py --days 14" in enabled["grade_completed_races"].bash_command
    assert enabled["grade_completed_races"].upstream == [] and enabled["grade_completed_races"].downstream == []
    assert "drain_horse_racing" not in enabled


def _script(name: str):
    pytest.importorskip("psycopg2")
    pytest.importorskip("requests")
    spec = importlib.util.spec_from_file_location(f"_dag_flags_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("state", ["dormant", "enabled"])
def test_the_racing_scripts_accept_every_flag_the_dag_passes(state, request):
    # An unknown flag is an argparse exit 2 on every tick; catch it here.
    parsers = {name: _script(name).parse_args for name in ("load_racing_api", "grade_completed_races")}
    seen = 0
    for task in request.getfixturevalue(state).values():
        for step in task.bash_command.split(";"):
            for name, parse in parsers.items():
                if f"/app/scripts/{name}.py" not in step:
                    continue
                args = step.split(f"/app/scripts/{name}.py", 1)[1].split("||")[0].split()
                # --results passes its dates as $(date ...) substitutions; parse with real dates.
                if "--results" in args:
                    args = ["--results", "--start", "2026-11-01", "--end", "2026-11-01"]
                parse([*args, "--database-url", "postgresql://x"])
                seen += 1
    assert seen == (2 if state == "dormant" else 3)
