"""auspex_pipeline: one refused fixture must never stop a sport's chain.

fetch_upcoming_* exit 3 after REFUSING a fixture whose identity drifted
(having committed the rest of the sweep). Static check of the DAG source:
every compute_features* task runs ALL_DONE and every fetch_upcoming* task
has retries=0 (a refusal is deterministic)."""

from __future__ import annotations

import ast
from pathlib import Path

DAG_FILE = Path(__file__).resolve().parents[2] / "dags" / "auspex_pipeline.py"
SPORT_SUFFIXES = ("", "_nhl", "_nba", "_nfl", "_tennis", "_mma")


def _operators() -> dict:
    tree = ast.parse(DAG_FILE.read_text())
    ops = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "BashOperator":
            kw = {k.arg: k.value for k in node.keywords}
            task_id = kw.get("task_id")
            if isinstance(task_id, ast.Constant):
                ops[task_id.value] = kw
    return ops


def test_every_compute_features_runs_all_done():
    ops = _operators()
    for sfx in SPORT_SUFFIXES:
        rule = ops[f"compute_features{sfx}"].get("trigger_rule")
        assert rule is not None and ast.unparse(rule) == "TriggerRule.ALL_DONE", sfx


def test_every_fetch_upcoming_has_no_retry():
    ops = _operators()
    for sfx in SPORT_SUFFIXES:
        retries = ops[f"fetch_upcoming{sfx}"].get("retries")
        assert isinstance(retries, ast.Constant) and retries.value == 0, sfx
