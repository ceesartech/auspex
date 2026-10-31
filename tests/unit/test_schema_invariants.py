"""Schema decisions that must not quietly revert. Static — no DB."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "services" / "data-ingestion" / "db" / "migrations"


def _last_definition(constraint: str) -> str:
    """The FK clause of the highest-numbered migration that (re)defines it."""
    last = ""
    for path in sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text())
        for m in re.finditer(rf"ADD CONSTRAINT {constraint}\s+(FOREIGN KEY[^;,]*)", sql):
            last = m.group(1)
    return last


def _last_function_body(name: str) -> tuple[str, str]:
    """(migration file, whitespace-collapsed body without comments) of the
    highest-numbered migration that (re)defines plpgsql function `name`."""
    last = ("", "")
    for path in sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text())
        for m in re.finditer(rf"CREATE OR REPLACE FUNCTION {name}\(\)(.*?)\$\$ LANGUAGE plpgsql", sql, re.S):
            last = (path.name, " ".join(m.group(1).split()))
    return last


def _str_constant(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            return node.value.value
    raise AssertionError(f"{name} not assigned in {path}")


def test_deleting_a_prediction_never_cascades_to_its_recs():
    # 2026-10-31: the CASCADE silently deleted 45 settled tennis recs (migration 025)
    clause = _last_definition("betting_recommendations_prediction_id_fkey")
    assert "REFERENCES predictions(id)" in clause and "ON DELETE NO ACTION" in clause
    model = (ROOT / "services" / "api" / "src" / "models" / "database_models.py").read_text()
    assert 'ForeignKey("predictions.id", ondelete="NO ACTION")' in model


def test_deleting_a_match_never_cascades_to_its_recs():
    # migration 026: a fixture with recs must have them re-pointed or removed first, explicitly
    clause = _last_definition("betting_recommendations_match_id_fkey")
    assert "REFERENCES matches(id)" in clause and "ON DELETE NO ACTION" in clause
    model = (ROOT / "services" / "api" / "src" / "models" / "database_models.py").read_text()
    assert 'ForeignKey("matches.id", ondelete="NO ACTION")' in model


def test_outcome_trigger_never_settles_rows_written_after_kickoff():
    # migration 027: the trigger used to mark post-kickoff recs won/lost and grade
    # post-kickoff predictions, which the Python grader voids / skips. Each guard
    # must sit in its own UPDATE, with the grader's comparison (> for
    # predictions, >= for recs).
    migration, body = _last_function_body("calculate_match_outcome")
    assert migration, "no migration defines calculate_match_outcome()"
    preds = re.search(r"UPDATE predictions p .*?;", body)
    recs = re.search(r"UPDATE betting_recommendations br .*?;", body)
    assert preds and recs, migration
    assert "AND (p.created_at > NEW.match_date) IS NOT TRUE" in preds.group(0), migration
    assert "AND (br.created_at >= NEW.match_date) IS NOT TRUE" in recs.group(0), migration


def test_accuracy_rollup_drops_the_graders_post_kickoff_void_label():
    # The API image cannot import scripts/, so the actual_result prefix is
    # duplicated; a drift would quietly put hindsight voids back into ROI.
    grader = _str_constant(ROOT / "scripts" / "grade_completed_matches.py", "POST_KICKOFF_VOID_RESULT")
    api = _str_constant(ROOT / "services" / "api" / "src" / "routes" / "accuracy.py", "POST_KICKOFF_VOID_PREFIX")
    assert api == grader == "void: written after kickoff"
