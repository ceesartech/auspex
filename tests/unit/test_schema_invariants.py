"""Schema decisions that must not quietly revert. Static — no DB."""

from __future__ import annotations

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
