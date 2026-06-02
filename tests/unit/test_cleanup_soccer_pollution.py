"""Unit tests for cleanup_soccer_pollution — pure pieces.

The script is one-shot data maintenance, but the SQL it issues is
load-bearing: a wrong filter could either (a) delete real soccer
predictions or (b) miss the pollution and leave the table cluttered.
Tests use a fake cursor to capture queries + assert intent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


csp = _load("cleanup_soccer_pollution", "cleanup_soccer_pollution.py")


class FakeCursor:
    """Captures executed queries + emits scripted fetch results."""

    def __init__(self, count_results=None, delete_rowcount=0):
        self.queries: list[tuple] = []
        self.count_results = count_results or []
        self.rowcount = delete_rowcount

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

    def fetchall(self):
        return self.count_results


class TestCountPollution:
    def test_query_filters_to_legacy_soccer_model_name(self):
        # The WHERE clause must check model_name = 'ensemble' so we
        # don't accidentally pick up NHL/NBA's own prediction rows
        # (which have model_name='ensemble_nhl_*' / 'ensemble_nba_*').
        cur = FakeCursor(count_results=[])
        csp.count_pollution(cur)
        sql, params = cur.queries[0]
        assert params == (csp.SOCCER_MODEL_NAME,) == ("ensemble",)
        assert "p.model_name = %s" in sql

    def test_query_excludes_soccer_matches_themselves(self):
        # The l.sport <> 'soccer' clause is the load-bearing filter:
        # real soccer ensemble predictions on real soccer matches must
        # stay put. Pollution is by definition (legacy soccer
        # ensemble × NON-soccer match).
        cur = FakeCursor()
        csp.count_pollution(cur)
        sql, _ = cur.queries[0]
        assert "l.sport <> 'soccer'" in sql

    def test_groups_by_sport_for_reporting(self):
        # The report buckets by sport so the operator sees "nhl: 600
        # rows, nba: 150 rows" before deciding to confirm.
        cur = FakeCursor(
            count_results=[
                {"sport": "nhl", "rows": 600},
                {"sport": "nba", "rows": 150},
            ]
        )
        result = csp.count_pollution(cur)
        assert result == {"nhl": 600, "nba": 150}


class TestDeletePollution:
    def test_delete_uses_same_model_name_pin(self):
        # The DELETE must use the SAME filter shape as count_pollution
        # so the "what we count" and "what we delete" agree.
        cur = FakeCursor(delete_rowcount=42)
        deleted = csp.delete_pollution(cur)
        assert deleted == 42
        sql, params = cur.queries[0]
        assert params == ("ensemble",)
        assert "DELETE FROM predictions" in sql
        assert "p.model_name = %s" in sql
        assert "l.sport <> 'soccer'" in sql

    def test_delete_protects_soccer_matches(self):
        # Defensive: the DELETE's sub-SELECT pins to non-soccer
        # matches. Any rename of this clause would risk deleting real
        # soccer recs. Lock it.
        cur = FakeCursor(delete_rowcount=0)
        csp.delete_pollution(cur)
        sql, _ = cur.queries[0]
        # The sub-SELECT carries the non-soccer filter:
        assert "WHERE l.sport <> 'soccer'" in sql


class TestCLI:
    def test_default_is_dry_run(self):
        # CRITICAL: the default behavior must NOT delete. A pipeline
        # operator running the script without thinking should see a
        # report, not lose data.
        args = csp.parse_args(["--database-url", "postgresql://x"])
        assert args.confirm is False

    def test_confirm_flag_enables_delete(self):
        args = csp.parse_args(["--database-url", "postgresql://x", "--confirm"])
        assert args.confirm is True


class TestConstants:
    def test_soccer_model_name_locked(self):
        # If TaskSpec.db_model_name for soccer ever changes from
        # 'ensemble' to something else, this script needs to update
        # too. Locked here so a rename is loud, not silent.
        assert csp.SOCCER_MODEL_NAME == "ensemble"


# Quiet unused-import lint.
_ = pytest
