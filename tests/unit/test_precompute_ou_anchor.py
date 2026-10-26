"""Serve-path wiring of the market over/under-2.5 scoreline anchor
(scripts/precompute_predictions.py, audit §7 2026-10-25 item 4).

The anchor must come from REAL multi-book prices (rec_gating's de-vigged
consensus over the odds table), never from features_cache — whose over/under
odds fall back to a NEUTRAL_DEFAULTS placeholder — and a failing lookup must
neither abort the run's transaction nor anchor to anything."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


precompute = _load("precompute_predictions", "precompute_predictions.py")


class ConsensusCursor:
    """Answers the rec_gating consensus query with canned odds rows and
    records every statement (SAVEPOINT handling included)."""

    def __init__(self, rows=None, fail=False):
        self.rows = rows or []
        self.fail = fail
        self.statements: list[str] = []
        self.params: list = []

    def execute(self, sql, params=None):
        self.statements.append(" ".join(str(sql).split())[:40])
        self.params.append(params)
        if self.fail and "FROM odds" in str(sql):
            raise RuntimeError("boom")

    def fetchall(self):
        return self.rows


def _book(name, over, under):
    return [
        {"bookmaker": name, "selection": "over", "line": 2.5, "odds_decimal": over},
        {"bookmaker": name, "selection": "under", "line": 2.5, "odds_decimal": under},
    ]


def test_anchor_is_devigged_multibook_consensus():
    rows = _book("A", 1.90, 1.95) + _book("B", 1.85, 2.00) + _book("C", 1.95, 1.90)
    cur = ConsensusCursor(rows)
    p = precompute.market_over25_anchor(cur, "m1")
    # Delegates to the ONE shared consensus definition (whatever de-vig method
    # rec_gating uses), not a private copy of it...
    shared = precompute.rec_gating.market_consensus_prob(ConsensusCursor(rows), "m1", "over_under", "over", 2.5)
    assert p == pytest.approx(shared, abs=1e-15)
    # ...which is a de-vigged two-way price close to the proportional one.
    proportional = sum((1 / o) / (1 / o + 1 / u) for o, u in ((1.90, 1.95), (1.85, 2.00), (1.95, 1.90))) / 3
    assert p == pytest.approx(proportional, abs=0.01)
    # the consensus query asked for over/under at 2.5
    q = [prm for prm in cur.params if isinstance(prm, dict) and "market_type" in prm][0]
    assert q["market_type"] == "over_under" and q["line"] == 2.5
    assert cur.statements[0].startswith("SAVEPOINT") and cur.statements[-1].startswith("RELEASE")


def test_too_few_books_means_no_anchor():
    cur = ConsensusCursor(_book("A", 1.90, 1.95) + _book("B", 1.85, 2.00))
    assert precompute.market_over25_anchor(cur, "m1") is None


def test_failed_lookup_rolls_back_and_returns_none():
    cur = ConsensusCursor(fail=True)
    assert precompute.market_over25_anchor(cur, "m1") is None
    assert any(s.startswith("ROLLBACK TO SAVEPOINT") for s in cur.statements)


class UpsertCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))


def test_market_rows_record_how_the_scoreline_was_reconciled():
    cur = UpsertCursor()
    markets = {"over_under": {"over_2.5": 0.55, "under_2.5": 0.45}, "btts": {"yes": 0.5, "no": 0.5}}
    meta = {"reconcile": "1x2+ou2.5", "market_over_2.5": 0.55}
    n = precompute.store_market_predictions(cur, "m1", markets, "v1", metadata=meta)
    assert n == 2
    for sql, params in cur.calls:
        assert json.loads(params["meta"]) == meta
        # merged, never overwritten
        assert "COALESCE(predictions.metadata" in sql


def test_market_rows_default_metadata_is_empty_object():
    cur = UpsertCursor()
    precompute.store_market_predictions(cur, "m1", {"btts": {"yes": 0.5, "no": 0.5}}, "v1")
    assert json.loads(cur.calls[0][1]["meta"]) == {}
