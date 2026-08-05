"""Unit tests for lottery_backtest.run_generate — one tracked line per
(game, strategy, target draw), lines logged visibly. Fake DB, no network."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load("lottery_backtest", REPO / "scripts" / "lottery_backtest.py")

# 40 plausible historical draws so generation has stats to work with.
FAKE_DRAWS = [
    {"numbers": [3 + i % 5, 11 + i % 7, 23 + i % 9, 40 + i % 11, 55 + i % 13], "bonus_number": 1 + i % 20}
    for i in range(40)
]


class FakeCursor:
    """Routes queries by shape: draw loads, dedup lookups, inserts."""

    def __init__(self, existing_line: dict | None):
        self.existing_line = existing_line
        self.inserts: list = []
        self._last = None

    def execute(self, sql, params=None):
        self._last = " ".join(sql.split())
        if self._last.startswith("INSERT INTO lottery_predictions"):
            self.inserts.append(params)

    def fetchall(self):
        assert "FROM lottery_draws" in self._last
        return FAKE_DRAWS

    def fetchone(self):
        assert "FROM lottery_predictions" in self._last
        return self.existing_line

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, existing_line):
        self.cur = FakeCursor(existing_line)
        self.committed = False

    def cursor(self, cursor_factory=None):
        return self.cur

    def commit(self):
        self.committed = True


def test_fresh_target_inserts_one_line_per_game_strategy():
    conn = FakeConn(existing_line=None)
    written = bt.run_generate(conn, today=date(2026, 8, 5), seed=7)
    n_expected = len(bt.GAME_CONFIG) * len(bt.STRATEGIES)
    assert written == n_expected
    assert len(conn.cur.inserts) == n_expected
    assert conn.committed


def test_already_tracked_target_inserts_nothing():
    existing = {"numbers": [4, 18, 26, 43, 51], "bonus_number": 4}
    conn = FakeConn(existing_line=existing)
    written = bt.run_generate(conn, today=date(2026, 8, 5), seed=7)
    assert written == 0
    assert conn.cur.inserts == []
    assert conn.committed  # still commits (a no-op transaction)
