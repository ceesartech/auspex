"""Unit tests for fetch_lottery_draws — row parsing + era validation + upsert
counting, no network or DB."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fetch = _load("fetch_lottery_draws", SCRIPTS_DIR / "fetch_lottery_draws.py")


class TestParsePowerball:
    def test_valid_row(self):
        row = {"draw_date": "2026-08-03T00:00:00.000", "winning_numbers": "08 30 41 48 54 04", "multiplier": "2"}
        d = fetch.parse_row("powerball", row)
        assert d == {
            "draw_date": date(2026, 8, 3),
            "numbers": [8, 30, 41, 48, 54],
            "bonus_number": 4,
            "multiplier": 2,
        }

    def test_wrong_count_skipped(self):
        row = {"draw_date": "2026-08-03T00:00:00.000", "winning_numbers": "08 30 41 48 54"}
        assert fetch.parse_row("powerball", row) is None

    def test_bonus_out_of_era_range_skipped(self):
        # PB 27 doesn't exist in the current 1-26 pool.
        row = {"draw_date": "2026-08-03T00:00:00.000", "winning_numbers": "08 30 41 48 54 27"}
        assert fetch.parse_row("powerball", row) is None

    def test_old_era_validates_against_its_own_matrix(self):
        # 2011 era was 5/59 + 1/39 — powerball 35 valid THEN, invalid today.
        row = {"draw_date": "2011-06-01T00:00:00.000", "winning_numbers": "05 12 33 41 59 35"}
        d = fetch.parse_row("powerball", row)
        assert d is not None and d["bonus_number"] == 35

    def test_mains_sorted(self):
        row = {"draw_date": "2026-08-03T00:00:00.000", "winning_numbers": "54 08 48 30 41 04"}
        assert fetch.parse_row("powerball", row)["numbers"] == [8, 30, 41, 48, 54]


class TestParseMegaMillions:
    def test_valid_row_current_era(self):
        row = {"draw_date": "2026-08-04T00:00:00.000", "winning_numbers": "14 21 51 55 65", "mega_ball": "21"}
        d = fetch.parse_row("mega_millions", row)
        assert d["numbers"] == [14, 21, 51, 55, 65]
        assert d["bonus_number"] == 21
        assert d["multiplier"] is None

    def test_megaball_25_invalid_after_apr_2025(self):
        row = {"draw_date": "2025-05-02T00:00:00.000", "winning_numbers": "14 21 51 55 65", "mega_ball": "25"}
        assert fetch.parse_row("mega_millions", row) is None

    def test_megaball_25_valid_before_apr_2025(self):
        row = {"draw_date": "2024-05-03T00:00:00.000", "winning_numbers": "14 21 51 55 65", "mega_ball": "25"}
        d = fetch.parse_row("mega_millions", row)
        assert d is not None and d["bonus_number"] == 25

    def test_megaplier_era_multiplier_kept(self):
        row = {
            "draw_date": "2024-05-03T00:00:00.000",
            "winning_numbers": "14 21 51 55 65",
            "mega_ball": "12",
            "multiplier": "4",
        }
        assert fetch.parse_row("mega_millions", row)["multiplier"] == 4

    def test_duplicate_mains_skipped(self):
        row = {"draw_date": "2026-08-04T00:00:00.000", "winning_numbers": "14 14 51 55 65", "mega_ball": "12"}
        assert fetch.parse_row("mega_millions", row) is None

    def test_prehistoric_draw_skipped(self):
        row = {"draw_date": "2001-01-05T00:00:00.000", "winning_numbers": "14 21 41 45 50", "mega_ball": "12"}
        assert fetch.parse_row("mega_millions", row) is None


class FakeCursor:
    """Counts inserts; every other row pretends to be a conflict no-op."""

    def __init__(self):
        self.calls = 0
        self.rowcount = 0

    def execute(self, _sql, _params):
        self.calls += 1
        self.rowcount = self.calls % 2  # alternate written / conflict-skipped


def test_store_draws_counts_only_new_rows():
    cur = FakeCursor()
    draws = [
        {"draw_date": date(2026, 8, d), "numbers": [1, 2, 3, 4, 5], "bonus_number": 1, "multiplier": None}
        for d in range(1, 5)
    ]
    written = fetch.store_draws(cur, "powerball", draws)
    assert cur.calls == 4
    assert written == 2  # two of four were conflict no-ops
