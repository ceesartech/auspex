"""Unit tests for the expanded soccer odds mapping in fetch_live_odds —
all totals lines kept (not just 2.5), spreads → asian_handicap, and the
secondary per-event markets (btts, double_chance, draw_no_bet), plus
defensive skipping of unknown markets/outcomes. No network or DB."""

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


flo = _load("fetch_live_odds", "fetch_live_odds.py")
mo = flo.map_outcome
H, A = "Home FC", "Away FC"


@pytest.mark.unit
class TestMarketBaskets:
    def test_soccer_adds_spreads(self):
        assert flo.SPORT_MARKETS["soccer"] == "h2h,totals,spreads"

    def test_nhl_unchanged(self):
        assert flo.SPORT_MARKETS["nhl"] == "h2h,spreads,totals"

    def test_additional_markets(self):
        # Additional-markets list grew 2026-06-04 to include the HT
        # trio (h2h_h1, totals_h1, btts_h1) so the recs engine can
        # consume our HT predictions (PR #11, migration 014).
        assert flo.SPORT_ADDITIONAL_MARKETS["soccer"] == ("btts,double_chance,draw_no_bet,h2h_h1,totals_h1,btts_h1")

    def test_known_keys(self):
        assert flo.KNOWN_MARKET_KEYS["soccer"] >= {
            "h2h",
            "totals",
            "spreads",
            "btts",
            "double_chance",
            "draw_no_bet",
        }


@pytest.mark.unit
class TestTotalsKeepsAllLines:
    @pytest.mark.parametrize("point", [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    def test_over_lines_kept(self, point):
        assert mo("soccer", "totals", "Over", H, A, point) == ("over_under", "over", True)

    @pytest.mark.parametrize("point", [1.5, 3.5])
    def test_under_lines_kept(self, point):
        assert mo("soccer", "totals", "Under", H, A, point) == ("over_under", "under", True)

    def test_missing_point_dropped(self):
        assert mo("soccer", "totals", "Over", H, A, None) == (None, None, False)


@pytest.mark.unit
class TestSpreads:
    def test_home_side(self):
        assert mo("soccer", "spreads", H, H, A, -0.5) == ("asian_handicap", "home", True)

    def test_away_side(self):
        assert mo("soccer", "spreads", A, H, A, 0.5) == ("asian_handicap", "away", True)

    def test_no_point_dropped(self):
        assert mo("soccer", "spreads", H, H, A, None) == (None, None, False)


@pytest.mark.unit
class TestSecondaryMarkets:
    def test_btts(self):
        assert mo("soccer", "btts", "Yes", H, A, None) == ("btts", "yes", False)
        assert mo("soccer", "btts", "No", H, A, None) == ("btts", "no", False)

    def test_double_chance(self):
        assert mo("soccer", "double_chance", "Home FC/Draw", H, A, None) == ("double_chance", "1X", False)
        assert mo("soccer", "double_chance", "Home FC/Away FC", H, A, None) == ("double_chance", "12", False)
        assert mo("soccer", "double_chance", "Draw/Away FC", H, A, None) == ("double_chance", "X2", False)

    def test_draw_no_bet(self):
        assert mo("soccer", "draw_no_bet", H, H, A, None) == ("draw_no_bet", "home", False)
        assert mo("soccer", "draw_no_bet", A, H, A, None) == ("draw_no_bet", "away", False)


@pytest.mark.unit
class TestHalftimeMarkets:
    """Halftime soccer markets (migration 016 + PR #11 predictions).
    h2h_h1 → match_result_ht (3-way), totals_h1 → over_under_ht
    (keep_line=True, books offer 0.5 / 1.5), btts_h1 → btts_ht."""

    def test_h2h_h1_home(self):
        assert mo("soccer", "h2h_h1", H, H, A, None) == ("match_result_ht", "home", False)

    def test_h2h_h1_draw(self):
        assert mo("soccer", "h2h_h1", "Draw", H, A, None) == ("match_result_ht", "draw", False)

    def test_h2h_h1_away(self):
        assert mo("soccer", "h2h_h1", A, H, A, None) == ("match_result_ht", "away", False)

    def test_h2h_h1_unknown_team(self):
        assert mo("soccer", "h2h_h1", "Some Other Team", H, A, None) == (None, None, False)

    def test_totals_h1_over_with_line(self):
        assert mo("soccer", "totals_h1", "Over", H, A, 0.5) == ("over_under_ht", "over", True)

    def test_totals_h1_under_with_line(self):
        assert mo("soccer", "totals_h1", "Under", H, A, 1.5) == ("over_under_ht", "under", True)

    def test_totals_h1_missing_line(self):
        # Without a line we can't pin which O/U bucket to store.
        assert mo("soccer", "totals_h1", "Over", H, A, None) == (None, None, False)

    def test_totals_h1_unknown_outcome(self):
        assert mo("soccer", "totals_h1", "Maybe", H, A, 0.5) == (None, None, False)

    def test_btts_h1_yes(self):
        assert mo("soccer", "btts_h1", "Yes", H, A, None) == ("btts_ht", "yes", False)

    def test_btts_h1_no(self):
        assert mo("soccer", "btts_h1", "No", H, A, None) == ("btts_ht", "no", False)

    def test_btts_h1_unknown_outcome(self):
        assert mo("soccer", "btts_h1", "Maybe", H, A, None) == (None, None, False)

    def test_h2h_h1_h2_not_yet_supported(self):
        # h2h_h2 (second-half-only moneyline) isn't in KNOWN_MARKET_KEYS
        # — the parser returns (None, None, False) and the fetcher
        # skips with a "unknown market" log line. Documented behaviour.
        assert mo("soccer", "h2h_h2", H, H, A, None) == (None, None, False)


@pytest.mark.unit
class TestDefensive:
    def test_unknown_market_key(self):
        assert mo("soccer", "alternate_totals", "Over", H, A, 2.5) == (None, None, False)

    def test_unparseable_double_chance(self):
        assert mo("soccer", "double_chance", "???", H, A, None) == (None, None, False)

    def test_unknown_btts_outcome(self):
        assert mo("soccer", "btts", "Maybe", H, A, None) == (None, None, False)

    def test_dnb_unknown_team(self):
        assert mo("soccer", "draw_no_bet", "Some Other Team", H, A, None) == (None, None, False)


@pytest.mark.unit
class TestNhlStillWorks:
    def test_moneyline(self):
        assert mo("nhl", "h2h", "Home FC", H, A, None) == ("moneyline", "home", False)

    def test_total(self):
        assert mo("nhl", "totals", "Over", H, A, 5.5) == ("total", "over", True)
