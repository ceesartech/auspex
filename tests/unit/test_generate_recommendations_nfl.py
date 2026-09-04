"""Unit tests for generate_recommendations_nfl — pure helpers.

Mirrors test_generate_recommendations_nba (the NFL engine is a near-direct
port). Same surface area: market dispatch, lined-bet display string,
risk-factor flags, shared math reuse, spread-line averaging pin.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("generate_recommendations", "generate_recommendations.py")
gr_nfl = _load("generate_recommendations_nfl", "generate_recommendations_nfl.py")


class TestMarketDispatch:
    def test_three_nfl_ensembles_registered(self):
        assert set(gr_nfl.NFL_MARKETS.keys()) == {
            "ensemble_nfl_ml",
            "ensemble_nfl_sp",
            "ensemble_nfl_tot",
        }

    def test_market_labels_match_taskspec(self):
        assert gr_nfl.NFL_MARKETS["ensemble_nfl_ml"][0] == "moneyline"
        assert gr_nfl.NFL_MARKETS["ensemble_nfl_sp"][0] == "spread"
        assert gr_nfl.NFL_MARKETS["ensemble_nfl_tot"][0] == "total"

    def test_labels_match_taskspec(self):
        assert gr_nfl.NFL_LABELS["moneyline"] == ["home", "away"]
        assert gr_nfl.NFL_LABELS["spread"] == ["home", "away"]
        assert gr_nfl.NFL_LABELS["total"] == ["over", "under"]


class TestSelectionWithLine:
    def test_moneyline_drops_line(self):
        assert gr_nfl._selection_with_line("moneyline", "home", None) == "home"
        assert gr_nfl._selection_with_line("moneyline", "away", -7.0) == "away"

    def test_spread_carries_signed_line(self):
        # NFL spreads frequently sit on key numbers (3, 7) — display
        # must encode the sign.
        assert gr_nfl._selection_with_line("spread", "home", -7.0) == "home_-7"
        assert gr_nfl._selection_with_line("spread", "away", 3.0) == "away_+3"
        assert gr_nfl._selection_with_line("spread", "home", 0.0) == "home_+0"

    def test_total_carries_unsigned_line(self):
        # NFL totals are typically in the 40-50 range — half-points
        # like 44.5 must round-trip through the display string.
        assert gr_nfl._selection_with_line("total", "over", 44.5) == "over_44.5"
        assert gr_nfl._selection_with_line("total", "under", 42.0) == "under_42"

    def test_missing_line_falls_back_to_bare_selection(self):
        assert gr_nfl._selection_with_line("spread", "home", None) == "home"


class TestRiskFactors:
    def test_longshot_flag_at_4x_or_more(self):
        # NFL has occasional +400 dogs (large divisional spread
        # mismatches) — flag them as longshot risk.
        assert "longshot" in gr_nfl._risk_factors(prob=0.25, odds_decimal=4.0)
        assert "longshot" in gr_nfl._risk_factors(prob=0.20, odds_decimal=5.0)
        assert "longshot" not in gr_nfl._risk_factors(prob=0.55, odds_decimal=1.85)

    def test_low_model_prob_flag_under_45(self):
        assert "low_model_probability" in gr_nfl._risk_factors(prob=0.40, odds_decimal=2.5)
        assert "low_model_probability" not in gr_nfl._risk_factors(prob=0.50, odds_decimal=2.0)

    def test_clean_pick_has_no_risks(self):
        assert gr_nfl._risk_factors(prob=0.55, odds_decimal=1.85) == []


class TestSharedMathReuse:
    def test_uses_soccer_kelly_and_ev(self):
        assert gr_nfl.expected_value(0.55, 2.0) == 0.55 * 2.0 - 1.0
        assert gr_nfl.kelly_fraction(0.55, 2.0) > 0
        assert gr_nfl.kelly_fraction(0.30, 2.0) == 0.0


class TestKellyConstant:
    def test_quarter_kelly(self):
        assert gr_nfl.KELLY_FRACTION == 0.25


class TestClosingLineForMatchSQL:
    """Same spread-line averaging pin as NBA — odds rows for home
    carry a SIGNED line (-7 for a 7-point favorite) and away the
    inverse (+7); averaging across both perspectives gives ~0 and
    fails the ±0.5 filter downstream. Pin to home for spread."""

    def test_spread_query_filters_to_home_perspective(self):
        captured = {}

        class FakeCursor:
            def execute(self, sql, params):
                captured["sql"] = sql
                captured["params"] = params

            def fetchone(self):
                return {"avg_line": -7.0}

        result = gr_nfl.closing_line_for_match(FakeCursor(), "match-id", "spread")
        assert result == -7.0
        assert "selection = 'home'" in captured["sql"]
        # the target line is averaged over lines still being QUOTED: a book
        # that moves its line leaves the abandoned one behind as its own key
        # forever (audit 2026-09).
        assert captured["params"] == ("match-id", "spread", gr_nfl.MAX_ODDS_AGE_HOURS)

    def test_total_query_omits_selection_filter(self):
        captured = {}

        class FakeCursor:
            def execute(self, sql, params):
                captured["sql"] = sql
                captured["params"] = params

            def fetchone(self):
                return {"avg_line": 44.5}

        result = gr_nfl.closing_line_for_match(FakeCursor(), "match-id", "total")
        assert result == 44.5
        assert "selection = 'home'" not in captured["sql"]

    def test_returns_none_when_no_odds(self):
        class FakeCursor:
            def execute(self, sql, params):
                pass

            def fetchone(self):
                return None

        assert gr_nfl.closing_line_for_match(FakeCursor(), "match-id", "spread") is None


class TestProbabilityCap:
    def test_cap_clips_overconfident_predictions(self):
        capped = gr_nfl.cap_prob(0.87)
        assert capped == gr_nfl.PROB_CAP_FOR_EV

    def test_cap_leaves_low_confidence_alone(self):
        assert gr_nfl.cap_prob(0.65) == 0.65

    def test_cap_exactly_at_threshold_unchanged(self):
        assert gr_nfl.cap_prob(0.80) == 0.80

    def test_cap_value_locked(self):
        assert 0.75 <= gr_nfl.PROB_CAP_FOR_EV <= 0.85

    def test_capped_ev_smaller_than_raw_ev(self):
        raw_ev = gr_nfl.expected_value(0.87, 1.95)
        capped = gr_nfl.cap_prob(0.87)
        capped_ev = gr_nfl.expected_value(capped, 1.95)
        assert capped_ev < raw_ev
        assert capped_ev > 0
