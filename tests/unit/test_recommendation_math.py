"""Unit tests for generate_recommendations pure logic — EV, Kelly, confidence
bucketing, the odds→model selection-key mapping (incl. the Asian-handicap sign
flip), push-aware value, and display formatting. No network or DB."""

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


gr = _load("generate_recommendations", "generate_recommendations.py")


@pytest.mark.unit
class TestExpectedValue:
    def test_positive_edge(self):
        assert abs(gr.expected_value(0.6, 2.0) - 0.2) < 1e-12

    def test_fair_price(self):
        assert abs(gr.expected_value(0.5, 2.0)) < 1e-12

    def test_negative_edge(self):
        assert gr.expected_value(0.4, 2.0) < 0


@pytest.mark.unit
class TestKelly:
    def test_known_value(self):
        # p=0.6, O=2.0 → f* = (1.2-1)/(2-1) = 0.2
        assert abs(gr.kelly_fraction(0.6, 2.0) - 0.2) < 1e-12

    def test_no_edge_is_zero(self):
        assert gr.kelly_fraction(0.5, 2.0) == 0.0

    def test_negative_clamped(self):
        assert gr.kelly_fraction(0.4, 2.0) == 0.0

    def test_odds_at_one_is_zero(self):
        assert gr.kelly_fraction(0.9, 1.0) == 0.0


@pytest.mark.unit
class TestConfidenceRating:
    def test_buckets(self):
        assert gr.confidence_rating(0.20, 0.60) == "very_high"
        assert gr.confidence_rating(0.20, 0.40) == "high"  # high EV but prob < 0.55
        assert gr.confidence_rating(0.12, 0.40) == "high"
        assert gr.confidence_rating(0.06, 0.40) == "medium"
        assert gr.confidence_rating(0.01, 0.90) == "low"


@pytest.mark.unit
class TestModelKeyForOdds:
    def test_simple_markets(self):
        assert gr.model_key_for_odds("1x2", "home", None) == "home"
        assert gr.model_key_for_odds("btts", "Yes", None) == "yes"
        assert gr.model_key_for_odds("double_chance", "1x", None) == "1X"
        assert gr.model_key_for_odds("draw_no_bet", "away", None) == "away"

    def test_over_under(self):
        assert gr.model_key_for_odds("over_under", "over", 2.5) == "over_2.5"
        assert gr.model_key_for_odds("over_under", "under", 1.5) == "under_1.5"
        assert gr.model_key_for_odds("over_under", "over", None) is None

    def test_asian_handicap_sign_flip(self):
        # the-odds-api reports each side's own signed point; the model indexes
        # by home-perspective line, so the away point is negated.
        assert gr.model_key_for_odds("asian_handicap", "home", -0.5) == "-0.5_home"
        assert gr.model_key_for_odds("asian_handicap", "away", 0.5) == "-0.5_away"
        assert gr.model_key_for_odds("asian_handicap", "home", -1.0) == "-1_home"
        assert gr.model_key_for_odds("asian_handicap", "away", 1.0) == "-1_away"

    def test_unpriced_market(self):
        assert gr.model_key_for_odds("team_total", "home_over", 1.5) is None

    def test_halftime_markets(self):
        # HT markets share the FT key conventions; the recs engine
        # routes them through the existing match-by-prediction-type
        # path without special handling.
        assert gr.model_key_for_odds("match_result_ht", "Home", None) == "home"
        assert gr.model_key_for_odds("match_result_ht", "Draw", None) == "draw"
        assert gr.model_key_for_odds("match_result_ht", "Away", None) == "away"
        assert gr.model_key_for_odds("btts_ht", "Yes", None) == "yes"
        assert gr.model_key_for_odds("btts_ht", "No", None) == "no"
        # over_under_ht: the HT_OU_LINES set is (0.5, 1.5, 2.5).
        assert gr.model_key_for_odds("over_under_ht", "Over", 0.5) == "over_0.5"
        assert gr.model_key_for_odds("over_under_ht", "Under", 1.5) == "under_1.5"

    def test_halftime_routes_via_odds_to_prediction_map(self):
        # Verifies the HT entries are present in ODDS_TO_PREDICTION
        # so the recs engine finds the right prediction_type row.
        # Each HT odds market_type maps to its identically-named
        # prediction_type.
        assert gr.ODDS_TO_PREDICTION["match_result_ht"] == "match_result_ht"
        assert gr.ODDS_TO_PREDICTION["over_under_ht"] == "over_under_ht"
        assert gr.ODDS_TO_PREDICTION["btts_ht"] == "btts_ht"

    def test_over_under_ht_in_lined_markets(self):
        # The lined-markets set drives display-string formatting
        # (selection_2.5 vs selection). Without the HT total here,
        # over_under_ht display lines would lose their line tag.
        assert "over_under_ht" in gr.LINED_MARKETS


@pytest.mark.unit
class TestSelectionValue:
    def test_simple_market(self):
        probs = {"home": 0.6, "draw": 0.25, "away": 0.15}
        pk, ev, raw = gr.selection_value("1x2", probs, "home", None, 2.0)
        assert abs(pk - 0.6) < 1e-12
        assert abs(ev - 0.2) < 1e-12
        assert abs(raw - 0.6) < 1e-12

    def test_asian_handicap_push(self):
        # raw win=0.45, push=0.10 at home-line 0; back home @ 2.1.
        probs = {"0_home": 0.45, "0_away": 0.45, "0_push": 0.10}
        pk, ev, raw = gr.selection_value("asian_handicap", probs, "home", 0.0, 2.1)
        assert abs(ev - (0.45 * 2.1 + 0.10 - 1.0)) < 1e-12
        assert abs(pk - (0.45 / 0.90)) < 1e-12  # no-push conditional
        assert abs(raw - 0.45) < 1e-12

    def test_unpriced_returns_none(self):
        assert gr.selection_value("1x2", {"draw": 1.0}, "home", None, 2.0) is None


@pytest.mark.unit
class TestDisplaySelection:
    def test_lined(self):
        assert gr.display_selection("over_under", "over", 2.5) == "over_2.5"
        assert gr.display_selection("asian_handicap", "home", -0.5) == "home_-0.5"

    def test_unlined(self):
        assert gr.display_selection("1x2", "home", None) == "home"
        assert gr.display_selection("btts", "yes", None) == "yes"

    def test_odds_to_prediction_covers_1x2(self):
        # The 1x2 headline maps to the match_result prediction row.
        assert gr.ODDS_TO_PREDICTION["1x2"] == "match_result"
