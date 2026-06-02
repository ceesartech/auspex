"""Unit tests for generate_recommendations_nhl — pure math + odds-to-prob
mapping. No DB or HTTP."""

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


# Load telegram_notify first because the NHL recs script imports it at load.
_load("telegram_notify", "telegram_notify.py")
nhl_recs = _load("generate_recommendations_nhl", "generate_recommendations_nhl.py")


# ── EV / Kelly math ──────────────────────────────────────────────────


class TestExpectedValue:
    def test_positive_edge(self):
        # 60% chance @ 2.0 odds → +20% EV
        assert nhl_recs.expected_value(0.60, 2.0) == pytest.approx(0.20)

    def test_zero_edge(self):
        # Fair price: prob = 1/odds → EV exactly 0
        assert nhl_recs.expected_value(0.50, 2.0) == 0.0

    def test_negative_edge(self):
        # 30% chance @ 2.0 odds → -40% EV
        assert nhl_recs.expected_value(0.30, 2.0) == pytest.approx(-0.40)


class TestKellyFraction:
    def test_positive_edge_returns_finite_fraction(self):
        # 60% @ 2.0 → b=1, f = (0.6*2 - 1)/1 = 0.20
        assert nhl_recs.kelly_fraction(0.60, 2.0) == pytest.approx(0.20)

    def test_no_edge_returns_zero(self):
        # EV 0 → Kelly 0 (no bet)
        assert nhl_recs.kelly_fraction(0.50, 2.0) == 0.0

    def test_negative_edge_clamped_to_zero(self):
        # Kelly is undefined for negative edge; clamp to 0 so we never
        # accidentally suggest a "fade your model" bet.
        assert nhl_recs.kelly_fraction(0.30, 2.0) == 0.0

    def test_odds_at_or_below_one_returns_zero(self):
        # 1.0 odds means zero gross profit on a win — Kelly would
        # divide by zero. Must clamp.
        assert nhl_recs.kelly_fraction(0.99, 1.0) == 0.0
        assert nhl_recs.kelly_fraction(0.99, 0.9) == 0.0


class TestConfidenceRating:
    def test_very_high_requires_both_ev_and_prob(self):
        # 18% EV + 60% prob → very_high
        assert nhl_recs.confidence_rating(0.18, 0.60) == "very_high"
        # 18% EV but only 45% prob → just "high" (longshot territory)
        assert nhl_recs.confidence_rating(0.18, 0.45) == "high"

    def test_high_medium_low_buckets(self):
        assert nhl_recs.confidence_rating(0.12, 0.5) == "high"
        assert nhl_recs.confidence_rating(0.07, 0.5) == "medium"
        assert nhl_recs.confidence_rating(0.01, 0.5) == "low"

    def test_thresholds_match_soccer_engine(self):
        # The frontend renders confidence_rating as a colored badge —
        # same colors for both sports. The cutoffs MUST match the
        # soccer engine's thresholds; if either drifts, the colors
        # would become sport-dependent. Locked here.
        assert nhl_recs.confidence_rating(0.05, 0.5) == "medium"  # 5% EV cutoff
        assert nhl_recs.confidence_rating(0.04, 0.5) == "low"
        assert nhl_recs.confidence_rating(0.10, 0.5) == "high"  # 10% EV cutoff


# ── Odds-row → model-probability mapping ─────────────────────────────


class TestModelProbForOdds:
    def test_moneyline_home(self):
        probs = {"home": 0.62, "away": 0.38}
        assert nhl_recs.model_prob_for_odds("moneyline", probs, "home", None) == 0.62

    def test_moneyline_away(self):
        probs = {"home": 0.62, "away": 0.38}
        assert nhl_recs.model_prob_for_odds("moneyline", probs, "away", None) == 0.38

    def test_moneyline_unknown_selection(self):
        # NHL has no draw outcome — unrecognized selection → None.
        probs = {"home": 0.62, "away": 0.38}
        assert nhl_recs.model_prob_for_odds("moneyline", probs, "draw", None) is None

    def test_puck_line_home_minus_1p5_maps_to_cover(self):
        # "cover" = home covers -1.5
        probs = {"cover": 0.55, "no_cover": 0.45}
        assert nhl_recs.model_prob_for_odds("spread", probs, "home", -1.5) == 0.55

    def test_puck_line_away_plus_1p5_maps_to_no_cover(self):
        # "no_cover" (home doesn't cover) = away covers +1.5
        probs = {"cover": 0.55, "no_cover": 0.45}
        assert nhl_recs.model_prob_for_odds("spread", probs, "away", 1.5) == 0.45

    def test_puck_line_non_1p5_line_returns_none(self):
        # Alternate puck lines (-2.5, +2.5) aren't in the model's
        # training scope — return None instead of guessing.
        probs = {"cover": 0.55, "no_cover": 0.45}
        assert nhl_recs.model_prob_for_odds("spread", probs, "home", -2.5) is None
        assert nhl_recs.model_prob_for_odds("spread", probs, "away", 2.5) is None

    def test_puck_line_missing_line_returns_none(self):
        probs = {"cover": 0.55, "no_cover": 0.45}
        assert nhl_recs.model_prob_for_odds("spread", probs, "home", None) is None

    def test_total_canonical_line(self):
        probs = {"over": 0.52, "under": 0.48}
        assert nhl_recs.model_prob_for_odds("total", probs, "over", 5.5) == 0.52
        assert nhl_recs.model_prob_for_odds("total", probs, "under", 5.5) == 0.48

    def test_total_alternate_line_returns_none(self):
        # 6.5, 4.5, etc. are alternates we didn't train on.
        probs = {"over": 0.52, "under": 0.48}
        assert nhl_recs.model_prob_for_odds("total", probs, "over", 6.5) is None

    def test_unknown_prediction_type_returns_none(self):
        # Defensive — regulation odds aren't priced by sportsbooks so
        # they don't enter this path, but we still want None over a
        # confident wrong answer.
        assert nhl_recs.model_prob_for_odds("match_result", {"home": 0.5}, "home", None) is None


# ── Selection display ────────────────────────────────────────────────


class TestDisplaySelection:
    def test_moneyline_just_side(self):
        assert nhl_recs.display_selection("moneyline", "home", None) == "home"

    def test_puck_line_embeds_signed_line(self):
        assert nhl_recs.display_selection("spread", "home", -1.5) == "home -1.5"
        assert nhl_recs.display_selection("spread", "away", 1.5) == "away +1.5"

    def test_total_embeds_line(self):
        assert nhl_recs.display_selection("total", "over", 5.5) == "over 5.5"
        assert nhl_recs.display_selection("total", "under", 5.5) == "under 5.5"


# ── Risk factors ─────────────────────────────────────────────────────


class TestRiskFactors:
    def test_longshot_only_above_6_decimal(self):
        assert nhl_recs.risk_factors(0.20, 6.0) == ["longshot"]
        assert nhl_recs.risk_factors(0.20, 5.99) == []

    def test_low_prob_only_below_15pct(self):
        assert nhl_recs.risk_factors(0.14, 2.0) == ["low_model_probability"]
        assert nhl_recs.risk_factors(0.15, 2.0) == []

    def test_both_flags_compose(self):
        assert nhl_recs.risk_factors(0.10, 8.0) == ["longshot", "low_model_probability"]

    def test_clean_pick_no_flags(self):
        assert nhl_recs.risk_factors(0.55, 2.10) == []


# ── Market label / bet_type mapping ──────────────────────────────────


class TestMarketLabelMapping:
    def test_every_prediction_type_has_a_label(self):
        # Mismatch here would leak raw "spread" / "total" strings into
        # the Telegram digest or the betting_recommendations.bet_type
        # column. Locked.
        for ptype in nhl_recs.NHL_PREDICTION_TYPES:
            assert ptype in nhl_recs.MARKET_LABEL_FOR_PREDICTION_TYPE

    def test_spread_renames_to_puck_line_for_db(self):
        # bet_type column gets "puck_line" even though odds.market_type
        # is "spread" — the user-facing identifier matches NHL vocab.
        # This is enforced at the recommend_for_match level (we test
        # the conditional via inspection of the label map's value).
        assert nhl_recs.MARKET_LABEL_FOR_PREDICTION_TYPE["spread"].startswith("Puck")
