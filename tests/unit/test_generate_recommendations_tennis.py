"""Unit tests for generate_recommendations_tennis — pure helpers.

Mirror of test_generate_recommendations_nfl. Tennis has a single
market in v1 (moneyline, no spread/total), so the test set is
smaller. Same shared math (Kelly + EV + prob cap) is exercised here
to confirm the engine reuses the soccer helpers correctly.
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
gr_tennis = _load("generate_recommendations_tennis", "generate_recommendations_tennis.py")


class TestMarketDispatch:
    def test_only_moneyline_ensemble_in_v1(self):
        assert set(gr_tennis.TENNIS_MARKETS.keys()) == {"ensemble_tennis_ml"}

    def test_market_label_matches_taskspec(self):
        assert gr_tennis.TENNIS_MARKETS["ensemble_tennis_ml"][0] == "moneyline"

    def test_display_label_is_match_winner(self):
        # Locked here so the recs reasoning string and the digest
        # share the same user-facing label.
        assert gr_tennis.TENNIS_MARKETS["ensemble_tennis_ml"][1] == "Match Winner"

    def test_labels_match_taskspec(self):
        # 1v1 → "home"/"away" maps to player1/player2 (positional
        # convention from process_event when is_individual=True).
        assert gr_tennis.TENNIS_LABELS["moneyline"] == ["home", "away"]


class TestRiskFactors:
    def test_longshot_flag_at_4x_or_more(self):
        # Tennis +400 underdogs (~20% implied) are rare but happen
        # at the top of the draw — top-10 vs qualifier.
        assert "longshot" in gr_tennis._risk_factors(prob=0.25, odds_decimal=4.0)
        assert "longshot" in gr_tennis._risk_factors(prob=0.20, odds_decimal=5.0)
        assert "longshot" not in gr_tennis._risk_factors(prob=0.55, odds_decimal=1.85)

    def test_low_model_prob_flag_under_45(self):
        assert "low_model_probability" in gr_tennis._risk_factors(prob=0.40, odds_decimal=2.5)
        assert "low_model_probability" not in gr_tennis._risk_factors(prob=0.50, odds_decimal=2.0)

    def test_clean_pick_has_no_risks(self):
        assert gr_tennis._risk_factors(prob=0.55, odds_decimal=1.85) == []


class TestSharedMathReuse:
    def test_uses_soccer_kelly_and_ev(self):
        assert gr_tennis.expected_value(0.55, 2.0) == 0.55 * 2.0 - 1.0
        assert gr_tennis.kelly_fraction(0.55, 2.0) > 0
        assert gr_tennis.kelly_fraction(0.30, 2.0) == 0.0


class TestKellyConstant:
    def test_quarter_kelly(self):
        assert gr_tennis.KELLY_FRACTION == 0.25


class TestProbabilityCap:
    def test_cap_clips_overconfident_predictions(self):
        # Tennis moneyline models often emit 0.90+ on heavy favorites
        # (Djokovic vs lower-ranked qualifier). Cap protects stake
        # sizing from over-confident tails.
        capped = gr_tennis.cap_prob(0.92)
        assert capped == gr_tennis.PROB_CAP_FOR_EV

    def test_cap_leaves_low_confidence_alone(self):
        assert gr_tennis.cap_prob(0.65) == 0.65

    def test_cap_exactly_at_threshold_unchanged(self):
        assert gr_tennis.cap_prob(0.80) == 0.80

    def test_cap_value_locked(self):
        # 0.80 leaves a safety margin against worst-bucket
        # overconfidence. Same value as NBA/NFL.
        assert 0.75 <= gr_tennis.PROB_CAP_FOR_EV <= 0.85

    def test_capped_ev_smaller_than_raw_ev(self):
        # End-to-end: a raw 92% pick at 1.50 odds emits +38% EV.
        # Capped at 80%, EV drops to (0.80 × 1.50 - 1) = +20%.
        # Still positive, smaller stake.
        raw_ev = gr_tennis.expected_value(0.92, 1.50)
        capped = gr_tennis.cap_prob(0.92)
        capped_ev = gr_tennis.expected_value(capped, 1.50)
        assert capped_ev < raw_ev
        assert capped_ev > 0
