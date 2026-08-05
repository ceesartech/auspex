"""Era-aware lottery rules: combinatorial tier odds must reproduce the
officially published odds figures exactly, and era lookup must respect the
Apr-2025 Mega Millions game change."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

_RULES_PATH = Path(__file__).parent.parent.parent / "src" / "services" / "lottery_rules.py"
_spec = importlib.util.spec_from_file_location("lottery_rules", _RULES_PATH)
lottery_rules = importlib.util.module_from_spec(_spec)
sys.modules["lottery_rules"] = lottery_rules
_spec.loader.exec_module(lottery_rules)

all_tier_probabilities = lottery_rules.all_tier_probabilities
analysis_window_starts = lottery_rules.analysis_window_starts
current_rules = lottery_rules.current_rules
jackpot_odds = lottery_rules.jackpot_odds
rules_for = lottery_rules.rules_for
tier_probability = lottery_rules.tier_probability


class TestPublishedOddsReproduction:
    """Every figure the lotteries publish must fall out of the hypergeometric
    math exactly — no hardcoded odds anywhere."""

    def test_powerball_jackpot_odds(self):
        assert jackpot_odds(current_rules("powerball")) == pytest.approx(292_201_338, abs=1)

    def test_powerball_published_tier_odds(self):
        era = current_rules("powerball")
        published = {  # powerball.com prize chart, 1-in-N
            "match_5": 11_688_053.52,
            "match_4_bonus": 913_129.18,
            "match_4": 36_525.17,
            "match_3_bonus": 14_494.11,
            "match_3": 579.76,
            "match_2_bonus": 701.33,
            "match_1_bonus": 91.98,
            "match_bonus": 38.32,
        }
        probs = all_tier_probabilities(era)
        for tier, one_in_n in published.items():
            # rel=5e-4 because the published figures are rounded to 2 decimals
            # (e.g. exact 38.32394 is published as 38.32).
            assert 1.0 / probs[tier] == pytest.approx(one_in_n, rel=5e-4), tier

    def test_mega_millions_jackpot_odds_post_2025(self):
        assert jackpot_odds(current_rules("mega_millions")) == pytest.approx(290_472_336, abs=1)

    def test_mega_millions_match5_exact(self):
        # C(70,5) * 24/23 = 12,629,232 exactly.
        p = tier_probability(current_rules("mega_millions"), 5, False)
        assert 1.0 / p == pytest.approx(12_629_232, abs=1)

    def test_mega_millions_10_dollar_amount_combined_odds(self):
        # The published "$10 at 1 in 318" combines the two $10 base tiers
        # (match_3 and match_2_bonus).
        era = current_rules("mega_millions")
        p = tier_probability(era, 3, False) + tier_probability(era, 2, True)
        assert 1.0 / p == pytest.approx(318, rel=0.01)

    def test_overall_any_prize_odds(self):
        pb = sum(all_tier_probabilities(current_rules("powerball")).values())
        mm = sum(all_tier_probabilities(current_rules("mega_millions")).values())
        assert 1.0 / pb == pytest.approx(24.87, rel=0.01)  # published "1 in 24.87"
        assert 1.0 / mm == pytest.approx(23.0, rel=0.02)  # published "1 in 23"


class TestEras:
    def test_mm_apr_2025_change(self):
        before = rules_for("mega_millions", date(2025, 4, 1))
        after = rules_for("mega_millions", date(2025, 4, 8))
        assert (before.bonus_max, before.ticket_price, before.expected_multiplier) == (25, 2.0, 1.0)
        assert (after.bonus_max, after.ticket_price, after.expected_multiplier) == (24, 5.0, 3.0)

    def test_mm_expected_multiplier_is_exactly_3(self):
        # Field of 32: fifteen 2x, ten 3x, four 4x, two 5x, one 10x.
        assert (15 * 2 + 10 * 3 + 4 * 4 + 2 * 5 + 1 * 10) / 32 == 3.0
        assert current_rules("mega_millions").expected_multiplier == 3.0

    def test_powerball_2015_matrix(self):
        era = rules_for("powerball", date(2016, 1, 1))
        assert (era.main_max, era.bonus_max) == (69, 26)
        old = rules_for("powerball", date(2013, 6, 1))
        assert (old.main_max, old.bonus_max) == (59, 35)

    def test_prehistoric_date_returns_none(self):
        assert rules_for("powerball", date(2000, 1, 1)) is None

    def test_unknown_game_raises(self):
        with pytest.raises(ValueError):
            rules_for("euromillions", date(2026, 1, 1))


class TestAnalysisWindows:
    def test_mm_mains_span_2017_but_bonus_starts_2025(self):
        main_start, bonus_start = analysis_window_starts("mega_millions")
        assert main_start == date(2017, 10, 31)  # 5/70 unchanged by the 2025 change
        assert bonus_start == date(2025, 4, 8)  # megaball pool shrank to 24

    def test_powerball_both_from_2015(self):
        main_start, bonus_start = analysis_window_starts("powerball")
        assert main_start == bonus_start == date(2015, 10, 7)
