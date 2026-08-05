"""Unit tests for the pure lottery engine (services.lottery_analysis).

No DB. Covers: game configs, generation validity/distinctness/determinism,
strategy effects (ev avoids common picks; hot favors frequent numbers),
scoring (popularity / ev / profile_fit), analysis output, and prize-tier
settlement. These assert engine behavior — NOT that any strategy beats the
odds (it can't; draws are random).
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

# The lottery engine is pure stdlib, so load it directly by file path. This
# avoids the `services` package-name clash between the repo-root services/ dir
# and services/api/src/services under pytest's rootdir importer. Register it in
# sys.modules before exec so dataclass introspection can resolve __module__.
_ENGINE_PATH = Path(__file__).parent.parent.parent / "src" / "services" / "lottery_analysis.py"
_spec = importlib.util.spec_from_file_location("lottery_analysis", _ENGINE_PATH)
lottery_analysis = importlib.util.module_from_spec(_spec)
sys.modules["lottery_analysis"] = lottery_analysis
_spec.loader.exec_module(lottery_analysis)

GAME_CONFIG = lottery_analysis.GAME_CONFIG
STRATEGIES = lottery_analysis.STRATEGIES
analyze = lottery_analysis.analyze
combination_features = lottery_analysis.combination_features
ev_score = lottery_analysis.ev_score
generate_combinations = lottery_analysis.generate_combinations
get_game_config = lottery_analysis.get_game_config
number_stats = lottery_analysis.number_stats
popularity_score = lottery_analysis.popularity_score
prize_tier = lottery_analysis.prize_tier
profile_fit = lottery_analysis.profile_fit
profile_stats = lottery_analysis.profile_stats
settle = lottery_analysis.settle

CFG = GAME_CONFIG["powerball"]


def _synthetic_history(n=300, hot=(), seed=1):
    """n Powerball draws. `hot` numbers (must be fewer than main_count) are
    forced into every draw so frequency/recency tests get a strong,
    deterministic signal; the remaining picks are uniformly random."""
    assert len(hot) < CFG.main_count
    rng = random.Random(seed)
    main, bonus = [], []
    for _ in range(n):
        pool = [x for x in range(1, 70) if x not in hot]
        rest = rng.sample(pool, CFG.main_count - len(hot))
        main.append(sorted(list(hot) + rest))
        bonus.append(rng.randint(1, CFG.bonus_max))
    return main, bonus


@pytest.fixture
def history():
    """Purely random history."""
    return _synthetic_history()


# Three numbers forced into every draw — for frequency/recency/hot tests.
FORCED_HOT = (62, 64, 66)


@pytest.fixture
def hot_history():
    return _synthetic_history(hot=FORCED_HOT)


@pytest.mark.unit
class TestConfig:
    def test_games(self):
        assert get_game_config("powerball").main_max == 69
        assert get_game_config("powerball").bonus_max == 26
        assert get_game_config("mega_millions").main_max == 70
        # 24 since the Apr-2025 game change (was 25) — sourced from the
        # era-aware rules registry, not a hardcoded literal.
        assert get_game_config("mega_millions").bonus_max == 24

    def test_unknown_game(self):
        with pytest.raises(ValueError):
            get_game_config("keno")


@pytest.mark.unit
class TestGeneration:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_valid_lines(self, history, strategy):
        main, bonus = history
        combos = generate_combinations(main, bonus, CFG, strategy=strategy, n=5, seed=7)
        assert len(combos) == 5
        for c in combos:
            assert len(c.numbers) == CFG.main_count
            assert len(set(c.numbers)) == CFG.main_count  # no intra-line dupes
            assert all(1 <= x <= CFG.main_max for x in c.numbers)
            assert 1 <= c.bonus_number <= CFG.bonus_max
            assert c.numbers == sorted(c.numbers)

    def test_lines_mutually_distinct(self, history):
        main, bonus = history
        combos = generate_combinations(main, bonus, CFG, strategy="blend", n=5, seed=7)
        sets = [set(c.numbers) for c in combos]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert len(sets[i] & sets[j]) <= CFG.main_count - 2

    def test_deterministic_with_seed(self, history):
        main, bonus = history
        a = generate_combinations(main, bonus, CFG, strategy="blend", n=5, seed=99)
        b = generate_combinations(main, bonus, CFG, strategy="blend", n=5, seed=99)
        assert [c.numbers for c in a] == [c.numbers for c in b]
        assert [c.bonus_number for c in a] == [c.bonus_number for c in b]

    def test_requested_count_respected(self, history):
        main, bonus = history
        assert len(generate_combinations(main, bonus, CFG, strategy="ev", n=3, seed=1)) == 3
        assert len(generate_combinations(main, bonus, CFG, strategy="ev", n=10, seed=1)) == 10

    def test_unknown_strategy_raises(self, history):
        main, bonus = history
        with pytest.raises(ValueError):
            generate_combinations(main, bonus, CFG, strategy="psychic", n=5)


@pytest.mark.unit
class TestStrategyEffects:
    def test_ev_avoids_popular_numbers(self, history):
        main, bonus = history
        ev = generate_combinations(main, bonus, CFG, strategy="ev", n=10, seed=3)
        rnd = generate_combinations(main, bonus, CFG, strategy="random", n=10, seed=3)
        ev_pop = sum(popularity_score(c.numbers, CFG) for c in ev) / len(ev)
        rnd_pop = sum(popularity_score(c.numbers, CFG) for c in rnd) / len(rnd)
        assert ev_pop < rnd_pop

    def test_hot_favors_frequent_numbers(self, hot_history):
        main, bonus = hot_history
        hot = generate_combinations(main, bonus, CFG, strategy="hot", n=5, seed=3)
        rnd = generate_combinations(main, bonus, CFG, strategy="random", n=5, seed=3)
        hot_comp = sum(c.features["hot"] for c in hot) / len(hot)
        rnd_comp = sum(c.features["hot"] for c in rnd) / len(rnd)
        assert hot_comp > rnd_comp


@pytest.mark.unit
class TestScoring:
    def test_number_stats(self, hot_history):
        main, bonus = hot_history
        stats = number_stats(main, CFG)
        assert stats["total_draws"] == len(main)
        # The forced-hot numbers appear in every draw.
        for n in FORCED_HOT:
            assert stats["frequency"][n] == len(main)
            assert stats["recency"][n] == 0

    def test_popularity_birthday_vs_spread(self):
        # An all-birthday, sequential line is far more commonly picked than a
        # spread-out high line.
        assert popularity_score([1, 2, 3, 4, 5], CFG) > popularity_score([13, 29, 41, 58, 67], CFG)

    def test_ev_is_complement_of_popularity(self):
        nums = [4, 17, 23, 38, 51]
        assert abs(ev_score(nums, CFG) - (1.0 - popularity_score(nums, CFG))) < 1e-9

    def test_profile_fit_prefers_typical_line(self, history):
        main, _ = history
        stats = profile_stats(main, CFG)
        # A balanced mid-range line (near a uniform draw's profile) should fit
        # better than a degenerate all-low sequence.
        typical = [8, 21, 34, 47, 60]
        assert profile_fit(typical, CFG, stats) > profile_fit([1, 2, 3, 4, 5], CFG, stats)

    def test_combination_features(self):
        feats = combination_features([2, 4, 10, 35, 36], CFG)
        assert feats["sum"] == 87.0
        assert feats["odd_count"] == 1.0  # only 35
        assert feats["even_count"] == 4.0
        assert feats["low_count"] == 3.0  # <= 34: 2,4,10
        assert feats["consecutive_pairs"] == 1.0  # 35,36


@pytest.mark.unit
class TestAnalyze:
    def test_keys_and_profile(self, hot_history):
        main, bonus = hot_history
        result = analyze(main, bonus, CFG)
        for key in ("hot_numbers", "cold_numbers", "overdue_numbers", "frequency_distribution", "profile"):
            assert key in result
        assert result["total_draws_analyzed"] == len(main)
        assert "sum" in result["profile"]
        assert "top_pairs" in result["profile"]
        # Forced-hot numbers dominate the hot list.
        hot_nums = {row["number"] for row in result["hot_numbers"]}
        assert set(FORCED_HOT).issubset(hot_nums)


@pytest.mark.unit
class TestSettlement:
    @pytest.mark.parametrize(
        "main_n,bonus,expected",
        [
            (5, True, "jackpot"),
            (5, False, "match_5"),
            (4, True, "match_4_bonus"),
            (4, False, "match_4"),
            (3, True, "match_3_bonus"),
            (1, True, "match_1_bonus"),
            (0, True, "match_bonus"),
            (2, False, None),
            (0, False, None),
        ],
    )
    def test_prize_tier(self, main_n, bonus, expected):
        assert prize_tier(main_n, bonus, "powerball") == expected

    def test_settle(self):
        res = settle([1, 2, 3, 4, 5], 10, [1, 2, 3, 9, 11], 10, "powerball")
        assert res["matched_main"] == 3
        assert res["matched_bonus"] is True
        assert res["prize_tier"] == "match_3_bonus"

    def test_settle_no_prize(self):
        res = settle([1, 2, 3, 4, 5], 10, [20, 30, 40, 50, 60], 11, "powerball")
        assert res["matched_main"] == 0
        assert res["matched_bonus"] is False
        assert res["prize_tier"] is None
