"""Lottery combination analysis + generation engine (pure, no DB).

HONEST FRAMING — read this before changing anything here.
Powerball / Mega Millions draws are uniform and independent. Nothing in this
module raises the probability of matching a draw; that is mathematically
impossible for a fair lottery. What it does do:

  1. Summarize historical draws across many features (frequency, recency, sum,
     odd/even, high/low, spread, decade spread, ...).
  2. Generate N distinct combinations that rank well under a chosen strategy.
  3. Optimize EXPECTED VALUE via jackpot-share avoidance — favor numbers and
     patterns humans under-pick (avoid birthdays <= 31, sequences, "lucky" 7,
     single-decade clusters), so that *if* a line wins it is less likely to be
     split. This raises expected payout, NOT the chance of winning.

`score` is an internal ranking signal in [0, 1]; it is never a win probability.
The functions are pure (lists in, dataclasses/dicts out) so they unit-test
without a database. A seeded `random.Random` makes generation reproducible.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ── Game configuration ────────────────────────────────────────────────────


@dataclass(frozen=True)
class GameConfig:
    key: str
    main_count: int  # how many main numbers are drawn
    main_max: int  # main numbers run 1..main_max
    bonus_max: int  # bonus ball runs 1..bonus_max
    bonus_name: str


GAME_CONFIG: Dict[str, GameConfig] = {
    "powerball": GameConfig("powerball", 5, 69, 26, "powerball"),
    "mega_millions": GameConfig("mega_millions", 5, 70, 25, "megaball"),
}

STRATEGIES: Tuple[str, ...] = ("blend", "statistical", "ev", "hot", "due", "random")

DISCLAIMER = (
    "Lottery draws are random and independent — no analysis can improve your odds "
    "of winning. These combinations are optimized for statistical profile and for "
    "expected value (avoiding commonly-picked numbers so a win is less likely to be "
    "shared), not to predict the draw. Play responsibly."
)

# Composite scoring weights per strategy, over (profile_fit, ev, hot, due).
STRATEGY_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "blend": (0.35, 0.35, 0.15, 0.15),
    "statistical": (0.80, 0.10, 0.05, 0.05),
    "ev": (0.15, 0.80, 0.025, 0.025),
    "hot": (0.20, 0.10, 0.70, 0.00),
    "due": (0.20, 0.10, 0.00, 0.70),
    "random": (0.0, 0.0, 0.0, 0.0),  # uniform — score is not used for ranking
}


@dataclass
class Combination:
    numbers: List[int]
    bonus_number: int
    score: float
    strategy: str
    rationale: str
    features: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "numbers": self.numbers,
            "bonus_number": self.bonus_number,
            "score": self.score,
            "strategy": self.strategy,
            "rationale": self.rationale,
            "features": self.features,
        }


def get_game_config(game: str) -> GameConfig:
    try:
        return GAME_CONFIG[game]
    except KeyError as exc:
        raise ValueError(f"Unknown lottery game {game!r}") from exc


# ── Per-number statistics ───────────────────────────────────────────────────


def number_stats(main_draws: Sequence[Sequence[int]], cfg: GameConfig) -> Dict:
    """Frequency and recency for every main number. `main_draws` is ordered
    most-recent-first. recency[n] = how many draws ago n last appeared
    (len(draws) if it never has)."""
    total = len(main_draws)
    freq = {n: 0 for n in range(1, cfg.main_max + 1)}
    recency = {n: total for n in range(1, cfg.main_max + 1)}
    seen: set = set()
    for i, draw in enumerate(main_draws):
        for n in draw:
            if 1 <= n <= cfg.main_max:
                freq[n] += 1
                if n not in seen:
                    recency[n] = i
                    seen.add(n)
    return {"frequency": freq, "recency": recency, "total_draws": total}


def _normalize(values: Dict[int, float], invert: bool = False) -> Dict[int, float]:
    """Min-max normalize a number->value map into 0..1 scores (0.5 flat if no
    spread). invert flips so a low raw value yields a high score."""
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    out = {}
    for n, v in values.items():
        s = (v - lo) / rng if rng > 0 else 0.5
        out[n] = (1.0 - s) if invert else s
    return out


# ── Combination-level features ──────────────────────────────────────────────


def combination_features(numbers: Sequence[int], cfg: GameConfig) -> Dict[str, float]:
    nums = sorted(numbers)
    n = len(nums)
    odd = sum(1 for x in nums if x % 2 == 1)
    low = sum(1 for x in nums if x <= cfg.main_max // 2)
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    return {
        "sum": float(sum(nums)),
        "odd_count": float(odd),
        "even_count": float(n - odd),
        "low_count": float(low),
        "high_count": float(n - low),
        "spread": float(nums[-1] - nums[0]) if n else 0.0,
        "consecutive_pairs": float(sum(1 for d in diffs if d == 1)),
        "decade_spread": float(len({(x - 1) // 10 for x in nums})),
        "distinct_last_digits": float(len({x % 10 for x in nums})),
    }


def popularity_score(numbers: Sequence[int], cfg: GameConfig) -> float:
    """Heuristic 0..1 estimate of how commonly humans pick this combination
    (higher = more shared = worse expected value). Calibrated to well-documented
    human biases: calendar/birthday numbers, arithmetic sequences, lucky 7,
    round multiples, single-decade clusters."""
    nums = sorted(numbers)
    n = len(nums)
    if n == 0:
        return 0.0
    score = 0.0

    # Calendar/birthday bias — the dominant tell. Numbers <= 31 are days; <= 12
    # double as months. Heavy <=31 lines are massively over-picked.
    score += 0.45 * (sum(1 for x in nums if x <= 31) / n)
    score += 0.10 * (sum(1 for x in nums if x <= 12) / n)

    # Arithmetic / sequence patterns (1-2-3-4-5, 5-10-15-20-25, ...).
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    if len(nums) >= 2 and len(set(diffs)) == 1:
        score += 0.25
    score += 0.10 * (sum(1 for d in diffs if d == 1) / max(1, n - 1))

    # "Lucky" 7 and round multiples of 5/10.
    if 7 in nums:
        score += 0.04
    score += 0.04 * (sum(1 for x in nums if x % 5 == 0) / n)

    # Single/double-decade cluster.
    if len({(x - 1) // 10 for x in nums}) <= 2:
        score += 0.08

    return min(1.0, score)


def ev_score(numbers: Sequence[int], cfg: GameConfig) -> float:
    """Jackpot-share-avoidance score in 0..1 (higher = less commonly picked =
    better expected payout if it wins). The only mathematically real edge."""
    return 1.0 - popularity_score(numbers, cfg)


# ── Historical profile + fit ────────────────────────────────────────────────

_PROFILE_KEYS = ("sum", "odd_count", "low_count", "spread")


def profile_stats(main_draws: Sequence[Sequence[int]], cfg: GameConfig) -> Dict[str, Tuple[float, float]]:
    """Mean and (population) std of the profile features across historical
    full-size draws. Used to score how 'typical' a candidate line looks."""
    feats = [combination_features(d, cfg) for d in main_draws if len(d) == cfg.main_count]
    stats: Dict[str, Tuple[float, float]] = {}
    for key in _PROFILE_KEYS:
        vals = [f[key] for f in feats]
        mean = statistics.fmean(vals) if vals else 0.0
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        stats[key] = (mean, std if std > 0 else 1.0)
    return stats


def profile_fit(numbers: Sequence[int], cfg: GameConfig, stats: Dict[str, Tuple[float, float]]) -> float:
    """0..1 Gaussian closeness of this combination's profile features to the
    historical means (1.0 = dead-on typical, →0 as it gets unusual)."""
    feats = combination_features(numbers, cfg)
    z_sq = 0.0
    for key, (mean, std) in stats.items():
        z = (feats[key] - mean) / std
        z_sq += z * z
    return math.exp(-0.5 * z_sq / max(1, len(stats)))


# ── Scoring ─────────────────────────────────────────────────────────────────


def score_combination(
    numbers: Sequence[int],
    cfg: GameConfig,
    *,
    stats: Dict[str, Tuple[float, float]],
    hot_scores: Dict[int, float],
    due_scores: Dict[int, float],
    weights: Tuple[float, float, float, float],
) -> Tuple[float, Dict[str, float]]:
    """Composite score in 0..1 + the component breakdown."""
    pf = profile_fit(numbers, cfg, stats)
    ev = ev_score(numbers, cfg)
    hot = statistics.fmean([hot_scores[n] for n in numbers]) if numbers else 0.0
    due = statistics.fmean([due_scores[n] for n in numbers]) if numbers else 0.0
    breakdown = {
        "profile_fit": round(pf, 4),
        "ev_score": round(ev, 4),
        "hot": round(hot, 4),
        "due": round(due, 4),
        "popularity": round(1.0 - ev, 4),
    }
    w_pf, w_ev, w_hot, w_due = weights
    wsum = w_pf + w_ev + w_hot + w_due
    if wsum == 0:  # random strategy — neutral score, ranking is shuffled
        return 0.5, breakdown
    composite = (w_pf * pf + w_ev * ev + w_hot * hot + w_due * due) / wsum
    return composite, breakdown


# ── Generation ──────────────────────────────────────────────────────────────


def _sampling_weights(
    strategy: str, cfg: GameConfig, hot_scores: Dict[int, float], due_scores: Dict[int, float]
) -> Tuple[List[int], List[float]]:
    """Per-number sampling weights that bias the candidate pool toward the
    strategy. Scoring then does the fine ranking."""
    nums = list(range(1, cfg.main_max + 1))
    if strategy == "hot":
        w = [hot_scores[n] + 0.05 for n in nums]
    elif strategy == "due":
        w = [due_scores[n] + 0.05 for n in nums]
    elif strategy == "ev":
        # Bias away from the birthday range so the pool is rich in high numbers.
        w = [1.0 if n > 31 else 0.25 for n in nums]
    elif strategy in ("statistical", "blend"):
        w = [0.5 + 0.5 * hot_scores[n] for n in nums]
    else:  # random / unknown -> uniform
        w = [1.0] * len(nums)
    return nums, w


def _weighted_sample_distinct(rng: random.Random, population: List[int], weights: List[float], k: int) -> List[int]:
    """Sample k distinct items with the given weights (sampling without
    replacement, weight-biased)."""
    chosen: set = set()
    guard = 0
    limit = max(1000, k * 200)
    while len(chosen) < k and guard < limit:
        guard += 1
        chosen.add(rng.choices(population, weights=weights, k=1)[0])
    # Fallback top-up if weights were degenerate.
    if len(chosen) < k:
        for x in population:
            chosen.add(x)
            if len(chosen) >= k:
                break
    return sorted(chosen)


def _pick_bonus(strategy: str, cfg: GameConfig, bonus_freq: Dict[int, int], rng: random.Random) -> int:
    if bonus_freq:
        if strategy == "hot":
            return max(bonus_freq, key=lambda n: bonus_freq.get(n, 0))
        if strategy == "due":
            return min(bonus_freq, key=lambda n: bonus_freq.get(n, 0))
    return rng.randint(1, cfg.bonus_max)


def _rationale(numbers: Sequence[int], cfg: GameConfig, breakdown: Dict[str, float]) -> str:
    feats = combination_features(numbers, cfg)
    return (
        f"sum {int(feats['sum'])}, {int(feats['odd_count'])} odd / {int(feats['even_count'])} even, "
        f"{int(feats['low_count'])} low / {int(feats['high_count'])} high, "
        f"spread {int(feats['spread'])}; EV score {breakdown['ev_score']:.2f} "
        f"(popularity {breakdown['popularity']:.2f}), profile fit {breakdown['profile_fit']:.2f}."
    )


def generate_combinations(
    main_draws: Sequence[Sequence[int]],
    bonus_draws: Sequence[int],
    cfg: GameConfig,
    *,
    strategy: str = "blend",
    n: int = 5,
    seed: Optional[int] = None,
    pool_size: int = 2000,
    max_overlap: Optional[int] = None,
) -> List[Combination]:
    """Generate `n` distinct, scored combinations for `strategy`.

    Builds a weight-biased candidate pool, scores each candidate by the
    strategy's composite, then greedily selects the top-scoring lines that don't
    overlap an already-selected line by more than `max_overlap` numbers (so the
    N returned lines are genuinely different). `seed` makes output reproducible.
    """
    if strategy not in STRATEGY_WEIGHTS:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if max_overlap is None:
        max_overlap = max(1, cfg.main_count - 2)  # e.g. 5-number game -> overlap <= 3

    rng = random.Random(seed)
    stats = number_stats(main_draws, cfg)
    hot_scores = _normalize(stats["frequency"])  # high frequency -> high
    due_scores = _normalize(stats["recency"])  # long since seen -> high
    prof = profile_stats(main_draws, cfg)
    weights = STRATEGY_WEIGHTS[strategy]

    bonus_freq: Dict[int, int] = {}
    for b in bonus_draws:
        if 1 <= b <= cfg.bonus_max:
            bonus_freq[b] = bonus_freq.get(b, 0) + 1

    nums, sw = _sampling_weights(strategy, cfg, hot_scores, due_scores)

    # Build a pool of unique candidate combinations.
    candidates: set = set()
    attempts = 0
    attempt_cap = pool_size * 20
    while len(candidates) < pool_size and attempts < attempt_cap:
        attempts += 1
        candidates.add(tuple(_weighted_sample_distinct(rng, nums, sw, cfg.main_count)))

    scored: List[Tuple[float, Tuple[int, ...], Dict[str, float]]] = []
    for combo in candidates:
        s, breakdown = score_combination(
            list(combo), cfg, stats=prof, hot_scores=hot_scores, due_scores=due_scores, weights=weights
        )
        scored.append((s, combo, breakdown))

    if strategy == "random":
        rng.shuffle(scored)
    else:
        scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Combination] = []
    for s, combo, breakdown in scored:
        combo_set = set(combo)
        if all(len(combo_set & set(c.numbers)) <= max_overlap for c in selected):
            selected.append(
                Combination(
                    numbers=list(combo),
                    bonus_number=_pick_bonus(strategy, cfg, bonus_freq, rng),
                    score=round(s, 4),
                    strategy=strategy,
                    rationale=_rationale(combo, cfg, breakdown),
                    features=breakdown,
                )
            )
        if len(selected) >= n:
            break

    return selected


# ── Analysis (frequency + overdue + historical profile) ─────────────────────


def analyze(main_draws: Sequence[Sequence[int]], bonus_draws: Sequence[int], cfg: GameConfig, top: int = 10) -> Dict:
    """Hot / cold / overdue numbers plus the historical combination profile.
    Mirrors (and supersedes) the inline analysis the route used to do."""
    stats = number_stats(main_draws, cfg)
    freq = stats["frequency"]
    recency = stats["recency"]
    total = stats["total_draws"]

    def _rows(items):
        return [
            {
                "number": num,
                "frequency": freq[num],
                "percentage": round(freq[num] / total * 100, 2) if total else 0.0,
            }
            for num in items
        ]

    by_freq_desc = sorted(freq, key=lambda x: freq[x], reverse=True)
    by_freq_asc = sorted(freq, key=lambda x: freq[x])
    overdue = sorted(recency, key=lambda x: recency[x], reverse=True)[:top]

    prof = profile_stats(main_draws, cfg)
    profile_block = {key: {"mean": round(m, 2), "std": round(s, 2)} for key, (m, s) in prof.items()}
    # Most common co-occurring pairs.
    pair_counts: Dict[Tuple[int, int], int] = {}
    for draw in main_draws:
        s = sorted(set(draw))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                key = (s[i], s[j])
                pair_counts[key] = pair_counts.get(key, 0) + 1
    top_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    profile_block["top_pairs"] = [{"pair": list(p), "count": c} for p, c in top_pairs]

    return {
        "total_draws_analyzed": total,
        "hot_numbers": _rows(by_freq_desc[:top]),
        "cold_numbers": _rows(by_freq_asc[:top]),
        "overdue_numbers": [
            {"number": num, "draws_since_last": recency[num], "frequency": freq[num]} for num in overdue
        ],
        "frequency_distribution": {str(num): freq[num] for num in sorted(freq)},
        "profile": profile_block,
    }


# ── Settlement (backtesting) ────────────────────────────────────────────────


def prize_tier(matched_main: int, matched_bonus: bool, game: str) -> Optional[str]:
    """Map (main matches, bonus match) to a prize tier label, or None for no
    prize. Tiers are identical for Powerball and Mega Millions."""
    if matched_main == 5 and matched_bonus:
        return "jackpot"
    if matched_main == 5:
        return "match_5"
    if matched_main == 4 and matched_bonus:
        return "match_4_bonus"
    if matched_main == 4:
        return "match_4"
    if matched_main == 3 and matched_bonus:
        return "match_3_bonus"
    if matched_main == 3:
        return "match_3"
    if matched_main == 2 and matched_bonus:
        return "match_2_bonus"
    if matched_main == 1 and matched_bonus:
        return "match_1_bonus"
    if matched_main == 0 and matched_bonus:
        return "match_bonus"
    return None


def settle(numbers: Sequence[int], bonus_number: int, draw_numbers: Sequence[int], draw_bonus: int, game: str) -> Dict:
    """Score one combination against an actual draw."""
    matched_main = len(set(numbers) & set(draw_numbers))
    matched_bonus = bonus_number == draw_bonus
    return {
        "matched_main": matched_main,
        "matched_bonus": matched_bonus,
        "prize_tier": prize_tier(matched_main, matched_bonus, game),
    }
