"""Era-aware game rules for Powerball / Mega Millions (pure, no DB).

Single source of truth for game structure: number matrices, ticket price,
prize tables, draw weekdays, and the built-in multiplier — versioned by
effective draw date so (a) historical draw ingestion validates against the
rules of ITS era, not today's, and (b) a future rule change is one new era
entry instead of silently-wrong odds math scattered across the codebase.
The Apr-2025 Mega Millions change (megaball 25 -> 24, $2 -> $5, Megaplier ->
built-in multiplier) went unnoticed here for over a year because the old
values were hardcoded; this module is the fix.

Tier probabilities are DERIVED combinatorially (math.comb), never hardcoded,
and unit-tested against the officially published odds — e.g. Mega Millions
match-5 at exactly 1 in 12,629,232 (= C(70,5) * 24/23) and Powerball
jackpot at 1 in 292,201,338 (= C(69,5) * 26).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import comb
from typing import Dict, List, Optional, Tuple

# Tier labels match lottery_analysis.prize_tier exactly.
TIER_LABELS = (
    "jackpot",
    "match_5",
    "match_4_bonus",
    "match_4",
    "match_3_bonus",
    "match_3",
    "match_2_bonus",
    "match_1_bonus",
    "match_bonus",
)


@dataclass(frozen=True)
class GameEra:
    """One era of a game's rules, effective for draws on/after `start`."""

    start: date
    main_count: int
    main_max: int
    bonus_max: int
    ticket_price: float
    # tier -> fixed base prize in dollars. Empty for validation-only eras
    # (pre-current history where we never compute EV). 'jackpot' is omitted —
    # it is variable and supplied per-draw.
    prizes: Dict[str, float] = field(default_factory=dict)
    # Expected value of the per-ticket multiplier applied to NON-JACKPOT
    # prizes. 1.0 when the game has no built-in multiplier (Powerball's
    # Power Play is a paid add-on we do not model in the base ticket).
    expected_multiplier: float = 1.0
    notes: str = ""


# Powerball: current matrix 5/69 + 1/26 since the 2015-10-07 draw; $2 since
# 2012-01-15. Older eras cover the NY Open Data history (starts 2010-02).
POWERBALL_ERAS: List[GameEra] = [
    GameEra(date(2009, 1, 7), 5, 59, 39, 1.0, notes="5/59 + 1/39"),
    GameEra(date(2012, 1, 15), 5, 59, 35, 2.0, notes="5/59 + 1/35, $2 ticket"),
    GameEra(
        date(2015, 10, 7),
        5,
        69,
        26,
        2.0,
        prizes={
            "match_5": 1_000_000.0,
            "match_4_bonus": 50_000.0,
            "match_4": 100.0,
            "match_3_bonus": 100.0,
            "match_3": 7.0,
            "match_2_bonus": 7.0,
            "match_1_bonus": 4.0,
            "match_bonus": 4.0,
        },
        notes="Current matrix 5/69 + 1/26; draws Mon/Wed/Sat",
    ),
]

# Mega Millions: the Apr-2025 game change (first draw 2025-04-08) cut the
# megaball pool to 24, raised the ticket to $5, and replaced the Megaplier
# add-on with a built-in multiplier drawn from a field of 32:
# fifteen 2x, ten 3x, four 4x, two 5x, one 10x -> E[multiplier] = 96/32 = 3.0
# exactly. It applies to every non-jackpot tier (match_5 caps at $10M = $1M x
# the max 10x). Published amounts are BASE amounts; the $10 advertised
# minimum win is the $5 base tier at the minimum 2x.
MEGA_MILLIONS_ERAS: List[GameEra] = [
    GameEra(date(2002, 5, 17), 5, 52, 52, 1.0, notes="5/52 + 1/52"),
    GameEra(date(2005, 6, 24), 5, 56, 46, 1.0, notes="5/56 + 1/46"),
    GameEra(date(2013, 10, 22), 5, 75, 15, 1.0, notes="5/75 + 1/15"),
    GameEra(date(2017, 10, 31), 5, 70, 25, 2.0, notes="5/70 + 1/25, Megaplier add-on era"),
    GameEra(
        date(2025, 4, 8),
        5,
        70,
        24,
        5.0,
        prizes={
            "match_5": 1_000_000.0,
            "match_4_bonus": 10_000.0,
            "match_4": 500.0,
            "match_3_bonus": 200.0,
            "match_3": 10.0,
            "match_2_bonus": 10.0,
            "match_1_bonus": 7.0,
            "match_bonus": 5.0,
        },
        expected_multiplier=3.0,
        notes="Current matrix 5/70 + 1/24; $5 ticket, built-in 2-10x multiplier; draws Tue/Fri",
    ),
]

GAME_ERAS: Dict[str, List[GameEra]] = {
    "powerball": POWERBALL_ERAS,
    "mega_millions": MEGA_MILLIONS_ERAS,
}

# Draw weekdays under CURRENT rules (Mon=0 .. Sun=6).
DRAW_WEEKDAYS: Dict[str, frozenset] = {
    "powerball": frozenset({0, 2, 5}),  # Mon/Wed/Sat (Monday added Aug 2021)
    "mega_millions": frozenset({1, 4}),  # Tue/Fri
}


def rules_for(game: str, draw_date: date) -> Optional[GameEra]:
    """The era in effect for a draw on `draw_date`; None if the date predates
    every known era (caller should skip-and-log, not guess)."""
    eras = GAME_ERAS.get(game)
    if eras is None:
        raise ValueError(f"Unknown lottery game {game!r}")
    current: Optional[GameEra] = None
    for era in eras:
        if draw_date >= era.start:
            current = era
        else:
            break
    return current


def current_rules(game: str) -> GameEra:
    eras = GAME_ERAS.get(game)
    if eras is None:
        raise ValueError(f"Unknown lottery game {game!r}")
    return eras[-1]


def analysis_window_starts(game: str) -> Tuple[date, date]:
    """(main_start, bonus_start): earliest draw dates whose MAIN matrix /
    BONUS pool match current rules. Frequency/recency/profile analysis must
    not mix matrices — a 'hot' megaball 25 is not even in today's pool. For
    Mega Millions the Apr-2025 change touched only the bonus pool, so mains
    can use the whole 5/70 history (2017+) while the bonus window starts at
    2025-04-08."""
    eras = GAME_ERAS.get(game)
    if eras is None:
        raise ValueError(f"Unknown lottery game {game!r}")
    cur = eras[-1]
    main_start = cur.start
    bonus_start = cur.start
    for era in reversed(eras):
        if era.main_count == cur.main_count and era.main_max == cur.main_max:
            main_start = era.start
        else:
            break
    for era in reversed(eras):
        if era.bonus_max == cur.bonus_max:
            bonus_start = era.start
        else:
            break
    return main_start, bonus_start


def tier_probability(era: GameEra, matched_main: int, matched_bonus: bool) -> float:
    """Exact probability of matching `matched_main` main numbers (and the
    bonus ball iff `matched_bonus`) on one line. Pure hypergeometric —
    reproduces every officially published odds figure."""
    k, n, m = era.main_count, era.main_max, matched_main
    if not 0 <= m <= k:
        raise ValueError(f"matched_main={m} out of range 0..{k}")
    p_main = comb(k, m) * comb(n - k, k - m) / comb(n, k)
    p_bonus = (1.0 / era.bonus_max) if matched_bonus else ((era.bonus_max - 1) / era.bonus_max)
    return p_main * p_bonus


def all_tier_probabilities(era: GameEra) -> Dict[str, float]:
    """Probability of every winning tier label for one line."""
    k = era.main_count
    out: Dict[str, float] = {
        "jackpot": tier_probability(era, k, True),
        f"match_{k}": tier_probability(era, k, False),
    }
    for m in range(1, k):
        out[f"match_{m}_bonus"] = tier_probability(era, m, True)
    out["match_bonus"] = tier_probability(era, 0, True)
    # Bonus-less low tiers only where the era actually pays them (match_4,
    # match_3 for both current games).
    for m in range(1, k):
        label = f"match_{m}"
        if label in era.prizes:
            out[label] = tier_probability(era, m, False)
    return out


def jackpot_odds(era: GameEra) -> float:
    """1-in-N odds of the jackpot (N)."""
    return 1.0 / tier_probability(era, era.main_count, True)
