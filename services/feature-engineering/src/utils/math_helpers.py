"""Mathematical and statistical helper functions for feature computation."""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------- Elo Rating ----------

ELO_K_FACTOR = 32
ELO_BASE_RATING = 1500


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating: float, expected: float, actual: float, k: float = ELO_K_FACTOR) -> float:
    """Update Elo rating after a match result."""
    return rating + k * (actual - expected)


def compute_elo_ratings(
    results: List[Tuple[float, float]],
    home_start: float = ELO_BASE_RATING,
    away_start: float = ELO_BASE_RATING,
    k: float = ELO_K_FACTOR,
) -> Tuple[float, float]:
    """Compute Elo ratings from a sequence of (home_actual, away_actual) results.

    actual: 1.0 = win, 0.5 = draw, 0.0 = loss
    Returns (home_rating, away_rating).
    """
    home_elo = home_start
    away_elo = away_start
    for home_actual, away_actual in results:
        exp_home = elo_expected(home_elo, away_elo)
        exp_away = 1.0 - exp_home
        home_elo = elo_update(home_elo, exp_home, home_actual, k)
        away_elo = elo_update(away_elo, exp_away, away_actual, k)
    return home_elo, away_elo


# ---------- Poisson Distribution ----------


def poisson_probability(lam: float, k: int) -> float:
    """Probability of exactly k events given rate lambda."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam**k * math.exp(-lam)) / math.factorial(k)


def poisson_match_probabilities(
    home_lambda: float, away_lambda: float, max_goals: int = 6
) -> Tuple[float, float, float]:
    """Calculate home win, draw, and away win probabilities from Poisson.

    Returns (p_home_win, p_draw, p_away_win).
    """
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_probability(home_lambda, i) * poisson_probability(away_lambda, j)
            if i > j:
                p_home_win += p
            elif i == j:
                p_draw += p
            else:
                p_away_win += p

    return p_home_win, p_draw, p_away_win


# ---------- Exponential Weighted Average ----------


def exponential_weighted_avg(values: List[float], span: Optional[int] = None, alpha: Optional[float] = None) -> float:
    """Compute exponential weighted average.

    More recent values have higher weight.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    if alpha is None:
        if span is None:
            span = len(values)
        alpha = 2.0 / (span + 1)

    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


# ---------- Linear Trend (Slope) ----------


def linear_trend(values: List[float]) -> float:
    """Calculate linear trend (slope) of a series of values.

    Positive = improving, negative = declining.
    """
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)

    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0

    slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
    return float(slope)


# ---------- Cyclical Encoding ----------


def cyclical_encode(value: float, period: float) -> Tuple[float, float]:
    """Encode a cyclical value (e.g. month, day of week) as sin/cos pair."""
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


# ---------- Implied Probability ----------


def decimal_odds_to_implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def remove_overround(probabilities: List[float]) -> List[float]:
    """Remove bookmaker overround to get fair probabilities."""
    total = sum(probabilities)
    if total == 0:
        return probabilities
    return [p / total for p in probabilities]


# ---------- Statistical Helpers ----------


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division avoiding ZeroDivisionError."""
    if denominator == 0:
        return default
    return numerator / denominator


def z_score(value: float, mean: float, std: float) -> float:
    """Calculate z-score of a value."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def coefficient_of_variation(values: List[float]) -> float:
    """Calculate coefficient of variation (CV = std/mean)."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = np.nanmean(arr)
    if mean == 0:
        return 0.0
    return float(np.nanstd(arr) / abs(mean))


def rolling_std(values: List[float]) -> float:
    """Calculate standard deviation of a series."""
    if len(values) < 2:
        return 0.0
    return float(np.nanstd(values, ddof=1))
