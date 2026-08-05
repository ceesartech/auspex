"""Per-ticket expected value for Powerball / Mega Millions (pure, no DB).

This is the only decision-relevant output a lottery feature can honestly
produce. Draws are uniform — nothing predicts numbers — but the EV of a
ticket is a function of OBSERVABLE state that genuinely varies 3-4x between
draws: advertised jackpot, its cash value, taxes, and expected co-winner
sharing (which depends on how many tickets the jackpot size attracts). The
verdict is almost always "don't play" — that is what makes it honest. Same
EV/no-bet discipline as the sports side, applied to a game whose house take
is ~50%+.

Model, per $1-of-face-value ticket line:
  ex-jackpot EV = E[multiplier] * sum(base_prize[t] * P[t]) * (1 - tax)
  jackpot EV    = cash_value * (1 - tax) * P(jackpot) * E[1/(1+K)]
where K ~ Poisson(lambda), lambda = other_tickets * P(jackpot), and
  E[1/(1+K)] = (1 - exp(-lambda)) / lambda   (the expected share factor).
All prize income is taxed at (federal + state) marginal rate — winnings are
ordinary income; applying the full rate to small tiers slightly overstates
tax for low earners, which is the conservative direction.

Ticket sales are NOT published pre-draw; `estimated_tickets` interpolates a
documented anchor curve of historical per-draw sales vs advertised jackpot
(rough figures from public per-draw sales trackers, e.g. lottoreport.com).
Pass `tickets_sold` explicitly to override. Once draw ingestion accumulates
winners_by_tier, the curve can be refit empirically: tickets ~= winners in
the 0+bonus tier x its 1-in-N odds (38.32 for PB, 35.3 for MM).
"""

from __future__ import annotations

from math import exp
from typing import Dict, List, Optional, Tuple

# Import shim: package import in the app, sibling file-path fallback for
# unit tests loading this module directly (see lottery_analysis.py).
try:
    from services.lottery_rules import GameEra, all_tier_probabilities, current_rules, jackpot_odds
except ImportError:  # pragma: no cover — file-path loaders
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path

    _rules = _sys.modules.get("lottery_rules")
    if _rules is None:
        _spec = _ilu.spec_from_file_location("lottery_rules", _Path(__file__).with_name("lottery_rules.py"))
        assert _spec and _spec.loader
        _rules = _ilu.module_from_spec(_spec)
        _sys.modules["lottery_rules"] = _rules
        _spec.loader.exec_module(_rules)
    GameEra = _rules.GameEra  # type: ignore[assignment, no-redef, misc]
    all_tier_probabilities = _rules.all_tier_probabilities  # type: ignore[assignment, no-redef]
    current_rules = _rules.current_rules  # type: ignore[assignment, no-redef]
    jackpot_odds = _rules.jackpot_odds  # type: ignore[assignment, no-redef]

# (advertised_jackpot_$M, tickets_per_draw) anchors, linearly interpolated.
#
# mega_millions: FITTED 2026-08-05 from 914 draws of real winners data
# (scripts/fit_lottery_sales_popularity.py; tickets inferred from the 0+MB
# tier via exact era odds — per-bin median jackpot vs median tickets). The
# hand-set priors were ~45% HIGH in the $500M-$1B band and ~26% low in the
# $1B+ frenzy band (n=12 there — the steep last segment is real: media
# coverage makes sales superlinear near records).
#
# powerball: UNFITTED hand estimates — powerball.com's CDN blocks
# non-browser clients, so no winners feed exists to fit against. The EV
# verdict is robust to this (sharing only matters at extreme jackpots;
# at a typical lambda~0.1 the share factor is ~0.95).
SALES_ANCHORS: Dict[str, List[Tuple[float, float]]] = {
    "powerball": [
        (20, 9e6),
        (100, 12e6),
        (300, 20e6),
        (500, 40e6),
        (800, 70e6),
        (1000, 110e6),
        (1500, 200e6),
        (2000, 300e6),
    ],
    "mega_millions": [
        (45, 9.44e6),
        (106, 10.98e6),
        (209, 13.89e6),
        (370, 18.92e6),
        (640, 28.84e6),
        (1175, 155.60e6),
    ],
}

# Advertised -> cash-value ratio fallback when the live cash value is not
# supplied. Recent draws run ~0.42-0.48 (rate-dependent); pass cash_value
# from the live feed whenever available instead of relying on this.
DEFAULT_CASH_RATIO = 0.45

DEFAULT_FEDERAL_TAX = 0.37


def estimated_tickets(game: str, advertised_jackpot: float) -> float:
    """Interpolated per-draw ticket sales for an advertised jackpot ($)."""
    anchors = SALES_ANCHORS[game]
    jm = advertised_jackpot / 1e6
    if jm <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if jm <= x1:
            return y0 + (y1 - y0) * (jm - x0) / (x1 - x0)
    # Extrapolate the last segment's slope beyond the final anchor.
    (x0, y0), (x1, y1) = anchors[-2], anchors[-1]
    return y1 + (y1 - y0) * (jm - x1) / (x1 - x0)


def share_factor(lam: float) -> float:
    """E[1/(1+K)] for K ~ Poisson(lam): the fraction of the jackpot you
    expect to keep given co-winners. 1.0 as lam -> 0."""
    if lam <= 0:
        return 1.0
    return (1.0 - exp(-lam)) / lam


def ev_report(
    game: str,
    advertised_jackpot: float,
    *,
    cash_value: Optional[float] = None,
    federal_tax: float = DEFAULT_FEDERAL_TAX,
    state_tax: float = 0.0,
    tickets_sold: Optional[float] = None,
    era: Optional[GameEra] = None,
) -> Dict:
    """Full EV breakdown for one ticket of `game` at the given jackpot.

    All dollar inputs/outputs are absolute dollars. Returns a dict ready for
    the API response: components, totals, breakeven, and a plain-language
    verdict."""
    if advertised_jackpot <= 0:
        raise ValueError("advertised_jackpot must be positive")
    era = era or current_rules(game)
    if not era.prizes:
        raise ValueError(f"No prize table for {game} era starting {era.start}")

    tax = federal_tax + state_tax
    if not 0 <= tax < 1:
        raise ValueError(f"Combined tax rate {tax} out of range [0, 1)")
    keep = 1.0 - tax

    cash = cash_value if cash_value is not None else advertised_jackpot * DEFAULT_CASH_RATIO
    tickets = tickets_sold if tickets_sold is not None else estimated_tickets(game, advertised_jackpot)

    probs = all_tier_probabilities(era)
    p_jackpot = probs["jackpot"]

    # Non-jackpot tiers: base amounts x expected multiplier, post-tax.
    ex_jackpot_ev = era.expected_multiplier * keep * sum(amount * probs[tier] for tier, amount in era.prizes.items())

    lam = tickets * p_jackpot
    share = share_factor(lam)
    jackpot_ev = cash * keep * p_jackpot * share

    total_ev = ex_jackpot_ev + jackpot_ev
    price = era.ticket_price

    breakeven = _breakeven_advertised(game, era, keep, cash / advertised_jackpot)

    if total_ev >= price:
        verdict = (
            "Positive EV at this jackpot — historically unprecedented; " "double-check the inputs before believing it."
        )
    else:
        loss_pct = (1.0 - total_ev / price) * 100
        verdict = (
            f"Don't play on EV grounds: a ${price:.0f} ticket returns ${total_ev:.2f} "
            f"expected ({loss_pct:.0f}% expected loss). "
            + (
                f"Breakeven needs a ~${breakeven / 1e9:.1f}B advertised jackpot, " "which has never happened."
                if breakeven is not None
                else "No advertised jackpot reaches breakeven under these assumptions "
                "(co-winner sharing caps the jackpot term)."
            )
        )

    return {
        "game": game,
        "ticket_price": price,
        "advertised_jackpot": advertised_jackpot,
        "cash_value": cash,
        "cash_ratio": cash / advertised_jackpot,
        "tax_rate": tax,
        "tickets_estimated": tickets,
        "tickets_source": "provided" if tickets_sold is not None else "sales-curve estimate",
        "expected_co_winners": lam,
        "share_factor": share,
        "jackpot_odds": jackpot_odds(era),
        "ev_ex_jackpot": ex_jackpot_ev,
        "ev_jackpot_term": jackpot_ev,
        "ev_total": total_ev,
        "ev_per_dollar": total_ev / price,
        "expected_loss_pct": max(0.0, (1.0 - total_ev / price) * 100),
        "breakeven_advertised_jackpot": breakeven,
        "expected_multiplier": era.expected_multiplier,
        "verdict": verdict,
    }


def _breakeven_advertised(game: str, era: GameEra, keep: float, cash_ratio: float) -> Optional[float]:
    """Smallest advertised jackpot where total EV >= ticket price, holding
    tax + cash ratio fixed and letting sales (and therefore sharing) respond
    to the jackpot via the sales curve. Returns None if sharing feedback
    prevents breakeven below $50B (effectively: never)."""
    probs = all_tier_probabilities(era)
    p_jackpot = probs["jackpot"]
    ex_jackpot_ev = era.expected_multiplier * keep * sum(amount * probs[tier] for tier, amount in era.prizes.items())
    lo, hi = 1e6, 50e9

    def ev_at(advertised: float) -> float:
        lam = estimated_tickets(game, advertised) * p_jackpot
        return ex_jackpot_ev + advertised * cash_ratio * keep * p_jackpot * share_factor(lam)

    if ev_at(hi) < era.ticket_price:
        return None
    # EV is monotonically increasing in the advertised jackpot (sharing grows
    # sub-linearly), so bisection is safe.
    for _ in range(200):
        mid = (lo + hi) / 2
        if ev_at(mid) >= era.ticket_price:
            hi = mid
        else:
            lo = mid
    return hi
