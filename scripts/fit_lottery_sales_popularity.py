"""Fit ticket sales + number-popularity models from Mega Millions winners data
(lottery v1.1 analysis harness — offline, run manually like the ab_* scripts).

Inputs: lottery_draws rows with winners_by_tier populated by
scripts/fetch_lottery_winners.py.

Three outputs, all printed as a report (this script never changes serving
code — update constants in lottery_ev.py / lottery_analysis.py by hand iff
the evidence is decisive):

1. TICKETS SOLD per draw, inferred as
       tickets ~= winners(0+MB tier) / P(0+MB | era)
   P from the era-aware rules registry (megaball 1/24 vs 1/25 matters).

2. SALES CURVE: median inferred tickets per advertised-jackpot bin, printed
   next to lottery_ev.SALES_ANCHORS so a divergence is obvious.

3. POPULARITY REGRESSION: for each draw, the excess-winners ratio at the
   mains-dependent tiers (4+MB, 4, 3+MB, 3, 2+MB, 1+MB):
       y = log( observed_winners / expected_winners )
   regressed (OLS) on human-bias features of the WINNING mains (birthday
   fractions, sequences, lucky-7, round numbers, decade clustering). Draws
   whose winning numbers look like human picks show excess winners at every
   tier — the size of that effect is the empirical grounding for
   lottery_analysis.popularity_score's hand-calibrated weights.

   Known caveat (stated in the report): tickets are inferred from the
   0-mains-match tier, which is itself depressed when popular mains are
   drawn, so measured excess is somewhat AMPLIFIED. Signs and feature
   ranking are robust; absolute magnitudes are upper-ish bounds.

Usage (inside the api container):
    python /app/scripts/fit_lottery_sales_popularity.py
    python /app/scripts/fit_lottery_sales_popularity.py --min-draws 100
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import sys
from datetime import date
from typing import Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fit_lottery_sales_popularity")


def _load_module(name: str, *path_candidates: str):
    if name in sys.modules:
        return sys.modules[name]
    for candidate in path_candidates:
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location(name, candidate)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"{name}.py not found")


_HERE = os.path.dirname(os.path.abspath(__file__))
_SVC = os.path.join(_HERE, "..", "services", "api", "src", "services")
lottery_rules = _load_module(
    "lottery_rules",
    "/app/services/api/src/services/lottery_rules.py",
    os.path.join(_SVC, "lottery_rules.py"),
)
lottery_ev = _load_module(
    "lottery_ev",
    "/app/services/api/src/services/lottery_ev.py",
    os.path.join(_SVC, "lottery_ev.py"),
)

MAINS_TIERS = ("match_4_bonus", "match_4", "match_3_bonus", "match_3", "match_2_bonus", "match_1_bonus")
TICKET_TIER = "match_bonus"  # 0 mains + megaball: the sales thermometer

FEATURE_NAMES = (
    "frac_le_31",
    "frac_le_12",
    "consec_frac",
    "frac_mult5",
    "has_7",
    "decades_le_2",
)


def line_features(numbers: list[int]) -> list[float]:
    """Human-bias features of a winning mains line (mirrors the components of
    lottery_analysis.popularity_score, kept standalone for the regression)."""
    nums = sorted(numbers)
    n = len(nums)
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    return [
        sum(1 for x in nums if x <= 31) / n,
        sum(1 for x in nums if x <= 12) / n,
        sum(1 for d in diffs if d == 1) / max(1, n - 1),
        sum(1 for x in nums if x % 5 == 0) / n,
        1.0 if 7 in nums else 0.0,
        1.0 if len({(x - 1) // 10 for x in nums}) <= 2 else 0.0,
    ]


def tier_probs_for(draw_date: date) -> Optional[dict]:
    """Complete tier probabilities for the era of `draw_date`, or None for
    eras without a prize table (pre-2017 — no unambiguous tier semantics,
    and no winners data there anyway)."""
    era = lottery_rules.rules_for("mega_millions", draw_date)
    if era is None or not era.prizes:
        return None
    return lottery_rules.all_tier_probabilities(era)


def load_draws(cur) -> list[dict]:
    cur.execute(
        """
        SELECT draw_date, numbers, jackpot_amount, winners_by_tier
        FROM lottery_draws
        WHERE game = 'mega_millions'
          AND winners_by_tier IS NOT NULL
          AND winners_by_tier <> '{}'::jsonb
        ORDER BY draw_date
        """
    )
    return [dict(r) for r in cur.fetchall()]


def build_dataset(rows: list[dict]) -> list[dict]:
    """Per-draw: inferred tickets, mains-tier excess, features."""
    out = []
    for r in rows:
        wbt = r["winners_by_tier"]
        probs = tier_probs_for(r["draw_date"])
        if probs is None:
            continue
        base_winners = wbt.get(TICKET_TIER) or 0
        if base_winners < 1000:
            # Too few 0+MB winners to anchor a sales estimate (partial or
            # corrupt row — a real draw has tens of thousands).
            continue
        tickets = base_winners / probs[TICKET_TIER]
        obs = sum(wbt.get(t) or 0 for t in MAINS_TIERS)
        exp = tickets * sum(probs[t] for t in MAINS_TIERS)
        if obs <= 0 or exp <= 0:
            continue
        out.append(
            {
                "draw_date": r["draw_date"],
                "jackpot": float(r["jackpot_amount"]) if r["jackpot_amount"] else None,
                "tickets": tickets,
                "log_excess": math.log(obs / exp),
                "features": line_features(list(r["numbers"])),
            }
        )
    return out


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Plain OLS with intercept; classical SEs."""
    n, k = X.shape
    Xi = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    resid = y - Xi @ beta
    dof = n - (k + 1)
    sigma2 = float(resid @ resid) / max(1, dof)
    cov = sigma2 * np.linalg.inv(Xi.T @ Xi)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": beta, "se": se, "r2": r2, "n": n}


def report_sales_curve(data: list[dict]) -> None:
    priced = [d for d in data if d["jackpot"]]
    print(f"\n== SALES CURVE (n={len(priced)} draws with jackpot amounts) ==")
    if not priced:
        print("no jackpot amounts available")
        return
    bins = [(0, 75e6), (75e6, 150e6), (150e6, 300e6), (300e6, 500e6), (500e6, 1e9), (1e9, 5e9)]
    print(f"{'jackpot bin':>22} | {'n':>4} | {'median tickets':>14} | {'current anchor est.':>19}")
    for lo, hi in bins:
        rows = [d for d in priced if lo <= d["jackpot"] < hi]
        if not rows:
            continue
        med_tickets = float(np.median([d["tickets"] for d in rows]))
        med_jackpot = float(np.median([d["jackpot"] for d in rows]))
        anchor = lottery_ev.estimated_tickets("mega_millions", med_jackpot)
        print(f"${lo / 1e6:>7.0f}M-${hi / 1e6:>5.0f}M | {len(rows):>4} | {med_tickets:>14,.0f} | {anchor:>19,.0f}")
    print("(update lottery_ev.SALES_ANCHORS by hand iff these diverge materially)")


def report_popularity_fit(data: list[dict]) -> None:
    y = np.array([d["log_excess"] for d in data])
    X = np.array([d["features"] for d in data])
    fit = ols(y, X)
    print(f"\n== POPULARITY REGRESSION (n={fit['n']}, R^2={fit['r2']:.3f}) ==")
    print("y = log(observed / expected winners) at mains-dependent tiers")
    print(f"{'term':>14} | {'coef':>8} | {'se':>7} | {'t':>6}")
    names = ("intercept",) + FEATURE_NAMES
    for name, b, s in zip(names, fit["beta"], fit["se"]):
        t = b / s if s > 0 else float("nan")
        print(f"{name:>14} | {b:>8.4f} | {s:>7.4f} | {t:>6.2f}")
    # Headline translation: birthday-heavy vs unpopular line.
    b = dict(zip(names, fit["beta"]))
    heavy = b["frac_le_31"] * 1.0 + b["frac_le_12"] * 0.6 + b.get("has_7", 0.0)
    light = b["frac_le_31"] * 0.0
    print(
        f"\nImplied excess-share multiplier, all-birthday line vs all->31 line: "
        f"e^({heavy:.3f} - {light:.3f}) = {math.exp(heavy - light):.2f}x"
    )
    print(
        "Caveat: tickets are inferred from the 0+MB tier, which popular drawn\n"
        "mains depress — treat magnitudes as upper-ish bounds; signs/ranking are robust."
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-draws", type=int, default=50, help="Refuse to fit below this many usable draws.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            rows = load_draws(cur)
    data = build_dataset(rows)
    print(f"Usable draws: {len(data)} (of {len(rows)} with winners data)")
    if len(data) < args.min_draws:
        logger.error("Only %d usable draws (< %d) — accumulate more winners data first", len(data), args.min_draws)
        return 2

    tickets = np.array([d["tickets"] for d in data])
    print(
        f"Inferred tickets/draw: median {np.median(tickets):,.0f}, "
        f"p10 {np.percentile(tickets, 10):,.0f}, p90 {np.percentile(tickets, 90):,.0f}"
    )

    report_sales_curve(data)
    report_popularity_fit(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
