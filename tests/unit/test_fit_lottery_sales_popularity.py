"""Unit tests for the sales/popularity fit harness — dataset construction,
sales inference math, and bias recovery on synthetic data. No DB."""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fit = _load("fit_lottery_sales_popularity", REPO / "scripts" / "fit_lottery_sales_popularity.py")
lottery_rules = sys.modules["lottery_rules"]  # loaded by the harness itself


class TestSalesInference:
    def test_tickets_from_match_bonus_tier_current_era(self):
        # Current era (megaball 1/24): P(0+MB) = C(65,5)/C(70,5) / 24.
        probs = fit.tier_probs_for(date(2026, 8, 1))
        p = probs["match_bonus"]
        assert p == (8_259_888 / 12_103_014) / 24
        rows = [
            {
                "draw_date": date(2026, 8, 1),
                "numbers": [40, 45, 50, 55, 60],
                "jackpot_amount": 300e6,
                "winners_by_tier": {t: 0 for t in fit.MAINS_TIERS} | {"match_bonus": 400_000, "match_3": 10_000},
            }
        ]
        data = fit.build_dataset(rows)
        assert len(data) == 1
        assert data[0]["tickets"] == 400_000 / p

    def test_pre_2017_draws_excluded(self):
        rows = [
            {
                "draw_date": date(2015, 1, 2),
                "numbers": [1, 2, 3, 4, 5],
                "jackpot_amount": 100e6,
                "winners_by_tier": {"match_bonus": 400_000, "match_3": 5_000},
            }
        ]
        assert fit.build_dataset(rows) == []

    def test_thin_rows_excluded(self):
        rows = [
            {
                "draw_date": date(2026, 8, 1),
                "numbers": [1, 2, 3, 4, 5],
                "jackpot_amount": 100e6,
                "winners_by_tier": {"match_bonus": 12, "match_3": 3},  # corrupt/partial
            }
        ]
        assert fit.build_dataset(rows) == []


class TestLineFeatures:
    def test_birthday_line_vs_high_line(self):
        birthday = fit.line_features([3, 7, 12, 21, 30])
        high = fit.line_features([37, 44, 52, 61, 68])
        assert birthday[0] == 1.0 and high[0] == 0.0  # frac <= 31
        assert birthday[4] == 1.0 and high[4] == 0.0  # has 7


class TestBiasRecovery:
    def test_ols_recovers_injected_birthday_effect(self):
        """Synthetic draws where log-excess is driven by frac<=31 with known
        coefficient 0.5: the fit must recover sign and rough magnitude."""
        rng = random.Random(42)
        era = lottery_rules.current_rules("mega_millions")
        probs = lottery_rules.all_tier_probabilities(era)
        p_ticket = probs["match_bonus"]
        p_mains = sum(probs[t] for t in fit.MAINS_TIERS)
        rows = []
        d0 = date(2025, 4, 8)
        for i in range(200):
            numbers = sorted(rng.sample(range(1, 71), 5))
            frac31 = sum(1 for x in numbers if x <= 31) / 5
            tickets = 12e6
            noise = rng.gauss(0, 0.02)
            excess = math.exp(0.5 * frac31 + noise)
            wbt = {t: 0 for t in fit.TICKET_TIER}
            wbt = {"match_bonus": int(tickets * p_ticket)}
            mains_total = tickets * p_mains * excess
            # Spread across mains tiers proportionally to their probabilities.
            for t in fit.MAINS_TIERS:
                wbt[t] = int(mains_total * probs[t] / p_mains)
            rows.append(
                {
                    "draw_date": d0 + timedelta(days=3 * i),
                    "numbers": numbers,
                    "jackpot_amount": 100e6,
                    "winners_by_tier": wbt,
                }
            )
        data = fit.build_dataset(rows)
        assert len(data) == 200
        y = np.array([d["log_excess"] for d in data])
        X = np.array([d["features"] for d in data])
        res = fit.ols(y, X)
        coef_frac31 = res["beta"][1]  # after intercept
        se_frac31 = res["se"][1]
        assert 0.3 < coef_frac31 < 0.7
        assert coef_frac31 / se_frac31 > 3  # decisively positive
        assert res["r2"] > 0.5
