"""Unit tests for backtest_recommendations — pure pieces.

DB walking + the full run() are covered by integration-style tests
elsewhere (or wouldn't be — running against postgres is e2e); these
lock down the per-rec math, the routing dispatch, and the report
rendering so a refactor of the live rec engine doesn't silently
change backtest outcomes.
"""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load dependencies first so the relative imports in backtest resolve.
_load("grading_outcomes", "grading_outcomes.py")
_load("generate_recommendations", "generate_recommendations.py")
_load("generate_recommendations_nba", "generate_recommendations_nba.py")
bt = _load("backtest_recommendations", "backtest_recommendations.py")


class TestSimulatePnl:
    """Exact P/L math per rec — locks the formula so live + backtest
    can't drift apart."""

    def test_winning_bet_pays_decimal_minus_one_times_stake(self):
        # 100 at 1.95 odds, selection 'home' matched actual 'home'.
        # P/L = 100 × (1.95 - 1) = 95.00.
        status, pnl = bt.simulate_pnl(
            actual="home",
            selection="home",
            stake=Decimal("100"),
            odds=Decimal("1.95"),
        )
        assert status == "won"
        assert pnl == Decimal("95.00")

    def test_losing_bet_returns_negative_stake(self):
        status, pnl = bt.simulate_pnl(
            actual="away",
            selection="home",
            stake=Decimal("100"),
            odds=Decimal("1.95"),
        )
        assert status == "lost"
        assert pnl == Decimal("-100.00")

    def test_push_returns_zero(self):
        # over_under at integer line that hit exact total.
        status, pnl = bt.simulate_pnl(
            actual="push_3",
            selection="over_3",
            stake=Decimal("50"),
            odds=Decimal("2.0"),
        )
        assert status == "void"
        assert pnl == Decimal("0")

    def test_bare_push_string_also_handled(self):
        # draw_no_bet returns bare "push" on a draw.
        status, pnl = bt.simulate_pnl(
            actual="push",
            selection="home",
            stake=Decimal("50"),
            odds=Decimal("2.0"),
        )
        assert status == "void"
        assert pnl == Decimal("0")

    def test_ungradable_returns_skip(self):
        # actual=None means the grader couldn't determine outcome
        # (e.g., NBA spread with missing closing line). Backtest must
        # SKIP — counting it as a loss would inflate the lose
        # column with rows where we'd never have placed in real life.
        status, pnl = bt.simulate_pnl(
            actual=None,
            selection="home",
            stake=Decimal("100"),
            odds=Decimal("1.95"),
        )
        assert status == "skip"
        assert pnl == Decimal("0")

    def test_pnl_quantizes_to_cents(self):
        # Stake × (odds - 1) at awkward odds (1.91) gives 0.91 × stake
        # which can drift past 2 decimal places if not quantized.
        status, pnl = bt.simulate_pnl(
            actual="home",
            selection="home",
            stake=Decimal("133.33"),
            odds=Decimal("1.91"),
        )
        # 133.33 × 0.91 = 121.3303 → 121.33
        assert status == "won"
        assert pnl == Decimal("121.33")


class TestApplyProbCap:
    """The NBA prob cap is part of the live rec engine — the backtest
    must apply the same cap when --nba-prob-cap is provided."""

    def test_nba_caps_high_prob(self):
        # Matches the 0.87 → 0.80 case observed on the Finals matchup.
        assert bt.apply_prob_cap("nba", raw_prob=0.87, nba_cap=0.80) == 0.80

    def test_nba_under_cap_unchanged(self):
        assert bt.apply_prob_cap("nba", raw_prob=0.65, nba_cap=0.80) == 0.65

    def test_soccer_unaffected_by_cap(self):
        # Soccer has no live cap; backtest must not apply one either,
        # else soccer ROI would drift vs prod.
        assert bt.apply_prob_cap("soccer", raw_prob=0.87, nba_cap=0.80) == 0.87

    def test_nhl_unaffected_by_cap(self):
        assert bt.apply_prob_cap("nhl", raw_prob=0.87, nba_cap=0.80) == 0.87


class TestMarketRouting:
    """_market_for_prediction translates (model_name, prediction_type)
    into the (sport, market, odds_market_type) routing used to fetch
    odds + look up labels. Wrong routing = backtest reads wrong odds
    rows = bogus ROI."""

    def test_nhl_moneyline_routes_to_moneyline_odds(self):
        result = bt._market_for_prediction("ensemble_nhl_ml", "moneyline")
        assert result == ("nhl", "moneyline", "moneyline")

    def test_nhl_puck_line_routes_to_spread_odds(self):
        # NHL puck line stores prediction_type='spread' in DB.
        result = bt._market_for_prediction("ensemble_nhl_pl", "spread")
        assert result == ("nhl", "puck_line", "spread")

    def test_nhl_total_routes_to_total_odds(self):
        result = bt._market_for_prediction("ensemble_nhl_tot", "total")
        assert result == ("nhl", "total", "total")

    def test_nhl_regulation_skipped(self):
        # Regulation isn't a bookable market — no live odds for it,
        # so backtesting it would just produce empty results.
        assert bt._market_for_prediction("ensemble_nhl_reg", "match_result") is None

    def test_nba_moneyline_routes(self):
        assert bt._market_for_prediction("ensemble_nba_ml", "moneyline") == (
            "nba",
            "moneyline",
            "moneyline",
        )

    def test_nba_spread_routes(self):
        assert bt._market_for_prediction("ensemble_nba_sp", "spread") == (
            "nba",
            "spread",
            "spread",
        )

    def test_nba_total_routes(self):
        assert bt._market_for_prediction("ensemble_nba_tot", "total") == (
            "nba",
            "total",
            "total",
        )

    def test_soccer_match_result_routes(self):
        # Soccer's headline market — uses model_name='ensemble' (the
        # legacy DB identifier) with prediction_type='match_result'
        # and odds market_type='1x2'.
        assert bt._market_for_prediction("ensemble", "match_result") == (
            "soccer",
            "match_result",
            "1x2",
        )

    def test_soccer_derived_markets_skipped_in_v1(self):
        # Derived soccer markets (over_under_2.5, asian_handicap, etc.)
        # are intentionally not backtested in v1 — their key-encoded
        # selection strings would complicate the odds-table join.
        # Production accuracy widget covers them.
        assert bt._market_for_prediction("ensemble", "over_under") is None
        assert bt._market_for_prediction("ensemble", "asian_handicap") is None

    def test_unknown_model_returns_none(self):
        # Defensive: an unknown ensemble (future sport added without
        # a routing entry) should be SKIPPED, not crash the backtest.
        assert bt._market_for_prediction("ensemble_nfl_ml", "moneyline") is None


class TestMarketAggregate:
    """Aggregator tally + derived ratios."""

    def test_hit_rate_excludes_voids(self):
        # Pushes don't count as either a win or a loss in hit-rate
        # math — same convention the live accuracy widget uses.
        agg = bt.MarketAggregate(sport="nba", market="spread")
        agg.won = 6
        agg.lost = 4
        agg.void = 2
        assert agg.hit_rate == 0.6  # 6 / (6+4), void excluded

    def test_hit_rate_zero_when_no_decided(self):
        # All voids → hit rate undefined; report 0.0 instead of NaN.
        agg = bt.MarketAggregate(sport="nba", market="spread")
        agg.void = 5
        assert agg.hit_rate == 0.0

    def test_roi_pct_calc(self):
        agg = bt.MarketAggregate(sport="nba", market="spread")
        agg.total_staked = Decimal("1000")
        agg.total_pnl = Decimal("85.50")
        assert agg.roi_pct == 8.55

    def test_roi_pct_zero_when_no_stakes(self):
        agg = bt.MarketAggregate(sport="nba", market="spread")
        assert agg.roi_pct == 0.0


class TestBacktestResultSerialize:
    def test_to_dict_round_trip(self):
        agg = bt.MarketAggregate(sport="nba", market="spread")
        agg.recs = 10
        agg.won = 6
        agg.lost = 4
        agg.total_staked = Decimal("500.00")
        agg.total_pnl = Decimal("47.50")

        result = bt.BacktestResult(
            start="2024-10-01",
            end="2025-06-30",
            params={
                "ev_threshold": 0.03,
                "prob_floor": 0.40,
                "kelly_fraction": 0.25,
                "nba_prob_cap": 0.80,
                "bankroll": 1000.0,
            },
            overall=agg,
            by_market=[agg],
        )

        d = result.to_dict()
        assert d["start"] == "2024-10-01"
        assert d["overall"]["recs"] == 10
        assert d["overall"]["total_pnl"] == 47.5
        assert d["overall"]["hit_rate"] == 0.6
        assert d["overall"]["roi_pct"] == 9.5  # 47.5 / 500 × 100
        assert len(d["by_market"]) == 1


class TestRenderMarkdown:
    def test_renders_per_market_rows(self):
        agg = bt.MarketAggregate(sport="nba", market="spread")
        agg.recs = 10
        agg.won = 6
        agg.lost = 4
        agg.total_staked = Decimal("500.00")
        agg.total_pnl = Decimal("47.50")

        result = bt.BacktestResult(
            start="2024-10-01",
            end="2025-06-30",
            params={
                "ev_threshold": 0.03,
                "prob_floor": 0.40,
                "kelly_fraction": 0.25,
                "nba_prob_cap": 0.80,
                "bankroll": 1000.0,
            },
            overall=agg,
            by_market=[agg],
        )

        md = bt.render_markdown(result)
        # Title with date window.
        assert "2024-10-01" in md and "2025-06-30" in md
        # Overall line with hit rate + ROI. f-string `${val:+,.2f}`
        # produces `$+47.50` (dollar sign first, then signed number),
        # not `+$47.50`.
        assert "60.0% hit" in md
        assert "$+47.50" in md
        # Per-market table header.
        assert "| Sport | Market |" in md
        # Per-market row with content (nba | spread | 10 | 6 | 4 ...).
        assert "nba" in md
        assert "spread" in md


class TestCliDefaults:
    """Default thresholds must match the live rec engine. If someone
    bumps the live KELLY_FRACTION but forgets to update the backtest
    default, runs will silently report results that don't reflect
    live behavior."""

    def test_ev_threshold_default(self):
        args = bt.parse_args(["--start", "2024-10-01", "--end", "2025-06-30"])
        # Matches generate_recommendations_nba.py CLI default.
        assert args.ev_threshold == 0.03

    def test_prob_floor_default(self):
        args = bt.parse_args(["--start", "2024-10-01", "--end", "2025-06-30"])
        assert args.prob_floor == 0.40

    def test_kelly_fraction_default(self):
        args = bt.parse_args(["--start", "2024-10-01", "--end", "2025-06-30"])
        # Quarter Kelly — matches generate_recommendations_nba.KELLY_FRACTION.
        assert args.kelly_fraction == 0.25

    def test_nba_cap_default(self):
        args = bt.parse_args(["--start", "2024-10-01", "--end", "2025-06-30"])
        # Matches generate_recommendations_nba.PROB_CAP_FOR_EV.
        assert args.nba_prob_cap == 0.80

    def test_bankroll_default(self):
        args = bt.parse_args(["--start", "2024-10-01", "--end", "2025-06-30"])
        assert args.bankroll == 1000.0
