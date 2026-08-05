"""Per-ticket EV math. Fixtures are hand-derived from the published prize
tables, NOT recomputed with the module's own formulas — a tautological test
would bless a broken formula."""

import importlib.util
import sys
from math import exp
from pathlib import Path

import pytest

_SVC = Path(__file__).parent.parent.parent / "src" / "services"
for _name in ("lottery_rules", "lottery_ev"):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, _SVC / f"{_name}.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)

lottery_ev = sys.modules["lottery_ev"]
estimated_tickets = lottery_ev.estimated_tickets
ev_report = lottery_ev.ev_report
share_factor = lottery_ev.share_factor

# Powerball ex-jackpot base EV per $2 ticket, summed by hand from the
# published prize chart (amount / published 1-in-N odds):
#   1,000,000/11,688,053.52 + 50,000/913,129.18 + 100/36,525.17
# + 100/14,494.11 + 7/579.76 + 7/701.33 + 4/91.98 + 4/38.32  = 0.31988
PB_EX_JACKPOT_PRETAX = 0.31988


class TestShareFactor:
    def test_limits(self):
        assert share_factor(0.0) == 1.0
        assert share_factor(1e-9) == pytest.approx(1.0, abs=1e-6)

    def test_known_value(self):
        # lam = 1 -> (1 - e^-1) / 1
        assert share_factor(1.0) == pytest.approx(1.0 - exp(-1.0), rel=1e-12)

    def test_monotone_decreasing(self):
        vals = [share_factor(x) for x in (0.1, 0.5, 1.0, 2.0, 5.0)]
        assert vals == sorted(vals, reverse=True)


class TestPowerballEV:
    def test_ex_jackpot_matches_hand_derivation(self):
        r = ev_report("powerball", 100e6, cash_value=45e6, federal_tax=0.0, tickets_sold=10e6)
        assert r["ev_ex_jackpot"] == pytest.approx(PB_EX_JACKPOT_PRETAX, rel=1e-3)

    def test_taxes_scale_every_term(self):
        r = ev_report("powerball", 100e6, cash_value=45e6, federal_tax=0.37, tickets_sold=10e6)
        assert r["ev_ex_jackpot"] == pytest.approx(PB_EX_JACKPOT_PRETAX * 0.63, rel=1e-3)

    def test_record_jackpot_case(self):
        # The Nov-2022-scale case: $2B advertised, $1.0B cash, 37% federal,
        # 300M tickets. lam = 300e6/292,201,338 = 1.026689,
        # share = (1-e^-lam)/lam = 0.625125,
        # jackpot term = 1e9 * 0.63 * (1/292,201,338) * 0.625125 = 1.34780.
        r = ev_report("powerball", 2e9, cash_value=1e9, federal_tax=0.37, tickets_sold=300e6)
        assert r["expected_co_winners"] == pytest.approx(1.026689, rel=1e-4)
        assert r["share_factor"] == pytest.approx(0.625125, rel=1e-4)
        assert r["ev_jackpot_term"] == pytest.approx(1.34780, rel=1e-3)
        assert r["ev_total"] == pytest.approx(1.34780 + PB_EX_JACKPOT_PRETAX * 0.63, rel=1e-3)
        # Still -EV on a $2 ticket even at the record jackpot.
        assert r["ev_total"] < r["ticket_price"]
        assert "Don't play" in r["verdict"]

    def test_positive_ev_possible_with_absurd_inputs(self):
        # Sanity that the branch exists: tax-free, tiny sales, huge cash.
        r = ev_report("powerball", 5e9, cash_value=3e9, federal_tax=0.0, tickets_sold=1e6)
        assert r["ev_total"] > r["ticket_price"]
        assert "double-check" in r["verdict"]


class TestMegaMillionsEV:
    def test_multiplier_applies_to_ex_jackpot_only(self):
        r = ev_report("mega_millions", 300e6, cash_value=130e6, federal_tax=0.0, tickets_sold=16e6)
        assert r["expected_multiplier"] == 3.0
        # Base ex-jackpot sum, hand-derived tier-by-tier from exact
        # hypergeometric probabilities (C(70,5)=12,103,014, megaball 1/24):
        #   match_5        1e6 * 1/12,629,232          = 0.0791821
        #   match_4_bonus  1e4 * 325/290,472,336       = 0.0111886
        #   match_4        500 * 325*23/290,472,336    = 0.0128670
        #   match_3_bonus  200 * 20,800/290,472,336    = 0.0143215
        #   match_3         10 * 20,800*23/290,472,336 = 0.0164697
        #   match_2_bonus   10 * 436,800/290,472,336   = 0.0150376
        #   match_1_bonus    7 * 3,385,200/290,472,336 = 0.0815788
        #   match_bonus      5 * 8,259,888/290,472,336 = 0.1421801
        #   sum = 0.3728254 -> x3 multiplier = 1.1184762
        assert r["ev_ex_jackpot"] == pytest.approx(1.1184762, rel=1e-4)

    def test_never_positive_at_plausible_jackpots(self):
        for advertised, cash in ((100e6, 45e6), (500e6, 230e6), (1e9, 460e6), (2e9, 900e6)):
            r = ev_report("mega_millions", advertised, cash_value=cash, federal_tax=0.37)
            assert r["ev_total"] < r["ticket_price"], f"${advertised:.0f}"


class TestBreakeven:
    def test_powerball_breakeven_is_astronomical_after_tax(self):
        r = ev_report("powerball", 500e6, federal_tax=0.37, state_tax=0.05)
        be = r["breakeven_advertised_jackpot"]
        # With sharing feedback + taxes, breakeven (if any) sits far beyond
        # every jackpot in history (record: ~$2.04B advertised).
        assert be is None or be > 2.5e9

    def test_breakeven_none_handled_in_verdict(self):
        r = ev_report("powerball", 500e6, federal_tax=0.37, state_tax=0.10)
        assert "Don't play" in r["verdict"]


class TestInputs:
    def test_sales_curve_interpolates_and_extrapolates(self):
        assert estimated_tickets("powerball", 20e6) == pytest.approx(9e6)
        mid = estimated_tickets("powerball", 400e6)
        assert 20e6 < mid < 40e6
        assert estimated_tickets("powerball", 3e9) > 300e6

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            ev_report("powerball", -1.0)
        with pytest.raises(ValueError):
            ev_report("powerball", 1e8, federal_tax=0.8, state_tax=0.3)
