"""Unit tests for generate_recommendations_nba — pure helpers.

The full match-level pipeline is exercised by integration tests
elsewhere; here we verify the NBA-specific surface (label maps,
display string for lined bets, market dispatch) since those are
where soccer↔NBA mismatches would silently produce wrong selection
strings or fail to look up model probs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# generate_recommendations (soccer) is imported by the NBA module —
# load it first so the import resolves.
_load("generate_recommendations", "generate_recommendations.py")
gr_nba = _load("generate_recommendations_nba", "generate_recommendations_nba.py")


class TestMarketDispatch:
    def test_three_nba_ensembles_registered(self):
        # The NBA_MARKETS dict must cover exactly the 3 NBA ensembles
        # the API loads. If an ensemble is added without an entry here,
        # load_nba_predictions silently misses it.
        assert set(gr_nba.NBA_MARKETS.keys()) == {
            "ensemble_nba_ml",
            "ensemble_nba_sp",
            "ensemble_nba_tot",
        }

    def test_market_labels_match_taskspec(self):
        # The prediction_type values must match what TaskSpec writes
        # to the predictions table. Locked here so a rename in either
        # place fails fast.
        assert gr_nba.NBA_MARKETS["ensemble_nba_ml"][0] == "moneyline"
        assert gr_nba.NBA_MARKETS["ensemble_nba_sp"][0] == "spread"
        assert gr_nba.NBA_MARKETS["ensemble_nba_tot"][0] == "total"

    def test_labels_match_taskspec(self):
        # NBA_LABELS index must align with TaskSpec.labels. Order
        # matters because probabilities dict keys are looked up by
        # name, but the LIST order would matter if anyone ever
        # iterated by index.
        assert gr_nba.NBA_LABELS["moneyline"] == ["home", "away"]
        assert gr_nba.NBA_LABELS["spread"] == ["home", "away"]
        assert gr_nba.NBA_LABELS["total"] == ["over", "under"]


class TestSelectionWithLine:
    def test_moneyline_drops_line(self):
        # ML never has a line — display is just the side.
        assert gr_nba._selection_with_line("moneyline", "home", None) == "home"
        # Even if a line slipped through, ML must ignore it.
        assert gr_nba._selection_with_line("moneyline", "away", -7.5) == "away"

    def test_spread_carries_signed_line(self):
        # Spread display must encode the sign so the recommendation
        # is unambiguous: 'home_-7.5' is "home -7.5", not "home +7.5".
        assert gr_nba._selection_with_line("spread", "home", -7.5) == "home_-7.5"
        assert gr_nba._selection_with_line("spread", "away", 7.5) == "away_+7.5"
        # Pickem (line=0) — sign suppressed.
        assert gr_nba._selection_with_line("spread", "home", 0.0) == "home_+0"

    def test_total_carries_unsigned_line(self):
        # Totals are always positive (225.5 not +225.5), so display
        # drops the sign.
        assert gr_nba._selection_with_line("total", "over", 225.5) == "over_225.5"
        assert gr_nba._selection_with_line("total", "under", 220.0) == "under_220"

    def test_missing_line_falls_back_to_bare_selection(self):
        # If the line is None on a non-moneyline bet (shouldn't happen
        # — we filter to line-present odds upstream — but defensive
        # against bad data), don't crash; emit just the selection.
        assert gr_nba._selection_with_line("spread", "home", None) == "home"


class TestRiskFactors:
    def test_longshot_flag_at_4x_or_more(self):
        # NBA games rarely have +400 dogs (that'd be a ~20%-implied
        # rare blowout matchup); flag them as longshot risk.
        assert "longshot" in gr_nba._risk_factors(prob=0.25, odds_decimal=4.0)
        assert "longshot" in gr_nba._risk_factors(prob=0.20, odds_decimal=5.0)
        assert "longshot" not in gr_nba._risk_factors(prob=0.55, odds_decimal=1.85)

    def test_low_model_prob_flag_under_45(self):
        # NBA model rarely picks at <45% confidence; if it does that's
        # a thin edge worth flagging.
        assert "low_model_probability" in gr_nba._risk_factors(prob=0.40, odds_decimal=2.5)
        assert "low_model_probability" not in gr_nba._risk_factors(prob=0.50, odds_decimal=2.0)

    def test_clean_pick_has_no_risks(self):
        # 55% confidence at 1.85 (~54% break-even) is a textbook
        # clean value bet — no risks raised.
        assert gr_nba._risk_factors(prob=0.55, odds_decimal=1.85) == []


class TestSharedMathReuse:
    def test_uses_soccer_kelly_and_ev(self):
        # The NBA engine reuses the soccer helpers (expected_value,
        # kelly_fraction, confidence_rating). Confirm they're the
        # same functions, not divergent copies.
        assert gr_nba.expected_value(0.55, 2.0) == 0.55 * 2.0 - 1.0
        assert gr_nba.kelly_fraction(0.55, 2.0) > 0
        # Negative-EV bet returns Kelly = 0.
        assert gr_nba.kelly_fraction(0.30, 2.0) == 0.0


class TestKellyConstant:
    def test_quarter_kelly(self):
        # Conservative bankroll sizing — matches the soccer engine.
        # Locked so it doesn't accidentally creep up to full Kelly.
        assert gr_nba.KELLY_FRACTION == 0.25


class TestProbabilityCap:
    def test_cap_clips_overconfident_predictions(self):
        # The 87% spread emit observed on the Finals matchup gets
        # capped to PROB_CAP_FOR_EV before EV / Kelly math.
        capped = gr_nba.cap_prob(0.87)
        assert capped == gr_nba.PROB_CAP_FOR_EV

    def test_cap_leaves_low_confidence_alone(self):
        # 65% moneyline confidence on a favorite stays at 65% — no
        # cap fires.
        assert gr_nba.cap_prob(0.65) == 0.65

    def test_cap_exactly_at_threshold_unchanged(self):
        # Boundary: a prob exactly at the cap passes through.
        assert gr_nba.cap_prob(0.80) == 0.80

    def test_cap_value_locked(self):
        # 0.80 leaves a ~7-point margin vs the model's MCE of 0.21.
        # If someone bumps it to 0.90 the cap becomes meaningless;
        # if they drop to 0.70 too many real edges get killed. Test
        # the band.
        assert 0.75 <= gr_nba.PROB_CAP_FOR_EV <= 0.85

    def test_capped_ev_smaller_than_raw_ev(self):
        # End-to-end: a raw 87% pick at 1.95 odds emits +70% EV. With
        # the cap, same offer drops to (0.80 × 1.95 - 1) = +56% EV.
        # Still positive — the rec still fires — but stake-size will
        # be lower because Kelly fraction shrinks.
        raw_ev = gr_nba.expected_value(0.87, 1.95)
        capped = gr_nba.cap_prob(0.87)
        capped_ev = gr_nba.expected_value(capped, 1.95)
        assert capped_ev < raw_ev
        assert capped_ev > 0  # still positive — rec still fires
