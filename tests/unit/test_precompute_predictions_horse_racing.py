"""Unit tests for the horse-racing market-consensus baseline.

The math under test is the devigging step:

    raw[i]      = 1 / morning_line_decimal[i]
    devigged[i] = raw[i] / sum(raw)

The invariants we care about (and lock down here):
  * Probabilities across a fully-priced race sum to 1.0.
  * Removing the vig BIASES individual probabilities up (since the
    pre-devig sum > 1.0 from bookmaker overround). Lock the direction.
  * Unpriced entrants get a uniform 1/field_size fallback, after which
    the whole field is renormalised to keep the row sum at 1.0.
  * Below MIN_PRICED_ENTRANTS, we don't trust the devig at all and
    fall through to uniform across the field.

The argparse + CLI plumbing is also tested so the DAG wiring doesn't
break silently.
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


pred = _load("precompute_predictions_horse_racing", "precompute_predictions_horse_racing.py")


_PROGRAM_COUNTER = [0]


def _ent(eid: str, ml=None, program=None):
    """Build a fake entrant row with the columns the predictor reads.
    program_number defaults to a monotonic counter so we don't have to
    parse it out of the entrant_id."""
    if program is None:
        _PROGRAM_COUNTER[0] += 1
        program = _PROGRAM_COUNTER[0]
    return {
        "entrant_id": eid,
        "program_number": program,
        "morning_line_odds": ml,
        "scratched": False,
        "horse_name": f"Horse {eid}",
    }


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_model_name_is_market_consensus_v1(self):
        # Locked because the DB unique key includes model_name. Renaming
        # creates a parallel set of rows rather than updating existing
        # ones; bump the version field instead.
        assert pred.MODEL_NAME == "market_consensus_v1"
        assert pred.MODEL_VERSION == "1.0.0"

    def test_min_priced_threshold_is_three(self):
        # Below 3 priced runners, devigging is just amplifying odds
        # noise — we'd rather output a uniform prior.
        assert pred.MIN_PRICED_ENTRANTS == 3


class TestUniformProb:
    def test_one_over_n(self):
        assert pred._uniform_prob(1) == 1.0
        assert pred._uniform_prob(2) == 0.5
        assert pred._uniform_prob(8) == 0.125

    def test_zero_field_does_not_divide_by_zero(self):
        # field_size=0 shouldn't crash — callers may pass it during
        # the empty-race short-circuit path.
        assert pred._uniform_prob(0) == 1.0


# ── Devig: happy path ──────────────────────────────────────────────


class TestDevigFullyPriced:
    def test_returns_one_prob_per_entrant(self):
        ents = [_ent("e1", 2.0), _ent("e2", 3.0), _ent("e3", 6.0)]
        probs = pred.devig(ents)
        assert set(probs.keys()) == {"e1", "e2", "e3"}

    def test_sums_to_one(self):
        # Classic bookmaker margin: 1/2 + 1/3 + 1/6 = 1.0 already (no
        # overround) — devig should leave the ratios untouched.
        ents = [_ent("e1", 2.0), _ent("e2", 3.0), _ent("e3", 6.0)]
        probs = pred.devig(ents)
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_devig_removes_overround(self):
        # 4 horses at 2.0 each = raw sum = 2.0 (100% overround).
        # Devigged each gets 0.25.
        ents = [_ent(f"e{i}", 2.0) for i in range(1, 5)]
        probs = pred.devig(ents)
        for p in probs.values():
            assert abs(p - 0.25) < 1e-9

    def test_favorite_gets_highest_prob(self):
        # Order invariant — lowest odds → highest probability.
        ents = [_ent("fav", 1.5), _ent("mid", 4.0), _ent("dog", 10.0)]
        probs = pred.devig(ents)
        assert probs["fav"] > probs["mid"] > probs["dog"]

    def test_probability_strictly_in_zero_one(self):
        # No prob should ever be 0.0 or 1.0 once we have multiple runners
        # — both are pathological in practice.
        ents = [_ent("fav", 1.10), _ent("dog1", 99.0), _ent("dog2", 99.0)]
        probs = pred.devig(ents)
        for p in probs.values():
            assert 0.0 < p < 1.0


# ── Devig: partial pricing ──────────────────────────────────────────


class TestDevigPartiallyPriced:
    def test_unpriced_entrants_get_uniform_then_renormalise(self):
        # 5-horse field, 3 priced + 2 unpriced. Sum should still be 1.0.
        ents = [
            _ent("e1", 2.0),
            _ent("e2", 4.0),
            _ent("e3", 8.0),
            _ent("e4", None),
            _ent("e5", None),
        ]
        probs = pred.devig(ents)
        assert len(probs) == 5
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_zero_morning_line_treated_as_unpriced(self):
        # Vendor sometimes sends 0.0 or 1.0 as a sentinel; both are
        # invalid decimal odds. Should drop through to the unpriced path.
        ents = [
            _ent("e1", 2.0),
            _ent("e2", 4.0),
            _ent("e3", 6.0),
            _ent("e4", 0.0),
            _ent("e5", 1.0),
        ]
        probs = pred.devig(ents)
        assert abs(sum(probs.values()) - 1.0) < 1e-6


class TestDevigUnderpriced:
    def test_one_priced_entrant_falls_back_to_uniform(self):
        # Only one runner has a morning line → uniform across the
        # entire field (everyone gets 1/N).
        ents = [_ent("e1", 2.0), _ent("e2"), _ent("e3"), _ent("e4")]
        probs = pred.devig(ents)
        for p in probs.values():
            assert abs(p - 0.25) < 1e-9

    def test_two_priced_entrants_falls_back_to_uniform(self):
        # MIN_PRICED_ENTRANTS=3, so 2 priced still triggers uniform.
        ents = [_ent("e1", 2.0), _ent("e2", 3.0), _ent("e3"), _ent("e4")]
        probs = pred.devig(ents)
        for p in probs.values():
            assert abs(p - 0.25) < 1e-9

    def test_zero_priced_entrants_falls_back_to_uniform(self):
        ents = [_ent("e1"), _ent("e2"), _ent("e3"), _ent("e4")]
        probs = pred.devig(ents)
        for p in probs.values():
            assert abs(p - 0.25) < 1e-9

    def test_empty_entrants_returns_empty(self):
        # A race with no non-scratched entrants — predictor should
        # short-circuit cleanly rather than crash on /0.
        assert pred.devig([]) == {}


# ── CLI plumbing ───────────────────────────────────────────────────


class TestCli:
    def test_default_days_is_two(self):
        args = pred.parse_args(["--database-url", "postgresql://x"])
        # 2-day window covers today + tomorrow's racecards on Racing
        # API Basic plan; longer windows would mostly be empty.
        assert args.days == 2
        assert args.race_ids is None

    def test_race_ids_passes_through(self):
        args = pred.parse_args(["--race-ids", "a,b,c", "--database-url", "x"])
        assert args.race_ids == "a,b,c"

    def test_days_can_be_overridden(self):
        args = pred.parse_args(["--days", "7", "--database-url", "x"])
        assert args.days == 7
