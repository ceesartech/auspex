"""Unit tests for compute_features_nba — pure helpers + dispatch.

DB/network calls aren't exercised here. Focus is the value-shape
invariants the trained model will depend on (every feature present,
all-finite, line-as-feature design preserved).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fnba = _load("compute_features_nba", "compute_features_nba.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_feature_set_name(self):
        # The training queries + predict-time lookups all key on this
        # exact pair — locked so a rename here forces a parallel update
        # in those places.
        assert fnba.FEATURE_SET == "nba_baseline"
        assert fnba.FEATURE_VERSION == "v1"

    def test_back_to_back_threshold_is_one_and_a_half_days(self):
        # 1.5 days catches NBA b2bs that span UTC+timezone variation
        # without bleeding into 2-days-rest games.
        assert fnba.BACK_TO_BACK_DAYS == 1.5

    def test_window_is_ten(self):
        # 10-game rolling window — standard NBA "last 10" cadence.
        assert fnba.WINDOW == 10


# ── Neutral defaults ────────────────────────────────────────────────


class TestNeutralDefaults:
    def test_every_default_is_finite_number(self):
        # If any default sneaks in as None / nan / str, the prediction
        # path will silently break when the model fills missing inputs.
        for k, v in fnba.NEUTRAL_DEFAULTS.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v).__name__}"
            assert v == v, f"{k} is NaN"  # NaN != NaN

    def test_moneyline_implied_probs_sum_to_one(self):
        # Devigging math invariant — the two implied probs should sum
        # to exactly 1.0 (within float epsilon).
        ph = fnba.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        pa = fnba.NEUTRAL_DEFAULTS["implied_prob_away_ml"]
        assert abs(ph + pa - 1.0) < 1e-9

    def test_home_court_advantage_baked_into_defaults(self):
        # League-average home win rate is ~55%; the defaults should
        # carry a modest home edge instead of 50/50 (which would
        # make every neutral-feature prediction emit "away" for
        # ties).
        ph = fnba.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        assert 0.50 < ph < 0.60

    def test_neutral_spread_is_zero(self):
        # closing_spread_home=0 means "no favorite" — the right
        # neutral value for an unknown matchup.
        assert fnba.NEUTRAL_DEFAULTS["closing_spread_home"] == 0.0

    def test_neutral_total_is_modern_league_modal(self):
        # 225 is the NBA's modal total this era (was ~210 a decade
        # ago — bumped with pace/3pt revolution).
        assert 220 <= fnba.NEUTRAL_DEFAULTS["closing_total_line"] <= 230

    def test_neutral_rolling_form_is_average(self):
        # Both teams at 5-5 over their last 10; both scoring/allowing
        # league average → zero margin. That's the correct neutral
        # for a "we know nothing about either team" prior.
        assert fnba.NEUTRAL_DEFAULTS["home_roll_wins"] == 5.0
        assert fnba.NEUTRAL_DEFAULTS["home_roll_margin"] == 0.0
        assert fnba.NEUTRAL_DEFAULTS["pts_scored_diff"] == 0.0


# ── _with_defaults: missing-value backfill ──────────────────────────


class TestWithDefaults:
    def test_missing_keys_filled(self):
        out = fnba._with_defaults({})
        # Every default key must be in the output.
        for k in fnba.NEUTRAL_DEFAULTS:
            assert k in out
        # And no missing values left.
        for k, v in out.items():
            assert isinstance(v, (int, float))

    def test_provided_values_override(self):
        out = fnba._with_defaults({"odds_home_ml": 2.5})
        assert out["odds_home_ml"] == 2.5
        # Other defaults still present.
        assert out["odds_away_ml"] == fnba.NEUTRAL_DEFAULTS["odds_away_ml"]

    def test_none_value_is_replaced_with_default(self):
        # A None coming from a missing odds row should fall back to
        # the league average, not propagate as None into the model.
        out = fnba._with_defaults({"odds_home_ml": None})
        assert out["odds_home_ml"] == fnba.NEUTRAL_DEFAULTS["odds_home_ml"]

    def test_string_value_is_replaced_with_default(self):
        # Defensive — a corrupt DB row shouldn't be able to ship a
        # string into model input space.
        out = fnba._with_defaults({"odds_home_ml": "1.85"})
        assert out["odds_home_ml"] == fnba.NEUTRAL_DEFAULTS["odds_home_ml"]

    def test_extra_keys_preserved(self):
        # Forward-compat: if a future feature lands before its default
        # is added, we keep it in the output instead of silently
        # dropping it. The training pipeline can pick it up.
        out = fnba._with_defaults({"experimental_pace": 100.5})
        assert out["experimental_pace"] == 100.5


# ── Diff helper ────────────────────────────────────────────────────


class TestDiffHelper:
    def test_writes_diff_when_both_values_present(self):
        features = {"home_x": 5.0, "away_x": 3.0}
        fnba._diff(features, "home_x", "away_x", "x_diff")
        assert features["x_diff"] == 2.0

    def test_skips_when_either_missing(self):
        # Diffs default to 0.0 via NEUTRAL_DEFAULTS at the final stage;
        # the per-call diff helper just no-ops if a side is missing
        # rather than writing a fake zero.
        features = {"home_x": 5.0}
        fnba._diff(features, "home_x", "away_x", "x_diff")
        assert "x_diff" not in features

    def test_handles_integer_inputs(self):
        # Wins-diff is integer-valued; verify int → float conversion.
        features = {"home_wins": 7, "away_wins": 4}
        fnba._diff(features, "home_wins", "away_wins", "wins_diff")
        assert features["wins_diff"] == 3.0
        assert isinstance(features["wins_diff"], float)


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_default_days_is_seven(self):
        args = fnba.parse_args(["--database-url", "postgresql://x"])
        assert args.days == 7
        assert args.force is False
        assert args.all_finished is False

    def test_match_ids_parses(self):
        args = fnba.parse_args(["--match-ids", "a,b,c", "--database-url", "x"])
        assert args.match_ids == "a,b,c"


# Quiet import-unused lint on pytest.
_ = pytest


# ── Cross-book SPREAD features ────────────────────────────────────
#
# Landed 2026-06-04 after scripts/ab_nba_cross_book.py verdict
# (ΔBrier -0.0061 on 2024-2025 walk-forward). NBA moneyline +
# total were tested in the same A/B and DROPPED — these tests
# guard the spread-only landing.


class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return list(self._rows)


class TestFetchSpreadCrossbookNBA:
    CROSSBOOK_KEYS = (
        "spread_book_count",
        "spread_consensus_mean",
        "spread_max_minus_min",
        "spread_std",
        "spread_consensus_implied_prob",
    )

    def test_no_rows_returns_empty_dict(self):
        out = fnba.fetch_spread_crossbook(_FakeCursor([]), "match-1")
        assert out == {}

    def test_single_book_zero_disagreement(self):
        cur = _FakeCursor([
            {"bookmaker": "DK", "p_line": -3.5,
             "p_odds": 1.91, "c_odds": 1.91},
        ])
        out = fnba.fetch_spread_crossbook(cur, "match-1")
        assert out["spread_book_count"] == 1.0
        assert out["spread_consensus_mean"] == -3.5
        assert out["spread_max_minus_min"] == 0.0
        assert out["spread_std"] == 0.0
        assert abs(out["spread_consensus_implied_prob"] - 0.5) < 1e-9

    def test_multi_book_max_min_and_std(self):
        # NBA spread CAN vary across books even within a single line
        # (some at -3.5, others at -4.0 / -4.5). Confirms the line-
        # statistic features actually move on NBA data.
        rows = [
            {"bookmaker": "A", "p_line": -3.5, "p_odds": 1.91, "c_odds": 1.91},
            {"bookmaker": "B", "p_line": -4.0, "p_odds": 1.95, "c_odds": 1.87},
            {"bookmaker": "C", "p_line": -4.5, "p_odds": 2.00, "c_odds": 1.83},
        ]
        cur = _FakeCursor(rows)
        out = fnba.fetch_spread_crossbook(cur, "match-1")
        assert out["spread_book_count"] == 3.0
        assert abs(out["spread_consensus_mean"] - (-4.0)) < 1e-9
        assert out["spread_max_minus_min"] == 1.0
        # ddof=0 std: lines [-3.5, -4.0, -4.5], mean -4.0,
        # sq diffs 0.25 + 0 + 0.25 = 0.5, var 0.5/3 = 0.1667, std 0.4082
        assert abs(out["spread_std"] - 0.40824829) < 1e-6

    def test_missing_counter_odds_keeps_line(self):
        # The line still counts toward consensus_mean even when
        # one book lacks the away odds (devig drops only that book).
        rows = [
            {"bookmaker": "Full", "p_line": -3.5,
             "p_odds": 1.91, "c_odds": 1.91},
            {"bookmaker": "OnlyHome", "p_line": -4.5,
             "p_odds": 1.91, "c_odds": None},
        ]
        cur = _FakeCursor(rows)
        out = fnba.fetch_spread_crossbook(cur, "match-1")
        assert out["spread_book_count"] == 2.0
        assert out["spread_consensus_mean"] == -4.0
        # Only Full contributes devigged prob.
        assert abs(out["spread_consensus_implied_prob"] - 0.5) < 1e-9

    def test_all_keys_in_neutral_defaults(self):
        for k in self.CROSSBOOK_KEYS:
            assert k in fnba.NEUTRAL_DEFAULTS

    def test_empty_fetch_filled_by_with_defaults(self):
        empty = fnba.fetch_spread_crossbook(_FakeCursor([]), "match-1")
        filled = fnba._with_defaults(empty)
        for k in self.CROSSBOOK_KEYS:
            assert filled[k] == fnba.NEUTRAL_DEFAULTS[k]
