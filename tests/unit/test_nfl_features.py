"""Unit tests for compute_features_nfl — pure helpers + defaults.

NFL feature shape differs from NBA's in three ways tested here:
  * Rolling window is 5 games (NBA's is 10) — NFL has fewer games
    per season so the window is shorter.
  * Schedule context has short_week / long_week flags (Thu games +
    bye-week games), not just back_to_back like NBA.
  * Modern NFL averages (22.5 pts/game/team, ~57% home win rate)
    differ from NBA (114 pts/game/team, ~55%).
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


fnfl = _load("compute_features_nfl", "compute_features_nfl.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_feature_set_name(self):
        # Training queries + predict-time lookups all key on this
        # exact pair — locked so a rename forces parallel updates.
        assert fnfl.FEATURE_SET == "nfl_baseline"
        assert fnfl.FEATURE_VERSION == "v1"

    def test_window_is_five(self):
        # 5-game rolling window — standard NFL "last 5" cadence. If
        # bumped to 10 it'd be more than half a season; if dropped
        # to 3 it'd be noisy on opening weeks.
        assert fnfl.WINDOW == 5

    def test_short_week_threshold(self):
        # Short-week threshold catches the Thursday game after a
        # Sunday — 3-4 days rest. NBA-style "back-to-back" doesn't
        # exist in NFL, but the short week is just as impactful.
        assert fnfl.SHORT_WEEK_DAYS == 4.0

    def test_long_week_threshold(self):
        # Long-week threshold catches bye-week games and international
        # series (10+ days rest). Models pick up the rested-team
        # advantage.
        assert fnfl.LONG_WEEK_DAYS == 10.0


# ── Neutral defaults ────────────────────────────────────────────────


class TestNeutralDefaults:
    def test_every_default_is_finite_number(self):
        # If any default sneaks in as None / nan / str, the prediction
        # path will silently break when the model fills missing inputs.
        for k, v in fnfl.NEUTRAL_DEFAULTS.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v).__name__}"
            assert v == v, f"{k} is NaN"  # NaN != NaN

    def test_moneyline_implied_probs_sum_to_one(self):
        # Devigging math invariant — the two implied probs should sum
        # to exactly 1.0 (within float epsilon).
        ph = fnfl.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        pa = fnfl.NEUTRAL_DEFAULTS["implied_prob_away_ml"]
        assert abs(ph + pa - 1.0) < 1e-9

    def test_home_advantage_higher_than_nba(self):
        # NFL home win rate ≈ 57% (NBA's is 55%). The bigger crowd
        # effect + travel for visiting teams produces a slightly
        # larger home edge in football.
        ph = fnfl.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        assert 0.55 <= ph <= 0.60

    def test_neutral_spread_is_zero(self):
        # 0 = no favorite. Same convention as NBA.
        assert fnfl.NEUTRAL_DEFAULTS["closing_spread_home"] == 0.0

    def test_neutral_total_is_modern_league_modal(self):
        # 45 is the modern NFL modal total (was ~42 a decade ago —
        # passing offense has crept it up).
        assert 42 <= fnfl.NEUTRAL_DEFAULTS["closing_total_line"] <= 48

    def test_neutral_pts_scored_is_modern_average(self):
        # NFL teams average ~22.5 points/game. Locked so a future
        # tweak to 30 (basketball-style numbers) fires.
        assert 20 <= fnfl.NEUTRAL_DEFAULTS["home_roll_pts_scored"] <= 25

    def test_neutral_wins_is_half_window(self):
        # 5-game window, .500 record → 2.5 wins. Same shape as NBA's
        # 10-game / 5-win neutral.
        assert fnfl.NEUTRAL_DEFAULTS["home_roll_wins"] == 2.5

    def test_neutral_days_rest_is_modal_seven(self):
        # NFL modal cadence is Sunday-to-Sunday = 7 days rest.
        # short_week and long_week flags default to 0 (modal game,
        # no schedule-context advantage).
        assert fnfl.NEUTRAL_DEFAULTS["home_days_rest"] == 7.0
        assert fnfl.NEUTRAL_DEFAULTS["home_short_week"] == 0.0
        assert fnfl.NEUTRAL_DEFAULTS["home_long_week"] == 0.0


# ── _with_defaults: missing-value backfill ──────────────────────────


class TestWithDefaults:
    def test_missing_keys_filled(self):
        out = fnfl._with_defaults({})
        for k in fnfl.NEUTRAL_DEFAULTS:
            assert k in out

    def test_provided_values_override(self):
        out = fnfl._with_defaults({"odds_home_ml": 2.5})
        assert out["odds_home_ml"] == 2.5
        assert out["odds_away_ml"] == fnfl.NEUTRAL_DEFAULTS["odds_away_ml"]

    def test_none_value_is_replaced_with_default(self):
        # A None from a missing odds row should fall back to league
        # average, not propagate as None into the model.
        out = fnfl._with_defaults({"closing_total_line": None})
        assert out["closing_total_line"] == fnfl.NEUTRAL_DEFAULTS["closing_total_line"]

    def test_extra_keys_preserved(self):
        out = fnfl._with_defaults({"experimental_drive_efficiency": 0.42})
        assert out["experimental_drive_efficiency"] == 0.42


# ── Diff helper ────────────────────────────────────────────────────


class TestDiffHelper:
    def test_writes_diff_when_both_values_present(self):
        features = {"home_x": 5.0, "away_x": 3.0}
        fnfl._diff(features, "home_x", "away_x", "x_diff")
        assert features["x_diff"] == 2.0

    def test_skips_when_either_missing(self):
        features = {"home_x": 5.0}
        fnfl._diff(features, "home_x", "away_x", "x_diff")
        assert "x_diff" not in features


# ── Line movement + derivative bookie margins ───────────────────────


class TestLineMovementDefaults:
    """The four new features added to fight NFL spread/total's
    coin-flip OOS performance — line movement carries sharp-action
    signal, derivative-market vigs distinguish efficient vs sloppy
    books."""

    def test_movement_defaults_to_zero(self):
        # 0.0 = "no movement" / "single snapshot only" — model treats
        # it as the neutral case.
        assert fnfl.NEUTRAL_DEFAULTS["spread_movement"] == 0.0
        assert fnfl.NEUTRAL_DEFAULTS["total_movement"] == 0.0

    def test_derivative_bookie_margins_default_to_zero(self):
        # 0.0 means "market unavailable" rather than implying a fair
        # market (which would be a misleading signal to the model).
        assert fnfl.NEUTRAL_DEFAULTS["spread_bookie_margin"] == 0.0
        assert fnfl.NEUTRAL_DEFAULTS["total_bookie_margin"] == 0.0

    def test_all_four_features_present_in_defaults(self):
        # Ensures _with_defaults() fills them even when the odds
        # query found no spread/total rows.
        for k in ("spread_movement", "total_movement", "spread_bookie_margin", "total_bookie_margin"):
            assert k in fnfl.NEUTRAL_DEFAULTS


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_default_days_is_seven(self):
        args = fnfl.parse_args(["--database-url", "postgresql://x"])
        assert args.days == 7
        assert args.force is False
        assert args.all_finished is False

    def test_match_ids_parses(self):
        args = fnfl.parse_args(["--match-ids", "a,b,c", "--database-url", "x"])
        assert args.match_ids == "a,b,c"
