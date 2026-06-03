"""Unit tests for compute_features_mma — pure helpers + defaults.

MMA feature shape mirrors tennis (1v1, single moneyline market,
H2H/rolling/schedule features) with MMA-specific magic numbers:

  * WINDOW = 5 (vs tennis's 10) — UFC fighters fight ~3-4 times/year
    so 5 fights covers ~14-18 months.
  * ACTIVE_DAYS = 365 (vs tennis's 30) — MMA layoffs of 6-12 months
    are routine; only longer layoffs flag as inactive.
  * Modal favorite implied prob ≈ 0.62 (vs tennis 0.65) — MMA upsets
    are more frequent.
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


fmma = _load("compute_features_mma", "compute_features_mma.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_feature_set_name(self):
        assert fmma.FEATURE_SET == "mma_baseline"
        assert fmma.FEATURE_VERSION == "v1"

    def test_window_is_five(self):
        # 5 fights ≈ 14-18 months of recent MMA form. Shorter than
        # tennis's 10 because MMA fight frequency is much lower.
        assert fmma.WINDOW == 5

    def test_active_threshold_is_one_year(self):
        # MMA layoffs of 6-12 months are routine; flag inactive only
        # beyond 365 days idle (signals borderline retirement / cage
        # rust).
        assert fmma.ACTIVE_DAYS == 365.0

    def test_active_threshold_much_larger_than_tennis(self):
        # Tennis ACTIVE_DAYS = 30 (tour cadence is weekly). MMA is
        # 365 — fighting frequency varies by 10-20x. If this drifts
        # toward tennis-like values, the model will over-flag every
        # MMA fighter as inactive.
        assert fmma.ACTIVE_DAYS > 180


# ── Neutral defaults ────────────────────────────────────────────────


class TestNeutralDefaults:
    def test_every_default_is_finite_number(self):
        for k, v in fmma.NEUTRAL_DEFAULTS.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v).__name__}"
            assert v == v, f"{k} is NaN"

    def test_moneyline_implied_probs_sum_to_one(self):
        ph = fmma.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        pa = fmma.NEUTRAL_DEFAULTS["implied_prob_away_ml"]
        assert abs(ph + pa - 1.0) < 1e-9

    def test_favorite_implied_prob_modal_mma_split(self):
        # MMA modal favorite is ~-160 / +135 = 62/38 split. Slightly
        # less heavy than tennis (~65/35) since MMA upsets are
        # more frequent.
        ph = fmma.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        assert 0.58 <= ph <= 0.66

    def test_rolling_defaults_at_break_even(self):
        # 2.5 wins in 5 fights = .500. Same shape as tennis (5/10 = .500).
        assert fmma.NEUTRAL_DEFAULTS["home_roll_wins"] == 2.5
        assert fmma.NEUTRAL_DEFAULTS["home_roll_matches"] == 5.0
        assert fmma.NEUTRAL_DEFAULTS["home_roll_win_pct"] == 0.50

    def test_h2h_defaults_to_never_fought(self):
        # MMA rematches are rare. 0/0/0 is the explicit "no prior
        # meeting" signal.
        assert fmma.NEUTRAL_DEFAULTS["h2h_home_wins"] == 0.0
        assert fmma.NEUTRAL_DEFAULTS["h2h_away_wins"] == 0.0
        assert fmma.NEUTRAL_DEFAULTS["h2h_matches"] == 0.0
        assert fmma.NEUTRAL_DEFAULTS["h2h_balance"] == 0.0

    def test_schedule_defaults_at_modal_inter_fight_gap(self):
        # ~120 days = 4 months, modal UFC inter-fight cadence.
        assert fmma.NEUTRAL_DEFAULTS["home_days_rest"] == 120.0
        assert fmma.NEUTRAL_DEFAULTS["away_days_rest"] == 120.0
        assert fmma.NEUTRAL_DEFAULTS["home_active"] == 1.0


# ── _with_defaults: missing-value backfill ──────────────────────────


class TestWithDefaults:
    def test_missing_keys_filled(self):
        out = fmma._with_defaults({})
        for k in fmma.NEUTRAL_DEFAULTS:
            assert k in out

    def test_provided_values_override(self):
        out = fmma._with_defaults({"odds_home_ml": 3.0})
        assert out["odds_home_ml"] == 3.0
        assert out["odds_away_ml"] == fmma.NEUTRAL_DEFAULTS["odds_away_ml"]

    def test_none_value_is_replaced_with_default(self):
        out = fmma._with_defaults({"home_roll_win_pct": None})
        assert out["home_roll_win_pct"] == fmma.NEUTRAL_DEFAULTS["home_roll_win_pct"]

    def test_extra_keys_preserved(self):
        out = fmma._with_defaults({"experimental_reach_diff": 2.5})
        assert out["experimental_reach_diff"] == 2.5


# ── Diff helper ────────────────────────────────────────────────────


class TestDiffHelper:
    def test_writes_diff_when_both_values_present(self):
        features = {"home_x": 5.0, "away_x": 3.0}
        fmma._diff(features, "home_x", "away_x", "x_diff")
        assert features["x_diff"] == 2.0

    def test_skips_when_either_missing(self):
        features = {"home_x": 5.0}
        fmma._diff(features, "home_x", "away_x", "x_diff")
        assert "x_diff" not in features


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_default_days_is_seven(self):
        args = fmma.parse_args(["--database-url", "postgresql://x"])
        assert args.days == 7
        assert args.force is False
        assert args.all_finished is False

    def test_all_finished_flag(self):
        args = fmma.parse_args(["--all-finished", "--database-url", "x"])
        assert args.all_finished is True
