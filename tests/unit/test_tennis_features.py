"""Unit tests for compute_features_tennis — pure helpers + defaults.

Tennis feature shape differs from the team sports in two structural
ways tested here:
  * No "pts scored" rolling form — tennis matches are binary win/lose.
    Rolling stat is win-rate, not points/game.
  * Head-to-head between this specific player pair is first-class
    (tennis matchups are stylistically dependent). NFL/NBA/NHL treat
    H2H in passing; tennis features include it as a core column.
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


ftennis = _load("compute_features_tennis", "compute_features_tennis.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_feature_set_name(self):
        # Training queries + predict-time lookups all key on this
        # exact pair — locked so a rename forces parallel updates.
        assert ftennis.FEATURE_SET == "tennis_baseline"
        assert ftennis.FEATURE_VERSION == "v1"

    def test_window_is_ten(self):
        # 10-match rolling window — tour players average ~70-90
        # matches/year so 10 ≈ 6-8 weeks of recent form.
        assert ftennis.WINDOW == 10

    def test_active_threshold_is_thirty_days(self):
        # 30+ days idle is the rough threshold for an injury-return
        # signal — tour regulars play more frequently than that.
        assert ftennis.ACTIVE_DAYS == 30.0


# ── Neutral defaults ────────────────────────────────────────────────


class TestNeutralDefaults:
    def test_every_default_is_finite_number(self):
        for k, v in ftennis.NEUTRAL_DEFAULTS.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v).__name__}"
            assert v == v, f"{k} is NaN"

    def test_moneyline_implied_probs_sum_to_one(self):
        # Devigging invariant.
        ph = ftennis.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        pa = ftennis.NEUTRAL_DEFAULTS["implied_prob_away_ml"]
        assert abs(ph + pa - 1.0) < 1e-9

    def test_favorite_implied_prob_at_modal_split(self):
        # Tour modal price is ~65/35 favorite/dog (-200 / +160). If
        # someone bumps this to 80/20 the default no longer represents
        # the typical match — would skew predictions on missing data.
        ph = ftennis.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        assert 0.60 <= ph <= 0.70

    def test_rolling_defaults_at_break_even(self):
        # 5 wins in 10 matches = .500 win rate — the tour ranking floor
        # for tracked players.
        assert ftennis.NEUTRAL_DEFAULTS["home_roll_wins"] == 5.0
        assert ftennis.NEUTRAL_DEFAULTS["home_roll_matches"] == 10.0
        assert ftennis.NEUTRAL_DEFAULTS["home_roll_win_pct"] == 0.50

    def test_h2h_defaults_to_never_played(self):
        # 0/0/0 = "no prior data" — common since the tour has 3000+
        # players and most pairings are first meetings.
        assert ftennis.NEUTRAL_DEFAULTS["h2h_home_wins"] == 0.0
        assert ftennis.NEUTRAL_DEFAULTS["h2h_away_wins"] == 0.0
        assert ftennis.NEUTRAL_DEFAULTS["h2h_matches"] == 0.0
        assert ftennis.NEUTRAL_DEFAULTS["h2h_balance"] == 0.0

    def test_schedule_defaults_are_typical(self):
        # 7 days rest is the modal inter-match cadence for tour
        # players outside of Grand Slam fortnights.
        assert ftennis.NEUTRAL_DEFAULTS["home_days_rest"] == 7.0
        assert ftennis.NEUTRAL_DEFAULTS["away_days_rest"] == 7.0
        # active=1 → "played in last 30 days" → typical for any tour
        # regular.
        assert ftennis.NEUTRAL_DEFAULTS["home_active"] == 1.0

    def test_implied_diff_matches_split(self):
        # 0.65 - 0.35 = 0.30. Locked here so the default odds
        # implied prob and the default odds_implied_diff stay
        # consistent.
        assert abs(ftennis.NEUTRAL_DEFAULTS["odds_implied_diff"] - 0.30) < 1e-9


# ── _with_defaults: missing-value backfill ──────────────────────────


class TestWithDefaults:
    def test_missing_keys_filled(self):
        out = ftennis._with_defaults({})
        for k in ftennis.NEUTRAL_DEFAULTS:
            assert k in out

    def test_provided_values_override(self):
        out = ftennis._with_defaults({"odds_home_ml": 2.20})
        assert out["odds_home_ml"] == 2.20
        # Untouched keys come from defaults.
        assert out["odds_away_ml"] == ftennis.NEUTRAL_DEFAULTS["odds_away_ml"]

    def test_none_value_is_replaced_with_default(self):
        # An odds-missing row should fall back to neutral default
        # rather than propagate None.
        out = ftennis._with_defaults({"home_roll_win_pct": None})
        assert out["home_roll_win_pct"] == ftennis.NEUTRAL_DEFAULTS["home_roll_win_pct"]

    def test_extra_keys_preserved(self):
        # Future features (e.g., surface_win_pct) should pass through
        # without being filtered out — keeps the path flexible for v2
        # additions without breaking compute_features_tennis.
        out = ftennis._with_defaults({"experimental_surface_pct": 0.72})
        assert out["experimental_surface_pct"] == 0.72


# ── Diff helper ────────────────────────────────────────────────────


class TestDiffHelper:
    def test_writes_diff_when_both_values_present(self):
        features = {"home_x": 0.62, "away_x": 0.48}
        ftennis._diff(features, "home_x", "away_x", "x_diff")
        assert abs(features["x_diff"] - 0.14) < 1e-9

    def test_skips_when_either_missing(self):
        features = {"home_x": 0.62}
        ftennis._diff(features, "home_x", "away_x", "x_diff")
        assert "x_diff" not in features

    def test_skips_when_value_non_numeric(self):
        # Defensive: a None or string slipping through shouldn't crash.
        features = {"home_x": None, "away_x": 0.5}
        ftennis._diff(features, "home_x", "away_x", "x_diff")
        assert "x_diff" not in features


# ── Argparse plumbing ──────────────────────────────────────────────


class TestWeatherDefaults:
    """Weather features (Phase 13). Tennis-specific defaults: ~20°C
    is typical Slam weather (AO/RG/W/USO combined), light wind, dry."""

    def test_temperature_default_is_typical_slam_temp(self):
        assert ftennis.NEUTRAL_DEFAULTS["weather_temp_c"] == 20.0

    def test_wind_default_is_light_breeze(self):
        assert ftennis.NEUTRAL_DEFAULTS["weather_wind_kmh"] == 10.0

    def test_precip_default_is_dry(self):
        assert ftennis.NEUTRAL_DEFAULTS["weather_precip_mm"] == 0.0

    def test_extreme_condition_flags_default_zero(self):
        assert ftennis.NEUTRAL_DEFAULTS["weather_high_wind"] == 0.0
        assert ftennis.NEUTRAL_DEFAULTS["weather_wet"] == 0.0
        assert ftennis.NEUTRAL_DEFAULTS["weather_hot"] == 0.0

    def test_indoor_default_zero(self):
        assert ftennis.NEUTRAL_DEFAULTS["weather_indoor"] == 0.0


class TestWeatherThresholds:
    def test_hot_threshold_matches_aussie_open_heat_policy(self):
        # AO's extreme-heat policy kicks in around 32°C. Tennis-
        # specific (NFL uses freezing, soccer uses 30°C hot).
        assert ftennis.HOT_TEMP_C == 32.0

    def test_high_wind_threshold_consistent_with_nfl(self):
        # Same 25 km/h cutoff as NFL — small ball + long flight
        # means wind disrupts tennis serves at similar speeds.
        assert ftennis.HIGH_WIND_KMH == 25.0


class TestCli:
    def test_default_days_is_seven(self):
        args = ftennis.parse_args(["--database-url", "postgresql://x"])
        assert args.days == 7
        assert args.force is False
        assert args.all_finished is False

    def test_match_ids_parses(self):
        args = ftennis.parse_args(["--match-ids", "a,b,c", "--database-url", "x"])
        assert args.match_ids == "a,b,c"

    def test_all_finished_flag(self):
        args = ftennis.parse_args(["--all-finished", "--database-url", "x"])
        assert args.all_finished is True
