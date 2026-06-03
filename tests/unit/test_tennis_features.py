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


# ── Weather integration (Phase 14 — Open-Meteo retry) ───────────────


class _FakeCursor:
    """See test_nfl_features._FakeCursor — same queue-of-fetchones
    stub. fetch_weather makes 1 or 2 execute() calls."""

    def __init__(self, responses: list):
        self._responses = list(responses)

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self._responses.pop(0) if self._responses else None


class TestNoWeatherFeaturesInDefaults:
    """Same load-bearing invariant as test_nfl_features: weather_*
    keys must NOT appear in NEUTRAL_DEFAULTS. Combined with the
    Phase A EXISTS-gate on TENNIS_MONEYLINE_TRAINING_QUERY, this
    means the model never trains on default weather values — no
    default-bias leakage like the reverted v3."""

    def test_no_weather_keys_in_defaults(self):
        weather_keys = [k for k in ftennis.NEUTRAL_DEFAULTS if k.startswith("weather_")]
        assert weather_keys == [], (
            f"weather_* keys in NEUTRAL_DEFAULTS reintroduce default-bias "
            f"leakage: {weather_keys}"
        )


class TestWeatherThresholds:
    def test_high_wind_kmh_is_25(self):
        # Same as NFL — wind disrupts ball toss + service drift.
        assert ftennis.HIGH_WIND_KMH == 25.0

    def test_wet_precip_mm_is_5(self):
        # Tour-stop covers roof / suspends play around 5mm/4h.
        assert ftennis.WET_PRECIP_MM == 5.0

    def test_hot_temp_c_is_32(self):
        # Australian Open extreme-heat policy threshold — extra rest
        # + roof closure protocols kick in above this.
        assert ftennis.HOT_TEMP_C == 32.0

    def test_no_freezing_threshold_exists(self):
        # Tennis is a no-winter-outdoor sport. The absence of
        # FREEZING_TEMP_C is intentional — adding one would imply
        # we're modelling matches the tour doesn't actually play.
        assert not hasattr(ftennis, "FREEZING_TEMP_C")


class TestCanonicalWeatherKeySet:
    """v4b invariant — see test_nfl_features.TestCanonicalWeatherKeySet
    docstring for the rationale. Tennis has its own canonical set
    (TENNIS_WEATHER_KEYS) that differs from NFL's by omitting
    weather_freezing and including weather_hot."""

    def test_outdoor_with_weather_emits_all_keys(self):
        cur = _FakeCursor(
            [
                {
                    "temperature_c": 22.0,
                    "wind_kmh": 8.0,
                    "precipitation_mm": 0.0,
                    "humidity_pct": 55.0,
                    "is_indoor": False,
                }
            ]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(ftennis.TENNIS_WEATHER_KEYS)

    def test_indoor_emits_all_keys(self):
        cur = _FakeCursor(
            [{"temperature_c": 24.0, "wind_kmh": 0.0, "precipitation_mm": 0.0, "humidity_pct": 50.0, "is_indoor": True}]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(ftennis.TENNIS_WEATHER_KEYS)

    def test_no_weather_known_outdoor_venue_emits_all_keys(self):
        cur = _FakeCursor([None, {"is_indoor": False}])
        out = ftennis.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(ftennis.TENNIS_WEATHER_KEYS)

    def test_no_weather_unknown_venue_emits_all_keys(self):
        cur = _FakeCursor([None, None])
        out = ftennis.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(ftennis.TENNIS_WEATHER_KEYS)


class TestFetchWeatherOutdoor:
    def test_mild_outdoor_match_values_populated(self):
        cur = _FakeCursor(
            [
                {
                    "temperature_c": 22.0,
                    "wind_kmh": 8.0,
                    "precipitation_mm": 0.0,
                    "humidity_pct": 55.0,
                    "is_indoor": False,
                }
            ]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_temp_c"] == 22.0
        assert out["weather_wind_kmh"] == 8.0
        assert out["weather_precip_mm"] == 0.0
        assert out["weather_humidity_pct"] == 55.0
        assert out["weather_indoor"] == 0.0
        assert out["weather_high_wind"] == 0.0
        assert out["weather_wet"] == 0.0
        assert out["weather_hot"] == 0.0

    def test_hot_flag_fires_above_32_c(self):
        cur = _FakeCursor(
            [{"temperature_c": 35.0, "wind_kmh": 5.0, "precipitation_mm": 0.0, "humidity_pct": 30.0, "is_indoor": False}]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_hot"] == 1.0

    def test_hot_flag_does_not_fire_at_32_c(self):
        cur = _FakeCursor(
            [{"temperature_c": 32.0, "wind_kmh": 5.0, "precipitation_mm": 0.0, "humidity_pct": 30.0, "is_indoor": False}]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_hot"] == 0.0

    def test_emits_no_freezing_feature_key(self):
        # Tennis canonical set omits freezing; sport-specific schema
        # divergence locked here.
        cur = _FakeCursor(
            [{"temperature_c": -3.0, "wind_kmh": 5.0, "precipitation_mm": 0.0, "humidity_pct": 70.0, "is_indoor": False}]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert "weather_freezing" not in out


class TestFetchWeatherIndoor:
    def test_indoor_venue_numerics_are_none(self):
        cur = _FakeCursor(
            [{"temperature_c": 24.0, "wind_kmh": 0.0, "precipitation_mm": 0.0, "humidity_pct": 50.0, "is_indoor": True}]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] == 1.0
        assert out["weather_temp_c"] is None
        assert out["weather_wind_kmh"] is None
        assert out["weather_hot"] is None

    def test_no_weather_indoor_venue(self):
        cur = _FakeCursor([None, {"is_indoor": True}])
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] == 1.0
        assert out["weather_temp_c"] is None


class TestFetchWeatherMissing:
    def test_no_weather_known_outdoor_venue(self):
        cur = _FakeCursor([None, {"is_indoor": False}])
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] == 0.0
        assert out["weather_temp_c"] is None

    def test_no_weather_unknown_venue(self):
        cur = _FakeCursor([None, None])
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] is None
        assert out["weather_temp_c"] is None

    def test_partial_nulls_keep_absent_numerics_as_none(self):
        cur = _FakeCursor(
            [{"temperature_c": 33.0, "wind_kmh": None, "precipitation_mm": None, "humidity_pct": None, "is_indoor": False}]
        )
        out = ftennis.fetch_weather(cur, "match-1")
        assert out["weather_temp_c"] == 33.0
        assert out["weather_hot"] == 1.0
        assert out["weather_wind_kmh"] is None
        assert out["weather_high_wind"] is None
        assert out["weather_precip_mm"] is None
        assert out["weather_wet"] is None
        assert out["weather_humidity_pct"] is None
