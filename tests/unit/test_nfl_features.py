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


# ── Weather integration (Phase 14 — Open-Meteo retry) ───────────────


class _FakeCursor:
    """Minimal cursor stub for fetch_weather tests. The function makes
    one query against match_weather_latest, optionally a second against
    venue_coords when no weather row is found. We seed fetchone() with
    a queue of return values."""

    def __init__(self, responses: list):
        self._responses = list(responses)

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self._responses.pop(0) if self._responses else None


class TestNoWeatherFeaturesInDefaults:
    """The whole point of the Phase 14 retry: NEUTRAL_DEFAULTS must
    contain NO weather keys. The reverted v3 attempt (commit 17b3eb6)
    added weather_temp_c=10.0 etc. as defaults, which the GBDT
    overfit as a "missing-data sentinel" when 39% of training rows
    used the defaults. With OMIT-on-missing + the EXISTS-gate on
    NFL training queries, weather features either come from real
    data or are simply absent from the dict — never defaulted."""

    def test_no_weather_keys_in_defaults(self):
        weather_keys = [k for k in fnfl.NEUTRAL_DEFAULTS if k.startswith("weather_")]
        assert weather_keys == [], (
            f"weather_* keys in NEUTRAL_DEFAULTS reintroduce the default-bias "
            f"leakage from commit 17b3eb6: {weather_keys}"
        )


class TestWeatherThresholds:
    """Sport-tuned thresholds locked here. Editing one of these is a
    real modelling change — the test is what makes that explicit."""

    def test_high_wind_kmh_is_25(self):
        # ~15 mph — degrades passing accuracy + kicking. Tests fail
        # loudly if someone bumps this without updating the model
        # commentary in the recent-form retro.
        assert fnfl.HIGH_WIND_KMH == 25.0

    def test_wet_precip_mm_is_5(self):
        # 5mm of rain/snow accumulated over a 4-hour kickoff window
        # is the "measurable precipitation during the game" threshold.
        assert fnfl.WET_PRECIP_MM == 5.0

    def test_freezing_temp_c_is_zero(self):
        # NFL-specific (tennis omits it). 0°C affects ball grip +
        # kicker leg strength; iconic cold-weather games (Lambeau in
        # January) cross this threshold.
        assert fnfl.FREEZING_TEMP_C == 0.0


class TestCanonicalWeatherKeySet:
    """Every fetch_weather return MUST contain every key in
    NFL_WEATHER_KEYS. This is the v4b invariant that fixes the v4a
    inference shape mismatch — train and predict DataFrames now have
    identical columns regardless of weather data availability.

    Missing values are None (pandas reads as NaN); GBDTs handle NaN
    as missing-direction in split learning, semantically distinct
    from any literal default value, so the v3 default-bias problem
    (commit 17b3eb6) cannot recur."""

    def test_outdoor_with_weather_emits_all_keys(self):
        cur = _FakeCursor(
            [
                {
                    "temperature_c": 15.0,
                    "wind_kmh": 12.0,
                    "precipitation_mm": 0.0,
                    "humidity_pct": 50.0,
                    "is_indoor": False,
                }
            ]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(fnfl.NFL_WEATHER_KEYS)

    def test_indoor_with_weather_row_emits_all_keys(self):
        cur = _FakeCursor(
            [{"temperature_c": 22.0, "wind_kmh": 0.0, "precipitation_mm": 0.0, "humidity_pct": 40.0, "is_indoor": True}]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(fnfl.NFL_WEATHER_KEYS)

    def test_no_weather_known_indoor_venue_emits_all_keys(self):
        cur = _FakeCursor([None, {"is_indoor": True}])
        out = fnfl.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(fnfl.NFL_WEATHER_KEYS)

    def test_no_weather_outdoor_venue_emits_all_keys(self):
        # v4a returned {} here; v4b emits every key so inference
        # shape matches training shape unconditionally.
        cur = _FakeCursor([None, {"is_indoor": False}])
        out = fnfl.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(fnfl.NFL_WEATHER_KEYS)

    def test_no_weather_unknown_venue_emits_all_keys(self):
        # Same v4a → v4b shape fix for the unknown-venue case.
        cur = _FakeCursor([None, None])
        out = fnfl.fetch_weather(cur, "match-1")
        assert set(out.keys()) == set(fnfl.NFL_WEATHER_KEYS)


class TestFetchWeatherOutdoor:
    """Outdoor venue with a weather row — every numeric + flag has a
    real value, weather_indoor=0."""

    def test_mild_outdoor_match_values_populated(self):
        # 15°C / 12 km/h wind / 0mm precip / 50% humidity — fair fall
        # game. All numerics populated, all flags 0.
        cur = _FakeCursor(
            [
                {
                    "temperature_c": 15.0,
                    "wind_kmh": 12.0,
                    "precipitation_mm": 0.0,
                    "humidity_pct": 50.0,
                    "is_indoor": False,
                }
            ]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_temp_c"] == 15.0
        assert out["weather_wind_kmh"] == 12.0
        assert out["weather_precip_mm"] == 0.0
        assert out["weather_humidity_pct"] == 50.0
        assert out["weather_indoor"] == 0.0
        assert out["weather_high_wind"] == 0.0
        assert out["weather_wet"] == 0.0
        assert out["weather_freezing"] == 0.0

    def test_high_wind_flag_fires_above_25_kmh(self):
        cur = _FakeCursor(
            [{"temperature_c": 10.0, "wind_kmh": 30.0, "precipitation_mm": 0.0, "humidity_pct": 60.0, "is_indoor": False}]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_high_wind"] == 1.0

    def test_high_wind_flag_does_not_fire_at_25_kmh(self):
        # Strict > threshold (not >=) — exactly 25 km/h is borderline,
        # not "high wind".
        cur = _FakeCursor(
            [{"temperature_c": 10.0, "wind_kmh": 25.0, "precipitation_mm": 0.0, "humidity_pct": 60.0, "is_indoor": False}]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_high_wind"] == 0.0

    def test_freezing_flag_fires_below_zero(self):
        cur = _FakeCursor(
            [
                {
                    "temperature_c": -2.0,
                    "wind_kmh": 10.0,
                    "precipitation_mm": 0.0,
                    "humidity_pct": 60.0,
                    "is_indoor": False,
                }
            ]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_freezing"] == 1.0

    def test_wet_flag_fires_above_5_mm(self):
        cur = _FakeCursor(
            [{"temperature_c": 8.0, "wind_kmh": 15.0, "precipitation_mm": 8.0, "humidity_pct": 90.0, "is_indoor": False}]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_wet"] == 1.0


class TestFetchWeatherIndoor:
    """Indoor games — every key emitted, only weather_indoor=1.0
    populated, all numerics + dependent flags stay None."""

    def test_indoor_with_weather_row_numerics_are_none(self):
        # Stale snapshot from before is_indoor was flagged is dropped
        # — venue signal is authoritative.
        cur = _FakeCursor(
            [{"temperature_c": 22.0, "wind_kmh": 0.0, "precipitation_mm": 0.0, "humidity_pct": 40.0, "is_indoor": True}]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] == 1.0
        assert out["weather_temp_c"] is None
        assert out["weather_wind_kmh"] is None
        assert out["weather_freezing"] is None

    def test_no_weather_indoor_venue_numerics_are_none(self):
        cur = _FakeCursor([None, {"is_indoor": True}])
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] == 1.0
        assert out["weather_temp_c"] is None


class TestFetchWeatherMissing:
    """No weather row, no actionable venue info: every numeric stays
    None. weather_indoor reflects what we KNOW (0 for known-outdoor,
    None for unknown-venue) — never invented."""

    def test_no_weather_known_outdoor_venue(self):
        # We know it's outdoor (venue seeded with is_indoor=false)
        # but lack measurements — weather_indoor=0, numerics NaN.
        cur = _FakeCursor([None, {"is_indoor": False}])
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] == 0.0
        assert out["weather_temp_c"] is None
        assert out["weather_wind_kmh"] is None

    def test_no_weather_unknown_venue(self):
        # We don't even know if it's indoor or outdoor — every key
        # None including weather_indoor.
        cur = _FakeCursor([None, None])
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_indoor"] is None
        assert out["weather_temp_c"] is None

    def test_partial_nulls_keep_absent_numerics_as_none(self):
        # Open-Meteo can return temp but NULL precipitation. Present
        # numerics populate; absent ones stay None; dependent flags
        # stay None too (no flag without a numeric to derive from).
        cur = _FakeCursor(
            [{"temperature_c": 5.0, "wind_kmh": None, "precipitation_mm": None, "humidity_pct": None, "is_indoor": False}]
        )
        out = fnfl.fetch_weather(cur, "match-1")
        assert out["weather_temp_c"] == 5.0
        assert out["weather_freezing"] == 0.0  # derived from temp
        assert out["weather_wind_kmh"] is None
        assert out["weather_high_wind"] is None
        assert out["weather_precip_mm"] is None
        assert out["weather_wet"] is None
        assert out["weather_humidity_pct"] is None
