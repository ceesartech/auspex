"""Unit tests for fetch_weather — pure parser helpers.

The HTTP and DB layers are integration territory (covered by a
follow-up e2e test against the docker postgres). These tests exercise
the pure helpers:

  * normalize_venue: same loose normalization shape as
    teams.normalized_name.
  * match_window_summary: 4-hour-window aggregation reducing
    Open-Meteo's hourly arrays to scalar features.
  * WMO_CODE_TO_CONDITIONS: the weather-code → label lookup.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load("fetch_weather", "fetch_weather.py")


class TestNormalizeVenue:
    def test_lowercases_and_trims(self):
        assert fw.normalize_venue("  Lambeau Field  ") == "lambeau field"

    def test_collapses_whitespace(self):
        assert fw.normalize_venue("Allianz   Arena") == "allianz arena"

    def test_handles_none_gracefully(self):
        assert fw.normalize_venue(None) == ""

    def test_handles_empty_string(self):
        assert fw.normalize_venue("") == ""


class TestMatchWindowSummary:
    def _hourly(self, **overrides):
        """Default hourly payload covering 24h around match_dt."""
        return {
            "time": [f"2024-09-08T{h:02d}:00:00Z" for h in range(24)],
            "temperature_2m": [10.0 + h * 0.5 for h in range(24)],
            "wind_speed_10m": [5.0 for _ in range(24)],
            "precipitation": [0.5 for _ in range(24)],
            "relative_humidity_2m": [70.0 for _ in range(24)],
            "weather_code": [61 for _ in range(24)],  # rain code
            **overrides,
        }

    def test_window_is_centered_on_match_dt(self):
        # 3h before kickoff to 1h after = window indices [13, 14, 15, 16, 17] for 16:00 kickoff.
        match_dt = datetime(2024, 9, 8, 16, 0, tzinfo=timezone.utc)
        summary = fw.match_window_summary(self._hourly(), match_dt)
        # Temperature is 10 + 0.5h linear. Window avg over indices
        # 13-17 = (16.5 + 17 + 17.5 + 18 + 18.5) / 5 = 17.5
        assert abs(summary["temperature_c"] - 17.5) < 0.01

    def test_precipitation_is_summed_not_averaged(self):
        # 0.5mm/h × 5 hours in window = 2.5mm total — sum, not mean.
        match_dt = datetime(2024, 9, 8, 16, 0, tzinfo=timezone.utc)
        summary = fw.match_window_summary(self._hourly(), match_dt)
        assert abs(summary["precipitation_mm"] - 2.5) < 0.01

    def test_conditions_uses_most_common_code(self):
        # Mixed: 3 hours of rain (code 61), 2 hours of clear (code 0).
        # Most common in window → rain.
        hourly = self._hourly()
        # Override the 5-hour window (indices 13-17) to mix codes.
        hourly["weather_code"] = [61] * 24
        hourly["weather_code"][16] = 0
        hourly["weather_code"][17] = 0
        match_dt = datetime(2024, 9, 8, 16, 0, tzinfo=timezone.utc)
        summary = fw.match_window_summary(hourly, match_dt)
        # Window indices 13-17: codes [61, 61, 61, 0, 0] → most common 61 → rain.
        assert summary["conditions"] == "rain"

    def test_empty_hourly_returns_empty(self):
        match_dt = datetime(2024, 9, 8, 16, 0, tzinfo=timezone.utc)
        assert fw.match_window_summary({}, match_dt) == {}

    def test_match_outside_hourly_returns_empty(self):
        # Match date with no overlap in the hourly window → no values.
        match_dt = datetime(2099, 1, 1, 12, 0, tzinfo=timezone.utc)
        assert fw.match_window_summary(self._hourly(), match_dt) == {}


class TestWmoCodeMap:
    def test_clear_codes(self):
        assert fw.WMO_CODE_TO_CONDITIONS[0] == "clear"
        assert fw.WMO_CODE_TO_CONDITIONS[1] == "mainly_clear"

    def test_rain_codes(self):
        for code in (61, 63, 65):
            assert fw.WMO_CODE_TO_CONDITIONS[code] == "rain"
        for code in (80, 81, 82):
            assert fw.WMO_CODE_TO_CONDITIONS[code] == "rain_showers"

    def test_snow_codes(self):
        for code in (71, 73, 75, 77):
            assert fw.WMO_CODE_TO_CONDITIONS[code] == "snow"

    def test_thunderstorm_codes(self):
        for code in (95, 96, 99):
            assert fw.WMO_CODE_TO_CONDITIONS[code] == "thunderstorm"


class TestCli:
    def test_default_days(self):
        args = fw.parse_args(["--database-url", "x"])
        assert args.days == 14
        assert args.backfill_days == 14
        assert args.sport is None

    def test_sport_filter(self):
        args = fw.parse_args(["--sport", "nfl", "--database-url", "x"])
        assert args.sport == "nfl"
