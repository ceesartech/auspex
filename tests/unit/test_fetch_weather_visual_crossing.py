"""Unit tests for scripts/fetch_weather_visual_crossing.py — JSON
shape parsing, match-window summary, and edge cases. No network
or DB."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
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


vc = _load("fetch_weather_visual_crossing", "fetch_weather_visual_crossing.py")


def _build_vc_response(hourly: list[dict]) -> dict:
    """Build a minimal Visual Crossing timeline response — one `days`
    entry containing a list of hourly samples."""
    return {
        "queryCost": 1,
        "latitude": 33.0,
        "longitude": -97.0,
        "days": [
            {
                "datetime": "2024-09-08",
                "hours": hourly,
            }
        ],
    }


@pytest.mark.unit
class TestMatchWindowSummaryBasics:
    """A standard match at 13:00 local time pulls hours 11:00..15:00
    (±2h window around the match hour). Test that values are
    averaged correctly per element."""

    def test_simple_average(self):
        hourly = []
        for hour in range(11, 16):
            hourly.append(
                {
                    "datetime": f"{hour:02d}:00:00",
                    "temp": float(hour),
                    "windspeed": 20.0,
                    "precip": 0.5,
                    "humidity": 60.0,
                    "conditions": "Partly cloudy",
                }
            )
        match_dt = datetime(2024, 9, 8, 13, 0, tzinfo=timezone.utc)
        out = vc.match_window_summary(_build_vc_response(hourly), match_dt)
        # Temp 11..15 → mean 13.0.
        assert out["temperature_c"] == 13.0
        # Wind constant 20.0 → mean 20.0.
        assert out["wind_kmh"] == 20.0
        # Precip summed (not averaged) over the window → 0.5 × 5 = 2.5.
        assert out["precipitation_mm"] == 2.5
        assert out["humidity_pct"] == 60.0
        assert out["conditions"] == "Partly cloudy"

    def test_window_pulls_only_in_range(self):
        # Hours 9, 10, 11, 12, 13, 14, 15, 16, 17. For match_dt at
        # 13:00 the parser keeps 11..15 (the ±2h window).
        hourly = []
        for hour in range(9, 18):
            hourly.append(
                {
                    "datetime": f"{hour:02d}:00:00",
                    "temp": 10.0 if 11 <= hour <= 15 else 100.0,
                    "windspeed": 0.0,
                    "precip": 0.0,
                    "humidity": 0.0,
                    "conditions": None,
                }
            )
        match_dt = datetime(2024, 9, 8, 13, 0, tzinfo=timezone.utc)
        out = vc.match_window_summary(_build_vc_response(hourly), match_dt)
        # If out-of-window hours leaked in, temp would shoot up.
        assert out["temperature_c"] == 10.0


@pytest.mark.unit
class TestMatchWindowSummaryEdges:
    def test_empty_days_returns_empty(self):
        assert vc.match_window_summary({"days": []}, datetime(2024, 9, 8, 13)) == {}

    def test_no_days_key_returns_empty(self):
        # Defensive: VC sometimes returns errors as a body without
        # `days`. Parser must not raise.
        assert vc.match_window_summary({}, datetime(2024, 9, 8, 13)) == {}

    def test_empty_hours_returns_empty(self):
        body = _build_vc_response([])
        assert vc.match_window_summary(body, datetime(2024, 9, 8, 13)) == {}

    def test_no_hours_in_window_returns_empty(self):
        # All hours far from the match hour → no window match.
        hourly = [
            {"datetime": "01:00:00", "temp": 0.0, "windspeed": 0.0, "precip": 0.0, "humidity": 0.0, "conditions": None},
            {"datetime": "02:00:00", "temp": 0.0, "windspeed": 0.0, "precip": 0.0, "humidity": 0.0, "conditions": None},
        ]
        body = _build_vc_response(hourly)
        out = vc.match_window_summary(body, datetime(2024, 9, 8, 20))
        assert out == {}

    def test_handles_missing_field_per_hour(self):
        # Some hours may be missing one field (sensor gap). Parser
        # should average over the hours that DO have it.
        hourly = [
            {
                "datetime": "11:00:00",
                "temp": 10.0,
                "windspeed": 15.0,
                "precip": 0.0,
                "humidity": 50.0,
                "conditions": "Clear",
            },
            {"datetime": "12:00:00", "windspeed": 20.0, "precip": 0.5, "humidity": 55.0, "conditions": None},  # no temp
            {
                "datetime": "13:00:00",
                "temp": 20.0,
                "windspeed": 25.0,
                "precip": 0.5,
                "humidity": 60.0,
                "conditions": "Cloudy",
            },
            {
                "datetime": "14:00:00",
                "temp": 22.0,
                "windspeed": 25.0,
                "precip": 0.5,
                "humidity": 65.0,
                "conditions": None,
            },
            {
                "datetime": "15:00:00",
                "temp": 18.0,
                "windspeed": 20.0,
                "precip": 0.0,
                "humidity": 70.0,
                "conditions": None,
            },
        ]
        body = _build_vc_response(hourly)
        out = vc.match_window_summary(body, datetime(2024, 9, 8, 13))
        # temp = (10 + 20 + 22 + 18) / 4 = 17.5
        assert out["temperature_c"] == 17.5
        # First-non-None condition picked deterministically.
        assert out["conditions"] == "Clear"

    def test_precip_summed_not_averaged(self):
        # A 1-hour rain burst at the match hour should NOT be
        # diluted by 4 dry hours in the average. The parser
        # SUMS precip across the window.
        hourly = [
            {
                "datetime": f"{h:02d}:00:00",
                "temp": 20.0,
                "windspeed": 0.0,
                "precip": (5.0 if h == 13 else 0.0),
                "humidity": 50.0,
                "conditions": None,
            }
            for h in range(11, 16)
        ]
        out = vc.match_window_summary(
            _build_vc_response(hourly),
            datetime(2024, 9, 8, 13),
        )
        assert out["precipitation_mm"] == 5.0

    def test_first_conditions_string_picked(self):
        # When multiple hours have conditions, the FIRST non-None
        # one wins so the output is deterministic.
        hourly = [
            {
                "datetime": "11:00:00",
                "temp": 10.0,
                "windspeed": 0.0,
                "precip": 0.0,
                "humidity": 50.0,
                "conditions": None,
            },
            {
                "datetime": "12:00:00",
                "temp": 10.0,
                "windspeed": 0.0,
                "precip": 0.0,
                "humidity": 50.0,
                "conditions": "Foggy",
            },
            {
                "datetime": "13:00:00",
                "temp": 10.0,
                "windspeed": 0.0,
                "precip": 0.0,
                "humidity": 50.0,
                "conditions": "Clear",
            },
        ]
        out = vc.match_window_summary(
            _build_vc_response(hourly),
            datetime(2024, 9, 8, 12),
        )
        # 12-2 = 10 floor, 12+2 = 14 ceiling, hours 11,12,13 included.
        # First non-None conditions = "Foggy".
        assert out["conditions"] == "Foggy"


@pytest.mark.unit
class TestApiKeyHelper:
    def test_returns_env_var(self, monkeypatch):
        monkeypatch.setenv("VISUAL_CROSSING_API_KEY", "test-key-12345")
        assert vc._api_key() == "test-key-12345"

    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.delenv("VISUAL_CROSSING_API_KEY", raising=False)
        assert vc._api_key() is None
