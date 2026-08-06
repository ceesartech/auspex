"""Unit tests for backfill_understat_xg — normalization, season parsing,
payload parsing. No network or DB."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from unittest.mock import Mock

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bx = _load("backfill_understat_xg", REPO / "scripts" / "backfill_understat_xg.py")


class TestNorm:
    def test_strips_accents_punct_and_filler(self):
        assert bx.norm("1. FC Köln") == "koln"
        assert bx.norm("AS Monaco") == "monaco"
        assert bx.norm("Athletic Club") == "ath bilbao"

    def test_aliases_applied_after_normalization(self):
        assert bx.norm("Wolverhampton Wanderers") == "wolves"
        assert bx.norm("Paris Saint Germain") == "paris sg"

    def test_plain_names_stable(self):
        assert bx.norm("Fulham") == "fulham"
        assert bx.norm("Real Madrid") == "real madrid"


class TestParseSeasons:
    def test_range_and_list_and_single(self):
        assert bx.parse_seasons("2016-2018") == [2016, 2017, 2018]
        assert bx.parse_seasons("2022,2024") == [2022, 2024]
        assert bx.parse_seasons("2024") == [2024]


class TestFetchSeason:
    def _resp(self, payload):
        m = Mock()
        m.json.return_value = payload
        m.raise_for_status.return_value = None
        return m

    def test_parses_real_shape(self):
        # Verbatim structure from getLeagueData/EPL/2024.
        payload = {
            "dates": [
                {
                    "id": "26602",
                    "isResult": True,
                    "h": {"id": "89", "title": "Manchester United"},
                    "a": {"id": "228", "title": "Fulham"},
                    "goals": {"h": "1", "a": "0"},
                    "xG": {"h": "2.04268", "a": "0.418711"},
                    "datetime": "2024-08-16 19:00:00",
                }
            ]
        }
        session = Mock(get=Mock(return_value=self._resp(payload)))
        rows = bx.fetch_season(session, "EPL", 2024)
        assert rows == [
            {
                "home": "Manchester United",
                "away": "Fulham",
                "date": date(2024, 8, 16),
                "home_goals": 1,
                "away_goals": 0,
                "home_xg": 2.04268,
                "away_xg": 0.418711,
            }
        ]

    def test_skips_unplayed_and_xg_less_rows(self):
        payload = {
            "dates": [
                {"isResult": False, "h": {"title": "A"}, "a": {"title": "B"}},
                {
                    "isResult": True,
                    "h": {"title": "A"},
                    "a": {"title": "B"},
                    "goals": {"h": "1", "a": "0"},
                    "xG": {"h": None, "a": None},
                    "datetime": "2024-08-16 19:00:00",
                },
            ]
        }
        session = Mock(get=Mock(return_value=self._resp(payload)))
        assert bx.fetch_season(session, "EPL", 2024) == []
