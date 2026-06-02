"""Unit tests for load_nfl_historical — pure ESPN-event parsing.

Same coverage shape as test_load_nba_historical: parse_score
quirks, finished-game extraction with state filters, date-range
iteration, CLI defaults. NFL's ESPN payload is structurally
identical to NBA's so the tests are parallel.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
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


_load("fetch_upcoming", "fetch_upcoming.py")
lnh = _load("load_nfl_historical", "load_nfl_historical.py")


def _make_event(
    state: str = "post",
    home_score: str = "31",
    away_score: str = "24",
    home_name: str = "Kansas City Chiefs",
    away_name: str = "Buffalo Bills",
    date_str: str = "2025-01-12T18:00Z",
    venue: str = "Arrowhead Stadium",
):
    """Minimal ESPN scoreboard event payload — only the fields the
    parser reaches into. Mirrors the real API's structure so a future
    response-shape drift breaks the test loudly."""
    return {
        "date": date_str,
        "status": {"type": {"state": state}},
        "competitions": [
            {
                "venue": {"fullName": venue},
                "competitors": [
                    {"homeAway": "home", "score": home_score, "team": {"displayName": home_name}},
                    {"homeAway": "away", "score": away_score, "team": {"displayName": away_name}},
                ],
            }
        ],
    }


# ── parse_score ─────────────────────────────────────────────────────


class TestParseScore:
    def test_string_int_parses(self):
        assert lnh.parse_score({"score": "31"}) == 31

    def test_missing_score_returns_none(self):
        # Postponed / cancelled games drop the score field entirely.
        assert lnh.parse_score({}) is None

    def test_empty_string_returns_none(self):
        assert lnh.parse_score({"score": ""}) is None

    def test_non_numeric_returns_none(self):
        # Defensive — if ESPN ever ships a non-numeric string we
        # refuse rather than mis-grading downstream.
        assert lnh.parse_score({"score": "TBD"}) is None


# ── extract_finished_game ───────────────────────────────────────────


class TestExtractFinishedGame:
    def test_finished_game_round_trips(self):
        out = lnh.extract_finished_game(_make_event())
        assert out["home_name"] == "Kansas City Chiefs"
        assert out["away_name"] == "Buffalo Bills"
        assert out["home_score"] == 31
        assert out["away_score"] == 24
        assert out["venue"] == "Arrowhead Stadium"
        assert out["match_dt"].year == 2025
        assert out["match_dt"].month == 1

    def test_pregame_skipped(self):
        # Backfill shouldn't write scheduled rows — that's
        # fetch_upcoming's job.
        assert lnh.extract_finished_game(_make_event(state="pre")) is None

    def test_in_progress_skipped(self):
        # Half-time games don't have final scores.
        assert lnh.extract_finished_game(_make_event(state="in")) is None

    def test_postponed_skipped(self):
        # NFL has occasional postponements (weather, COVID legacy).
        # Don't write rows for them.
        assert lnh.extract_finished_game(_make_event(state="postponed")) is None

    def test_final_state_alias_accepted(self):
        out = lnh.extract_finished_game(_make_event(state="final"))
        assert out is not None
        assert out["home_score"] == 31

    def test_missing_score_drops_event(self):
        # "Final" but no score = data quality bug. Refuse to write
        # a NULL-score row.
        ev = _make_event()
        ev["competitions"][0]["competitors"][0]["score"] = ""
        assert lnh.extract_finished_game(ev) is None

    def test_missing_team_drops_event(self):
        ev = _make_event()
        ev["competitions"][0]["competitors"][0]["team"] = {}
        assert lnh.extract_finished_game(ev) is None

    def test_missing_homeaway_field_drops_event(self):
        # If neither competitor is marked home/away we can't make a
        # match row.
        ev = _make_event()
        for c in ev["competitions"][0]["competitors"]:
            c.pop("homeAway", None)
        assert lnh.extract_finished_game(ev) is None

    def test_bad_date_format_drops_event(self):
        # ESPN format drift — skip rather than insert garbage.
        ev = _make_event(date_str="not-a-date")
        assert lnh.extract_finished_game(ev) is None

    def test_no_competitions_array_drops_event(self):
        ev = {"status": {"type": {"state": "post"}}, "date": "2025-01-12T18:00Z"}
        assert lnh.extract_finished_game(ev) is None

    def test_venue_optional(self):
        # NFL has international + neutral-site games (London, Mexico
        # City) that occasionally drop the venue field.
        ev = _make_event(venue=None)
        ev["competitions"][0]["venue"] = {}
        out = lnh.extract_finished_game(ev)
        assert out is not None
        assert out["venue"] is None


# ── iter_dates ──────────────────────────────────────────────────────


class TestIterDates:
    def test_inclusive_range(self):
        days = list(lnh.iter_dates(date(2024, 9, 5), date(2024, 9, 7)))
        assert days == [date(2024, 9, 5), date(2024, 9, 6), date(2024, 9, 7)]

    def test_single_day(self):
        assert list(lnh.iter_dates(date(2025, 1, 12), date(2025, 1, 12))) == [date(2025, 1, 12)]

    def test_reversed_range_yields_nothing(self):
        # Defensive against misuse — main() rejects this but
        # iter_dates also short-circuits so no infinite loops.
        assert list(lnh.iter_dates(date(2025, 1, 12), date(2024, 9, 5))) == []


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_dates_required(self):
        with pytest.raises(SystemExit):
            lnh.parse_args([])

    def test_full_args(self):
        args = lnh.parse_args(
            [
                "--start-date",
                "2022-09-01",
                "--end-date",
                "2025-02-15",
                "--database-url",
                "postgresql://x",
            ]
        )
        assert args.start_date == "2022-09-01"
        assert args.end_date == "2025-02-15"
        assert args.database_url == "postgresql://x"
