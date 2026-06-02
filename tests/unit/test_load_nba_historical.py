"""Unit tests for load_nba_historical — pure ESPN-event parsing.

DB and HTTP layers aren't exercised here. We focus on the
extract_finished_game / parse_score helpers because those are where
the schema-vs-API mismatches usually bite: ESPN's payload shape has
shifted on us before, and a missed field silently drops thousands of
games during a backfill.
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


# fetch_upcoming exposes the ensure_team / ensure_league helpers + the
# NBA_LEAGUES / SPORT_CONFIGS that load_nba_historical imports.
_load("fetch_upcoming", "fetch_upcoming.py")
lnh = _load("load_nba_historical", "load_nba_historical.py")


def _make_event(
    state: str = "post",
    home_score: str = "108",
    away_score: str = "104",
    home_name: str = "Boston Celtics",
    away_name: str = "Dallas Mavericks",
    date_str: str = "2025-06-13T00:30Z",
    venue: str = "TD Garden",
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
        # ESPN ships scores as strings; the parser must coerce.
        assert lnh.parse_score({"score": "108"}) == 108

    def test_missing_score_returns_none(self):
        # Postponed games drop the score field entirely.
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
        assert out["home_name"] == "Boston Celtics"
        assert out["away_name"] == "Dallas Mavericks"
        assert out["home_score"] == 108
        assert out["away_score"] == 104
        assert out["venue"] == "TD Garden"
        # ISO date with Z → tz-aware datetime
        assert out["match_dt"].year == 2025
        assert out["match_dt"].month == 6

    def test_pregame_skipped(self):
        # Backfill shouldn't write scheduled rows — that's fetch_upcoming's
        # job. We skip cleanly so the caller increments "skipped".
        assert lnh.extract_finished_game(_make_event(state="pre")) is None

    def test_in_progress_skipped(self):
        # Half-time and in-progress games don't have final scores.
        assert lnh.extract_finished_game(_make_event(state="in")) is None

    def test_postponed_skipped(self):
        # Postponed games have state="postponed" — fall outside
        # FINAL_STATES → skipped.
        assert lnh.extract_finished_game(_make_event(state="postponed")) is None

    def test_cancelled_skipped(self):
        assert lnh.extract_finished_game(_make_event(state="cancelled")) is None

    def test_final_state_alias_accepted(self):
        # "post" is canonical, but defensive guard for "final" alias.
        out = lnh.extract_finished_game(_make_event(state="final"))
        assert out is not None
        assert out["home_score"] == 108

    def test_missing_score_drops_event(self):
        # "Final" but no score — almost certainly a data quality bug.
        # Refuse to write a NULL-score row.
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
        # ESPN format drift — we'd rather skip the row than insert
        # with garbage.
        ev = _make_event(date_str="not-a-date")
        assert lnh.extract_finished_game(ev) is None

    def test_no_competitions_array_drops_event(self):
        # Defensive — some scoreboard responses ship empty competitions
        # for placeholder events.
        ev = {"status": {"type": {"state": "post"}}, "date": "2025-06-13T00:30Z"}
        assert lnh.extract_finished_game(ev) is None

    def test_venue_optional(self):
        # ESPN sometimes drops venue for neutral-court games.
        ev = _make_event(venue=None)
        ev["competitions"][0]["venue"] = {}
        out = lnh.extract_finished_game(ev)
        assert out is not None
        assert out["venue"] is None


# ── iter_dates ──────────────────────────────────────────────────────


class TestIterDates:
    def test_inclusive_range(self):
        days = list(lnh.iter_dates(date(2025, 6, 10), date(2025, 6, 12)))
        assert days == [date(2025, 6, 10), date(2025, 6, 11), date(2025, 6, 12)]

    def test_single_day(self):
        assert list(lnh.iter_dates(date(2025, 6, 10), date(2025, 6, 10))) == [date(2025, 6, 10)]

    def test_reversed_range_yields_nothing(self):
        # Defensive — main() rejects this, but iter_dates also
        # short-circuits so a misuse can't loop forever.
        assert list(lnh.iter_dates(date(2025, 6, 11), date(2025, 6, 10))) == []


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_dates_required(self):
        with pytest.raises(SystemExit):
            lnh.parse_args([])

    def test_full_args(self):
        args = lnh.parse_args(
            [
                "--start-date",
                "2024-10-22",
                "--end-date",
                "2025-06-30",
                "--database-url",
                "postgresql://x",
            ]
        )
        assert args.start_date == "2024-10-22"
        assert args.end_date == "2025-06-30"
        assert args.database_url == "postgresql://x"
