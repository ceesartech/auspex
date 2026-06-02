"""Unit tests for load_tennis_historical — pure ESPN-event parsing.

Same coverage shape as test_load_nfl_historical: parse_sets quirks,
finished-match extraction with state filters, date-range iteration,
CLI defaults. Tennis ESPN payload uses 'athlete' not 'team' and lacks
homeAway — covered explicitly here.
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
lth = _load("load_tennis_historical", "load_tennis_historical.py")


def _make_event(
    state: str = "post",
    home_score: str = "3",
    away_score: str = "1",
    home_name: str = "Novak Djokovic",
    away_name: str = "Carlos Alcaraz",
    date_str: str = "2024-09-09T18:00Z",
    venue: str = "Arthur Ashe Stadium",
):
    """Minimal ESPN tennis scoreboard event payload — only the fields
    the parser reaches into. Tennis competitors use 'athlete' instead
    of 'team' and have NO homeAway field (positional ordering)."""
    return {
        "date": date_str,
        "status": {"type": {"state": state}},
        "competitions": [
            {
                "venue": {"fullName": venue},
                "competitors": [
                    {"score": home_score, "athlete": {"displayName": home_name}},
                    {"score": away_score, "athlete": {"displayName": away_name}},
                ],
            }
        ],
    }


# ── parse_sets ──────────────────────────────────────────────────────


class TestParseSets:
    def test_string_int_parses(self):
        assert lth.parse_sets({"score": "3"}) == 3

    def test_missing_score_returns_none(self):
        # Retired / walkover matches sometimes drop the score.
        assert lth.parse_sets({}) is None

    def test_empty_string_returns_none(self):
        assert lth.parse_sets({"score": ""}) is None

    def test_non_numeric_returns_none(self):
        # Defensive — if ESPN ever ships "W" for a walkover, refuse
        # rather than crash.
        assert lth.parse_sets({"score": "W"}) is None


# ── extract_finished_match ──────────────────────────────────────────


class TestExtractFinishedMatch:
    def test_finished_match_round_trips(self):
        out = lth.extract_finished_match(_make_event())
        assert out["home_name"] == "Novak Djokovic"
        assert out["away_name"] == "Carlos Alcaraz"
        assert out["home_score"] == 3
        assert out["away_score"] == 1
        assert out["venue"] == "Arthur Ashe Stadium"
        assert out["match_dt"].year == 2024
        assert out["match_dt"].month == 9

    def test_pregame_skipped(self):
        assert lth.extract_finished_match(_make_event(state="pre")) is None

    def test_in_progress_skipped(self):
        assert lth.extract_finished_match(_make_event(state="in")) is None

    def test_postponed_skipped(self):
        # Tennis postponements happen (rain delay across days).
        # Don't write rows for them.
        assert lth.extract_finished_match(_make_event(state="postponed")) is None

    def test_walkover_skipped(self):
        # ESPN tags some walkovers / retirements with non-final states.
        assert lth.extract_finished_match(_make_event(state="walkover")) is None

    def test_retired_match_with_partial_score_skipped(self):
        # If a player retires mid-match the final score on ESPN is
        # often missing or marked with non-numeric values — already
        # tested at parse_sets level; the extract path skips when
        # parse_sets returns None.
        ev = _make_event()
        ev["competitions"][0]["competitors"][0]["score"] = ""
        assert lth.extract_finished_match(ev) is None

    def test_final_state_alias_accepted(self):
        out = lth.extract_finished_match(_make_event(state="final"))
        assert out is not None
        assert out["home_score"] == 3

    def test_missing_athlete_drops_event(self):
        ev = _make_event()
        ev["competitions"][0]["competitors"][0]["athlete"] = {}
        assert lth.extract_finished_match(ev) is None

    def test_fewer_than_two_competitors_drops(self):
        ev = _make_event()
        ev["competitions"][0]["competitors"] = ev["competitions"][0]["competitors"][:1]
        assert lth.extract_finished_match(ev) is None

    def test_bad_date_format_drops_event(self):
        ev = _make_event(date_str="not-a-date")
        assert lth.extract_finished_match(ev) is None

    def test_no_competitions_array_drops_event(self):
        ev = {"status": {"type": {"state": "post"}}, "date": "2024-09-09T18:00Z"}
        assert lth.extract_finished_match(ev) is None

    def test_venue_optional(self):
        # Qualifying rounds + early-round tournament matches occasionally
        # drop the court (venue) field.
        ev = _make_event(venue=None)
        ev["competitions"][0]["venue"] = {}
        out = lth.extract_finished_match(ev)
        assert out is not None
        assert out["venue"] is None

    def test_positional_ordering_no_homeaway_field(self):
        # Tennis events don't carry homeAway. Index 0 → home, 1 → away.
        # If this test breaks because someone added homeAway parsing,
        # tennis matches will go missing.
        ev = _make_event()
        for c in ev["competitions"][0]["competitors"]:
            assert "homeAway" not in c
        out = lth.extract_finished_match(ev)
        assert out["home_name"] == "Novak Djokovic"  # positional


# ── iter_dates ──────────────────────────────────────────────────────


class TestIterDates:
    def test_inclusive_range(self):
        days = list(lth.iter_dates(date(2024, 9, 5), date(2024, 9, 7)))
        assert days == [date(2024, 9, 5), date(2024, 9, 6), date(2024, 9, 7)]

    def test_single_day(self):
        assert list(lth.iter_dates(date(2024, 9, 9), date(2024, 9, 9))) == [date(2024, 9, 9)]

    def test_reversed_range_yields_nothing(self):
        assert list(lth.iter_dates(date(2024, 9, 9), date(2024, 9, 5))) == []


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_dates_required(self):
        with pytest.raises(SystemExit):
            lth.parse_args([])

    def test_default_tour_is_both(self):
        args = lth.parse_args(["--start-date", "2024-01-01", "--end-date", "2024-12-31"])
        assert args.tour == "both"

    def test_atp_only(self):
        args = lth.parse_args(["--start-date", "2024-01-01", "--end-date", "2024-12-31", "--tour", "atp"])
        assert args.tour == "atp"

    def test_wta_only(self):
        args = lth.parse_args(["--start-date", "2024-01-01", "--end-date", "2024-12-31", "--tour", "wta"])
        assert args.tour == "wta"

    def test_invalid_tour_rejected(self):
        with pytest.raises(SystemExit):
            lth.parse_args(["--start-date", "2024-01-01", "--end-date", "2024-12-31", "--tour", "itf"])

    def test_full_args(self):
        args = lth.parse_args(
            [
                "--start-date",
                "2022-01-01",
                "--end-date",
                "2024-12-31",
                "--tour",
                "atp",
                "--database-url",
                "postgresql://x",
            ]
        )
        assert args.start_date == "2022-01-01"
        assert args.end_date == "2024-12-31"
        assert args.database_url == "postgresql://x"
        assert args.tour == "atp"
