"""Unit tests for load_tennis_historical — pure ESPN-event parsing.

ESPN tennis events are TOURNAMENTS, not matches. The structure is:
    events[] (tournaments)
      groupings[] (brackets: men's/women's singles, doubles)
        competitions[] (individual matches)
            competitors[] (athletes with winner + homeAway)

These tests cover the leaf-competition parser (`extract_finished_match`)
plus the helper that flattens the nested structure
(`iter_match_competitions`).

The score-comparison approach used by team sports doesn't apply to
tennis — competitor.score is None. Instead each competitor carries
a `winner` boolean, exactly one of which is True for a finished
match (no draws). The parser sets home_score=1 / away_score=0 (or
vice versa) based on which side won.
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


def _make_competition(
    state: str = "post",
    home_winner: bool = True,
    away_winner: bool = False,
    home_name: str = "Novak Djokovic",
    away_name: str = "Carlos Alcaraz",
    date_str: str = "2024-09-09T18:00Z",
    venue: str = "Arthur Ashe Stadium",
    set_homeaway: bool = True,
):
    """Minimal ESPN tennis competition (leaf-level match) payload —
    only the fields the parser reaches into. Each competitor carries
    a `winner` boolean; `score` is None on tennis (per-set scores
    live in linescores which v1 doesn't parse)."""
    home = {
        "score": None,
        "winner": home_winner,
        "athlete": {"displayName": home_name},
    }
    away = {
        "score": None,
        "winner": away_winner,
        "athlete": {"displayName": away_name},
    }
    if set_homeaway:
        home["homeAway"] = "home"
        away["homeAway"] = "away"
    return {
        "date": date_str,
        "status": {"type": {"state": state}},
        "venue": {"fullName": venue},
        "competitors": [home, away],
    }


def _make_tournament(competitions: list, name: str = "Wimbledon", venue: str | None = None):
    """ESPN tennis event shape — a tournament that nests competitions
    inside groupings. The actual structure has multiple groupings
    (men's/women's singles, doubles); we put all competitions under
    one grouping for tests since the parser doesn't care about the
    grouping label."""
    out = {
        "name": name,
        "groupings": [{"competitions": competitions}],
    }
    if venue:
        out["venue"] = {"fullName": venue}
    return out


# ── iter_match_competitions ─────────────────────────────────────────


class TestIterMatchCompetitions:
    def test_walks_nested_structure(self):
        # Wimbledon-style: one event, 3 groupings, multiple matches each.
        ev = {
            "groupings": [
                {"competitions": [{"id": "1"}, {"id": "2"}]},
                {"competitions": [{"id": "3"}]},
                {"competitions": []},
            ]
        }
        ids = [c["id"] for c in lth.iter_match_competitions(ev)]
        assert ids == ["1", "2", "3"]

    def test_handles_missing_groupings(self):
        # Some ESPN payloads omit groupings (off-day, no matches).
        assert list(lth.iter_match_competitions({})) == []
        assert list(lth.iter_match_competitions({"groupings": []})) == []

    def test_handles_grouping_without_competitions(self):
        # A bracket may exist but have no scheduled matches for the day.
        ev = {"groupings": [{"competitions": None}]}
        assert list(lth.iter_match_competitions(ev)) == []


# ── extract_finished_match ──────────────────────────────────────────


class TestExtractFinishedMatch:
    def test_finished_match_round_trips(self):
        out = lth.extract_finished_match(_make_competition())
        assert out["home_name"] == "Novak Djokovic"
        assert out["away_name"] == "Carlos Alcaraz"
        assert out["home_score"] == 1
        assert out["away_score"] == 0
        assert out["venue"] == "Arthur Ashe Stadium"
        assert out["match_dt"].year == 2024

    def test_away_winner_round_trips(self):
        out = lth.extract_finished_match(_make_competition(home_winner=False, away_winner=True))
        assert out["home_score"] == 0
        assert out["away_score"] == 1

    def test_pregame_skipped(self):
        assert lth.extract_finished_match(_make_competition(state="pre")) is None

    def test_in_progress_skipped(self):
        assert lth.extract_finished_match(_make_competition(state="in")) is None

    def test_postponed_skipped(self):
        assert lth.extract_finished_match(_make_competition(state="postponed")) is None

    def test_walkover_skipped(self):
        # ESPN sometimes tags walkovers / retirements with non-final states.
        assert lth.extract_finished_match(_make_competition(state="walkover")) is None

    def test_final_state_alias_accepted(self):
        out = lth.extract_finished_match(_make_competition(state="final"))
        assert out is not None

    def test_both_winners_true_dropped(self):
        # Data quality bug — exactly one winner is the contract.
        # Both true (or both false) means we can't tell who won;
        # skip rather than insert an ambiguous row.
        assert lth.extract_finished_match(_make_competition(home_winner=True, away_winner=True)) is None
        assert lth.extract_finished_match(_make_competition(home_winner=False, away_winner=False)) is None

    def test_missing_athlete_drops(self):
        comp = _make_competition()
        comp["competitors"][0]["athlete"] = {}
        assert lth.extract_finished_match(comp) is None

    def test_fewer_than_two_competitors_drops(self):
        comp = _make_competition()
        comp["competitors"] = comp["competitors"][:1]
        assert lth.extract_finished_match(comp) is None

    def test_bad_date_format_drops(self):
        comp = _make_competition(date_str="not-a-date")
        assert lth.extract_finished_match(comp) is None

    def test_missing_date_drops(self):
        comp = _make_competition()
        del comp["date"]
        assert lth.extract_finished_match(comp) is None

    def test_homeaway_used_when_present(self):
        # ESPN orders tennis competitors by seed and carries homeAway.
        # Even if positional order differs, homeAway wins.
        comp = _make_competition()
        comp["competitors"].reverse()
        # After reverse: competitors[0] is away (away seed). homeAway
        # value pins it to the right side.
        out = lth.extract_finished_match(comp)
        assert out["home_name"] == "Novak Djokovic"  # found via homeAway
        assert out["away_name"] == "Carlos Alcaraz"

    def test_positional_fallback_when_homeaway_absent(self):
        # If a future ESPN payload drops homeAway, fall back to index
        # 0 = home / 1 = away.
        comp = _make_competition(set_homeaway=False)
        out = lth.extract_finished_match(comp)
        assert out is not None
        assert out["home_name"] == "Novak Djokovic"  # positional

    def test_venue_fallback_to_tournament(self):
        # Match-level venue can be missing on lesser-tracked matches;
        # fall back to the parent tournament's venue.
        comp = _make_competition(venue=None)
        comp["venue"] = {}
        out = lth.extract_finished_match(comp, venue_fallback="All England Club")
        assert out["venue"] == "All England Club"


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


# Suppress the _make_tournament unused-warning — it's a doc helper for
# future tests that exercise the orchestration layer.
_ = _make_tournament
