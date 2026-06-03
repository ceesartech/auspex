"""Unit tests for load_mma_historical — pure ESPN-event parsing.

ESPN MMA events are CARDS (one event = one fight night) with
competitions[] holding individual fights. No groupings layer like
tennis. Competitors carry `winner` boolean + `athlete.displayName`
but lack `homeAway` (positional ordering).
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
lmh = _load("load_mma_historical", "load_mma_historical.py")


def _make_fight(
    state: str = "post",
    home_winner: bool = True,
    away_winner: bool = False,
    home_name: str = "Israel Adesanya",
    away_name: str = "Dricus Du Plessis",
    date_str: str = "2024-08-17T22:00Z",
    venue: str = "Sphere",
):
    """Minimal ESPN MMA competition (leaf-level fight)."""
    return {
        "date": date_str,
        "status": {"type": {"state": state}},
        "venue": {"fullName": venue},
        "competitors": [
            {"winner": home_winner, "athlete": {"displayName": home_name}},
            {"winner": away_winner, "athlete": {"displayName": away_name}},
        ],
    }


class TestExtractFinishedFight:
    def test_finished_fight_round_trips(self):
        out = lmh.extract_finished_fight(_make_fight())
        assert out["home_name"] == "Israel Adesanya"
        assert out["away_name"] == "Dricus Du Plessis"
        assert out["home_score"] == 1
        assert out["away_score"] == 0
        assert out["venue"] == "Sphere"
        assert out["match_dt"].year == 2024

    def test_away_winner_round_trips(self):
        out = lmh.extract_finished_fight(_make_fight(home_winner=False, away_winner=True))
        assert out["home_score"] == 0
        assert out["away_score"] == 1

    def test_pregame_skipped(self):
        assert lmh.extract_finished_fight(_make_fight(state="pre")) is None

    def test_in_progress_skipped(self):
        assert lmh.extract_finished_fight(_make_fight(state="in")) is None

    def test_cancelled_skipped(self):
        # Weight miss / injury withdrawal → never happens, not a result.
        assert lmh.extract_finished_fight(_make_fight(state="cancelled")) is None

    def test_final_state_alias_accepted(self):
        out = lmh.extract_finished_fight(_make_fight(state="final"))
        assert out is not None

    def test_draw_dropped(self):
        # MMA draws happen (~1%). Both winners false → skip; the
        # 2-class model can't represent draws anyway.
        assert lmh.extract_finished_fight(_make_fight(home_winner=False, away_winner=False)) is None

    def test_both_winners_true_dropped(self):
        # Data corruption case. Skip.
        assert lmh.extract_finished_fight(_make_fight(home_winner=True, away_winner=True)) is None

    def test_missing_athlete_drops(self):
        comp = _make_fight()
        comp["competitors"][0]["athlete"] = {}
        assert lmh.extract_finished_fight(comp) is None

    def test_fewer_than_two_competitors_drops(self):
        comp = _make_fight()
        comp["competitors"] = comp["competitors"][:1]
        assert lmh.extract_finished_fight(comp) is None

    def test_bad_date_format_drops(self):
        comp = _make_fight(date_str="not-a-date")
        assert lmh.extract_finished_fight(comp) is None

    def test_no_homeaway_uses_positional(self):
        # MMA competitors never carry homeAway. Index 0 = home, 1 = away
        # is the only signal. Regression guard: if a future ESPN change
        # adds homeAway and we add it back to the dispatch, ensure the
        # behavior is still consistent.
        comp = _make_fight()
        for c in comp["competitors"]:
            assert "homeAway" not in c
        out = lmh.extract_finished_fight(comp)
        assert out["home_name"] == "Israel Adesanya"


class TestIterDates:
    def test_inclusive_range(self):
        days = list(lmh.iter_dates(date(2024, 9, 5), date(2024, 9, 7)))
        assert days == [date(2024, 9, 5), date(2024, 9, 6), date(2024, 9, 7)]

    def test_single_day(self):
        assert list(lmh.iter_dates(date(2024, 9, 9), date(2024, 9, 9))) == [date(2024, 9, 9)]

    def test_reversed_range_yields_nothing(self):
        assert list(lmh.iter_dates(date(2024, 9, 9), date(2024, 9, 5))) == []


class TestCli:
    def test_dates_required(self):
        with pytest.raises(SystemExit):
            lmh.parse_args([])

    def test_default_league_is_ufc(self):
        args = lmh.parse_args(["--start-date", "2024-01-01", "--end-date", "2024-12-31"])
        assert args.league == "ufc"

    def test_invalid_league_rejected(self):
        with pytest.raises(SystemExit):
            lmh.parse_args(["--start-date", "2024-01-01", "--end-date", "2024-12-31", "--league", "bellator"])

    def test_full_args(self):
        args = lmh.parse_args(
            [
                "--start-date",
                "2022-01-01",
                "--end-date",
                "2024-12-31",
                "--league",
                "ufc",
                "--database-url",
                "postgresql://x",
            ]
        )
        assert args.start_date == "2022-01-01"
        assert args.end_date == "2024-12-31"
        assert args.database_url == "postgresql://x"
        assert args.league == "ufc"
