"""Unit tests for the NHL live-season enrichment (audit 2026-10-25).

Locks the pure rules the live `enrich_nhl_stats` task and the serve-time
goalie projection depend on:
  * fixture attach/dedup — an NHL-API game attaches to the existing
    ESPN-created fixture instead of minting a second row,
  * accent-insensitive team identity ('Montreal' == 'Montréal'),
  * season resolution for --season current,
  * the probable-starter rule,
  * play-by-play team orientation for 5v5 rows.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


lnh = _load("load_nhl_historical", "load_nhl_historical.py")
cfn = _load("compute_features_nhl", "compute_features_nhl.py")
b5 = _load("backfill_nhl_5v5_stats", "backfill_nhl_5v5_stats.py")

UTC = dt.timezone.utc


def _t(y, mo, d, h=0, mi=0):
    return dt.datetime(y, mo, d, h, mi, tzinfo=UTC)


# ── Team identity ────────────────────────────────────────────────────


class TestTeamMatchKey:
    def test_accent_insensitive(self):
        assert lnh.team_match_key("Montréal Canadiens") == lnh.team_match_key("Montreal Canadiens")

    def test_case_space_and_dots(self):
        assert lnh.team_match_key("  St. Louis   Blues ") == lnh.team_match_key("st louis blues")

    def test_different_clubs_differ(self):
        assert lnh.team_match_key("Utah Mammoth") != lnh.team_match_key("Utah Utah Hockey Club")

    def test_empty(self):
        assert lnh.team_match_key(None) == ""


# ── Season resolution ────────────────────────────────────────────────


class TestSeasons:
    def test_october_is_new_season(self):
        assert lnh.current_nhl_season(dt.date(2026, 10, 25)) == "2026-2027"

    def test_spring_is_previous_start_year(self):
        assert lnh.current_nhl_season(dt.date(2027, 4, 1)) == "2026-2027"

    def test_june_playoffs(self):
        assert lnh.current_nhl_season(dt.date(2027, 6, 20)) == "2026-2027"

    def test_resolve_current_and_passthrough(self):
        assert lnh.resolve_season_arg("current", dt.date(2026, 10, 25)) == "2026-2027"
        assert lnh.resolve_season_arg("2024-2025") == "2024-2025"
        assert lnh.resolve_season_arg(None) is None

    def test_season_from_api(self):
        assert lnh.season_from_api(20262027) == "2026-2027"
        assert lnh.season_from_api("bad") is None


# ── Fixture attach / dedup ───────────────────────────────────────────


def _row(id_, when, status="finished", **ext):
    return {"id": id_, "match_date": when, "status": status, "external_ids": ext}


class TestChooseFixture:
    def test_exact_nhl_id_wins_even_if_far(self):
        rows = [_row("espn", _t(2026, 10, 24, 23)), _row("mine", _t(2026, 10, 20), nhl="2026020168")]
        assert lnh.choose_fixture(rows, _t(2026, 10, 24, 23), 2026020168)["id"] == "mine"

    def test_attaches_to_espn_row_with_shifted_kickoff(self):
        # ESPN realigned the kickoff by 30 minutes — still the same game.
        rows = [_row("espn", _t(2026, 10, 24, 23, 30), espn="401892539")]
        assert lnh.choose_fixture(rows, _t(2026, 10, 24, 23), 2026020168)["id"] == "espn"

    def test_none_when_outside_tolerance(self):
        # Same pairing a day later is a different game (2020-21 back-to-backs).
        rows = [_row("next_day", _t(2026, 10, 25, 23))]
        assert lnh.choose_fixture(rows, _t(2026, 10, 24, 23), 1) is None

    def test_closest_wins(self):
        rows = [_row("far", _t(2026, 10, 24, 14)), _row("near", _t(2026, 10, 24, 22))]
        assert lnh.choose_fixture(rows, _t(2026, 10, 24, 23), 1)["id"] == "near"

    def test_finished_beats_stale_scheduled_placeholder_on_tie(self):
        rows = [
            _row("placeholder", _t(2026, 10, 24, 22), status="scheduled"),
            _row("real", _t(2026, 10, 25, 0), status="finished", espn="1"),
        ]
        assert lnh.choose_fixture(rows, _t(2026, 10, 24, 23), 1)["id"] == "real"

    def test_never_steals_a_row_owned_by_another_nhl_game(self):
        rows = [_row("other", _t(2026, 10, 24, 23), nhl="2026020001")]
        assert lnh.choose_fixture(rows, _t(2026, 10, 24, 23), 2026020168) is None

    def test_naive_datetimes_treated_as_utc(self):
        rows = [_row("espn", dt.datetime(2026, 10, 24, 23))]
        assert lnh.choose_fixture(rows, _t(2026, 10, 24, 23), 5)["id"] == "espn"


# ── Probable starter ─────────────────────────────────────────────────


class TestPickProbableStarter:
    def test_no_history(self):
        assert cfn.pick_probable_starter([], _t(2026, 10, 25)) is None

    def test_most_starts_wins(self):
        recent = [("B", _t(2026, 10, 20)), ("A", _t(2026, 10, 18)), ("A", _t(2026, 10, 16))]
        assert cfn.pick_probable_starter(recent, _t(2026, 10, 25)) == "A"

    def test_tie_goes_to_most_recent_start(self):
        recent = [("B", _t(2026, 10, 20)), ("A", _t(2026, 10, 18))]
        assert cfn.pick_probable_starter(recent, _t(2026, 10, 25)) == "B"

    def test_back_to_back_rests_the_number_one(self):
        recent = [("A", _t(2026, 10, 24, 23)), ("A", _t(2026, 10, 22)), ("B", _t(2026, 10, 20))]
        assert cfn.pick_probable_starter(recent, _t(2026, 10, 25, 23)) == "B"

    def test_back_to_back_after_backup_started_returns_number_one(self):
        recent = [("B", _t(2026, 10, 24, 23)), ("A", _t(2026, 10, 22)), ("A", _t(2026, 10, 20))]
        assert cfn.pick_probable_starter(recent, _t(2026, 10, 25, 23)) == "A"

    def test_back_to_back_with_single_goalie_keeps_him(self):
        recent = [("A", _t(2026, 10, 24, 23)), ("A", _t(2026, 10, 22))]
        assert cfn.pick_probable_starter(recent, _t(2026, 10, 25, 23)) == "A"

    def test_null_goalies_ignored(self):
        recent = [(None, _t(2026, 10, 24)), ("A", _t(2026, 10, 20))]
        assert cfn.pick_probable_starter(recent, _t(2026, 10, 25, 23)) == "A"


class TestGoalieModes:
    def test_none_mode_skips_lookup(self):
        assert cfn._find_starting_goalie(None, "t", _t(2026, 10, 25), "none") is None

    def test_unknown_mode_is_loud(self):
        try:
            cfn._find_starting_goalie(None, "t", _t(2026, 10, 25), "bogus")
        except ValueError:
            return
        raise AssertionError("unknown goalie mode should raise")


# ── 5v5 orientation ──────────────────────────────────────────────────


class TestMapPbpTeamIds:
    def test_maps_to_fixture_team_ids(self):
        payload = {"homeTeam": {"id": 8}, "awayTeam": {"id": 12}}
        assert b5.map_pbp_team_ids(payload, "espn-montreal", "car") == {8: "espn-montreal", 12: "car"}

    def test_missing_blocks(self):
        assert b5.map_pbp_team_ids({}, "h", "a") == {}
