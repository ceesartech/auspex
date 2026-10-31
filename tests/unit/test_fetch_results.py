"""Unit tests for the results-ingestion path in fetch_upcoming.py
(--results mode, audit doc §2.1).

Locks the behaviors the graders depend on:
  * only completed post-state competitions are recorded,
  * team sports store real scores; tennis/MMA store winner-flag 1/0
    (the corpus convention),
  * NHL games decided past regulation write metadata.regulation_winner
    = 'tie' (grading_outcomes.nhl_regulation_outcome reads it),
  * the fixtures path still skips non-'pre' events (the two paths
    split on state, never double-handle),
  * a 'cancelled' row is never flipped to 'finished' (the upsert runs
    for real against SQLite, as in test_promote_raw).
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming_results", "fetch_upcoming.py")
_REAL_INSERT_FINISHED_MATCH = fu.insert_finished_match


def _team_per_name(cur, cfg, name, league_id):
    """ensure_team stub: one id per name (a single id for both sides is now
    refused as a team playing itself)."""
    return f"team-{name}"


def _comp(
    state="post", completed=True, detail="Final", home_score="3", away_score="1", home_winner=None, away_winner=None
):
    home = {"homeAway": "home", "team": {"displayName": "Home FC"}, "score": home_score}
    away = {"homeAway": "away", "team": {"displayName": "Away FC"}, "score": away_score}
    if home_winner is not None:
        home["winner"] = home_winner
    if away_winner is not None:
        away["winner"] = away_winner
    return {
        "competitors": [home, away],
        "date": "2026-07-12T18:00:00Z",
        "status": {"type": {"state": state, "completed": completed, "detail": detail}},
        "venue": {"fullName": "Test Park"},
    }


class TestFinalScores:
    def test_team_sport_numeric_scores(self):
        cfg = fu.SPORT_CONFIGS["soccer"]
        comp = _comp()
        parts = fu._competition_parts(comp, cfg)
        home, away = parts[0], parts[1]
        assert fu._final_scores(cfg, home, away) == (3, 1)

    def test_individual_sport_winner_flag(self):
        cfg = fu.SPORT_CONFIGS["tennis"]
        comp = _comp(home_winner=False, away_winner=True, home_score=None, away_score=None)
        # tennis competitors use 'athlete'; reuse team shape via displayName
        home, away = comp["competitors"]
        assert fu._final_scores(cfg, home, away) == (0, 1)

    def test_individual_no_winner_flag_returns_none(self):
        cfg = fu.SPORT_CONFIGS["mma"]
        home = {"score": None}
        away = {"score": None}
        assert fu._final_scores(cfg, home, away) is None

    def test_team_sport_unparseable_score_returns_none(self):
        cfg = fu.SPORT_CONFIGS["soccer"]
        assert fu._final_scores(cfg, {"score": None}, {"score": "2"}) is None


class TestRecordResult:
    def _run(self, cfg, comp):
        cur = MagicMock()
        # ensure_team + insert_finished_match hit the DB — stub them.
        fu.ensure_team = MagicMock(side_effect=_team_per_name)
        captured = {}

        def fake_insert(cur, cfg, league_id, home_id, away_id, match_dt, venue, hs, as_, meta, halftime=None):
            captured.update(home_score=hs, away_score=as_, meta=meta)
            return 1

        fu.insert_finished_match = fake_insert
        ok = fu._record_result(cur, cfg, "league-uuid", comp)
        return ok, captured

    def test_skips_pre_state(self):
        cfg = fu.SPORT_CONFIGS["soccer"]
        ok, _ = self._run(cfg, _comp(state="pre", completed=False))
        assert ok is False

    def test_skips_in_progress(self):
        cfg = fu.SPORT_CONFIGS["soccer"]
        ok, _ = self._run(cfg, _comp(state="in", completed=False))
        assert ok is False

    def test_records_completed_final(self):
        cfg = fu.SPORT_CONFIGS["soccer"]
        ok, captured = self._run(cfg, _comp())
        assert ok is True
        assert (captured["home_score"], captured["away_score"]) == (3, 1)
        assert captured["meta"]["result_source"] == "espn_scoreboard"

    def test_nhl_overtime_writes_regulation_tie(self):
        # 'tie' is the regulation model's label and the training query's
        # value; the pre-2026-10 'draw' could never be predicted correctly.
        cfg = fu.SPORT_CONFIGS["nhl"]
        ok, captured = self._run(cfg, _comp(detail="Final/OT", home_score="4", away_score="3"))
        assert ok is True
        assert captured["meta"]["regulation_winner"] == "tie"

    def test_nhl_regulation_win_writes_winner(self):
        cfg = fu.SPORT_CONFIGS["nhl"]
        ok, captured = self._run(cfg, _comp(detail="Final", home_score="4", away_score="3"))
        assert ok is True
        assert captured["meta"]["regulation_winner"] == "home"


class TestFixturesPathUnchanged:
    def test_process_competition_still_skips_post(self):
        cfg = fu.SPORT_CONFIGS["soccer"]
        cur = MagicMock()
        assert fu._process_competition(cur, cfg, "league-uuid", _comp(state="post")) is False


def _goal(team_id, clock, shootout=False, scoring=True):
    return {
        "scoringPlay": scoring,
        "shootout": shootout,
        "team": {"id": team_id},
        "clock": {"displayValue": clock},
    }


def _soccer_comp(home_score, away_score, details):
    comp = _comp(home_score=str(home_score), away_score=str(away_score))
    comp["competitors"][0]["id"] = "10"
    comp["competitors"][1]["id"] = "20"
    comp["details"] = details
    return comp


class TestHalftimeScores:
    def _ht(self, comp):
        home, away = comp["competitors"]
        final = (int(home["score"]), int(away["score"]))
        return fu.halftime_scores(comp, home, away, final)

    def test_first_half_includes_stoppage_and_own_goals(self):
        # Own goals arrive credited to the side they benefit.
        comp = _soccer_comp(
            2,
            1,
            [_goal("10", "17'"), _goal("20", "45'+2'"), _goal("10", "46'"), {"scoringPlay": False, "clock": {}}],
        )
        assert self._ht(comp) == (1, 1)

    def test_shootout_kicks_are_not_goals(self):
        comp = _soccer_comp(1, 1, [_goal("10", "30'"), _goal("20", "88'"), _goal("10", "120'", shootout=True)])
        assert self._ht(comp) == (1, 0)

    def test_goalless_draw_without_plays(self):
        assert self._ht(_soccer_comp(0, 0, [])) == (0, 0)

    def test_plays_that_do_not_reproduce_the_final_are_rejected(self):
        # One goal missing from the feed: grading four HT markets off it would be a guess.
        assert self._ht(_soccer_comp(2, 0, [_goal("10", "12'")])) is None

    def test_unknown_team_or_clock_is_rejected(self):
        assert self._ht(_soccer_comp(1, 0, [_goal("99", "12'")])) is None
        assert self._ht(_soccer_comp(1, 0, [_goal("10", "")])) is None

    def test_missing_competitor_ids(self):
        comp = _comp()
        home, away = comp["competitors"]
        assert fu.halftime_scores(comp, home, away, (3, 1)) is None


class TestRecordResultHalftime:
    def _record(self, monkeypatch, cfg, comp):
        monkeypatch.setattr(fu, "ensure_team", MagicMock(side_effect=_team_per_name))
        captured = {}

        def fake_insert(cur, cfg, league_id, home_id, away_id, match_dt, venue, hs, as_, meta, halftime=None):
            captured.update(meta=meta, halftime=halftime)
            return 1

        monkeypatch.setattr(fu, "insert_finished_match", fake_insert)
        assert fu._record_result(MagicMock(), cfg, "league-uuid", comp) is True
        return captured

    def test_soccer_result_carries_halftime_to_the_columns(self, monkeypatch):
        comp = _soccer_comp(2, 1, [_goal("10", "17'"), _goal("20", "50'"), _goal("10", "80'")])
        got = self._record(monkeypatch, fu.SPORT_CONFIGS["soccer"], comp)
        assert got["halftime"] == (1, 0)
        assert "halftime_home_score" not in got["meta"]  # the retired metadata keys

    def test_soccer_result_without_usable_plays_has_no_halftime(self, monkeypatch):
        got = self._record(monkeypatch, fu.SPORT_CONFIGS["soccer"], _soccer_comp(2, 1, [_goal("10", "17'")]))
        assert got["halftime"] is None

    def test_nhl_result_has_no_halftime(self, monkeypatch):
        got = self._record(monkeypatch, fu.SPORT_CONFIGS["nhl"], _soccer_comp(4, 3, [_goal("10", "17'")]))
        assert got["halftime"] is None


class TestInsertFinishedMatchHalftime:
    def _sql_and_params(self, halftime):
        cur = MagicMock()
        kickoff = datetime(2026, 10, 25, 15, 0, tzinfo=timezone.utc)
        # the real function: other tests here replace fu.insert_finished_match
        _REAL_INSERT_FINISHED_MATCH(
            cur, fu.SPORT_CONFIGS["soccer"], "lg", "h", "a", kickoff, None, 2, 1, {}, halftime=halftime
        )
        return cur.execute.call_args.args

    def test_halftime_fills_the_columns(self):
        sql, params = self._sql_and_params((1, 0))
        assert "half_time_home, half_time_away" in sql and params[-2:] == (1, 0)

    def test_an_existing_pair_is_never_overwritten_or_split(self):
        sql, _ = self._sql_and_params((1, 0))
        assert sql.count("CASE WHEN matches.half_time_home IS NULL") == 2

    def test_no_halftime_writes_nulls(self):
        _, params = self._sql_and_params(None)
        assert params[-2:] == (None, None)


class _SqliteCursor:
    """Just enough of a RealDictCursor over SQLite for insert_finished_match.
    SQLite 3.35+ speaks the same INSERT ... ON CONFLICT DO UPDATE ... WHERE
    ... RETURNING; only the placeholder, NOW() and the jsonb casts are
    translated (the metadata merge becomes a text concat, which these tests
    do not read)."""

    def __init__(self, conn: sqlite3.Connection):
        self._cur = conn.cursor()

    def execute(self, sql, params=()):
        sql = sql.replace("%s", "?").replace("NOW()", "CURRENT_TIMESTAMP").replace("::jsonb", "")
        self._cur.execute(sql, [p.isoformat() if isinstance(p, datetime) else p for p in params])

    def fetchone(self):
        row = self._cur.fetchone()
        return None if row is None else {"id": row[0]}


RESULT_KICKOFF = datetime(2025, 12, 5, 19, 0, tzinfo=timezone.utc)


@pytest.fixture
def results_db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE matches (
            id INTEGER PRIMARY KEY,
            league_id TEXT, home_team_id TEXT, away_team_id TEXT, match_date TEXT,
            status TEXT CHECK (status IN ('scheduled', 'live', 'finished', 'postponed', 'cancelled')),
            venue TEXT, season TEXT, home_score INTEGER, away_score INTEGER, metadata TEXT,
            half_time_home INTEGER, half_time_away INTEGER, updated_at TEXT,
            UNIQUE (home_team_id, away_team_id, match_date)
        )
        """
    )
    yield conn
    conn.close()


class TestInsertFinishedMatchLeavesCancelledRows:
    """Bastia v Red Star (2025-12-05, ESPN fra.2 747597) was abandoned at 58'
    and awarded, yet ESPN reports STATUS_FULL_TIME 0-0. An operator recorded
    'cancelled' on purpose; a deep results backfill must not flip it to
    'finished' and hand the graders a 0-0."""

    def _record(self, db, home_score=0, away_score=0):
        return _REAL_INSERT_FINISHED_MATCH(
            _SqliteCursor(db),
            fu.SPORT_CONFIGS["soccer"],
            "lg",
            "bastia",
            "red-star",
            RESULT_KICKOFF,
            None,
            home_score,
            away_score,
            {"result_detail": "FT", "result_source": "espn_scoreboard"},
        )

    def _existing(self, db, status):
        db.execute(
            "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season) "
            "VALUES ('lg', 'bastia', 'red-star', ?, ?, '2025-2026')",
            (RESULT_KICKOFF.isoformat(), status),
        )

    def _row(self, db):
        return db.execute("SELECT status, home_score, away_score FROM matches").fetchone()

    def test_a_cancelled_row_is_left_alone_and_reported_as_not_written(self, results_db):
        self._existing(results_db, "cancelled")
        assert self._record(results_db) == 0
        assert self._row(results_db) == ("cancelled", None, None)

    def test_a_scheduled_row_becomes_finished(self, results_db):
        self._existing(results_db, "scheduled")
        assert self._record(results_db, 2, 1) == 1
        assert self._row(results_db) == ("finished", 2, 1)

    def test_a_postponed_row_still_becomes_finished(self, results_db):
        self._existing(results_db, "postponed")
        assert self._record(results_db, 2, 1) == 1
        assert self._row(results_db) == ("finished", 2, 1)

    def test_an_unseen_fixture_is_inserted_finished(self, results_db):
        assert self._record(results_db, 3, 1) == 1
        assert self._row(results_db) == ("finished", 3, 1)

    def test_the_guard_mirrors_promote_raw(self):
        cur = MagicMock()
        _REAL_INSERT_FINISHED_MATCH(cur, fu.SPORT_CONFIGS["soccer"], "lg", "h", "a", RESULT_KICKOFF, None, 1, 0, {})
        sql = " ".join(cur.execute.call_args.args[0].split())
        assert "WHERE matches.status <> 'cancelled' RETURNING id" in sql


class TestFindTeamNeverWrites:
    def test_find_team_returns_none_without_inserting(self, monkeypatch):
        cur = MagicMock()
        cur.fetchone.return_value = None
        cur.fetchall.return_value = []
        monkeypatch.setattr(fu.team_identity, "find_team_accent_insensitive", lambda *a: None)
        assert fu.find_team(cur, fu.SPORT_CONFIGS["soccer"], "Nowhere FC", "league-uuid") is None
        assert not any("INSERT" in str(c.args[0]) for c in cur.execute.call_args_list)
