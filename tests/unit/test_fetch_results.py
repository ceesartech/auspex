"""Unit tests for the results-ingestion path in fetch_upcoming.py
(--results mode, audit doc §2.1).

Locks the behaviors the graders depend on:
  * only completed post-state competitions are recorded,
  * team sports store real scores; tennis/MMA store winner-flag 1/0
    (the corpus convention),
  * NHL games decided past regulation write metadata.regulation_winner
    = 'draw' (grading_outcomes.nhl_regulation_outcome reads it),
  * the fixtures path still skips non-'pre' events (the two paths
    split on state, never double-handle).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming_results", "fetch_upcoming.py")


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
        fu.ensure_team = MagicMock(return_value="team-uuid")
        captured = {}

        def fake_insert(cur, cfg, league_id, home_id, away_id, match_dt, venue, hs, as_, meta):
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

    def test_nhl_overtime_writes_regulation_draw(self):
        cfg = fu.SPORT_CONFIGS["nhl"]
        ok, captured = self._run(cfg, _comp(detail="Final/OT", home_score="4", away_score="3"))
        assert ok is True
        assert captured["meta"]["regulation_winner"] == "draw"

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
