"""Unit tests for the results-ingestion path in fetch_upcoming.py
(--results mode, audit doc §2.1).

Locks the behaviors the graders depend on:
  * only completed post-state competitions are recorded,
  * team sports store real scores; tennis/MMA store winner-flag 1/0
    (the corpus convention),
  * NHL games decided past regulation write metadata.regulation_winner
    = 'tie' (grading_outcomes.nhl_regulation_outcome reads it),
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
    def _meta(self, cfg, comp):
        fu.ensure_team = MagicMock(return_value="team-uuid")
        captured = {}

        def fake_insert(cur, cfg, league_id, home_id, away_id, match_dt, venue, hs, as_, meta):
            captured.update(meta=meta)
            return 1

        fu.insert_finished_match = fake_insert
        assert fu._record_result(MagicMock(), cfg, "league-uuid", comp) is True
        return captured["meta"]

    def test_soccer_result_carries_halftime(self):
        comp = _soccer_comp(2, 1, [_goal("10", "17'"), _goal("20", "50'"), _goal("10", "80'")])
        meta = self._meta(fu.SPORT_CONFIGS["soccer"], comp)
        assert (meta["halftime_home_score"], meta["halftime_away_score"]) == (1, 0)

    def test_soccer_result_without_usable_plays_has_no_halftime(self):
        meta = self._meta(fu.SPORT_CONFIGS["soccer"], _soccer_comp(2, 1, [_goal("10", "17'")]))
        assert "halftime_home_score" not in meta

    def test_nhl_result_has_no_halftime(self):
        comp = _soccer_comp(4, 3, [_goal("10", "17'")])
        meta = self._meta(fu.SPORT_CONFIGS["nhl"], comp)
        assert "halftime_home_score" not in meta


class TestFindTeamNeverWrites:
    def test_find_team_returns_none_without_inserting(self, monkeypatch):
        cur = MagicMock()
        cur.fetchone.return_value = None
        cur.fetchall.return_value = []
        monkeypatch.setattr(fu.team_identity, "find_team_accent_insensitive", lambda *a: None)
        assert fu.find_team(cur, fu.SPORT_CONFIGS["soccer"], "Nowhere FC", "league-uuid") is None
        assert not any("INSERT" in str(c.args[0]) for c in cur.execute.call_args_list)
