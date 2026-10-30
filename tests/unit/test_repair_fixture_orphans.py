"""Unit tests for repair_fixture_orphans — the plan (which orphan merges
where, which is deleted, which is left alone) and the safety invariants of
its SQL. No DB."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
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


rfo = _load("repair_fixture_orphans", "repair_fixture_orphans.py")

KICK = datetime(2026, 10, 28, 18, 15, tzinfo=timezone.utc)


def _cand(cid, hours_before=1.0, has_recs=False):
    return {
        "id": cid,
        "home_team_id": "fritz",
        "away_team_id": "hijikata",
        "match_date": KICK - timedelta(hours=hours_before),
        "status": "scheduled",
        "has_recs": has_recs,
    }


def _twin(tid, when=KICK, score=(1, 0)):
    return {
        "twin_id": tid,
        "home_team_id": "fritz",
        "away_team_id": "hijikata",
        "home_score": score[0],
        "away_score": score[1],
        "match_date": when,
    }


class TestPlan:
    def test_every_slide_of_one_fixture_merges_onto_its_finished_twin(self):
        cands = [_cand("o1", 38), _cand("o2", 1.25), _cand("o3", 0.75), _cand("o4", 0.17)]
        p = rfo.plan(cands, lambda c: [_twin("real")])
        assert p["groups"] == {"real": ["o1", "o2", "o3", "o4"]}
        assert p["kickoff"]["real"] == KICK
        assert p["counts"]["merge"] == 4 and p["counts"]["twins"] == 1

    def test_a_pairing_that_never_played_is_deleted(self):
        p = rfo.plan([_cand("redrawn")], lambda c: [])
        assert p["no_twin"] == ["redrawn"] and p["groups"] == {}

    def test_ambiguous_twins_are_left_alone(self):
        twins = [_twin("a", score=(1, 0)), _twin("b", when=KICK + timedelta(hours=2), score=(0, 1))]
        p = rfo.plan([_cand("o")], lambda c: twins)
        assert p["groups"] == {} and p["no_twin"] == []
        assert p["counts"]["skipped_ambiguous_scores"] == 1

    def test_an_orphan_holding_recs_is_never_touched(self):
        calls = []
        p = rfo.plan([_cand("o", has_recs=True)], lambda c: calls.append(c) or [_twin("real")])
        assert p["groups"] == {} and p["no_twin"] == [] and calls == []
        assert p["counts"]["kept_has_recs"] == 1


class TestSqlInvariants:
    def test_only_pre_kickoff_rows_move(self):
        assert '"timestamp" < %(kickoff)s' in rfo.MOVE_ODDS_SQL
        assert "captured_at < %(kickoff)s" in rfo.MOVE_SNAPSHOTS_SQL
        assert "COALESCE(updated_at, created_at) < %(kickoff)s" in rfo.MOVE_PREDICTIONS_SQL

    def test_the_twins_own_rows_always_win(self):
        for sql in (rfo.MOVE_ODDS_SQL, rfo.MOVE_SNAPSHOTS_SQL, rfo.MOVE_PREDICTIONS_SQL):
            assert "NOT EXISTS" in sql and "t.match_id = %(twin)s::uuid" in sql

    def test_odds_keep_the_freshest_price_per_current_price_key(self):
        # the key mirrors uq_odds_current_price_key, so a move can never violate it
        assert "COALESCE(line, -1), COALESCE(is_opening, false), COALESCE(is_live, false)" in rfo.MOVE_ODDS_SQL
        assert '"timestamp" DESC' in rfo.MOVE_ODDS_SQL

    def test_delete_spares_recs_history_and_played_rows(self):
        sql = rfo.DELETE_ORPHANS_SQL
        assert "betting_recommendations" in sql and "betting_history" in sql
        assert "m.status IN ('scheduled', 'cancelled')" in sql


class TestArgs:
    def test_dry_run_is_the_default(self):
        assert rfo.parse_args(["--sport", "tennis"]).apply is False

    def test_cannot_race_the_results_look_back(self):
        with pytest.raises(rfo.UnsafeRequest):
            rfo.parse_args(["--sport", "tennis", "--stale-after-hours", "24"])

    def test_twin_window_is_bounded(self):
        with pytest.raises(rfo.UnsafeRequest):
            rfo.parse_args(["--sport", "tennis", "--twin-window-hours", "500"])

    def test_team_sports_are_refused(self):
        with pytest.raises(SystemExit):
            rfo.parse_args(["--sport", "nba"])
