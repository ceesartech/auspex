"""Unit tests for grade_completed_races — pure helpers + the two
core grading branches (predictions + recs settlement). The DB
helpers are exercised through a fake cursor that captures execute()
calls + lets the test seed fetchone() / fetchall() results.
"""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gcr = _load("grade_completed_races", "grade_completed_races.py")


class _FakeCursor:
    """Captures execute() args + lets tests seed fetchone() and
    fetchall() responses in order. .rowcount is settable per
    execute() call via a parallel responses list."""

    def __init__(self):
        self.executions: list[tuple[str, tuple]] = []
        self._fetchone_responses: list = []
        self._fetchall_responses: list = []
        self._rowcount_responses: list = []
        self.rowcount = 0

    def queue_fetchone(self, *responses):
        self._fetchone_responses.extend(responses)

    def queue_fetchall(self, *responses):
        self._fetchall_responses.extend(responses)

    def queue_rowcount(self, *counts):
        self._rowcount_responses.extend(counts)

    def execute(self, sql, params=None):
        self.executions.append((sql, params or ()))
        if self._rowcount_responses:
            self.rowcount = self._rowcount_responses.pop(0)

    def fetchone(self):
        return self._fetchone_responses.pop(0) if self._fetchone_responses else None

    def fetchall(self):
        return self._fetchall_responses.pop(0) if self._fetchall_responses else []


# ── CLI ─────────────────────────────────────────────────────────────


class TestCli:
    def test_defaults(self):
        args = gcr.parse_args(["--database-url", "x"])
        # 14-day lookback matches the team-sport grader's default —
        # enough to catch late results without re-scanning months
        # of already-graded races each tick.
        assert args.days == 14
        assert args.race_id is None

    def test_race_id_parses(self):
        args = gcr.parse_args(["--race-id", "abc-123", "--database-url", "x"])
        assert args.race_id == "abc-123"

    def test_days_parses_as_int(self):
        args = gcr.parse_args(["--days", "30", "--database-url", "x"])
        assert args.days == 30


# ── fetch_race_outcome / fetch_consensus_favourite ─────────────────


class TestFetchRaceOutcome:
    def test_returns_winner_when_finish_position_one_exists(self):
        cur = _FakeCursor()
        cur.queue_fetchone({"winner_entrant_id": "e1", "winner_horse_name": "Big Boy"})
        out = gcr.fetch_race_outcome(cur, "race-1")
        assert out == {"winner_entrant_id": "e1", "winner_horse_name": "Big Boy"}

    def test_returns_none_when_no_winner(self):
        # Abandoned / void race — no row has finish_position=1.
        # Grader handles the None case by voiding recs + setting
        # is_correct=FALSE on predictions.
        cur = _FakeCursor()
        cur.queue_fetchone(None)
        assert gcr.fetch_race_outcome(cur, "race-1") is None


class TestFetchConsensusFavourite:
    def test_returns_highest_confidence_entrant(self):
        cur = _FakeCursor()
        cur.queue_fetchone({"entrant_id": "fav-1"})
        assert gcr.fetch_consensus_favourite(cur, "race-1") == "fav-1"

    def test_returns_none_when_no_predictions(self):
        # Backfilled finished race with no predictions yet — happens
        # when precompute hadn't run before the result landed.
        cur = _FakeCursor()
        cur.queue_fetchone(None)
        assert gcr.fetch_consensus_favourite(cur, "race-1") is None


# ── grade_race_predictions: per-row + per-race semantics ───────────


class TestGradeRacePredictions:
    def test_winner_known_favourite_won_sets_is_correct_true(self):
        # Consensus pick matched the actual winner. UPDATE runs with
        # actual_outcome derived per-row (1 for winner, 0 elsewhere)
        # and is_correct=TRUE for every row in the race.
        cur = _FakeCursor()
        cur.queue_rowcount(7)  # 7 entrants in the race
        n = gcr.grade_race_predictions(cur, "race-1", winner_entrant_id="w1", favourite_entrant_id="w1")
        assert n == 7
        sql, params = cur.executions[0]
        assert "UPDATE race_predictions" in sql
        assert "is_correct     = %s" in sql
        # Winner id, is_correct=True, race_id
        assert params == ("w1", True, "race-1")

    def test_winner_known_favourite_lost_sets_is_correct_false(self):
        cur = _FakeCursor()
        cur.queue_rowcount(5)
        gcr.grade_race_predictions(cur, "race-1", winner_entrant_id="w1", favourite_entrant_id="other")
        _, params = cur.executions[0]
        assert params == ("w1", False, "race-1")

    def test_winner_known_no_favourite_treats_as_lost(self):
        # No predictions written for the race ⇒ favourite_entrant_id
        # is None. We still grade per-entrant outcomes but is_correct
        # falls back to False — "the model picked the winner" is
        # vacuously false when the model didn't pick anyone.
        cur = _FakeCursor()
        cur.queue_rowcount(8)
        gcr.grade_race_predictions(cur, "race-1", winner_entrant_id="w1", favourite_entrant_id=None)
        _, params = cur.executions[0]
        assert params == ("w1", False, "race-1")

    def test_winner_unknown_sets_is_correct_false_only(self):
        # Void / abandoned race — no winner. Mark is_correct=FALSE so
        # the row doesn't sit in the ungraded queue forever but leave
        # actual_outcome NULL so downstream accuracy queries can
        # filter it out.
        cur = _FakeCursor()
        cur.queue_rowcount(9)
        n = gcr.grade_race_predictions(cur, "race-1", winner_entrant_id=None, favourite_entrant_id="fav")
        assert n == 9
        sql, _ = cur.executions[0]
        assert "UPDATE race_predictions" in sql
        # actual_outcome NOT touched in the void-race branch.
        assert "actual_outcome" not in sql
        assert "is_correct = FALSE" in sql


# ── settle_recommendations: status + profit_loss math ──────────────


class TestSettleRecommendations:
    def _rec(self, *, rec_id="r1", entrant_id="e1", selection="Big Boy", stake="10.00", odds="4.50", scratched=False):
        return {
            "rec_id": rec_id,
            "entrant_id": entrant_id,
            "selection": selection,
            "recommended_stake": Decimal(stake),
            "odds_at_recommendation": Decimal(odds),
            "bet_type": "win",
            "entrant_scratched": scratched,
            "finish_position": None,
        }

    def test_winning_rec_settles_with_positive_profit(self):
        cur = _FakeCursor()
        cur.queue_fetchall([self._rec(entrant_id="w1", stake="10.00", odds="4.50")])
        cur._fetchone_responses.append({"n": 10})
        counts = gcr.settle_recommendations(cur, "race-1", "w1", "Big Boy")
        assert counts == {"won": 1, "lost": 0, "void": 0}
        # Find the UPDATE call.
        update_sql, params = cur.executions[-1]
        assert "UPDATE race_recommendations" in update_sql
        # status, actual_result, profit_loss, rec_id
        status, actual_result, profit_loss, rec_id = params
        assert status == "won"
        assert actual_result == "Big Boy"
        # 10 × (4.5 - 1) = 35.00
        assert profit_loss == Decimal("35.00")

    def test_losing_rec_settles_with_negative_profit(self):
        cur = _FakeCursor()
        cur.queue_fetchall([self._rec(entrant_id="loser", stake="8.50", odds="3.00")])
        cur._fetchone_responses.append({"n": 10})
        counts = gcr.settle_recommendations(cur, "race-1", "winner", "Big Boy")
        assert counts == {"won": 0, "lost": 1, "void": 0}
        _, params = cur.executions[-1]
        status, actual_result, profit_loss, _ = params
        assert status == "lost"
        assert actual_result == "Big Boy"
        # Lose the stake.
        assert profit_loss == Decimal("-8.50")

    def test_scratched_entrant_voids_with_zero_profit(self):
        # Entrant scratched after the rec was written — refund. Still
        # records the winning horse name so the user can see what
        # actually won.
        cur = _FakeCursor()
        cur.queue_fetchall([self._rec(entrant_id="other", scratched=True, stake="5.00")])
        cur._fetchone_responses.append({"n": 10})
        counts = gcr.settle_recommendations(cur, "race-1", "winner", "Big Boy")
        assert counts == {"won": 0, "lost": 0, "void": 1}
        _, params = cur.executions[-1]
        status, actual_result, profit_loss, _ = params
        assert status == "void"
        assert actual_result == "Big Boy"
        assert profit_loss == Decimal("0")

    def test_race_with_no_winner_voids_all_recs(self):
        # Race abandoned — refund every pending rec regardless of
        # selection.
        cur = _FakeCursor()
        cur.queue_fetchall(
            [
                self._rec(rec_id="r1", entrant_id="a", stake="10.00"),
                self._rec(rec_id="r2", entrant_id="b", stake="20.00"),
            ]
        )
        cur._fetchone_responses.append({"n": 10})
        counts = gcr.settle_recommendations(cur, "race-1", None, None)
        assert counts == {"won": 0, "lost": 0, "void": 2}
        # Both UPDATEs marked void with profit_loss=0.
        for ex in cur.executions:
            sql, params = ex
            if "UPDATE race_recommendations" in sql:
                status, actual_result, profit_loss, _ = params
                assert status == "void"
                assert actual_result == "void"
                assert profit_loss == Decimal("0")

    def test_no_recs_returns_empty_counts(self):
        # Nothing pending — no settlement work to do.
        cur = _FakeCursor()
        cur.queue_fetchall([])
        cur._fetchone_responses.append({"n": 10})
        counts = gcr.settle_recommendations(cur, "race-1", "winner", "Big Boy")
        assert counts == {"won": 0, "lost": 0, "void": 0}

    def test_zero_stake_or_odds_treated_as_zero(self):
        # Defensive: if a rec slipped through with NULL stake / odds,
        # we shouldn't crash on Decimal × None. Coerce to 0 silently.
        cur = _FakeCursor()
        rec = self._rec(entrant_id="winner")
        rec["recommended_stake"] = None
        rec["odds_at_recommendation"] = None
        cur.queue_fetchall([rec])
        cur._fetchone_responses.append({"n": 10})
        counts = gcr.settle_recommendations(cur, "race-1", "winner", "Big Boy")
        assert counts == {"won": 1, "lost": 0, "void": 0}
        _, params = cur.executions[-1]
        _, _, profit_loss, _ = params
        # 0 × (0 - 1) = 0
        assert profit_loss == Decimal("0")


class TestPlaceSettlement:
    def _place_rec(self, fp, scratched=False):
        return {
            "rec_id": "p1",
            "entrant_id": "e1",
            "selection": "Big Boy",
            "recommended_stake": Decimal("10.00"),
            "odds_at_recommendation": Decimal("1.70"),
            "bet_type": "place",
            "entrant_scratched": scratched,
            "finish_position": fp,
        }

    def _settle(self, fp, field=10, scratched=False, winner="other"):
        cur = _FakeCursor()
        cur.queue_fetchall([self._place_rec(fp, scratched)])
        cur._fetchone_responses.append({"n": field})
        counts = gcr.settle_recommendations(cur, "race-1", winner, "Winner Horse")
        _, params = cur.executions[-1]
        return counts, params

    def test_third_place_wins_in_big_field(self):
        counts, params = self._settle(fp=3, field=10)
        assert counts["won"] == 1
        assert params[0] == "won"
        assert params[2] == Decimal("7.000")  # 10 x (1.70 - 1)

    def test_third_place_loses_in_seven_runner_field(self):
        # 5-7 runners pay 2 places only.
        counts, params = self._settle(fp=3, field=7)
        assert counts["lost"] == 1 and params[0] == "lost"

    def test_unplaced_and_dnf_lose(self):
        assert self._settle(fp=6, field=10)[1][0] == "lost"
        assert self._settle(fp=None, field=10)[1][0] == "lost"

    def test_scratched_place_rec_voids(self):
        counts, params = self._settle(fp=None, field=10, scratched=True)
        assert counts["void"] == 1 and params[0] == "void"


class TestEwPlaces:
    def test_terms_by_field_size(self):
        assert gcr.ew_places(4) == 0
        assert gcr.ew_places(5) == 2
        assert gcr.ew_places(7) == 2
        assert gcr.ew_places(8) == 3
        assert gcr.ew_places(16) == 3
