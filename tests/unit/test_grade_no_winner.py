"""Grading a bout that ended with no winner (an MMA draw or no contest).

fetch_upcoming records it finished with NULL scores and metadata.no_winner;
the convention (grading_outcomes.NO_WINNER_KEY):
  * a prediction is neither right nor wrong: actual_outcome 'draw' /
    'no_contest', is_correct NULL, logged once, never read back;
  * an open rec is VOID: profit_loss 0, actual_result 'void: draw' / 'void: no
    contest' — after the post-kickoff and postponement voids, which win;
  * a kind nobody knows grades and settles nothing, loudly;
  * the pure graders never produce a verdict for it, whatever the scores say.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from decimal import Decimal
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parents[1] / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


go = _load("grading_outcomes", SCRIPTS_DIR / "grading_outcomes.py")
gcm = _load("grade_completed_matches", SCRIPTS_DIR / "grade_completed_matches.py")
# The grader's in-memory cursor (not collected again: only the class is used).
_fake = _load("_grade_fake_cursor", TESTS_DIR / "test_grade_completed_matches.py")

DRAW_META = {"no_winner": "draw", "no_winner_detail": "majority draw; judges 28-28, 27-29, 28-28; ended R3 5:00"}


class NoWinnerCursor(_fake.FakeCursor):
    """FakeCursor that honours NO_WINNER_RECORDED_SQL in the ungraded-prediction
    listing, but only when the query carries it: dropping the clause from
    list_ungraded_predictions makes a second run read the rows back."""

    def __init__(self, predictions, recs, match_meta):
        super().__init__(predictions, recs)
        self.match_meta = match_meta

    def execute(self, sql, params=None):
        super().execute(sql, params)
        sql_norm = " ".join(sql.split())
        if self._last and self._last[0] == "preds" and gcm.NO_WINNER_RECORDED_SQL in sql_norm:
            recorded = {
                pid
                for pid, p in self.predictions.items()
                if self.match_meta.get("no_winner") is not None and p.get("actual_outcome") is not None
            }
            self._last = ("preds", [r for r in self._last[1] if r["prediction_id"] not in recorded])


def _match(meta=None, match_id="m1"):
    return {"match_id": match_id, "sport": "mma", "home_score": None, "away_score": None, "metadata": meta}


def _preds():
    return [
        _fake._pred("p1", "m1", "moneyline", "ensemble_mma_ml", "away"),
        _fake._pred("p2", "m1", "moneyline", "ensemble_mma_ml", "home"),
    ]


def _rec(rid="r1", **kw):
    return _fake._rec(rid, "m1", "p1", 25, 1.8, bet_type="moneyline", selection="away", sport="mma", **kw)


class TestGradeMatch:
    def test_predictions_are_recorded_as_a_draw_and_left_ungraded(self, caplog):
        cur = NoWinnerCursor(_preds(), [], DRAW_META)
        stats: dict = {}
        with caplog.at_level(logging.WARNING):
            counts = gcm.grade_match(cur, _match(DRAW_META), stats)
        assert {p: (r["actual_outcome"], r["is_correct"]) for p, r in cur.predictions.items()} == {
            "p1": ("draw", None),
            "p2": ("draw", None),
        }
        assert counts == {"predictions_graded": 0, "predictions_skipped": 0, "recs_settled": 0}
        assert stats["no_winner_predictions"] == 2
        assert "ended with no winner (draw; majority draw" in caplog.text
        assert "2 prediction(s) recorded with actual_outcome 'draw' and left ungraded" in caplog.text

    def test_a_second_run_reads_nothing_back(self, caplog):
        cur = NoWinnerCursor(_preds(), [], DRAW_META)
        gcm.grade_match(cur, _match(DRAW_META), {})
        stats: dict = {}
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            gcm.grade_match(cur, _match(DRAW_META), stats)
        assert "no_winner_predictions" not in stats and "ended with no winner" not in caplog.text

    @pytest.mark.parametrize(
        "kind,result",
        [("draw", "void: draw"), ("no_contest", "void: no contest")],
    )
    @pytest.mark.parametrize("status", ["pending", "placed"])
    def test_an_open_rec_is_void_with_its_stake_returned(self, kind, result, status, caplog):
        meta = {"no_winner": kind}
        cur = NoWinnerCursor(_preds(), [_rec(status=status)], meta)
        stats: dict = {}
        with caplog.at_level(logging.WARNING):
            counts = gcm.grade_match(cur, _match(meta), stats)
        rec = cur.recs["r1"]
        assert (rec["status"], rec["actual_result"], rec["profit_loss"]) == ("void", result, Decimal("0"))
        assert stats["no_winner_rec_voids"] == 1 and counts["recs_settled"] == 0
        assert f"the bout ended with no winner ({kind})" in caplog.text

    def test_a_rec_written_after_kickoff_is_voided_as_hindsight_first(self):
        cur = NoWinnerCursor(_preds(), [_rec(after_kickoff=True)], DRAW_META)
        stats: dict = {}
        gcm.grade_match(cur, _match(DRAW_META), stats)
        assert cur.recs["r1"]["actual_result"] == gcm.POST_KICKOFF_VOID_RESULT
        assert stats["post_kickoff_rec_voids"] == 1 and "no_winner_rec_voids" not in stats

    def test_a_rec_already_settled_without_money_is_left_loudly(self, caplog):
        cur = NoWinnerCursor(_preds(), [_rec(status="won", actual_result="away")], DRAW_META)
        with caplog.at_level(logging.ERROR):
            gcm.grade_match(cur, _match(DRAW_META), {})
        assert cur.recs["r1"]["status"] == "won" and cur.recs["r1"]["profit_loss"] is None
        assert "is 'won' without a profit_loss on a bout that ended with no winner" in caplog.text

    def test_an_unknown_kind_grades_and_settles_nothing(self, caplog):
        meta = {"no_winner": "technical_decision"}
        cur = NoWinnerCursor(_preds(), [_rec()], meta)
        with caplog.at_level(logging.ERROR):
            counts = gcm.grade_match(cur, _match(meta), {})
        assert all(p["actual_outcome"] is None and p["is_correct"] is None for p in cur.predictions.values())
        assert cur.recs["r1"]["status"] == "pending"
        assert counts["predictions_skipped"] == 2
        assert "no winner of an unknown kind (metadata.no_winner='technical_decision'" in caplog.text

    def test_a_decisive_mma_bout_still_grades(self):
        cur = NoWinnerCursor(_preds(), [_rec()], {})
        match = {**_match({}), "home_score": 0, "away_score": 1}
        counts = gcm.grade_match(cur, match, {})
        assert cur.predictions["p1"]["is_correct"] is True and cur.predictions["p2"]["is_correct"] is False
        assert counts["recs_settled"] == 1 and cur.recs["r1"]["status"] == "won"


class TestComputePredictionGrade:
    PRED = {"prediction_type": "moneyline", "model_name": "ensemble_mma_ml", "predicted_outcome": "home"}

    def test_a_no_winner_bout(self):
        assert gcm.compute_prediction_grade(self.PRED, _match(DRAW_META), {}) == (None, "draw", None)
        nc = _match({"no_winner": "no_contest"})
        assert gcm.compute_prediction_grade(self.PRED, nc, {}) == (None, "no_contest", None)

    def test_whatever_the_scores_say(self):
        match = {**_match(DRAW_META), "home_score": 1, "away_score": 0}
        assert gcm.compute_prediction_grade(self.PRED, match, {}) == (None, "draw", None)

    def test_written_after_kickoff_stays_hindsight(self):
        pred = {**self.PRED, "after_kickoff": True}
        assert gcm.compute_prediction_grade(pred, _match(DRAW_META), {}) == (gcm.SKIP_AFTER_KICKOFF, None, None)

    def test_an_unknown_kind(self):
        match = _match({"no_winner": "??"})
        assert gcm.compute_prediction_grade(self.PRED, match, {}) == (gcm.SKIP_UNGRADABLE, None, None)

    def test_regrade_finds_a_recorded_row_unchanged(self):
        # regrade_predictions re-runs compute_prediction_grade over rows that
        # carry an actual_outcome: a recorded no-winner row must read as it is.
        rg = _load("regrade_predictions", SCRIPTS_DIR / "regrade_predictions.py")
        row = {
            **self.PRED,
            "prediction_id": "p1",
            "match_id": "m1",
            "probabilities": None,
            "actual_outcome": "draw",
            "is_correct": None,
            "after_kickoff": False,
            "home_score": None,
            "away_score": None,
            "metadata": DRAW_META,
            "half_time_home": None,
            "half_time_away": None,
        }
        changes, counts = rg.plan_regrade([row], {})
        assert changes == [] and counts["unchanged"] == 1


class TestListingQueries:
    class _Capture:
        def __init__(self):
            self.sql = ""

        def execute(self, sql, params=None):
            self.sql = " ".join(sql.split())

        def fetchall(self):
            return []

    def test_the_window_listing_skips_recorded_no_winner_predictions(self):
        cur = self._Capture()
        gcm.list_finished_matches(cur, 14, None)
        assert (
            "WHERE p.match_id = m.id AND p.is_correct IS NULL AND NOT "
            "(m.metadata->>'no_winner' IS NOT NULL AND p.actual_outcome IS NOT NULL)"
        ) in cur.sql

    def test_the_ungraded_listing_skips_them_too(self):
        cur = self._Capture()
        gcm.list_ungraded_predictions(cur, "m1")
        assert "AND NOT (m.metadata->>'no_winner' IS NOT NULL AND p.actual_outcome IS NOT NULL)" in cur.sql

    def test_the_stat_keys(self):
        assert {"no_winner_predictions", "no_winner_rec_voids"} <= set(gcm.STAT_KEYS)


class TestPureGraders:
    def test_no_winner_result(self):
        assert go.no_winner_result(None) is None
        assert go.no_winner_result({}) is None
        assert go.no_winner_result({"no_winner": "draw"}) == "draw"
        assert go.no_winner_result({"no_winner": 7}) == "7"  # kept as it is: callers refuse it

    @pytest.mark.parametrize("scores", [(None, None), (1, 0), (0, 0)])
    def test_actual_outcome_is_none_for_every_market(self, scores):
        for ptype in ("moneyline", "match_result", "over_under", "btts"):
            assert (
                go.actual_outcome(
                    prediction_type=ptype,
                    model_name="ensemble_mma_ml",
                    predicted_outcome="home",
                    home_score=scores[0],
                    away_score=scores[1],
                    metadata=DRAW_META,
                )
                is None
            )

    def test_a_rec_is_never_settled_by_the_pure_grader(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert go.grade_rec_selection("moneyline", "away", 0, 0, sport="mma", metadata=DRAW_META) is None
        assert "recorded with no winner (draw)" in caplog.text

    def test_without_the_marker_a_0_0_mma_rec_would_have_lost(self):
        # Why NULL scores and an explicit marker, never 0-0: the 1x2-shaped
        # moneyline settlement reads 0-0 as a draw and an 'away' pick as lost.
        assert go.grade_rec_selection("moneyline", "away", 0, 0, sport="mma", metadata={}) == go.REC_LOST

    def test_the_void_results(self):
        assert go.NO_WINNER_VOID_RESULTS == {"draw": "void: draw", "no_contest": "void: no contest"}
        assert all(v.startswith(gcm.SCORE_INDEPENDENT_VOID_PREFIX) for v in go.NO_WINNER_VOID_RESULTS.values())
        assert not any(v.startswith(gcm.POST_KICKOFF_VOID_RESULT) for v in go.NO_WINNER_VOID_RESULTS.values())
