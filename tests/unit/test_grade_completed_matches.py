"""End-to-end grade_match() behavior against a mocked DB cursor.

The cursor stub intercepts SQL, dispatches on the leading verb, and
keeps in-memory tables for predictions + betting_recommendations so
we can assert on the post-grade state without standing up Postgres."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
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


_load("grading_outcomes", "grading_outcomes.py")
gcm = _load("grade_completed_matches", "grade_completed_matches.py")


class FakeCursor:
    """Minimal stub: respond to the four SQL shapes the grader uses
    (list ungraded preds, update pred, list open recs, settle rec)."""

    def __init__(self, predictions, recs):
        # predictions: list of dicts keyed by id
        self.predictions = {p["id"]: dict(p) for p in predictions}
        self.recs = {r["id"]: dict(r) for r in recs}
        self._last = None

    def execute(self, sql, params=None):
        sql_norm = " ".join(sql.split())
        if "FROM predictions" in sql_norm and "is_correct IS NULL" in sql_norm:
            match_id = params[0]
            rows = []
            for p in self.predictions.values():
                if p["match_id"] != match_id or p.get("is_correct") is not None:
                    continue
                rows.append(
                    {
                        "prediction_id": p["id"],
                        "prediction_type": p["prediction_type"],
                        "model_name": p["model_name"],
                        "predicted_outcome": p["predicted_outcome"],
                    }
                )
            self._last = ("preds", rows)
            return
        if sql_norm.startswith("UPDATE predictions"):
            actual, correct, pred_id = params
            self.predictions[pred_id]["actual_outcome"] = actual
            self.predictions[pred_id]["is_correct"] = correct
            self._last = ("update_pred", None)
            return
        if "FROM betting_recommendations br" in sql_norm and "LEFT JOIN predictions" in sql_norm:
            match_id = params[0]
            rows = []
            for r in self.recs.values():
                if r["match_id"] != match_id:
                    continue
                if r["status"] not in ("pending", "placed"):
                    continue
                p = self.predictions.get(r["prediction_id"], {})
                rows.append(
                    {
                        "rec_id": r["id"],
                        "prediction_id": r["prediction_id"],
                        "recommended_stake": r["recommended_stake"],
                        "odds_at_recommendation": r["odds_at_recommendation"],
                        "status": r["status"],
                        "p_actual_outcome": p.get("actual_outcome"),
                        "p_is_correct": p.get("is_correct"),
                    }
                )
            self._last = ("recs", rows)
            return
        if sql_norm.startswith("UPDATE betting_recommendations"):
            new_status, actual_result, pl, rec_id = params
            self.recs[rec_id]["status"] = new_status
            self.recs[rec_id]["actual_result"] = actual_result
            self.recs[rec_id]["profit_loss"] = pl
            self._last = ("settle", None)
            return
        raise AssertionError(f"Unexpected SQL: {sql_norm[:80]}")

    def fetchall(self):
        kind, rows = self._last or (None, [])
        if kind in ("preds", "recs"):
            return rows
        return []


def _pred(pid, match_id, ptype, model_name, predicted_outcome, is_correct=None, actual_outcome=None):
    return {
        "id": pid,
        "match_id": match_id,
        "prediction_type": ptype,
        "model_name": model_name,
        "predicted_outcome": predicted_outcome,
        "is_correct": is_correct,
        "actual_outcome": actual_outcome,
    }


def _rec(rid, match_id, prediction_id, stake, odds, status="pending"):
    return {
        "id": rid,
        "match_id": match_id,
        "prediction_id": prediction_id,
        "recommended_stake": Decimal(str(stake)),
        "odds_at_recommendation": Decimal(str(odds)),
        "status": status,
        "actual_result": None,
        "profit_loss": None,
    }


# ── grade_match end-to-end ───────────────────────────────────────────


class TestGradeMatchSoccer:
    def test_soccer_match_result_correct_pick(self):
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "match_result", "ensemble", "home")]
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0)]
        cur = FakeCursor(preds, recs)

        counts = gcm.grade_match(cur, match)

        assert counts == {
            "predictions_graded": 1,
            "predictions_skipped": 0,
            "recs_settled": 1,
        }
        assert cur.predictions["p1"]["is_correct"] is True
        assert cur.predictions["p1"]["actual_outcome"] == "home"
        assert cur.recs["r1"]["status"] == "won"
        # P/L = 25 × (2.0 - 1) = 25.00
        assert cur.recs["r1"]["profit_loss"] == Decimal("25.00")

    def test_soccer_match_result_wrong_pick(self):
        match = {"match_id": "m1", "home_score": 0, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "match_result", "ensemble", "home")]
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0)]
        cur = FakeCursor(preds, recs)

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["is_correct"] is False
        assert cur.recs["r1"]["status"] == "lost"
        assert cur.recs["r1"]["profit_loss"] == Decimal("-25.00")

    def test_over_under_push_marks_rec_void(self):
        # Soccer over 3.0 with exactly 3 goals = push.
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "over_under", "ensemble", "over_3")]
        recs = [_rec("r1", "m1", "p1", stake=25, odds=1.90)]
        cur = FakeCursor(preds, recs)

        gcm.grade_match(cur, match)

        # is_correct stays NULL on push — accuracy stats shouldn't
        # count this as right or wrong.
        assert cur.predictions["p1"]["is_correct"] is None
        assert cur.predictions["p1"]["actual_outcome"] == "push_3"
        # Rec settles as void with zero P/L (stake refunded).
        assert cur.recs["r1"]["status"] == "void"
        assert cur.recs["r1"]["profit_loss"] == Decimal("0")


class TestGradeMatchNhl:
    def test_nhl_regulation_via_metadata(self):
        # NHL game where regulation ended tied but home won in OT.
        # nhl:regulation predicted "home" but the regulation_winner
        # field is "tie" → the rec was wrong.
        match = {
            "match_id": "m1",
            "home_score": 4,
            "away_score": 3,
            "metadata": {"regulation_winner": "tie"},
        }
        preds = [_pred("p1", "m1", "match_result", "ensemble_nhl_reg", "home")]
        recs = [_rec("r1", "m1", "p1", stake=20, odds=2.20)]
        cur = FakeCursor(preds, recs)

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["actual_outcome"] == "tie"
        assert cur.predictions["p1"]["is_correct"] is False
        assert cur.recs["r1"]["status"] == "lost"
        assert cur.recs["r1"]["profit_loss"] == Decimal("-20.00")

    def test_nhl_moneyline_won_after_ot(self):
        # Home won 4-3 (OT) — moneyline picks home → correct.
        match = {
            "match_id": "m1",
            "home_score": 4,
            "away_score": 3,
            "metadata": {"regulation_winner": "tie"},
        }
        preds = [_pred("p1", "m1", "moneyline", "ensemble_nhl_ml", "home")]
        recs = [_rec("r1", "m1", "p1", stake=20, odds=1.80)]
        cur = FakeCursor(preds, recs)

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["is_correct"] is True
        assert cur.recs["r1"]["status"] == "won"
        # P/L = 20 × 0.80 = 16.00
        assert cur.recs["r1"]["profit_loss"] == Decimal("16.00")

    def test_nhl_puck_line_no_cover_on_one_goal_win(self):
        # Home wins 3-2: covers moneyline but doesn't cover -1.5.
        match = {"match_id": "m1", "home_score": 3, "away_score": 2, "metadata": {}}
        preds = [_pred("p1", "m1", "spread", "ensemble_nhl_pl", "cover")]
        cur = FakeCursor(preds, [])

        gcm.grade_match(cur, match)
        assert cur.predictions["p1"]["actual_outcome"] == "no_cover"
        assert cur.predictions["p1"]["is_correct"] is False


# ── Edge cases ───────────────────────────────────────────────────────


class TestGradeMatchEdgeCases:
    def test_already_graded_predictions_skipped(self):
        # is_correct is NOT NULL → grade_match should leave it alone.
        # The FakeCursor's filter respects is_correct IS NULL semantics.
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        preds = [
            _pred("p1", "m1", "match_result", "ensemble", "home", is_correct=True, actual_outcome="home"),
        ]
        recs = []
        cur = FakeCursor(preds, recs)

        counts = gcm.grade_match(cur, match)
        # Nothing to grade — counts stay zero.
        assert counts["predictions_graded"] == 0

    def test_ungradable_market_increments_skipped_counter(self):
        # asian_handicap isn't in the v1 dispatch — actual_outcome
        # returns None and the counter ticks "skipped".
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "asian_handicap", "ensemble", "-0.5_home")]
        cur = FakeCursor(preds, [])

        counts = gcm.grade_match(cur, match)

        assert counts["predictions_skipped"] == 1
        assert counts["predictions_graded"] == 0
        # No UPDATE happened.
        assert cur.predictions["p1"]["is_correct"] is None

    def test_rec_without_graded_prediction_stays_pending(self):
        # If the prediction is ungradable, the rec it references can't
        # be settled either. Status stays 'pending'.
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "asian_handicap", "ensemble", "-0.5_home")]
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0)]
        cur = FakeCursor(preds, recs)

        gcm.grade_match(cur, match)
        assert cur.recs["r1"]["status"] == "pending"
        assert cur.recs["r1"]["profit_loss"] is None


# ── profit_loss math ─────────────────────────────────────────────────


class TestProfitLossMath:
    def test_win_pays_decimal_minus_one_times_stake(self):
        status, pl = gcm.profit_loss_for(True, False, Decimal("100"), Decimal("2.50"))
        assert status == "won"
        assert pl == Decimal("150.00")

    def test_loss_subtracts_stake(self):
        status, pl = gcm.profit_loss_for(False, False, Decimal("100"), Decimal("2.50"))
        assert status == "lost"
        assert pl == Decimal("-100.00")

    def test_push_returns_zero(self):
        # Push always voids regardless of is_correct.
        status, pl = gcm.profit_loss_for(True, True, Decimal("100"), Decimal("2.50"))
        assert status == "void"
        assert pl == Decimal("0")

    def test_quantizes_to_cents(self):
        # 33 × 1.75 = 57.75 — should be exactly two decimal places.
        status, pl = gcm.profit_loss_for(True, False, Decimal("33"), Decimal("1.75"))
        assert pl == Decimal("24.75")
        # Hard cents check (no fractional cent residue).
        assert pl.as_tuple().exponent == -2


# ── Smoke: imports and signatures ────────────────────────────────────


class TestModuleSurface:
    def test_required_helpers_exist(self):
        # Lock the names CronExpression-grading-DAG and the API
        # accuracy endpoint will import in Commit D. If these get
        # renamed we want the import error to surface here, not at
        # DAG load time on prod.
        assert callable(gcm.grade_match)
        assert callable(gcm.run)
        assert callable(gcm.profit_loss_for)
        assert callable(gcm.list_finished_matches)


# Quiet a flake8 import-unused warning on pytest in test discovery.
_ = pytest
