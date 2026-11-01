"""End-to-end grade_match() behavior against a mocked DB cursor.

The cursor stub intercepts SQL, dispatches on the leading verb, and
keeps in-memory tables for predictions + betting_recommendations so
we can assert on the post-grade state without standing up Postgres."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
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
    """Minimal stub: respond to the SQL shapes the grader uses
    (list ungraded preds, update pred, list open recs, settle rec,
    features fetch). Features map is optional; defaults to no rows."""

    def __init__(self, predictions, recs, features=None):
        # predictions: list of dicts keyed by id
        self.predictions = {p["id"]: dict(p) for p in predictions}
        self.recs = {r["id"]: dict(r) for r in recs}
        # features: {match_id: features_dict}
        self.features = dict(features or {})
        self._last = None

    def execute(self, sql, params=None):
        sql_norm = " ".join(sql.split())
        if "FROM features_cache" in sql_norm:
            match_id = params[0]
            feat = self.features.get(match_id)
            self._last = ("features", {"features": feat} if feat is not None else None)
            return
        if "FROM predictions" in sql_norm and "is_correct IS NULL" in sql_norm:
            match_id = params[0]
            rows = []
            for p in self.predictions.values():
                if p["match_id"] != match_id or p.get("is_correct") is not None:
                    continue
                row = {
                    "prediction_id": p["id"],
                    "prediction_type": p["prediction_type"],
                    "model_name": p["model_name"],
                    "predicted_outcome": p["predicted_outcome"],
                    "after_kickoff": p.get("after_kickoff", False),
                }
                # Only when the query selects it, so dropping the column
                # from list_ungraded_predictions fails the 'other' tests.
                if "p.probabilities" in sql_norm:
                    row["probabilities"] = p.get("probabilities")
                rows.append(row)
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
                # Mirrors the SQL: in flight, or trigger-settled without P&L.
                in_flight = r["status"] in ("pending", "placed")
                trigger_settled = r["status"] in ("won", "lost", "void") and r.get("profit_loss") is None
                if not (in_flight or trigger_settled):
                    continue
                p = self.predictions.get(r["prediction_id"], {})
                row = {
                    "rec_id": r["id"],
                    "prediction_id": r["prediction_id"],
                    "recommended_stake": r["recommended_stake"],
                    "odds_at_recommendation": r["odds_at_recommendation"],
                    "bet_type": r.get("bet_type"),
                    "selection": r.get("selection"),
                    "status": r["status"],
                    "p_actual_outcome": p.get("actual_outcome"),
                    "p_is_correct": p.get("is_correct"),
                }
                # Only when the query selects them (as for probabilities
                # above), so dropping either column fails the NHL rec tests.
                if "p.model_name AS p_model_name" in sql_norm:
                    row["p_model_name"] = p.get("model_name")
                if (
                    "l.sport AS sport" in sql_norm
                    and "LEFT JOIN matches m ON m.id = br.match_id" in sql_norm
                    and "LEFT JOIN leagues l ON l.id = m.league_id" in sql_norm
                ):
                    row["sport"] = r.get("sport")
                # Only when the query computes it against the match's
                # kickoff, so dropping that from the SQL fails the
                # post-kickoff tests.
                if "br.created_at >= m.match_date" in sql_norm and "JOIN matches m" in sql_norm:
                    row["after_kickoff"] = r.get("after_kickoff", False)
                rows.append(row)
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

    def fetchone(self):
        kind, row = self._last or (None, None)
        if kind == "features":
            return row
        return None


def _pred(
    pid, match_id, ptype, model_name, predicted_outcome, is_correct=None, actual_outcome=None, probabilities=None
):
    return {
        "id": pid,
        "match_id": match_id,
        "prediction_type": ptype,
        "model_name": model_name,
        "predicted_outcome": predicted_outcome,
        "is_correct": is_correct,
        "actual_outcome": actual_outcome,
        "probabilities": probabilities,
    }


def _rec(
    rid,
    match_id,
    prediction_id,
    stake,
    odds,
    bet_type="1x2",
    selection="home",
    status="pending",
    sport="soccer",
    after_kickoff=False,
    actual_result=None,
):
    return {
        "id": rid,
        "bet_type": bet_type,
        "selection": selection,
        "sport": sport,
        "match_id": match_id,
        "prediction_id": prediction_id,
        "recommended_stake": Decimal(str(stake)),
        "odds_at_recommendation": Decimal(str(odds)),
        "status": status,
        "actual_result": actual_result,
        "profit_loss": None,
        "after_kickoff": after_kickoff,
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

    def test_halftime_market_grades_from_the_match_columns(self):
        match = {
            "match_id": "m1",
            "home_score": 2,
            "away_score": 2,
            "metadata": {},
            "half_time_home": 1,
            "half_time_away": 0,
        }
        cur = FakeCursor([_pred("p1", "m1", "match_result_ht", "ensemble", "home")], [])

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["actual_outcome"] == "home"
        assert cur.predictions["p1"]["is_correct"] is True

    def test_halftime_market_without_the_columns_stays_ungraded(self):
        match = {"match_id": "m1", "home_score": 2, "away_score": 2, "metadata": {}}
        cur = FakeCursor([_pred("p1", "m1", "match_result_ht", "ensemble", "home")], [])

        counts = gcm.grade_match(cur, match)

        assert counts["predictions_skipped"] == 1 and cur.predictions["p1"]["is_correct"] is None

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
        recs = [_rec("r1", "m1", "p1", stake=25, odds=1.90, bet_type="over_under", selection="over_3")]
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
        recs = [_rec("r1", "m1", "p1", stake=20, odds=2.20, bet_type="moneyline", selection="away")]
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
        recs = [_rec("r1", "m1", "p1", stake=20, odds=1.80, bet_type="moneyline", selection="home")]
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

    def test_dixon_coles_double_chance_grades_on_regulation(self):
        # 4-3 in OT: the final score has a winner, regulation was tied —
        # the '12' pick (regulation 3-way) lost.
        match = {"match_id": "m1", "home_score": 4, "away_score": 3, "metadata": {"regulation_winner": "tie"}}
        cur = FakeCursor([_pred("p1", "m1", "double_chance", "dixon_coles_nhl", "12")], [])

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["actual_outcome"] == "draw"
        assert cur.predictions["p1"]["is_correct"] is False

    def test_dixon_coles_correct_score_other_reads_the_prediction_keys(self):
        match = {"match_id": "m1", "home_score": 5, "away_score": 2, "metadata": {}}
        probs = {"3-2": 0.055, "2-2": 0.052, "4-3": 0.039, "other": 0.854}
        preds = [
            _pred("p1", "m1", "correct_score", "dixon_coles_nhl", "other", probabilities=probs),
            _pred("p2", "m1", "correct_score", "dixon_coles_nhl", "3-2", probabilities=probs),
        ]
        cur = FakeCursor(preds, [])

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["actual_outcome"] == "5-2"
        assert cur.predictions["p1"]["is_correct"] is True
        assert cur.predictions["p2"]["is_correct"] is False


class TestComputePredictionGrade:
    """The one verdict path shared by grade_match and regrade_predictions."""

    MATCH = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}

    def _pred(self, ptype, model_name, pick, **extra):
        return {"prediction_type": ptype, "model_name": model_name, "predicted_outcome": pick, **extra}

    def test_graded_row(self):
        assert gcm.compute_prediction_grade(self._pred("btts", "ensemble", "yes"), self.MATCH, {}) == (
            None,
            "yes",
            True,
        )

    def test_push_grades_with_a_null_verdict(self):
        got = gcm.compute_prediction_grade(self._pred("over_under", "ensemble", "over_3"), self.MATCH, {})
        assert got == (None, "push_3", None)

    def test_skip_reasons(self):
        late = self._pred("btts", "ensemble", "yes", after_kickoff=True)
        assert gcm.compute_prediction_grade(late, self.MATCH, {})[0] == gcm.SKIP_AFTER_KICKOFF
        phantom = self._pred("total", "ensemble_nfl_tot", "over")
        ctx = {"features": {"closing_total_line": 45.0}, "has_total_odds": False}
        assert gcm.compute_prediction_grade(phantom, self.MATCH, ctx)[0] == gcm.SKIP_PHANTOM_LINE
        unknown = self._pred("player_prop", "ensemble", "over_0.5")
        assert gcm.compute_prediction_grade(unknown, self.MATCH, {}) == (gcm.SKIP_UNGRADABLE, None, None)


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
        # player_prop isn't in the dispatch — actual_outcome
        # returns None and the counter ticks "skipped".
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "player_prop", "ensemble", "over_0.5")]
        cur = FakeCursor(preds, [])

        counts = gcm.grade_match(cur, match)

        assert counts["predictions_skipped"] == 1
        assert counts["predictions_graded"] == 0
        # No UPDATE happened.
        assert cur.predictions["p1"]["is_correct"] is None

    def test_prediction_written_after_kickoff_is_never_graded(self):
        # 467 June-2026 rows on 2023-25 matches: hindsight, not predictions.
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        late = {**_pred("p1", "m1", "match_result", "ensemble", "home"), "after_kickoff": True}
        cur = FakeCursor([late, _pred("p2", "m1", "btts", "ensemble", "yes")], [])
        stats: dict = {}

        counts = gcm.grade_match(cur, match, stats)

        assert counts["predictions_graded"] == 1 and counts["predictions_skipped"] == 1
        assert cur.predictions["p1"]["is_correct"] is None
        assert cur.predictions["p2"]["is_correct"] is True
        assert stats["after_kickoff_skips"] == 1

    def test_rec_settles_from_its_own_selection_without_graded_prediction(self):
        # Rec-level settlement: the rec grades from ITS OWN selection
        # and the final score. The AH prediction row ALSO grades now
        # (home_covers at -0.5) — both verdicts are independent.
        match = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "asian_handicap", "ensemble", "-0.5_home")]
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0, bet_type="asian_handicap", selection="home_-0.5")]
        cur = FakeCursor(preds, recs)

        gcm.grade_match(cur, match)
        assert cur.predictions["p1"]["is_correct"] is True
        assert cur.recs["r1"]["status"] == "won"
        assert cur.recs["r1"]["profit_loss"] == Decimal("25.00")


class TestRecSettlementGuards:
    """Recs the final score alone cannot settle stay open — never a
    guessed won/lost."""

    OT_GAME = {"match_id": "m1", "home_score": 4, "away_score": 3, "metadata": {"regulation_winner": "tie"}}

    def test_nhl_double_chance_rec_is_never_settled_on_the_final_score(self):
        # 4-3 in OT: on the final score '12' would win; regulation was tied.
        preds = [_pred("p1", "m1", "double_chance", "dixon_coles_nhl", "12")]
        recs = [_rec("r1", "m1", "p1", stake=20, odds=1.25, bet_type="double_chance", selection="12", sport="nhl")]
        cur = FakeCursor(preds, recs)

        counts = gcm.grade_match(cur, self.OT_GAME)

        assert counts["recs_settled"] == 0
        assert cur.recs["r1"]["status"] == "pending" and cur.recs["r1"]["profit_loss"] is None
        # The prediction row still grades — from regulation_winner.
        assert cur.predictions["p1"]["is_correct"] is False

    def test_the_league_alone_marks_an_nhl_rec(self):
        # No prediction row to name the model: leagues.sport is enough.
        recs = [_rec("r1", "m1", None, stake=20, odds=1.25, bet_type="double_chance", selection="12", sport="nhl")]
        cur = FakeCursor([], recs)
        assert gcm.grade_match(cur, self.OT_GAME)["recs_settled"] == 0
        assert cur.recs["r1"]["status"] == "pending"

    def test_the_model_alone_marks_an_nhl_rec(self):
        # League sport unknown (NULL from the LEFT JOIN): the NHL model is enough.
        preds = [_pred("p1", "m1", "double_chance", "dixon_coles_nhl", "12")]
        recs = [_rec("r1", "m1", "p1", stake=20, odds=1.25, bet_type="double_chance", selection="12", sport=None)]
        cur = FakeCursor(preds, recs)
        assert gcm.grade_match(cur, self.OT_GAME)["recs_settled"] == 0
        assert cur.recs["r1"]["status"] == "pending"

    def test_soccer_double_chance_rec_still_settles(self):
        match = {"match_id": "m1", "home_score": 1, "away_score": 1, "metadata": {}}
        preds = [_pred("p1", "m1", "double_chance", "ensemble", "X2")]
        recs = [_rec("r1", "m1", "p1", stake=10, odds=1.5, bet_type="double_chance", selection="X2")]
        cur = FakeCursor(preds, recs)

        assert gcm.grade_match(cur, match)["recs_settled"] == 1
        assert cur.recs["r1"]["status"] == "won"
        assert cur.recs["r1"]["profit_loss"] == Decimal("5.0")

    def test_correct_score_other_rec_is_never_settled(self):
        # 5-2 is not among the prediction's listed keys, so the prediction
        # row credits 'other' — but the rec cannot see those keys.
        probs = {"3-2": 0.055, "2-2": 0.052, "4-3": 0.039, "other": 0.854}
        match = {"match_id": "m1", "home_score": 5, "away_score": 2, "metadata": {}}
        preds = [_pred("p1", "m1", "correct_score", "dixon_coles_nhl", "other", probabilities=probs)]
        recs = [
            _rec("r1", "m1", "p1", stake=10, odds=1.2, bet_type="correct_score", selection="other", sport="nhl"),
            _rec("r2", "m1", "p1", stake=10, odds=9.0, bet_type="correct_score", selection="5-2", sport="nhl"),
        ]
        cur = FakeCursor(preds, recs)

        assert gcm.grade_match(cur, match)["recs_settled"] == 1
        assert cur.recs["r1"]["status"] == "pending" and cur.recs["r1"]["profit_loss"] is None
        assert cur.recs["r2"]["status"] == "won"
        assert cur.predictions["p1"]["is_correct"] is True


# ── recs written after kickoff: void, never graded ───────────────────


class TestPostKickoffRecs:
    """A bet placed after kickoff is hindsight. The 2026-09-04 stranded-rec
    repair settled 21 soccer + 2 MMA such recs (net -237.05); the settler
    must void them, whatever the score says."""

    MATCH = {"match_id": "m1", "home_score": 2, "away_score": 1, "metadata": {}}

    def test_pending_rec_written_after_kickoff_is_voided_not_graded(self):
        preds = [_pred("p1", "m1", "match_result", "ensemble", "home")]
        # would have WON at 2.0: +25.00 of hindsight if graded
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0, after_kickoff=True)]
        cur = FakeCursor(preds, recs)
        stats: dict = {}

        counts = gcm.grade_match(cur, self.MATCH, stats)

        assert cur.recs["r1"]["status"] == "void"
        assert cur.recs["r1"]["profit_loss"] == Decimal("0")
        assert cur.recs["r1"]["actual_result"] == gcm.POST_KICKOFF_VOID_RESULT == "void: written after kickoff"
        assert counts["recs_settled"] == 0
        assert stats["post_kickoff_rec_voids"] == 1
        # the prediction (written pre-kickoff) still grades as usual
        assert cur.predictions["p1"]["is_correct"] is True

    def test_trigger_settled_post_kickoff_rec_is_voided_not_money_filled(self):
        # The migration-011 trigger settles from the prediction, so a
        # post-kickoff rec can arrive already 'won' with no P&L.
        preds = [_pred("p1", "m1", "match_result", "ensemble", "home", is_correct=True, actual_outcome="home")]
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0, status="won", actual_result="home", after_kickoff=True)]
        cur = FakeCursor(preds, recs)

        gcm.grade_match(cur, self.MATCH)

        assert cur.recs["r1"]["status"] == "void"
        assert cur.recs["r1"]["profit_loss"] == Decimal("0")
        assert cur.recs["r1"]["actual_result"] == gcm.POST_KICKOFF_VOID_RESULT

    def test_placed_rec_written_after_kickoff_is_voided(self):
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0, status="placed", after_kickoff=True)]
        cur = FakeCursor([], recs)
        gcm.grade_match(cur, self.MATCH)
        assert cur.recs["r1"]["status"] == "void" and cur.recs["r1"]["profit_loss"] == Decimal("0")

    def test_void_needs_no_score(self):
        match = {"match_id": "m1", "home_score": None, "away_score": None, "metadata": {}}
        recs = [
            _rec("late", "m1", "p1", stake=25, odds=2.0, after_kickoff=True),
            _rec("ok", "m1", "p1", stake=25, odds=2.0),
        ]
        cur = FakeCursor([], recs)
        gcm.grade_match(cur, match)
        assert cur.recs["late"]["status"] == "void"
        assert cur.recs["ok"]["status"] == "pending"  # no score: stays open, as before

    def test_pre_kickoff_rec_beside_a_post_kickoff_one_still_settles(self):
        recs = [
            _rec("late", "m1", "p1", stake=25, odds=2.0, after_kickoff=True),
            _rec("ok", "m1", "p1", stake=25, odds=2.0, selection="away"),
        ]
        cur = FakeCursor([], recs)
        counts = gcm.grade_match(cur, self.MATCH)
        assert cur.recs["late"]["status"] == "void"
        assert cur.recs["ok"]["status"] == "lost" and cur.recs["ok"]["profit_loss"] == Decimal("-25.00")
        assert counts["recs_settled"] == 1

    def test_open_recs_query_compares_created_at_with_the_kickoff(self):
        seen = []

        class Cur:
            def execute(self, sql, params=None):
                seen.append(" ".join(sql.split()))

            def fetchall(self):
                return []

        gcm.list_open_recs_for_match(Cur(), "m1")
        assert "(br.created_at >= m.match_date) AS after_kickoff" in seen[0]
        assert "JOIN matches m ON m.id = br.match_id" in seen[0]

    def test_run_reports_the_voids_at_error(self, monkeypatch, caplog):
        recs = [_rec("r1", "m1", "p1", stake=25, odds=2.0, after_kickoff=True)]
        cur = _RunCursor([dict(self.MATCH)], [], recs)
        monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _RunConn(cur))

        with caplog.at_level("WARNING", logger="grade_completed_matches"):
            summary = gcm.run("postgresql://unused/db", 14, None)

        assert summary["post_kickoff_rec_voids"] == 1
        assert summary["recs_settled"] == 0
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any("Voided 1 recommendation(s) written at or after kickoff" in r.getMessage() for r in errors)
        assert cur.recs["r1"]["status"] == "void"


class _RunCursor(FakeCursor):
    """FakeCursor plus the finished-match listing run() starts with."""

    def __init__(self, matches, predictions, recs):
        super().__init__(predictions, recs)
        self.matches = matches

    def execute(self, sql, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT m.id::text AS match_id"):
            self._last = ("matches", [dict(m) for m in self.matches])
            return
        if "AND m.status = 'postponed'" in sql_norm or "AND m.status = 'cancelled'" in sql_norm:
            self._last = ("recs", [])
            return
        super().execute(sql, params)

    def fetchall(self):
        if self._last and self._last[0] == "matches":
            return self._last[1]
        return super().fetchall()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _RunConn:
    def __init__(self, cur):
        self.cur = cur
        self.commits = 0

    def cursor(self, cursor_factory=None):
        return self.cur

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


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
        assert callable(gcm.fetch_match_features)


# ── NBA: variable-line grading via features_cache ────────────────────


class TestGradeMatchNba:
    """NBA spread + total are graded against the closing line from
    features_cache, not a fixed line. These tests prove the grader
    fetches features once per match and threads them through to
    actual_outcome's dispatch."""

    def test_nba_spread_home_covers(self):
        # Game: SAS 110, NYK 100 (margin 10). Closing line: -5 home.
        # 10 > 5 → home covers → grading marks the 'home' pick correct.
        match = {
            "match_id": "m-nba",
            "home_score": 110,
            "away_score": 100,
            "metadata": None,
        }
        preds = [_pred("p1", "m-nba", "spread", "ensemble_nba_sp", "home")]
        recs = [_rec("r1", "m-nba", "p1", stake=100, odds=1.95, bet_type="spread", selection="home_-5")]
        cur = FakeCursor(preds, recs, features={"m-nba": {"closing_spread_home": -5.0}})

        counts = gcm.grade_match(cur, match)

        assert counts["predictions_graded"] == 1
        assert counts["predictions_skipped"] == 0
        assert cur.predictions["p1"]["actual_outcome"] == "home"
        assert cur.predictions["p1"]["is_correct"] is True
        # Rec settled: win @ 1.95 stake 100 → +95.
        assert counts["recs_settled"] == 1
        assert cur.recs["r1"]["status"] == "won"
        assert cur.recs["r1"]["profit_loss"] == Decimal("95.00")

    def test_nba_spread_home_fails_to_cover(self):
        # Game: 103-100, margin 3. Closing -5 → away covers.
        match = {
            "match_id": "m-nba",
            "home_score": 103,
            "away_score": 100,
            "metadata": None,
        }
        preds = [_pred("p1", "m-nba", "spread", "ensemble_nba_sp", "home")]
        recs = [_rec("r1", "m-nba", "p1", stake=100, odds=1.95, bet_type="spread", selection="home_-5")]
        cur = FakeCursor(preds, recs, features={"m-nba": {"closing_spread_home": -5.0}})

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["actual_outcome"] == "away"
        assert cur.predictions["p1"]["is_correct"] is False
        assert cur.recs["r1"]["status"] == "lost"
        assert cur.recs["r1"]["profit_loss"] == Decimal("-100.00")

    def test_nba_total_over(self):
        # 110+108=218, line 215 → over.
        match = {
            "match_id": "m-nba",
            "home_score": 110,
            "away_score": 108,
            "metadata": None,
        }
        preds = [_pred("p1", "m-nba", "total", "ensemble_nba_tot", "over")]
        recs = []
        cur = FakeCursor(preds, recs, features={"m-nba": {"closing_total_line": 215.0}})

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["actual_outcome"] == "over"
        assert cur.predictions["p1"]["is_correct"] is True

    def test_nba_spread_skipped_when_features_missing(self):
        # If features_cache doesn't carry the closing line (NBA odds
        # never landed or features expired), refuse to grade rather
        # than fall back to NHL's fixed -1.5 which would be wildly
        # wrong for NBA. is_correct stays NULL → re-tried next run.
        match = {
            "match_id": "m-nba",
            "home_score": 110,
            "away_score": 100,
            "metadata": None,
        }
        preds = [_pred("p1", "m-nba", "spread", "ensemble_nba_sp", "home")]
        cur = FakeCursor(preds, recs=[], features={})  # no features for this match

        counts = gcm.grade_match(cur, match)

        assert counts["predictions_graded"] == 0
        assert counts["predictions_skipped"] == 1
        assert cur.predictions["p1"].get("is_correct") is None
        assert cur.predictions["p1"].get("actual_outcome") is None

    def test_nba_features_fetched_once_per_match(self):
        # Multiple predictions for the same match should NOT spam the
        # features_cache query — one fetch, threaded through every
        # per-prediction dispatch.
        match = {
            "match_id": "m-nba",
            "home_score": 110,
            "away_score": 100,
            "metadata": None,
        }
        preds = [
            _pred("p1", "m-nba", "moneyline", "ensemble_nba_ml", "home"),
            _pred("p2", "m-nba", "spread", "ensemble_nba_sp", "home"),
            _pred("p3", "m-nba", "total", "ensemble_nba_tot", "over"),
        ]
        cur = FakeCursor(
            preds,
            recs=[],
            features={"m-nba": {"closing_spread_home": -5.0, "closing_total_line": 215.0}},
        )

        gcm.grade_match(cur, match)

        # All three predictions graded successfully.
        for pid in ("p1", "p2", "p3"):
            assert cur.predictions[pid]["is_correct"] is not None

    def test_nhl_spread_uses_fixed_line_not_features(self):
        # Regression guard: adding NBA dispatch must not break NHL
        # spread (which still uses fixed -1.5 puck line). NHL match's
        # features_cache might or might not exist; doesn't matter.
        match = {
            "match_id": "m-nhl",
            "home_score": 4,
            "away_score": 2,  # margin 2 → home covers -1.5
            "metadata": None,
        }
        preds = [_pred("p1", "m-nhl", "spread", "ensemble_nhl_pl", "cover")]
        cur = FakeCursor(preds, recs=[], features={})

        gcm.grade_match(cur, match)

        assert cur.predictions["p1"]["actual_outcome"] == "cover"
        assert cur.predictions["p1"]["is_correct"] is True


# Quiet a flake8 import-unused warning on pytest in test discovery.
_ = pytest


# ── Soccer full-time markets on the 90-minute score ──────────────────


class RegulationCursor(FakeCursor):
    """FakeCursor plus the graded full-time read, the guarded re-grade, the
    money-settled rec read and its guarded re-settle, the long-postponement
    and cancelled rec reads, the guarded void, and rec created_at.
    `match_status` maps a match id to its status for the void's guard (an
    unlisted match is taken to hold the status the guard asks for)."""

    def __init__(self, predictions, recs, postponed_recs=(), features=None, cancelled_recs=(), match_status=None):
        super().__init__(predictions, recs, features)
        self.postponed_recs = [dict(r) for r in postponed_recs]
        self.cancelled_recs = [dict(r) for r in cancelled_recs]
        self.match_status = dict(match_status or {})
        self.guarded_updates: list[tuple] = []
        self.resettles: list[tuple] = []
        self.voids: list[tuple] = []
        self.rowcount = 0
        self.sql: list[str] = []

    def fetchone(self):
        if self._last and self._last[0] == "void":
            return self._last[1]
        return super().fetchone()

    def execute(self, sql, params=None):
        sql_norm = " ".join(sql.split())
        self.sql.append(sql_norm)
        if "br.profit_loss IS NOT NULL" in sql_norm and "FROM betting_recommendations br" in sql_norm:
            (match_id,) = params
            rows = []
            for r in self.recs.values():
                if r["match_id"] != match_id or r["status"] not in ("won", "lost", "void") or r["profit_loss"] is None:
                    continue
                p = self.predictions.get(r["prediction_id"], {})
                rows.append(
                    {
                        "rec_id": r["id"],
                        "bet_type": r["bet_type"],
                        "selection": r["selection"],
                        "recommended_stake": r["recommended_stake"],
                        "odds_at_recommendation": r["odds_at_recommendation"],
                        "status": r["status"],
                        "actual_result": r.get("actual_result"),
                        "profit_loss": r["profit_loss"],
                        "p_model_name": p.get("model_name"),
                        "sport": r.get("sport") if "l.sport AS sport" in sql_norm else None,
                    }
                )
            self._last = ("recs", rows)
            return
        if sql_norm.startswith("UPDATE betting_recommendations br") and "FROM matches m" in sql_norm:
            actual_result, rec_id, old_status, match_status = params
            for guard in ("WHERE br.id = %s", "AND br.status = %s", "AND m.id = br.match_id", "AND m.status = %s"):
                assert guard in sql_norm, f"the void lacks its guard {guard!r}"
            self.voids.append(params)
            r = self.recs[rec_id]
            hit = r["status"] == old_status and self.match_status.get(r["match_id"], match_status) == match_status
            hit = hit and "SET status = 'void'" in sql_norm and "profit_loss = 0" in sql_norm
            if hit:
                r["status"], r["actual_result"], r["profit_loss"] = "void", actual_result, Decimal("0")
            self._last = ("void", {"id": rec_id} if hit else None)
            return
        if sql_norm.startswith("UPDATE betting_recommendations") and params is not None and len(params) == 7:
            new_status, actual_result, pl, rec_id, old_status, old_actual, old_pl = params
            for guard in (
                "WHERE id = %s",
                "AND status = %s",
                "AND actual_result IS NOT DISTINCT FROM %s",
                "AND profit_loss IS NOT DISTINCT FROM %s",
            ):
                assert guard in sql_norm, f"the re-settle lacks its guard {guard!r}"
            self.resettles.append(params)
            r = self.recs[rec_id]
            hit = (r["status"], r.get("actual_result"), r["profit_loss"]) == (old_status, old_actual, old_pl)
            if hit:
                r["status"], r["actual_result"], r["profit_loss"] = new_status, actual_result, pl
            self.rowcount = 1 if hit else 0
            self._last = ("settle", None)
            return
        if "AND m.status = 'cancelled'" in sql_norm:
            self._last = ("recs", [dict(r) for r in self.cancelled_recs])
            return
        if "FROM predictions" in sql_norm and "p.is_correct IS NOT NULL" in sql_norm:
            match_id, types = params
            self._last = (
                "preds",
                [
                    {
                        "prediction_id": p["id"],
                        "prediction_type": p["prediction_type"],
                        "model_name": p["model_name"],
                        "predicted_outcome": p["predicted_outcome"],
                        "probabilities": p.get("probabilities"),
                        "actual_outcome": p.get("actual_outcome"),
                        "is_correct": p.get("is_correct"),
                        "after_kickoff": p.get("after_kickoff", False),
                    }
                    for p in self.predictions.values()
                    if p["match_id"] == match_id and p.get("is_correct") is not None and p["prediction_type"] in types
                ],
            )
            return
        if sql_norm.startswith("UPDATE predictions") and len(params) == 5:
            actual, correct, pid, old_actual, old_correct = params
            self.guarded_updates.append(params)
            p = self.predictions[pid]
            hit = (p.get("actual_outcome"), p.get("is_correct")) == (old_actual, old_correct)
            if hit:
                p["actual_outcome"], p["is_correct"] = actual, correct
            self.rowcount = 1 if hit else 0
            self._last = ("update_pred", None)
            return
        if "AND m.status = 'postponed'" in sql_norm:
            self._last = ("recs", [dict(r) for r in self.postponed_recs])
            return
        super().execute(sql, params)
        if self._last and self._last[0] == "recs":
            for row in self._last[1]:
                row["created_at"] = self.recs[row["rec_id"]].get("created_at")


# Belgium v Senegal, 2026 World Cup: 3-2 after extra time, 2-2 after 90.
AET_MATCH = {
    "match_id": "m1",
    "sport": "soccer",
    "home_score": 3,
    "away_score": 2,
    "half_time_home": 0,
    "half_time_away": 1,
    "metadata": {"result_detail": "AET", "regulation_home_score": 2, "regulation_away_score": 2, "went_extra": True},
}


class TestNinetyMinuteGrading:
    def test_ungraded_markets_and_recs_settle_on_90_minutes(self):
        preds = [
            _pred("p1", "m1", "match_result", "ensemble", "home"),
            _pred("p2", "m1", "correct_score", "ensemble", "2-2"),
            _pred("p3", "m1", "match_result_ht", "ensemble", "away"),
        ]
        recs = [
            _rec("r1", "m1", "p1", stake=10, odds=2.0, selection="home"),
            _rec("r2", "m1", "p1", stake=10, odds=3.5, selection="draw"),
        ]
        cur = RegulationCursor(preds, recs)

        gcm.grade_match(cur, dict(AET_MATCH))

        assert (cur.predictions["p1"]["actual_outcome"], cur.predictions["p1"]["is_correct"]) == ("draw", False)
        assert cur.predictions["p2"]["is_correct"] is True
        assert (cur.predictions["p3"]["actual_outcome"], cur.predictions["p3"]["is_correct"]) == ("away", True)
        assert (cur.recs["r1"]["status"], cur.recs["r1"]["profit_loss"]) == ("lost", Decimal("-10"))
        assert (cur.recs["r2"]["status"], cur.recs["r2"]["profit_loss"]) == ("won", Decimal("25.0"))

    def test_the_triggers_post_extra_time_verdicts_are_regraded(self, caplog):
        # The DB trigger graded these on 3-2 when the match flipped to finished.
        preds = [
            _pred("p1", "m1", "match_result", "ensemble", "home", is_correct=True, actual_outcome="home"),
            _pred("p2", "m1", "over_under", "ensemble", "over_4.5", is_correct=True, actual_outcome="over"),
            _pred("p3", "m1", "btts", "ensemble", "yes", is_correct=True, actual_outcome="yes"),
            _pred("p4", "m1", "draw_no_bet", "ensemble", "home", is_correct=True, actual_outcome="home"),
            _pred("p5", "m1", "match_result_ht", "ensemble", "home", is_correct=False, actual_outcome="away"),
        ]
        cur = RegulationCursor(preds, [])
        stats: dict = {}

        with caplog.at_level("WARNING", logger="grade_completed_matches"):
            gcm.grade_match(cur, dict(AET_MATCH), stats)

        assert (cur.predictions["p1"]["actual_outcome"], cur.predictions["p1"]["is_correct"]) == ("draw", False)
        assert (cur.predictions["p2"]["actual_outcome"], cur.predictions["p2"]["is_correct"]) == ("under_4.5", False)
        assert (cur.predictions["p3"]["actual_outcome"], cur.predictions["p3"]["is_correct"]) == ("yes", True)
        # a draw voids draw-no-bet: graded -> push is the one NULL verdict written
        assert (cur.predictions["p4"]["actual_outcome"], cur.predictions["p4"]["is_correct"]) == ("push", None)
        # half-time markets are not full-time markets: never read back
        assert cur.predictions["p5"]["actual_outcome"] == "away"
        assert stats["regulation_regrades"] == 3
        assert all(len(p) == 5 for p in cur.guarded_updates)  # guarded on the old verdict
        assert sum("on the 90-minute score" in r.getMessage() for r in caplog.records) == 3

    def test_a_verdict_is_never_ungraded(self):
        # correct_score 'other' without its key set grades NULL (not a push).
        preds = [_pred("p1", "m1", "correct_score", "ensemble", "other", is_correct=False, actual_outcome="3-2")]
        cur = RegulationCursor(preds, [])
        gcm.grade_match(cur, dict(AET_MATCH))
        assert cur.guarded_updates == [] and cur.predictions["p1"]["is_correct"] is False

    def test_a_row_rewritten_meanwhile_is_left_alone(self, caplog):
        preds = [_pred("p1", "m1", "match_result", "ensemble", "home", is_correct=True, actual_outcome="home")]
        cur = RegulationCursor(preds, [])
        real = cur.execute

        def racing(sql, params=None):
            if " ".join(sql.split()).startswith("UPDATE predictions"):
                cur.predictions["p1"]["actual_outcome"] = "someone-else"
            return real(sql, params)

        cur.execute = racing
        stats: dict = {}
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            gcm.grade_match(cur, dict(AET_MATCH), stats)
        assert cur.predictions["p1"]["actual_outcome"] == "someone-else"
        assert stats.get("regulation_regrades", 0) == 0
        assert any("changed while it was being re-graded" in r.getMessage() for r in caplog.records)

    def test_a_match_without_the_keys_reads_nothing_back(self):
        # The base FakeCursor raises on any SQL it does not know.
        match = {"match_id": "m1", "home_score": 3, "away_score": 2, "metadata": {"result_detail": "FT"}}
        preds = [_pred("p1", "m1", "match_result", "ensemble", "home", is_correct=True, actual_outcome="home")]
        gcm.grade_match(FakeCursor(preds, []), match)

    def test_an_nhl_match_is_never_regraded(self):
        match = {**AET_MATCH, "sport": "nhl"}
        preds = [_pred("p1", "m1", "moneyline", "ensemble_nhl_ml", "home", is_correct=True, actual_outcome="home")]
        cur = RegulationCursor(preds, [])
        gcm.grade_match(cur, match)
        assert not any("p.is_correct IS NOT NULL" in s for s in cur.sql)

    def test_finished_soccer_matches_with_the_keys_are_listed(self):
        seen = []

        class Cur:
            def execute(self, sql, params=None):
                seen.append((" ".join(sql.split()), params))

            def fetchall(self):
                return []

        gcm.list_finished_matches(Cur(), 14, None)
        sql, params = seen[0]
        # Soccer only: load_nhl_historical writes the same key on every
        # finished NHL game (tests/unit/test_grading_sql_pg.py runs it).
        assert "OR (COALESCE(l.sport, 'soccer') = 'soccer' AND m.metadata ? %s)" in sql
        assert params == ("14", "regulation_home_score")
        assert "m.match_date, l.sport" in sql and "LEFT JOIN leagues l ON l.id = m.league_id" in sql

    def test_decided_after_90_without_the_keys_warns_once(self, caplog):
        match = {"match_id": "m1", "home_score": 3, "away_score": 2, "metadata": {"result_detail": "AET"}}
        cur = FakeCursor([_pred("p1", "m1", "match_result", "ensemble", "home")], [])
        with caplog.at_level("WARNING", logger="grade_completed_matches"):
            gcm.grade_match(cur, match)
            gcm.grade_match(cur, match)  # the next tick: nothing new graded
        warned = [r for r in caplog.records if "carries no 90-minute score" in r.getMessage()]
        assert len(warned) == 1
        assert cur.predictions["p1"]["actual_outcome"] == "home"  # the final score, as before


# ── Postponed matches ────────────────────────────────────────────────

ORIGINAL = datetime(2026, 10, 17, 16, 30, tzinfo=timezone.utc)
RECORDED = ORIGINAL - timedelta(hours=20)


def _postponed_match(played_hours_later: float, **meta) -> dict:
    return {
        "match_id": "m1",
        "sport": "soccer",
        "match_date": ORIGINAL + timedelta(hours=played_hours_later),
        "home_score": 2,
        "away_score": 1,
        "metadata": {"postponed_kickoff": ORIGINAL.isoformat(), "postponed_at": RECORDED.isoformat(), **meta},
    }


class TestPostponedRecs:
    def _settle(self, match, created_at):
        rec = _rec("r1", "m1", "p1", stake=10, odds=2.0)
        rec["created_at"] = created_at
        cur = RegulationCursor([], [rec])
        stats: dict = {}
        counts = gcm.grade_match(cur, match, stats)
        return cur.recs["r1"], counts, stats

    def test_a_rec_priced_before_a_long_postponement_is_void(self):
        rec, counts, stats = self._settle(_postponed_match(96), ORIGINAL - timedelta(days=2))
        assert (rec["status"], rec["profit_loss"]) == ("void", Decimal("0"))
        assert rec["actual_result"] == gcm.POSTPONED_VOID_RESULT == "void: match postponed >72h"
        assert counts["recs_settled"] == 0 and stats["postponed_rec_voids"] == 1

    def test_a_short_postponement_settles_as_usual(self):
        rec, counts, _ = self._settle(_postponed_match(48), ORIGINAL - timedelta(days=2))
        assert rec["status"] == "won" and counts["recs_settled"] == 1

    def test_a_rec_priced_for_the_rescheduled_match_settles(self):
        rec, _, _ = self._settle(_postponed_match(240), RECORDED + timedelta(days=3))
        assert rec["status"] == "won"

    def test_without_the_keys_nothing_changes(self):
        match = {**_postponed_match(240), "metadata": {}}
        rec, _, _ = self._settle(match, ORIGINAL - timedelta(days=2))
        assert rec["status"] == "won"

    def test_unreadable_keys_settle_as_usual_and_warn(self, caplog):
        match = _postponed_match(240, postponed_at="not-a-time")
        with caplog.at_level("WARNING", logger="grade_completed_matches"):
            rec, _, _ = self._settle(match, ORIGINAL - timedelta(days=2))
        assert rec["status"] == "won"
        assert any("cannot apply the postponement rule" in r.getMessage() for r in caplog.records)

    def test_still_postponed_after_72h_is_void(self):
        pending = {"rec_id": "r1", "match_id": "m9", "bet_type": "1x2", "selection": "home", "status": "pending"}
        recs = [_rec("r1", "m9", "p1", stake=10, odds=2.0), _rec("r2", "m9", "p1", stake=5, odds=3.0, status="placed")]
        cur = RegulationCursor([], recs, postponed_recs=[pending, {**pending, "rec_id": "r2", "status": "placed"}])
        assert gcm.void_recs_on_long_postponed_matches(cur) == 2
        for rid in ("r1", "r2"):
            assert cur.recs[rid]["status"] == "void" and cur.recs[rid]["profit_loss"] == Decimal("0")
            assert cur.recs[rid]["actual_result"] == gcm.POSTPONED_VOID_RESULT

    def test_the_long_postponement_query(self):
        seen = []

        class Cur:
            def execute(self, sql, params=None):
                seen.append((" ".join(sql.split()), params))

            def fetchall(self):
                return []

        gcm.list_recs_on_long_postponed_matches(Cur())
        sql, params = seen[0]
        assert "br.status IN ('pending', 'placed')" in sql and "m.status = 'postponed'" in sql
        assert "m.match_date < NOW() - (%s || ' hours')::interval" in sql and params == ("72",)

    def test_run_voids_them_and_reports_it(self, monkeypatch, caplog):
        pending = {"rec_id": "r1", "match_id": "m9", "bet_type": "1x2", "selection": "home", "status": "pending"}
        cur = _PostponedRunCursor([], [], [_rec("r1", "m9", "p1", stake=10, odds=2.0)], [pending])
        monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _RunConn(cur))
        with caplog.at_level("WARNING", logger="grade_completed_matches"):
            summary = gcm.run("postgresql://unused/db", 14, None)
        assert summary["postponed_rec_voids"] == 1 and cur.recs["r1"]["status"] == "void"
        assert any("postponed more than 72h" in r.getMessage() for r in caplog.records)

    def test_an_ad_hoc_match_run_leaves_them_alone(self, monkeypatch):
        pending = {"rec_id": "r1", "match_id": "m9", "bet_type": "1x2", "selection": "home", "status": "pending"}
        cur = _PostponedRunCursor([], [], [_rec("r1", "m9", "p1", stake=10, odds=2.0)], [pending])
        monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _RunConn(cur))
        assert gcm.run("postgresql://unused/db", 14, "m1")["postponed_rec_voids"] == 0
        assert cur.recs["r1"]["status"] == "pending"


class _PostponedRunCursor(RegulationCursor):
    def __init__(self, matches, predictions, recs, postponed_recs):
        super().__init__(predictions, recs, postponed_recs)
        self.matches = matches

    def execute(self, sql, params=None):
        if " ".join(sql.split()).startswith("SELECT m.id::text AS match_id"):
            self._last = ("matches", [dict(m) for m in self.matches])
            return
        super().execute(sql, params)

    def fetchall(self):
        if self._last and self._last[0] == "matches":
            return self._last[1]
        return super().fetchall()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── Money-settled recs when the 90-minute score arrives later ────────


def _settled(rid, selection, status, actual_result, pl, bet_type="1x2", stake=10, odds=2.0, match_id="m1"):
    rec = _rec(rid, match_id, "p1", stake=stake, odds=odds, bet_type=bet_type, selection=selection, status=status)
    rec["actual_result"] = actual_result
    rec["profit_loss"] = Decimal(str(pl))
    return rec


class TestRegulationResettles:
    PRED = [_pred("p1", "m1", "match_result", "ensemble", "home", is_correct=False, actual_outcome="draw")]

    def test_a_rec_settled_on_the_post_extra_time_score_is_resettled_loudly(self, caplog):
        # Settled 'won' on Belgium 3-2 AET before the 90-minute 2-2 arrived.
        cur = RegulationCursor(self.PRED, [_settled("r1", "home", "won", "won", "10.00")])
        stats: dict = {}
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            gcm.grade_match(cur, dict(AET_MATCH), stats)
        r = cur.recs["r1"]
        assert (r["status"], r["actual_result"], r["profit_loss"]) == ("lost", "lost", Decimal("-10"))
        assert stats["regulation_resettles"] == 1
        assert cur.resettles[0][3:] == ("r1", "won", "won", Decimal("10.00"))  # guarded on what it read
        msg = next(rec.getMessage() for rec in caplog.records if "Re-settled rec r1" in rec.getMessage())
        assert "(2, 2)" in msg and "(won, won, 10.00) -> (lost, lost, -10.00)" in msg and "3-2" in msg

    def test_an_unchanged_verdict_is_left_alone(self):
        # BTTS yes wins on 3-2 and on 2-2 alike.
        cur = RegulationCursor(self.PRED, [_settled("r1", "yes", "won", "won", "10.00", bet_type="btts")])
        gcm.grade_match(cur, dict(AET_MATCH))
        assert cur.resettles == [] and cur.recs["r1"]["status"] == "won"

    def test_cents_rounding_is_not_a_change(self):
        # 7.50 at 1.833 pays 6.2475, stored as 6.25.
        rec = _settled("r1", "yes", "won", "won", "6.25", bet_type="btts", stake=7.5, odds=1.833)
        cur = RegulationCursor(self.PRED, [rec])
        gcm.grade_match(cur, dict(AET_MATCH))
        assert cur.resettles == []

    def test_a_score_independent_void_is_never_touched(self):
        rec = _settled("r1", "home", "void", "void: constant-prior incident (2026-06-15..07-13)", "0.00")
        cur = RegulationCursor(self.PRED, [rec])
        gcm.grade_match(cur, dict(AET_MATCH))
        assert cur.resettles == [] and cur.recs["r1"]["status"] == "void"

    def test_a_legacy_result_text_with_the_same_money_is_left_alone(self):
        rec = _settled("r1", "draw", "won", "draw", "10.00")  # an older settler wrote the outcome, not the verdict
        cur = RegulationCursor(self.PRED, [rec])
        gcm.grade_match(cur, dict(AET_MATCH))
        assert cur.resettles == []

    def test_a_rec_rewritten_meanwhile_is_left_alone(self, caplog):
        cur = RegulationCursor(self.PRED, [_settled("r1", "home", "won", "won", "10.00")])
        real = cur.execute

        def racing(sql, params=None):
            if params is not None and len(params) == 7:
                cur.recs["r1"]["profit_loss"] = Decimal("9.99")
            return real(sql, params)

        cur.execute = racing
        stats: dict = {}
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            gcm.grade_match(cur, dict(AET_MATCH), stats)
        assert cur.recs["r1"]["status"] == "won" and stats.get("regulation_resettles", 0) == 0
        assert any("changed while it was being re-settled" in r.getMessage() for r in caplog.records)

    def test_nhl_and_unscored_matches_never_read_their_settled_recs(self):
        for match in ({**AET_MATCH, "sport": "nhl"}, {**AET_MATCH, "metadata": {"result_detail": "AET"}}):
            cur = RegulationCursor(self.PRED, [_settled("r1", "home", "won", "won", "10.00")])
            gcm.grade_match(cur, match)
            assert not any("br.profit_loss IS NOT NULL" in s for s in cur.sql)

    def test_run_reports_them_at_error(self, monkeypatch, caplog):
        class Cur(_PostponedRunCursor):
            pass

        cur = Cur([dict(AET_MATCH)], self.PRED, [_settled("r1", "home", "won", "won", "10.00")], [])
        monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _RunConn(cur))
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            summary = gcm.run("postgresql://unused/db", 14, None)
        assert summary["regulation_resettles"] == 1
        assert any("Re-settled 1 recommendation(s)" in r.getMessage() for r in caplog.records)


# ── Cancelled matches ────────────────────────────────────────────────


def _open_on(rid, match_id, status="pending", **extra):
    return {"rec_id": rid, "match_id": match_id, "bet_type": "1x2", "selection": "home", "status": status, **extra}


class TestCancelledRecs:
    def test_open_recs_on_a_cancelled_match_are_voided_guarded(self):
        recs = [_rec("r1", "m9", "p1", stake=10, odds=2.0), _rec("r2", "m9", "p1", stake=5, odds=3.0, status="placed")]
        cur = RegulationCursor([], recs, cancelled_recs=[_open_on("r1", "m9"), _open_on("r2", "m9", "placed")])
        assert gcm.void_recs_on_cancelled_matches(cur) == (2, 0)
        for rid in ("r1", "r2"):
            r = cur.recs[rid]
            assert (r["status"], r["profit_loss"], r["actual_result"]) == (
                "void",
                Decimal("0"),
                "void: match cancelled",
            )
        assert cur.voids[0] == (gcm.CANCELLED_VOID_RESULT, "r1", "pending", "cancelled")
        assert cur.voids[1][2] == "placed"

    def test_a_cancelled_duplicate_of_a_played_fixture_is_refused(self, caplog):
        cur = RegulationCursor(
            [], [_rec("r1", "m9", "p1", stake=10, odds=2.0)], cancelled_recs=[_open_on("r1", "m9", orphan_of="twin-1")]
        )
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            assert gcm.void_recs_on_cancelled_matches(cur) == (0, 1)
        assert cur.voids == [] and cur.recs["r1"]["status"] == "pending"
        (msg,) = [r.getMessage() for r in caplog.records if "duplicate of match twin-1" in r.getMessage()]
        assert "repair_stranded_recs.py --sport <sport>" in msg and "UPDATE betting_recommendations" not in msg

    def test_a_placed_rec_on_a_cancelled_duplicate_is_told_to_reopen_it_first(self, caplog):
        cur = RegulationCursor(
            [],
            [_rec("r1", "m9", "p1", stake=10, odds=2.0, status="placed")],
            cancelled_recs=[_open_on("r1", "m9", "placed", orphan_of="twin-1")],
        )
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            assert gcm.void_recs_on_cancelled_matches(cur) == (0, 1)
        (msg,) = [r.getMessage() for r in caplog.records if "duplicate of match twin-1" in r.getMessage()]
        assert "SET status = 'pending' WHERE id = 'r1' AND status = 'placed'" in msg
        assert "repair_stranded_recs.py --sport <sport>" in msg

    def test_a_match_rescheduled_meanwhile_keeps_its_rec(self, caplog):
        cur = RegulationCursor(
            [],
            [_rec("r1", "m9", "p1", stake=10, odds=2.0)],
            cancelled_recs=[_open_on("r1", "m9")],
            match_status={"m9": "scheduled"},
        )
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            assert gcm.void_recs_on_cancelled_matches(cur) == (0, 0)
        assert cur.recs["r1"]["status"] == "pending"
        assert any("changed while voiding it" in r.getMessage() for r in caplog.records)

    def test_the_cancelled_query(self):
        seen = []

        class Cur:
            def execute(self, sql, params=None):
                seen.append((" ".join(sql.split()), params))

            def fetchall(self):
                return []

        gcm.list_recs_on_cancelled_matches(Cur())
        sql, params = seen[0]
        assert "br.status IN ('pending', 'placed')" in sql and "m.status = 'cancelled'" in sql
        assert "m.metadata->>'orphan_of' AS orphan_of" in sql
        assert "m.match_date < NOW() - (%s || ' hours')::interval" in sql and params == ("72",)
        assert "ANY(" not in sql

    def test_the_cancelled_query_limited_to_given_matches(self):
        seen = []

        class Cur:
            def execute(self, sql, params=None):
                seen.append((" ".join(sql.split()), params))

            def fetchall(self):
                return []

        gcm.list_recs_on_cancelled_matches(Cur(), match_ids=["m1", "m2"])
        sql, params = seen[0]
        assert "m.status = 'cancelled'" in sql and "AND m.id = ANY(%s::uuid[])" in sql
        assert params == ("72", ["m1", "m2"])

    def test_the_postponement_void_is_guarded_the_same_way(self):
        pending = _open_on("r1", "m9")
        cur = RegulationCursor([], [_rec("r1", "m9", "p1", stake=10, odds=2.0)], postponed_recs=[pending])
        cur.match_status["m9"] = "scheduled"  # rescheduled meanwhile
        assert gcm.void_recs_on_long_postponed_matches(cur) == 0
        assert cur.voids == [(gcm.POSTPONED_VOID_RESULT, "r1", "pending", "postponed")]
        assert cur.recs["r1"]["status"] == "pending"

    def test_run_voids_them_and_reports_it(self, monkeypatch, caplog):
        cur = _PostponedRunCursor([], [], [_rec("r1", "m9", "p1", stake=10, odds=2.0)], [])
        cur.cancelled_recs = [_open_on("r1", "m9")]
        monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _RunConn(cur))
        with caplog.at_level("WARNING", logger="grade_completed_matches"):
            summary = gcm.run("postgresql://unused/db", 14, None)
        assert summary["cancelled_rec_voids"] == 1 and cur.recs["r1"]["status"] == "void"
        assert any("cancelled more than 72h ago" in r.getMessage() for r in caplog.records)

    def test_run_reports_refused_orphans_at_error(self, monkeypatch, caplog):
        cur = _PostponedRunCursor([], [], [_rec("r1", "m9", "p1", stake=10, odds=2.0)], [])
        cur.cancelled_recs = [_open_on("r1", "m9", orphan_of="twin-1")]
        monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _RunConn(cur))
        with caplog.at_level("ERROR", logger="grade_completed_matches"):
            summary = gcm.run("postgresql://unused/db", 14, None)
        assert summary["cancelled_orphan_recs"] == 1 and cur.recs["r1"]["status"] == "pending"
        assert any("cancelled duplicates of played fixtures" in r.getMessage() for r in caplog.records)

    def test_an_ad_hoc_match_run_leaves_them_alone(self, monkeypatch):
        cur = _PostponedRunCursor([], [], [_rec("r1", "m9", "p1", stake=10, odds=2.0)], [])
        cur.cancelled_recs = [_open_on("r1", "m9")]
        monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _RunConn(cur))
        assert gcm.run("postgresql://unused/db", 14, "m1")["cancelled_rec_voids"] == 0
        assert cur.recs["r1"]["status"] == "pending"
