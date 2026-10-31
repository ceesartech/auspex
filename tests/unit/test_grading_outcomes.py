"""Per-market grading math — pure functions, no DB or HTTP."""

from __future__ import annotations

import importlib.util
import sys
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


go = _load("grading_outcomes", "grading_outcomes.py")


# ── Soccer match_result (1X2) ────────────────────────────────────────


class TestSoccerMatchResult:
    def test_home_win(self):
        assert go.soccer_match_result(2, 1) == "home"

    def test_away_win(self):
        assert go.soccer_match_result(0, 3) == "away"

    def test_draw(self):
        assert go.soccer_match_result(1, 1) == "draw"
        assert go.soccer_match_result(0, 0) == "draw"


# ── NHL regulation (3-way via metadata) ──────────────────────────────


class TestNhlRegulation:
    def test_reads_metadata_field(self):
        assert go.nhl_regulation_outcome({"regulation_winner": "home"}) == "home"
        assert go.nhl_regulation_outcome({"regulation_winner": "tie"}) == "tie"
        assert go.nhl_regulation_outcome({"regulation_winner": "away"}) == "away"

    def test_legacy_draw_reads_as_tie(self):
        # The ESPN results path wrote 'draw' until 2026-10; the model's
        # label for a regulation tie is 'tie'.
        assert go.nhl_regulation_outcome({"regulation_winner": "draw"}) == "tie"

    def test_missing_field_returns_none(self):
        # Older NHL rows that haven't been backfilled with regulation
        # split don't have the field. Returning None leaves is_correct
        # NULL and the grader will retry next run after backfill.
        assert go.nhl_regulation_outcome({}) is None
        assert go.nhl_regulation_outcome(None) is None


# ── NHL moneyline / puck_line / total ────────────────────────────────


class TestNhlMoneyline:
    def test_home_win_after_ot(self):
        # NHL moneyline scores from final (OT/SO) score.
        assert go.nhl_moneyline_outcome(4, 3) == "home"

    def test_away_win(self):
        assert go.nhl_moneyline_outcome(2, 5) == "away"

    def test_tie_returns_none(self):
        # NHL has no regular-season ties post-2005, but if the data
        # shows one (likely a logging bug), refuse to call it either way.
        assert go.nhl_moneyline_outcome(2, 2) is None


class TestNhlPuckLine:
    def test_cover_when_home_wins_by_2(self):
        assert go.nhl_puck_line_outcome(3, 1) == "cover"

    def test_cover_when_home_wins_by_more(self):
        assert go.nhl_puck_line_outcome(5, 0) == "cover"

    def test_no_cover_when_home_wins_by_1(self):
        # 1-goal win doesn't cover -1.5.
        assert go.nhl_puck_line_outcome(3, 2) == "no_cover"

    def test_no_cover_when_away_wins(self):
        assert go.nhl_puck_line_outcome(1, 3) == "no_cover"

    def test_no_cover_when_tied(self):
        # Tie at final (data issue) doesn't cover.
        assert go.nhl_puck_line_outcome(2, 2) == "no_cover"


class TestNhlTotal:
    def test_over_at_canonical_line(self):
        assert go.nhl_total_outcome(4, 3) == "over"
        # 6 goals > 5.5 → over.
        assert go.nhl_total_outcome(3, 3) == "over"

    def test_under_at_canonical_line(self):
        assert go.nhl_total_outcome(2, 2) == "under"
        # 5 goals < 5.5 → under.
        assert go.nhl_total_outcome(5, 0) == "under"

    def test_alternate_line_param(self):
        # Defensive — function accepts a line override for future
        # alternate-total support.
        assert go.nhl_total_outcome(4, 4, line=7.5) == "over"
        assert go.nhl_total_outcome(3, 3, line=6.5) == "under"


# ── Soccer derived markets ───────────────────────────────────────────


class TestBtts:
    def test_yes_when_both_score(self):
        assert go.btts_outcome(2, 1) == "yes"

    def test_no_when_clean_sheet(self):
        assert go.btts_outcome(3, 0) == "no"
        assert go.btts_outcome(0, 2) == "no"

    def test_no_when_both_zero(self):
        assert go.btts_outcome(0, 0) == "no"


class TestOverUnder:
    def test_over_25_at_3_goals(self):
        assert go._ou_outcome_for_line(2, 1, 2.5) == "over_2.5"

    def test_under_25_at_2_goals(self):
        assert go._ou_outcome_for_line(1, 1, 2.5) == "under_2.5"

    def test_push_at_integer_line(self):
        # 3.0 line with exactly 3 goals → push (over_under is the only
        # bet where pushes are common — line is usually X.5 but books
        # do offer X.0).
        assert go._ou_outcome_for_line(2, 1, 3.0) == "push_3"

    def test_extract_line_from_predicted_outcome(self):
        # The grading dispatcher pulls the line off the predicted_outcome
        # string. Locked here so the parsing pattern doesn't drift.
        assert go._extract_over_under_line("over_2.5") == 2.5
        assert go._extract_over_under_line("under_3") == 3.0
        assert go._extract_over_under_line("under_4.5") == 4.5

    def test_extract_line_returns_none_on_garbage(self):
        assert go._extract_over_under_line(None) is None
        assert go._extract_over_under_line("home") is None
        assert go._extract_over_under_line("over") is None
        assert go._extract_over_under_line("over_abc") is None


class TestDoubleChance:
    """actual_outcome stores the BASE result; correctness is coverage-set
    membership via grade_prediction(prediction_type='double_chance').
    The old single-label convention ('1X' for home-or-draw) mis-graded
    every '12' pick and every 'X2'-pick-on-a-draw."""

    def test_actual_is_base_result(self):
        assert go.double_chance_outcome(3, 1) == "home"
        assert go.double_chance_outcome(1, 1) == "draw"
        assert go.double_chance_outcome(0, 2) == "away"

    def test_coverage_grading_every_cell(self):
        # (predicted, actual) -> expected verdict, all 9 combinations.
        expected = {
            ("1X", "home"): True,
            ("1X", "draw"): True,
            ("1X", "away"): False,
            ("12", "home"): True,
            ("12", "draw"): False,
            ("12", "away"): True,
            ("X2", "home"): False,
            ("X2", "draw"): True,
            ("X2", "away"): True,
        }
        for (pred, actual), want in expected.items():
            got = go.grade_prediction(pred, actual, prediction_type="double_chance")
            assert got is want, f"{pred} vs {actual}: got {got}, want {want}"

    def test_malformed_selection_grades_null(self):
        assert go.grade_prediction("2X", "home", prediction_type="double_chance") is None

    def test_other_markets_still_grade_by_equality(self):
        assert go.grade_prediction("home", "home", prediction_type="match_result") is True
        assert go.grade_prediction("home", "away", prediction_type="match_result") is False
        # And without prediction_type (back-compat callers).
        assert go.grade_prediction("2-1", "2-1") is True


class TestDrawNoBet:
    def test_home_win(self):
        assert go.draw_no_bet_outcome(2, 0) == "home"

    def test_away_win(self):
        assert go.draw_no_bet_outcome(1, 3) == "away"

    def test_draw_is_push(self):
        assert go.draw_no_bet_outcome(1, 1) == "push"


class TestCorrectScore:
    def test_exact_scoreline(self):
        assert go.correct_score_outcome(2, 1) == "2-1"
        assert go.correct_score_outcome(0, 0) == "0-0"


# ── Top-level dispatch ───────────────────────────────────────────────


class TestActualOutcome:
    def test_routes_soccer_match_result_via_scores(self):
        assert (
            go.actual_outcome(
                prediction_type="match_result",
                model_name="ensemble",
                predicted_outcome="home",
                home_score=2,
                away_score=1,
                metadata={},
            )
            == "home"
        )

    def test_routes_nhl_regulation_via_metadata(self):
        # Same prediction_type as soccer match_result but the model
        # name distinguishes — should pull from metadata not scores.
        assert (
            go.actual_outcome(
                prediction_type="match_result",
                model_name="ensemble_nhl_reg",
                predicted_outcome="home",
                home_score=4,
                away_score=3,  # final score (after OT)
                metadata={"regulation_winner": "tie"},
            )
            == "tie"
        )

    def test_returns_none_when_scores_missing(self):
        # Postponed / cancelled matches that flipped to finished
        # without scores → ungradable, leave is_correct NULL.
        assert (
            go.actual_outcome(
                prediction_type="moneyline",
                model_name="ensemble_nhl_ml",
                predicted_outcome="home",
                home_score=None,
                away_score=None,
                metadata=None,
            )
            is None
        )

    def test_routes_nhl_moneyline(self):
        assert (
            go.actual_outcome(
                prediction_type="moneyline",
                model_name="ensemble_nhl_ml",
                predicted_outcome="home",
                home_score=4,
                away_score=3,
                metadata=None,
            )
            == "home"
        )

    def test_routes_nhl_puck_line(self):
        assert (
            go.actual_outcome(
                prediction_type="spread",
                model_name="ensemble_nhl_pl",
                predicted_outcome="cover",
                home_score=4,
                away_score=1,
                metadata=None,
            )
            == "cover"
        )

    def test_routes_nhl_total(self):
        assert (
            go.actual_outcome(
                prediction_type="total",
                model_name="ensemble_nhl_tot",
                predicted_outcome="over",
                home_score=4,
                away_score=2,
                metadata=None,
            )
            == "over"
        )

    def test_routes_over_under_with_line_extraction(self):
        # Soccer over_under emits predicted_outcome="over_2.5"; the
        # dispatcher parses the line out and computes the matching
        # actual_outcome shape.
        assert (
            go.actual_outcome(
                prediction_type="over_under",
                model_name="ensemble",
                predicted_outcome="over_2.5",
                home_score=2,
                away_score=1,
                metadata=None,
            )
            == "over_2.5"
        )

    def test_returns_none_for_ungraded_markets(self):
        # player_prop isn't in the dispatch table — must fall through to
        # None so grading skips it cleanly. (asian_handicap and the
        # derived markets graduated to real branches.)
        assert (
            go.actual_outcome(
                prediction_type="player_prop",
                model_name="ensemble",
                predicted_outcome="home_over_1.5",
                home_score=2,
                away_score=1,
                metadata=None,
            )
            is None
        )


# ── Grading verdict (is_correct + push) ──────────────────────────────


class TestGradePrediction:
    def test_correct_pick(self):
        assert go.grade_prediction("home", "home") is True

    def test_incorrect_pick(self):
        assert go.grade_prediction("home", "away") is False

    def test_push_returns_none(self):
        # Pushes aren't right or wrong — is_correct stays NULL so
        # accuracy stats don't count them as misses.
        assert go.grade_prediction("over_3", "push_3") is None
        assert go.grade_prediction("home", "push") is None

    def test_ungradable_actual_returns_none(self):
        assert go.grade_prediction("home", None) is None


class TestIsPush:
    def test_bare_push(self):
        assert go.is_push("push") is True

    def test_lined_push(self):
        assert go.is_push("push_3") is True
        assert go.is_push("push_2.5") is True

    def test_non_push_outcomes(self):
        assert go.is_push("over_2.5") is False
        assert go.is_push("home") is False
        assert go.is_push(None) is False


# ── NBA spread (variable closing line, model_name dispatch) ──────────


class TestNbaSpread:
    """NBA spread differs from NHL puck_line: variable closing line
    stored in features_cache, dispatched by model_name='ensemble_nba_sp'.
    The bet is bidirectional (home / away / push) per game's actual
    line, not a fixed -1.5."""

    def test_home_covers_when_margin_beats_line(self):
        # Home -5 favored. Final 110-100, margin = 10 > 5 → home covers.
        assert go.nba_spread_outcome(110, 100, -5.0) == "home"

    def test_home_fails_to_cover_when_margin_below_line(self):
        # Home -5 favored, wins 103-100. Margin 3 < 5 → away covers.
        assert go.nba_spread_outcome(103, 100, -5.0) == "away"

    def test_home_dog_covers_with_close_loss(self):
        # Home +7 dog, loses 100-105. Margin -5 > -7 → home covers.
        assert go.nba_spread_outcome(100, 105, 7.0) == "home"

    def test_push_at_integer_line(self):
        # Half-point lines never push; integer lines (rare) can.
        # Home -3, wins by exactly 3 → push.
        assert go.nba_spread_outcome(103, 100, -3.0) == "push"

    def test_half_point_line_no_push(self):
        # The .5 half-point design eliminates pushes. Either home
        # covers or doesn't.
        assert go.nba_spread_outcome(108, 100, -7.5) == "home"
        assert go.nba_spread_outcome(106, 100, -7.5) == "away"

    def test_dispatch_reads_features_for_nba(self):
        # End-to-end through actual_outcome: NBA spread reads the
        # closing_spread_home from features instead of using NHL's
        # fixed -1.5.
        result = go.actual_outcome(
            prediction_type="spread",
            model_name="ensemble_nba_sp",
            predicted_outcome="home",
            home_score=110,
            away_score=100,
            features={"closing_spread_home": -5.0},
        )
        assert result == "home"

    def test_dispatch_returns_none_when_features_missing(self):
        # Defensive: if features_cache row is missing (or doesn't
        # carry the line), refuse to grade — would otherwise grade
        # the NBA pick at NHL's fixed -1.5 puck line, completely
        # wrong.
        result = go.actual_outcome(
            prediction_type="spread",
            model_name="ensemble_nba_sp",
            predicted_outcome="home",
            home_score=110,
            away_score=100,
            features=None,
        )
        assert result is None
        result = go.actual_outcome(
            prediction_type="spread",
            model_name="ensemble_nba_sp",
            predicted_outcome="home",
            home_score=110,
            away_score=100,
            features={},  # features row exists but no line key
        )
        assert result is None

    def test_dispatch_still_returns_nhl_for_nhl_spread(self):
        # NHL spread (model_name=ensemble_nhl_pl) must still use the
        # fixed -1.5 puck-line logic regardless of any features
        # passed in. Tests that adding NBA dispatch didn't break NHL.
        result = go.actual_outcome(
            prediction_type="spread",
            model_name="ensemble_nhl_pl",
            predicted_outcome="cover",
            home_score=4,
            away_score=2,  # home wins by 2 → covers -1.5
            features={"closing_spread_home": 999.0},  # noise — should be ignored
        )
        assert result == "cover"


# ── NBA total (variable closing line, model_name dispatch) ───────────


class TestNbaTotal:
    """NBA total differs from NHL: variable line (typical 215-235)
    stored in features_cache, dispatched by model_name='ensemble_nba_tot'.
    """

    def test_over_at_variable_line(self):
        # Total 110+108=218 > line 215 → over.
        assert go.nba_total_outcome(110, 108, 215.0) == "over"

    def test_under_at_variable_line(self):
        # Total 100+105=205 < line 220 → under.
        assert go.nba_total_outcome(100, 105, 220.0) == "under"

    def test_push_at_integer_line(self):
        assert go.nba_total_outcome(110, 110, 220.0) == "push"

    def test_half_point_line_no_push(self):
        assert go.nba_total_outcome(110, 108, 217.5) == "over"
        assert go.nba_total_outcome(108, 108, 217.5) == "under"

    def test_dispatch_reads_features_for_nba(self):
        result = go.actual_outcome(
            prediction_type="total",
            model_name="ensemble_nba_tot",
            predicted_outcome="under",
            home_score=100,
            away_score=105,
            features={"closing_total_line": 218.5},
        )
        # 205 < 218.5 → under
        assert result == "under"

    def test_dispatch_returns_none_when_features_missing(self):
        # Same defensive case as spread: refuse to grade NBA total
        # at NHL's fixed 5.5.
        result = go.actual_outcome(
            prediction_type="total",
            model_name="ensemble_nba_tot",
            predicted_outcome="under",
            home_score=100,
            away_score=105,
            features=None,
        )
        assert result is None

    def test_dispatch_still_returns_nhl_for_nhl_total(self):
        # NHL total (ensemble_nhl_tot) keeps fixed 5.5 logic.
        result = go.actual_outcome(
            prediction_type="total",
            model_name="ensemble_nhl_tot",
            predicted_outcome="over",
            home_score=4,
            away_score=3,  # 7 > 5.5 → over
            features=None,
        )
        assert result == "over"


# ── NFL spread + total (mirrors NBA — variable closing line dispatch) ──


class TestNflSpread:
    """NFL spread uses the same line-as-feature design + variable
    closing line as NBA. Sport-agnostic math (reuses nba_spread_outcome).
    Dispatched by model_name='ensemble_nfl_sp'."""

    def test_dispatch_reads_features_for_nfl(self):
        # Home -7 favored. Final 31-21, margin = 10 > 7 → home covers.
        result = go.actual_outcome(
            prediction_type="spread",
            model_name="ensemble_nfl_sp",
            predicted_outcome="home",
            home_score=31,
            away_score=21,
            features={"closing_spread_home": -7.0},
        )
        assert result == "home"

    def test_dispatch_returns_none_when_features_missing(self):
        # Without the closing line, refuse to grade — would otherwise
        # fall through to NHL puck-line logic which is wrong for NFL.
        result = go.actual_outcome(
            prediction_type="spread",
            model_name="ensemble_nfl_sp",
            predicted_outcome="home",
            home_score=31,
            away_score=21,
            features=None,
        )
        assert result is None

    def test_dispatch_distinguishes_from_nba(self):
        # NFL spread should use features_cache the same way NBA does.
        # Defensive: a stray NBA-specific dispatch path wouldn't catch
        # NFL model_name and would grade as NHL puck-line.
        result = go.actual_outcome(
            prediction_type="spread",
            model_name="ensemble_nfl_sp",
            predicted_outcome="home",
            home_score=24,
            away_score=27,  # NFL margin -3
            features={"closing_spread_home": 2.5},  # NFL home dog by 2.5
        )
        # -3 > -2.5 is false → away covers
        assert result == "away"


class TestNflTotal:
    """NFL total mirrors NBA — variable line (typical 40-55) stored
    in features_cache, dispatched by model_name='ensemble_nfl_tot'."""

    def test_dispatch_reads_features_for_nfl(self):
        # Total 31+24=55 > line 44.5 → over.
        result = go.actual_outcome(
            prediction_type="total",
            model_name="ensemble_nfl_tot",
            predicted_outcome="over",
            home_score=31,
            away_score=24,
            features={"closing_total_line": 44.5},
        )
        assert result == "over"

    def test_dispatch_returns_none_when_features_missing(self):
        result = go.actual_outcome(
            prediction_type="total",
            model_name="ensemble_nfl_tot",
            predicted_outcome="under",
            home_score=17,
            away_score=14,
            features=None,
        )
        assert result is None

    def test_under_at_variable_line(self):
        # 17+14=31 < 44.5 → under
        result = go.actual_outcome(
            prediction_type="total",
            model_name="ensemble_nfl_tot",
            predicted_outcome="under",
            home_score=17,
            away_score=14,
            features={"closing_total_line": 44.5},
        )
        assert result == "under"


class TestNflMoneylineGrading:
    """NFL moneyline reuses nhl_moneyline_outcome — same final-score
    winner math. NFL ties (rare but possible) leave the row ungraded."""

    def test_home_win(self):
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_nfl_ml",
            predicted_outcome="home",
            home_score=31,
            away_score=24,
        )
        assert result == "home"

    def test_away_win(self):
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_nfl_ml",
            predicted_outcome="away",
            home_score=17,
            away_score=24,
        )
        assert result == "away"

    def test_tie_leaves_ungraded(self):
        # NFL ties happen (e.g. Steelers-Lions 16-16 in 2022). 2-class
        # classifier can't represent them — moneyline_outcome returns
        # None so the row stays ungraded.
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_nfl_ml",
            predicted_outcome="home",
            home_score=16,
            away_score=16,
        )
        assert result is None


class TestTennisMoneylineGrading:
    """Tennis moneyline reuses the shared 2-way grading path. No
    schema or dispatch changes were needed — the existing
    `prediction_type='moneyline'` branch handles tennis correctly via
    nhl_moneyline_outcome (returns None on ties; tennis matches never
    tie so this is belt-and-suspenders).

    Lock the dispatch here so a future refactor that introduces
    sport-specific moneyline grading doesn't accidentally exclude
    tennis."""

    def test_player1_wins(self):
        # home_score = sets won by player1; tennis best-of-3
        # final 2-1 → home (player1) won.
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_tennis_ml",
            predicted_outcome="home",
            home_score=2,
            away_score=1,
        )
        assert result == "home"

    def test_player2_wins(self):
        # Best-of-5 final 2-3 → away (player2) won.
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_tennis_ml",
            predicted_outcome="away",
            home_score=2,
            away_score=3,
        )
        assert result == "away"

    def test_retired_or_data_quality_tie_leaves_ungraded(self):
        # Tennis matches never tie in completion. A 0-0 row here would
        # mean a retirement-before-play or data quality bug — the
        # grader returns None so we don't write a false win/loss.
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_tennis_ml",
            predicted_outcome="home",
            home_score=0,
            away_score=0,
        )
        assert result is None


class TestMmaMoneylineGrading:
    """MMA moneyline reuses the shared 2-way grading path. No
    schema or dispatch changes were needed — the existing
    `prediction_type='moneyline'` branch handles MMA correctly via
    nhl_moneyline_outcome. MMA draws (~1% of decisions) are dropped
    at ingest, so the None-on-tie fallback is belt-and-suspenders
    against data quality issues."""

    def test_fighter1_wins(self):
        # home_score=1, away_score=0 → fighter1 (home) won.
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_mma_ml",
            predicted_outcome="home",
            home_score=1,
            away_score=0,
        )
        assert result == "home"

    def test_fighter2_wins(self):
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_mma_ml",
            predicted_outcome="away",
            home_score=0,
            away_score=1,
        )
        assert result == "away"

    def test_draw_or_data_quality_tie_leaves_ungraded(self):
        # MMA draws are dropped at ingest, but if one slipped through
        # the grader returns None (no false win/loss).
        result = go.actual_outcome(
            prediction_type="moneyline",
            model_name="ensemble_mma_ml",
            predicted_outcome="home",
            home_score=0,
            away_score=0,
        )
        assert result is None


# ── Model-name dispatch helpers ──────────────────────────────────────


class TestModelNameHelpers:
    def test_is_nba_spread(self):
        assert go.is_nba_spread("ensemble_nba_sp") is True
        assert go.is_nba_spread("ensemble_nhl_pl") is False
        assert go.is_nba_spread("ensemble_nba_tot") is False

    def test_is_nba_total(self):
        assert go.is_nba_total("ensemble_nba_tot") is True
        assert go.is_nba_total("ensemble_nhl_tot") is False
        assert go.is_nba_total("ensemble_nba_sp") is False

    def test_is_nhl_regulation_unchanged(self):
        # Existing helper still works (regression guard).
        assert go.is_nhl_regulation("ensemble_nhl_reg") is True
        assert go.is_nhl_regulation("ensemble_nba_sp") is False

    def test_is_nfl_spread(self):
        assert go.is_nfl_spread("ensemble_nfl_sp") is True
        assert go.is_nfl_spread("ensemble_nba_sp") is False
        assert go.is_nfl_spread("ensemble_nfl_tot") is False

    def test_is_nfl_total(self):
        assert go.is_nfl_total("ensemble_nfl_tot") is True
        assert go.is_nfl_total("ensemble_nba_tot") is False
        assert go.is_nfl_total("ensemble_nfl_sp") is False


# ── Rec-level settlement (grade_rec_selection) ───────────────────────


class TestGradeRecSelection:
    def test_moneyline_and_1x2(self):
        assert go.grade_rec_selection("1x2", "away", 0, 1) == go.REC_WON
        assert go.grade_rec_selection("1x2", "draw", 1, 1) == go.REC_WON
        assert go.grade_rec_selection("moneyline", "home", 2, 3) == go.REC_LOST

    def test_totals_with_line_and_push(self):
        assert go.grade_rec_selection("over_under", "over_2.5", 2, 1) == go.REC_WON
        assert go.grade_rec_selection("total", "under_5.5", 2, 3) == go.REC_WON
        assert go.grade_rec_selection("over_under", "over_3", 2, 1) == go.REC_PUSH

    def test_handicap_full_and_half_lines(self):
        # home -0.5, home wins by 1 -> covers.
        assert go.grade_rec_selection("asian_handicap", "home_-0.5", 2, 1) == go.REC_WON
        # away +1.5 (own line), away loses by 1 -> covers.
        assert go.grade_rec_selection("spread", "away_1.5", 3, 2) == go.REC_WON
        # home -2 exactly won by 2 -> push.
        assert go.grade_rec_selection("puck_line", "home_-2", 4, 2) == go.REC_PUSH

    def test_quarter_lines_produce_half_outcomes(self):
        # home -0.75 wins by exactly 1: win at -0.5, push at -1 -> half win.
        assert go.grade_rec_selection("asian_handicap", "home_-0.75", 2, 1) == go.REC_HALF_WON
        # home -0.25 draws: push at 0, lose at -0.5 -> half loss.
        assert go.grade_rec_selection("asian_handicap", "home_-0.25", 1, 1) == go.REC_HALF_LOST
        # home -0.75 wins by 2: clean win on both sub-lines.
        assert go.grade_rec_selection("asian_handicap", "home_-0.75", 3, 1) == go.REC_WON

    def test_coverage_and_push_markets(self):
        assert go.grade_rec_selection("double_chance", "12", 2, 0) == go.REC_WON
        assert go.grade_rec_selection("double_chance", "X2", 1, 1) == go.REC_WON
        assert go.grade_rec_selection("draw_no_bet", "home", 1, 1) == go.REC_PUSH
        assert go.grade_rec_selection("btts", "no", 3, 0) == go.REC_WON
        assert go.grade_rec_selection("correct_score", "2-1", 2, 1) == go.REC_WON

    def test_space_separated_selections(self):
        # The June-2026 NHL generator wrote 'home -1.5' / 'over 5.5'. Real
        # rows: home -1.5 at 4-2 (settled 'lost' by the old argmax path) and
        # away +1.5 at 4-3, over 5.5 at 0-3.
        assert go.grade_rec_selection("puck_line", "home -1.5", 4, 2) == go.REC_WON
        assert go.grade_rec_selection("puck_line", "away +1.5", 4, 3) == go.REC_WON
        assert go.grade_rec_selection("total", "over 5.5", 0, 3) == go.REC_LOST
        # A name with a space is still a bare selection, not (side, line).
        assert go._split_sel_line("Carlos Alcaraz") == ("carlos alcaraz", None)

    def test_unknown_formats_stay_open(self):
        assert go.grade_rec_selection("over_under_ht", "over_1.5", 2, 1) is None
        assert go.grade_rec_selection("team_total", "home_over_1.5", 2, 1) is None
        assert go.grade_rec_selection("asian_handicap", "garbage", 2, 1) is None

    def test_nhl_double_chance_is_never_settled_on_the_final_score(self):
        # 4-3 after OT: the final score has a winner, so '12' would always
        # win. It is the regulation 3-way — the final score cannot settle it.
        assert go.grade_rec_selection("double_chance", "12", 4, 3, sport="nhl") is None
        assert go.grade_rec_selection("double_chance", "1X", 4, 3, sport=" NHL ") is None
        assert go.grade_rec_selection("double_chance", "X2", 2, 3, model_name="dixon_coles_nhl") is None
        assert go.grade_rec_selection("double_chance", "12", 4, 3, sport=None, model_name="ensemble_nhl_ml") is None
        # Either signal is enough, even when the other says soccer.
        assert go.grade_rec_selection("double_chance", "12", 4, 3, sport="soccer", model_name="dixon_coles_nhl") is None

    def test_soccer_double_chance_settles_as_before(self):
        # Default (no sport/model) keeps soccer behaviour, and so does soccer.
        assert go.grade_rec_selection("double_chance", "12", 4, 3) == go.REC_WON
        assert go.grade_rec_selection("double_chance", "12", 1, 1, sport="soccer", model_name="ensemble") == go.REC_LOST
        assert go.grade_rec_selection("double_chance", "X2", 1, 1, sport="soccer") == go.REC_WON

    def test_nhl_markets_other_than_double_chance_still_settle(self):
        assert go.grade_rec_selection("moneyline", "home", 4, 3, sport="nhl") == go.REC_WON
        assert go.grade_rec_selection("puck_line", "home -1.5", 4, 3, sport="nhl") == go.REC_LOST
        assert go.grade_rec_selection("total", "over_5.5", 4, 3, sport="nhl") == go.REC_WON

    def test_correct_score_other_is_never_settled(self):
        # 'other' covers whatever that prediction did not list; the rec
        # does not carry the key set, so neither won nor lost is knowable.
        assert go.grade_rec_selection("correct_score", "other", 5, 2) is None
        assert go.grade_rec_selection("correct_score", " Other ", 0, 0, sport="soccer") is None
        assert go.grade_rec_selection("correct_score", "other", 5, 2, sport="nhl", model_name="dixon_coles_nhl") is None
        # Scoreline selections still settle, NHL included.
        assert go.grade_rec_selection("correct_score", "5-2", 5, 2, sport="nhl") == go.REC_WON
        assert go.grade_rec_selection("correct_score", "2-1", 5, 2) == go.REC_LOST

    def test_sport_and_model_are_keyword_only(self):
        with pytest.raises(TypeError):
            go.grade_rec_selection("double_chance", "12", 4, 3, "nhl")

    def test_status_and_pl_money_math(self):
        assert go.rec_status_and_pl(go.REC_WON, 25, 2.0) == ("won", 25)
        assert go.rec_status_and_pl(go.REC_HALF_WON, 100, 1.9) == ("won", 45)
        status, pl = go.rec_status_and_pl(go.REC_HALF_LOST, 100, 1.9)
        assert (status, pl) == ("lost", -50)
        assert go.rec_status_and_pl(go.REC_PUSH, 100, 1.9)[0] == "void"


class TestAsianHandicapPredictionGrading:
    def test_actual_outcome_and_grade(self):
        actual = go.actual_outcome(
            prediction_type="asian_handicap",
            model_name="ensemble",
            predicted_outcome="-0.5_home",
            home_score=2,
            away_score=1,
            metadata=None,
        )
        assert actual == "home_covers_-0.5"
        assert go.grade_prediction("-0.5_home", actual, prediction_type="asian_handicap") is True
        assert go.grade_prediction("-0.5_away", actual, prediction_type="asian_handicap") is False

    def test_quarter_and_push_grade_null(self):
        # Quarter-line half outcomes and pushes -> push_<line> -> NULL.
        a = go.actual_outcome(
            prediction_type="asian_handicap",
            model_name="ensemble",
            predicted_outcome="-0.75_home",
            home_score=2,
            away_score=1,
            metadata=None,
        )
        assert a == "push_-0.75"
        assert go.grade_prediction("-0.75_home", a, prediction_type="asian_handicap") is None


def _actual(prediction_type, predicted, h, a, model_name="ensemble", metadata=None, halftime=None):
    return go.actual_outcome(
        prediction_type=prediction_type,
        model_name=model_name,
        predicted_outcome=predicted,
        home_score=h,
        away_score=a,
        metadata=metadata,
        halftime=halftime,
    )


class TestDerivedMarketGrading:
    """market_derivation.py keys, graded from the final score."""

    def test_clean_sheet_is_graded_for_the_picked_team(self):
        assert _actual("clean_sheet", "home_no", 2, 0) == "home_yes"
        assert _actual("clean_sheet", "home_no", 2, 1) == "home_no"
        assert _actual("clean_sheet", "away_yes", 2, 0) == "away_no"
        assert go.grade_prediction("home_no", _actual("clean_sheet", "home_no", 2, 0)) is False
        assert go.grade_prediction("away_no", _actual("clean_sheet", "away_no", 2, 0)) is True

    def test_win_to_nil(self):
        assert _actual("win_to_nil", "home_no", 2, 0) == "home_yes"
        assert _actual("win_to_nil", "home_no", 0, 0) == "home_no"  # nil-nil is not a win
        assert _actual("win_to_nil", "away_no", 0, 1) == "away_yes"
        assert _actual("win_to_nil", "away_no", 1, 2) == "away_no"

    def test_team_total_uses_the_picked_line(self):
        assert _actual("team_total", "home_over_0.5", 1, 3) == "home_over_0.5"
        assert _actual("team_total", "away_under_2.5", 1, 3) == "away_over_2.5"
        assert go.grade_prediction("away_under_2.5", "away_over_2.5") is False
        # integer line hit exactly -> push -> NULL
        assert _actual("team_total", "home_over_1", 1, 0) == "push"
        assert go.grade_prediction("home_over_1", "push") is None

    def test_odd_even_and_total_goals_bands(self):
        assert _actual("odd_even", "odd", 2, 1) == "odd"
        assert _actual("odd_even", "odd", 0, 0) == "even"
        assert _actual("total_goals", "2", 1, 1) == "2"
        assert _actual("total_goals", "6+", 4, 3) == "6+"
        # NHL bands run to 8+: 7 goals is its own key, 9 folds into 8+.
        assert _actual("total_goals", "8+", 4, 3, model_name="dixon_coles_nhl") == "7"
        assert _actual("total_goals", "8+", 5, 4, model_name="dixon_coles_nhl") == "8+"

    def test_winning_margin(self):
        assert _actual("winning_margin", "draw", 1, 1) == "draw"
        assert _actual("winning_margin", "home_1", 2, 1) == "home_1"
        assert _actual("winning_margin", "home_1", 0, 2) == "away_2"
        assert _actual("winning_margin", "home_1", 5, 1) == "home_3plus"

    def test_result_combos(self):
        assert _actual("result_btts", "home_yes", 2, 1) == "home_yes"
        assert _actual("result_btts", "home_yes", 1, 1) == "draw_yes"
        assert _actual("result_btts", "home_yes", 0, 0) == "draw_no"
        assert _actual("result_over_under", "home_over", 2, 1) == "home_over"
        assert _actual("result_over_under", "home_over", 1, 0) == "home_under"

    def test_malformed_team_picks_stay_ungraded(self):
        assert _actual("clean_sheet", "yes", 1, 0) is None
        assert _actual("win_to_nil", "draw_no", 1, 0) is None
        assert _actual("team_total", "home_over", 1, 0) is None


class TestNhlDerivedMarketGrading:
    def test_regulation_1x2_maps_tie_to_draw(self):
        md = {"regulation_winner": "tie"}
        assert _actual("regulation_1x2", "home", 4, 3, model_name="dixon_coles_nhl", metadata=md) == "draw"
        md = {"regulation_winner": "draw"}  # the pre-2026-10 ESPN label
        assert _actual("regulation_1x2", "home", 4, 3, model_name="dixon_coles_nhl", metadata=md) == "draw"
        md = {"regulation_winner": "away"}
        assert _actual("regulation_1x2", "away", 1, 3, model_name="dixon_coles_nhl", metadata=md) == "away"

    def test_regulation_1x2_without_regulation_winner_is_ungraded(self):
        assert _actual("regulation_1x2", "home", 4, 3, model_name="dixon_coles_nhl", metadata={}) is None

    def test_puck_line_uses_the_handicap_keys(self):
        # '-1.5_away' = away +1.5 against a home -1.5: a one-goal home win covers.
        a = _actual("puck_line", "-1.5_away", 4, 3, model_name="dixon_coles_nhl")
        assert a == "away_covers_-1.5"
        assert go.grade_prediction("-1.5_away", a, prediction_type="puck_line") is True
        # '1.5_home' = home +1.5: a two-goal home loss fails.
        a = _actual("puck_line", "1.5_home", 1, 3, model_name="dixon_coles_nhl")
        assert a == "away_covers_1.5"
        assert go.grade_prediction("1.5_home", a, prediction_type="puck_line") is False


class TestHalftimeMarketGrading:
    HT = (1, 0)

    def test_without_halftime_scores_every_ht_market_is_ungraded(self):
        for pt, pick in (
            ("match_result_ht", "home"),
            ("btts_ht", "no"),
            ("over_under_ht", "over_0.5"),
            ("ht_ft_double_result", "home_home"),
        ):
            assert _actual(pt, pick, 2, 1, halftime=None) is None

    def test_halftime_markets(self):
        assert _actual("match_result_ht", "draw", 2, 2, halftime=self.HT) == "home"
        assert _actual("btts_ht", "no", 2, 2, halftime=self.HT) == "no"
        assert _actual("over_under_ht", "over_0.5", 2, 2, halftime=self.HT) == "over_0.5"
        assert _actual("over_under_ht", "under_1.5", 2, 2, halftime=self.HT) == "under_1.5"
        assert _actual("ht_ft_double_result", "home_home", 2, 2, halftime=self.HT) == "home_draw"

    def test_halftime_above_final_is_a_data_error(self):
        assert _actual("match_result_ht", "home", 2, 1, halftime=(3, 0)) is None

    def test_halftime_comes_from_the_matches_columns_only(self):
        assert go.halftime_scores({"half_time_home": 1, "half_time_away": 0}) == (1, 0)
        assert go.halftime_scores({"half_time_home": 1, "half_time_away": None}) is None
        # the retired metadata keys are not a source any more
        assert go.halftime_scores({"metadata": {"halftime_home_score": 1, "halftime_away_score": 0}}) is None
        assert go.halftime_scores(None) is None


class TestNhlDoubleChanceGradesOnRegulation:
    """dixon_coles_nhl prices double_chance over the REGULATION 3-way (its X
    is 'tied after regulation'). Graded on the final score, which is never a
    draw in the NHL, every '12' pick won: 286/286 to 2026-10-31."""

    NHL = "dixon_coles_nhl"

    def _dc(self, pick, h, a, regulation_winner):
        md = {} if regulation_winner is None else {"regulation_winner": regulation_winner}
        actual = _actual("double_chance", pick, h, a, model_name=self.NHL, metadata=md)
        return actual, go.grade_prediction(pick, actual, prediction_type="double_chance")

    def test_regulation_tie_is_a_draw_even_when_ot_decided_it(self):
        # 4-3 final after OT: the final score says 'home', regulation says draw.
        assert self._dc("12", 4, 3, "tie") == ("draw", False)
        assert self._dc("1X", 4, 3, "tie") == ("draw", True)
        assert self._dc("X2", 4, 3, "tie") == ("draw", True)

    def test_legacy_draw_label_reads_the_same(self):
        assert self._dc("12", 2, 3, "draw") == ("draw", False)

    def test_regulation_winner_decides(self):
        assert self._dc("12", 4, 2, "home") == ("home", True)
        assert self._dc("1X", 4, 2, "home") == ("home", True)
        assert self._dc("X2", 4, 2, "home") == ("home", False)
        assert self._dc("X2", 1, 3, "away") == ("away", True)

    def test_missing_or_unknown_regulation_winner_is_ungraded(self):
        assert self._dc("12", 4, 3, None) == (None, None)
        assert self._dc("12", 4, 3, "overtime") == (None, None)

    def test_soccer_double_chance_is_unchanged(self):
        # Soccer grades the final score; a stray regulation_winner is ignored.
        md = {"regulation_winner": "home"}
        for (h, a), base in {(1, 1): "draw", (2, 0): "home", (0, 1): "away"}.items():
            assert _actual("double_chance", "12", h, a, metadata=md) == base
        assert go.grade_prediction("12", "draw", prediction_type="double_chance") is False
        assert go.grade_prediction("1X", "draw", prediction_type="double_chance") is True

    def test_is_nhl_model(self):
        assert go.is_nhl_model("dixon_coles_nhl") and go.is_nhl_model("ensemble_nhl_ml")
        assert not go.is_nhl_model("ensemble") and not go.is_nhl_model(None)


class TestCorrectScoreOther:
    """'other' covers every scoreline the SAME prediction did not list, so
    it is graded against that prediction's own keys — string equality never
    credited it (0/281 dixon_coles_nhl rows, 146 of them winners)."""

    NHL_KEYS = {
        "2-2": 0.052,
        "3-2": 0.055,
        "2-3": 0.047,
        "3-3": 0.049,
        "2-1": 0.039,
        "3-1": 0.041,
        "4-2": 0.043,
        "1-2": 0.033,
        "2-4": 0.032,
        "3-4": 0.033,
        "4-1": 0.032,
        "4-3": 0.039,
        "other": 0.505,
    }

    def _grade(self, pick, h, a, probs, model_name="dixon_coles_nhl"):
        actual = _actual("correct_score", pick, h, a, model_name=model_name)
        return actual, go.grade_prediction(pick, actual, prediction_type="correct_score", probabilities=probs)

    def test_other_credited_when_the_score_was_not_listed(self):
        assert self._grade("other", 5, 2, self.NHL_KEYS) == ("5-2", True)
        assert self._grade("other", 0, 1, self.NHL_KEYS) == ("0-1", True)

    def test_other_not_credited_when_the_score_was_listed(self):
        assert self._grade("other", 4, 3, self.NHL_KEYS) == ("4-3", False)

    def test_other_without_its_key_set_is_ungraded(self):
        assert self._grade("other", 5, 2, None) == ("5-2", None)
        assert self._grade("other", 5, 2, {"other": 1.0}) == ("5-2", None)
        assert self._grade("other", 5, 2, "not-a-dict") == ("5-2", None)

    def test_scoreline_picks_still_grade_by_equality(self):
        assert self._grade("3-2", 3, 2, self.NHL_KEYS) == ("3-2", True)
        assert self._grade("3-2", 4, 3, self.NHL_KEYS) == ("4-3", False)
        assert self._grade("3-2", 3, 2, None) == ("3-2", True)

    def test_nhl_tie_keys_can_never_win_on_the_final_score(self):
        # Graded on the FINAL score (OT/SO included), which is never tied.
        assert self._grade("2-2", 3, 2, self.NHL_KEYS) == ("3-2", False)

    def test_soccer_is_unchanged_and_other_is_correct_for_it_too(self):
        keys = {"1-1": 0.12, "1-0": 0.11, "0-0": 0.09, "2-1": 0.08, "other": 0.6}
        assert self._grade("1-1", 1, 1, keys, model_name="ensemble") == ("1-1", True)
        assert self._grade("1-0", 2, 2, keys, model_name="ensemble") == ("2-2", False)
        assert self._grade("other", 2, 2, keys, model_name="ensemble") == ("2-2", True)
        assert self._grade("other", 0, 0, keys, model_name="ensemble") == ("0-0", False)
