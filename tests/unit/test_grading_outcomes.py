"""Per-market grading math — pure functions, no DB or HTTP."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
    def test_home_win_is_1X(self):
        assert go.double_chance_outcome(3, 1) == "1X"

    def test_draw_is_1X(self):
        assert go.double_chance_outcome(1, 1) == "1X"

    def test_away_win_is_X2(self):
        assert go.double_chance_outcome(0, 2) == "X2"


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
        # asian_handicap isn't in the v1 dispatch table — must fall
        # through to None so grading skips it cleanly.
        assert (
            go.actual_outcome(
                prediction_type="asian_handicap",
                model_name="ensemble",
                predicted_outcome="-0.5_home",
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
