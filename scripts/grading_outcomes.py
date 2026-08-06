"""Pure per-market grading: actual_outcome given finished-match data.

Every function here is a pure mapping from a match's final state +
prediction context to the canonical actual_outcome string that the
trigger/grading script will write to predictions.actual_outcome and
betting_recommendations.actual_result.

Dispatch table — what each prediction_type means and how it grades:

  prediction_type   model_name prefix              graded from
  ────────────────  ──────────────────────────     ────────────────────────
  match_result      ensemble (soccer 1X2)          home_score vs away_score
  match_result      ensemble_nhl_reg (regulation)  metadata.regulation_winner
  moneyline         ensemble_nhl_ml                home_score vs away_score (final)
  spread            ensemble_nhl_pl                home_score − away_score (puck line)
  total             ensemble_nhl_tot               home_score + away_score (5.5)
  btts              ensemble (soccer)              both scores > 0
  over_under        ensemble (soccer derived)      home_score + away_score vs <line>
  double_chance     ensemble (soccer derived)      1X / 12 / X2 from match result
  draw_no_bet       ensemble (soccer derived)      home/away (push on draw)
  correct_score     ensemble (soccer derived)      exact <h>-<a> string match

Returns None when the outcome can't be determined — caller leaves
is_correct NULL and the grading script tries again on the next run.
Push semantics: actual_outcome="push" → grading marks
betting_recommendations status='void' with profit_loss=0; predictions
is_correct stays NULL (a push isn't right or wrong).

Markets not in the dispatch table (asian_handicap, team_total,
clean_sheet, win_to_nil, odd_even, winning_margin, total_goals,
result_btts, result_over_under, player_prop, lottery) are NOT graded
in v1 — they pass through with actual_outcome=None. Add coverage
when needed.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Sentinel used in match_result dispatch to detect NHL regulation. The
# NHL regulation TaskSpec writes prediction_type="match_result" (no
# dedicated CHECK-constraint value yet) so we disambiguate by the
# trained-ensemble registry name.
NHL_REGULATION_MODEL_NAME = "ensemble_nhl_reg"

# NBA spread + total use the SAME prediction_type values as NHL
# (spread / total) but with VARIABLE closing lines. NHL uses fixed
# lines (puck_line -1.5, total 5.5); NBA's line is whatever the book
# closed at and lives in features_cache.closing_spread_home /
# closing_total_line. Disambiguate by model_name.
NBA_SPREAD_MODEL_NAME = "ensemble_nba_sp"
NBA_TOTAL_MODEL_NAME = "ensemble_nba_tot"

# NFL spread + total mirror NBA: same prediction_type values + variable
# closing lines stored in features_cache. NFL moneyline grades the same
# as NHL/NBA — final score winner; ties (rare but possible in NFL)
# leave the row ungraded via nhl_moneyline_outcome's None return.
NFL_SPREAD_MODEL_NAME = "ensemble_nfl_sp"
NFL_TOTAL_MODEL_NAME = "ensemble_nfl_tot"


def is_nhl_regulation(model_name: str) -> bool:
    return model_name == NHL_REGULATION_MODEL_NAME


def is_nba_spread(model_name: str) -> bool:
    return model_name == NBA_SPREAD_MODEL_NAME


def is_nba_total(model_name: str) -> bool:
    return model_name == NBA_TOTAL_MODEL_NAME


def is_nfl_spread(model_name: str) -> bool:
    return model_name == NFL_SPREAD_MODEL_NAME


def is_nfl_total(model_name: str) -> bool:
    return model_name == NFL_TOTAL_MODEL_NAME


def _extract_over_under_line(predicted_outcome: Optional[str]) -> Optional[float]:
    """Pull the numeric line off an over_under predicted_outcome like
    'over_2.5' / 'under_3'. Returns None if no line is encoded."""
    if not predicted_outcome:
        return None
    parts = predicted_outcome.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _ou_outcome_for_line(home_score: int, away_score: int, line: float) -> str:
    """over_<line> / under_<line> / push_<line>. The predicted_outcome
    is reconstructed in the same shape so equality testing works."""
    total = home_score + away_score
    if total > line:
        return f"over_{_fmt_line(line)}"
    if total < line:
        return f"under_{_fmt_line(line)}"
    return f"push_{_fmt_line(line)}"


def _fmt_line(line: float) -> str:
    """Format a line so '2.5' stays '2.5' but '3.0' becomes '3' —
    matches the keys precompute writes into predictions.probabilities."""
    f = float(line)
    return str(int(f)) if f.is_integer() else str(f)


# ── Per-sport, per-market outcome calculators ───────────────────────


def soccer_match_result(home_score: int, away_score: int) -> str:
    """Soccer 1X2 — three-way."""
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def nhl_regulation_outcome(metadata: dict) -> Optional[str]:
    """NHL regulation 3-class: read metadata.regulation_winner.
    'tie' covers regulation-tied games (decided in OT/SO). Returns
    None if the loader hasn't backfilled regulation_winner — caller
    leaves the prediction ungraded rather than guessing."""
    return (metadata or {}).get("regulation_winner")


def nhl_moneyline_outcome(home_score: int, away_score: int) -> Optional[str]:
    """NHL moneyline — final score (incl. OT/SO). Returns None if the
    scores tied (shouldn't happen in NHL after OT/SO regular season
    play, but defensive: a data-quality issue shouldn't grade as a win
    or loss either way)."""
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return None


def nhl_puck_line_outcome(home_score: int, away_score: int) -> str:
    """NHL puck line at -1.5 home / +1.5 away. 'cover' = home wins by
    >= 2 goals (the canonical puck-line bet). 'no_cover' otherwise.
    The ±1.5 half-line never pushes."""
    return "cover" if (home_score - away_score) >= 2 else "no_cover"


def nhl_total_outcome(home_score: int, away_score: int, line: float = 5.5) -> str:
    """NHL total over/under 5.5 (default). 5.5 is the canonical NHL
    book line and never pushes."""
    return "over" if (home_score + away_score) > line else "under"


def nba_spread_outcome(home_score: int, away_score: int, line: float) -> str:
    """NBA spread at the closing `line` (home perspective; -5 means
    home was favored by 5). Home covers iff (home - away) > -line.
    Integer lines can push, half-point lines (.5) never do — same
    convention as the rec engine."""
    margin = home_score - away_score
    if margin > -line:
        return "home"
    if margin < -line:
        return "away"
    return "push"


def nba_total_outcome(home_score: int, away_score: int, line: float) -> str:
    """NBA total over/under at the closing `line`. Over iff total
    points > line. Integer lines (rare in NBA, usually .5) push."""
    total = home_score + away_score
    if total > line:
        return "over"
    if total < line:
        return "under"
    return "push"


def btts_outcome(home_score: int, away_score: int) -> str:
    """Both Teams To Score — yes if both > 0, else no."""
    return "yes" if home_score > 0 and away_score > 0 else "no"


# Double-chance selections cover SETS of base results — a home win
# satisfies both '1X' and '12', so no single actual label can grade the
# market under string equality. actual_outcome stores the BASE result
# ('home'/'draw'/'away') and grade_prediction() tests set membership.
# The old single-label convention ('1X' for home-or-draw, 'X2' for away)
# mis-graded every '12' pick and every 'X2'-pick-on-a-draw, deflating
# accuracy to 48% and firing a false ECE-0.28 monitor alarm.
DOUBLE_CHANCE_COVERS = {
    "1X": {"home", "draw"},
    "12": {"home", "away"},
    "X2": {"draw", "away"},
}


def double_chance_outcome(home_score: int, away_score: int) -> str:
    """Base match result ('home'/'draw'/'away') — the truthful 'what
    happened' for a coverage market. Correctness is decided in
    grade_prediction() via DOUBLE_CHANCE_COVERS membership, not label
    equality."""
    return soccer_match_result(home_score, away_score)


def draw_no_bet_outcome(home_score: int, away_score: int) -> str:
    """Draw-no-bet: home / away on decisive results, push on a draw."""
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "push"


def correct_score_outcome(home_score: int, away_score: int) -> str:
    """Exact-score: '<home>-<away>' string (no zero-padding). The
    correct_score market in the derivation engine emits keys like
    '2-1' so the equality test is straightforward."""
    return f"{home_score}-{away_score}"


# ── Top-level dispatch ──────────────────────────────────────────────


def actual_outcome(
    *,
    prediction_type: str,
    model_name: str,
    predicted_outcome: Optional[str],
    home_score: Optional[int],
    away_score: Optional[int],
    metadata: Optional[dict] = None,
    features: Optional[dict] = None,
) -> Optional[str]:
    """Compute the canonical actual_outcome for one prediction. Returns
    None for ungradable rows (missing scores, regulation_winner not
    backfilled, missing closing line for an NBA spread/total, market
    not in the dispatch table). The grading script treats None as
    "leave is_correct NULL; try again next run".

    `features` (NEW): the features_cache JSON for this match. Used by
    NBA spread/total to read the closing line the model conditioned
    on (which is also the line the bet was effectively placed at).
    NHL fixed lines and other markets ignore it.
    """
    # NHL regulation gets scored from metadata, not the goal columns —
    # check it FIRST since it shares prediction_type='match_result'
    # with soccer 1X2.
    if prediction_type == "match_result" and is_nhl_regulation(model_name):
        return nhl_regulation_outcome(metadata or {})

    # The remaining markets all need both score columns populated.
    if home_score is None or away_score is None:
        return None

    if prediction_type == "match_result":
        return soccer_match_result(home_score, away_score)
    if prediction_type == "moneyline":
        # Same dispatch for NHL + NBA — both use final score.
        return nhl_moneyline_outcome(home_score, away_score)
    if prediction_type == "spread":
        # NHL: fixed puck line -1.5. NBA + NFL: variable line stored in
        # features_cache.closing_spread_home. Distinguish by model_name
        # — same prediction_type, same line-as-feature math (reuses
        # nba_spread_outcome which is sport-agnostic), different
        # ensemble artifact.
        if is_nba_spread(model_name) or is_nfl_spread(model_name):
            line = (features or {}).get("closing_spread_home")
            if line is None:
                return None
            return nba_spread_outcome(home_score, away_score, float(line))
        return nhl_puck_line_outcome(home_score, away_score)
    if prediction_type == "total":
        # Same pattern: NHL fixed at 5.5, NBA + NFL read from features.
        if is_nba_total(model_name) or is_nfl_total(model_name):
            line = (features or {}).get("closing_total_line")
            if line is None:
                return None
            return nba_total_outcome(home_score, away_score, float(line))
        return nhl_total_outcome(home_score, away_score)
    if prediction_type == "btts":
        return btts_outcome(home_score, away_score)
    if prediction_type == "over_under":
        line = _extract_over_under_line(predicted_outcome)
        if line is None:
            return None
        return _ou_outcome_for_line(home_score, away_score, line)
    if prediction_type == "double_chance":
        return double_chance_outcome(home_score, away_score)
    if prediction_type == "draw_no_bet":
        return draw_no_bet_outcome(home_score, away_score)
    if prediction_type == "correct_score":
        return correct_score_outcome(home_score, away_score)
    if prediction_type == "asian_handicap":
        # predicted_outcome is the model key '{home_line}_{side}' (e.g.
        # '-0.5_home'); grade the HOME-perspective line, side decides
        # correctness in grade_prediction. Quarter-line half outcomes and
        # pushes grade to NULL (not right or wrong).
        line = _extract_ah_line(predicted_outcome)
        if line is None:
            return None
        verdict = _handicap_outcome(home_score - away_score, line)
        if verdict == REC_WON:
            return f"home_covers_{_fmt_line(line)}"
        if verdict == REC_LOST:
            return f"away_covers_{_fmt_line(line)}"
        return f"push_{_fmt_line(line)}"

    # Markets we don't grade in v1 (asian_handicap, team_total,
    # clean_sheet, etc.) fall through to None.
    return None


def is_push(actual: Optional[str]) -> bool:
    """True if the outcome string represents a push (over_under or
    draw_no_bet at the exact line). Used by the rec settler to
    transition status='void' with profit_loss=0."""
    if actual is None:
        return False
    return actual == "push" or actual.startswith("push_")


def grade_prediction(
    predicted: Optional[str],
    actual: Optional[str],
    prediction_type: Optional[str] = None,
) -> Optional[bool]:
    """Final correctness verdict. NULL on ungradable outcomes (pushes
    or actual=None) so the predictions.is_correct column distinguishes
    'we don't know' from 'we know it was wrong'.

    double_chance grades by coverage-set membership (a '12' pick wins on
    a home OR away result); everything else is label equality. An
    unknown double_chance selection grades NULL, not False — it means
    the stored pick is malformed, which is 'we don't know'."""
    if actual is None or is_push(actual):
        return None
    if predicted is None:
        return None
    if prediction_type == "asian_handicap":
        # predicted '{line}_{side}' vs actual '{home|away}_covers_{line}'.
        if actual.startswith("push_"):
            return None
        side = predicted.rsplit("_", 1)[-1]
        if side not in ("home", "away"):
            return None
        return actual.startswith(f"{side}_covers_")
    if prediction_type == "double_chance":
        covers = DOUBLE_CHANCE_COVERS.get(predicted)
        if covers is None:
            return None
        return actual in covers
    return predicted == actual


# ── Rec-level settlement (grades the REC's own selection, not the model's
#    argmax — see audit §7 2026-08-06: the old settle path mis-graded any
#    rec on a non-argmax selection and every lined market) ────────────────

REC_WON, REC_LOST, REC_PUSH, REC_HALF_WON, REC_HALF_LOST = ("won", "lost", "push", "half_won", "half_lost")

_MONEYLINE_TYPES = {"1x2", "match_result", "moneyline", "h2h"}
_HANDICAP_TYPES = {"asian_handicap", "spread", "puck_line"}
_TOTAL_TYPES = {"over_under", "total"}


def _split_sel_line(selection: str):
    """'home_-0.75' / 'over_2.5' -> (side, line) or (selection, None)."""
    if "_" in (selection or ""):
        side, _, raw = selection.partition("_")
        try:
            return side.lower(), float(raw)
        except ValueError:
            pass
    return (selection or "").strip().lower(), None


def _handicap_outcome(margin: float, line: float):
    """Home-perspective handicap: adjusted = margin + line. Quarter lines
    split the stake across the two adjacent half-lines."""
    if line % 0.5 == 0.25 or line % 0.5 == -0.25:
        lo, hi = _handicap_outcome(margin, line - 0.25), _handicap_outcome(margin, line + 0.25)
        pair = {lo, hi}
        if pair == {REC_WON}:
            return REC_WON
        if pair == {REC_LOST}:
            return REC_LOST
        if pair == {REC_WON, REC_PUSH}:
            return REC_HALF_WON
        if pair == {REC_LOST, REC_PUSH}:
            return REC_HALF_LOST
        return REC_PUSH
    adj = margin + line
    if adj > 0:
        return REC_WON
    if adj < 0:
        return REC_LOST
    return REC_PUSH


def _extract_ah_line(predicted_outcome):
    """'-0.5_home' -> -0.5 (home-perspective line from the model key)."""
    if not predicted_outcome or "_" not in predicted_outcome:
        return None
    raw = predicted_outcome.rsplit("_", 1)[0]
    try:
        return float(raw)
    except ValueError:
        return None


def grade_rec_selection(bet_type: str, selection: str, home_score: int, away_score: int):
    """Settlement verdict for one recommendation from ITS OWN selection and
    the final score. Returns REC_* or None (ungradable: unknown format, or
    a market needing data we don't grade yet — *_ht, team_total)."""
    bt = (bet_type or "").strip().lower()
    side, line = _split_sel_line(selection)
    margin = home_score - away_score

    if bt in _MONEYLINE_TYPES:
        if side not in ("home", "draw", "away"):
            return None
        actual = soccer_match_result(home_score, away_score)
        return REC_WON if side == actual else REC_LOST

    if bt in _TOTAL_TYPES:
        if side not in ("over", "under") or line is None:
            return None
        total = home_score + away_score
        if total == line:
            return REC_PUSH
        hit = total > line if side == "over" else total < line
        return REC_WON if hit else REC_LOST

    if bt in _HANDICAP_TYPES:
        if side not in ("home", "away") or line is None:
            return None
        # Selection stores each side's OWN signed line; normalize to
        # home-perspective margin for the math.
        eff_margin = margin if side == "home" else -margin
        return _handicap_outcome(eff_margin, line)

    if bt == "btts":
        if side not in ("yes", "no"):
            return None
        actual = btts_outcome(home_score, away_score)
        return REC_WON if side == actual else REC_LOST

    if bt == "double_chance":
        covers = DOUBLE_CHANCE_COVERS.get((selection or "").strip().upper())
        if covers is None:
            return None
        return REC_WON if soccer_match_result(home_score, away_score) in covers else REC_LOST

    if bt == "draw_no_bet":
        if side not in ("home", "away"):
            return None
        actual = draw_no_bet_outcome(home_score, away_score)
        if actual == "push":
            return REC_PUSH
        return REC_WON if side == actual else REC_LOST

    if bt == "correct_score":
        sel = (selection or "").strip()
        return REC_WON if sel == correct_score_outcome(home_score, away_score) else REC_LOST

    # *_ht, team_total, anything unknown: leave open rather than guess.
    return None


def rec_status_and_pl(outcome: str, stake, odds):
    """(status, profit_loss) for a REC_* verdict. Half outcomes keep the
    schema's won/lost statuses with half-stake money (actual_result records
    the half-ness)."""
    from decimal import Decimal

    stake = Decimal(str(stake or 0))
    odds = Decimal(str(odds or 0))
    if outcome == REC_WON:
        return "won", stake * (odds - 1)
    if outcome == REC_LOST:
        return "lost", -stake
    if outcome == REC_PUSH:
        return "void", Decimal("0")
    if outcome == REC_HALF_WON:
        return "won", stake / 2 * (odds - 1)
    if outcome == REC_HALF_LOST:
        return "lost", -stake / 2
    raise ValueError(f"Unknown rec outcome {outcome!r}")
