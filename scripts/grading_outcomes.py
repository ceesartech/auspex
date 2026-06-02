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


def is_nhl_regulation(model_name: str) -> bool:
    return model_name == NHL_REGULATION_MODEL_NAME


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


def btts_outcome(home_score: int, away_score: int) -> str:
    """Both Teams To Score — yes if both > 0, else no."""
    return "yes" if home_score > 0 and away_score > 0 else "no"


def double_chance_outcome(home_score: int, away_score: int) -> str:
    """1X / 12 / X2 — the outcome categories ALWAYS-include-draw form.
    Matches the keys precompute writes (upper-case in the soccer
    market-derivation engine).
    """
    base = soccer_match_result(home_score, away_score)
    if base == "home":
        # home win covers 1X (home or draw) and 12 (home or away).
        # Convention: return "1X" since it's the "no-away" cover.
        return "1X"
    if base == "draw":
        # draw covers 1X and X2.
        return "1X"
    # away win covers 12 and X2.
    return "X2"


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
) -> Optional[str]:
    """Compute the canonical actual_outcome for one prediction. Returns
    None for ungradable rows (missing scores, regulation_winner not
    backfilled, market not in the dispatch table). The grading script
    treats None as "leave is_correct NULL; try again next run".
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
        return nhl_moneyline_outcome(home_score, away_score)
    if prediction_type == "spread":
        return nhl_puck_line_outcome(home_score, away_score)
    if prediction_type == "total":
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


def grade_prediction(predicted: Optional[str], actual: Optional[str]) -> Optional[bool]:
    """Final correctness verdict. NULL on ungradable outcomes (pushes
    or actual=None) so the predictions.is_correct column distinguishes
    'we don't know' from 'we know it was wrong'."""
    if actual is None or is_push(actual):
        return None
    if predicted is None:
        return None
    return predicted == actual
