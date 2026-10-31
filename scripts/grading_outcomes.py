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
  asian_handicap    ensemble (soccer derived)      '{home_line}_{side}' vs margin
  puck_line         dixon_coles_nhl                same keys + math as asian_handicap
  regulation_1x2    dixon_coles_nhl                metadata.regulation_winner (tie→draw)
  clean_sheet       soccer / NHL derived           '{team}_{yes|no}' for the picked team
  win_to_nil        soccer / NHL derived           '{team}_{yes|no}' for the picked team
  team_total        ensemble (soccer derived)      '{team}_{over|under}_{line}', picked team+line
  odd_even          ensemble (soccer derived)      parity of the total
  total_goals       soccer / NHL derived           exact total, top band 6+ (NHL 8+)
  winning_margin    ensemble (soccer derived)      draw / {side}_{1|2|3plus}
  result_btts       ensemble (soccer derived)      {result}_{yes|no}
  result_over_under ensemble (soccer derived)      {result}_{over|under} at 2.5
  *_ht, ht_ft_…     ensemble (soccer halftime)     matches.half_time_{home,away}

Returns None when the outcome can't be determined — caller leaves
is_correct NULL and the grading script tries again on the next run.
Push semantics: actual_outcome="push" → grading marks
betting_recommendations status='void' with profit_loss=0; predictions
is_correct stays NULL (a push isn't right or wrong).

Markets not in the dispatch table (player_prop, lottery) are NOT
graded — they pass through with actual_outcome=None. The derived
markets were added 2026-10-30: until then ~92k soccer + 1.4k NHL
derived predictions sat is_correct NULL on every grading run.
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
    leaves the prediction ungraded rather than guessing.

    'draw' is read as 'tie': the ESPN results path wrote 'draw' until
    2026-10, which the regulation model (labels home/tie/away) can never
    predict, so every correct tie pick on those rows graded as a miss."""
    winner = (metadata or {}).get("regulation_winner")
    return "tie" if winner == "draw" else winner


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


# ── Derived markets (market_derivation.py, soccer + NHL) ────────────
#
# Several of these markets pack more than one binary proposition into a
# single key set (clean_sheet carries home_yes/home_no/away_yes/away_no),
# so no single "what happened" label grades the argmax by equality. As
# with over_under, the actual is computed FOR the picked proposition: a
# 'home_no' clean-sheet pick gets 'home_yes' or 'home_no' back.

HANDICAP_PREDICTION_TYPES = frozenset({"asian_handicap", "puck_line"})
HALFTIME_TYPES = frozenset({"match_result_ht", "btts_ht", "over_under_ht", "ht_ft_double_result"})

# Top band of the exact-total market (derive_soccer_markets emits '6+',
# derive_hockey_markets '8+').
TOTAL_GOALS_CAP_SOCCER = 6
TOTAL_GOALS_CAP_NHL = 8


def _split_team_pick(predicted_outcome: Optional[str]) -> Optional[tuple[str, str]]:
    """'home_yes' -> ('home', 'yes'); None unless '{home|away}_<rest>'."""
    if not predicted_outcome or "_" not in predicted_outcome:
        return None
    team, _, rest = predicted_outcome.partition("_")
    if team not in ("home", "away") or not rest:
        return None
    return team, rest


def clean_sheet_outcome(predicted_outcome: Optional[str], home_score: int, away_score: int) -> Optional[str]:
    """'{team}_yes' when the picked team conceded nothing, else '{team}_no'."""
    pick = _split_team_pick(predicted_outcome)
    if pick is None:
        return None
    team = pick[0]
    conceded = away_score if team == "home" else home_score
    return f"{team}_{'yes' if conceded == 0 else 'no'}"


def win_to_nil_outcome(predicted_outcome: Optional[str], home_score: int, away_score: int) -> Optional[str]:
    """'{team}_yes' when the picked team won without conceding."""
    pick = _split_team_pick(predicted_outcome)
    if pick is None:
        return None
    team = pick[0]
    scored, conceded = (home_score, away_score) if team == "home" else (away_score, home_score)
    return f"{team}_{'yes' if scored > conceded and conceded == 0 else 'no'}"


def team_total_outcome(predicted_outcome: Optional[str], home_score: int, away_score: int) -> Optional[str]:
    """'{team}_{over|under}_{line}' for the picked team and line. 'push'
    on an integer line hit exactly (the derivation offers .5 lines only)."""
    pick = _split_team_pick(predicted_outcome)
    if pick is None:
        return None
    team, rest = pick
    line = _extract_over_under_line(rest)
    if line is None:
        return None
    goals = home_score if team == "home" else away_score
    if goals == line:
        return "push"
    return f"{team}_{'over' if goals > line else 'under'}_{_fmt_line(line)}"


def odd_even_outcome(home_score: int, away_score: int) -> str:
    return "odd" if (home_score + away_score) % 2 else "even"


def total_goals_outcome(model_name: str, home_score: int, away_score: int) -> str:
    """Exact total as the derivation keys it: '0'..'5' then '6+' for
    soccer, '0'..'7' then '8+' for the NHL Dixon-Coles markets."""
    cap = TOTAL_GOALS_CAP_NHL if "nhl" in (model_name or "") else TOTAL_GOALS_CAP_SOCCER
    total = home_score + away_score
    return f"{cap}+" if total >= cap else str(total)


def winning_margin_outcome(home_score: int, away_score: int) -> str:
    """'draw', or '{home|away}_{1|2|3plus}'."""
    margin = home_score - away_score
    if margin == 0:
        return "draw"
    side = "home" if margin > 0 else "away"
    n = abs(margin)
    return f"{side}_{n if n < 3 else '3plus'}"


def result_btts_outcome(home_score: int, away_score: int) -> str:
    return f"{soccer_match_result(home_score, away_score)}_{btts_outcome(home_score, away_score)}"


def result_over_under_outcome(home_score: int, away_score: int) -> str:
    """Result + over/under 2.5 — the only line the derivation offers."""
    side = "over" if home_score + away_score > 2.5 else "under"
    return f"{soccer_match_result(home_score, away_score)}_{side}"


def halftime_scores(match: Optional[dict]) -> Optional[tuple[int, int]]:
    """(home, away) at half-time from a matches row's half_time_home /
    half_time_away — the one home of the half-time score (football-data
    loads and the ESPN results path both write it; the HT trainers and the
    match API read it). None when absent, so HT markets stay ungraded
    rather than guessed."""
    row = match or {}
    h, a = row.get("half_time_home"), row.get("half_time_away")
    if h is None or a is None:
        return None
    try:
        return int(h), int(a)
    except (TypeError, ValueError):
        return None


def halftime_outcome(
    prediction_type: str,
    predicted_outcome: Optional[str],
    home_score: int,
    away_score: int,
    halftime: Optional[tuple[int, int]],
) -> Optional[str]:
    if halftime is None:
        return None
    ht_home, ht_away = halftime
    if ht_home > home_score or ht_away > away_score:
        # A half-time score above the final one is a data error — don't
        # grade four markets off it.
        logger.warning(
            "half-time %d-%d exceeds final %d-%d; HT markets left ungraded", ht_home, ht_away, home_score, away_score
        )
        return None
    if prediction_type == "match_result_ht":
        return soccer_match_result(ht_home, ht_away)
    if prediction_type == "btts_ht":
        return btts_outcome(ht_home, ht_away)
    if prediction_type == "over_under_ht":
        line = _extract_over_under_line(predicted_outcome)
        return None if line is None else _ou_outcome_for_line(ht_home, ht_away, line)
    if prediction_type == "ht_ft_double_result":
        return f"{soccer_match_result(ht_home, ht_away)}_{soccer_match_result(home_score, away_score)}"
    return None


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
    halftime: Optional[tuple[int, int]] = None,
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
    if prediction_type == "regulation_1x2":
        # dixon_coles_nhl labels the regulation 3-way home/DRAW/away,
        # unlike ensemble_nhl_reg's home/tie/away.
        winner = nhl_regulation_outcome(metadata or {})
        return "draw" if winner == "tie" else winner

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
    if prediction_type in HANDICAP_PREDICTION_TYPES:
        # predicted_outcome is the model key '{home_line}_{side}' (e.g.
        # '-0.5_home'); grade the HOME-perspective line, side decides
        # correctness in grade_prediction. Quarter-line half outcomes and
        # pushes grade to NULL (not right or wrong). dixon_coles_nhl's
        # puck_line uses the same keys ('-1.5_home', '1.5_away', ...).
        line = _extract_ah_line(predicted_outcome)
        if line is None:
            return None
        verdict = _handicap_outcome(home_score - away_score, line)
        if verdict == REC_WON:
            return f"home_covers_{_fmt_line(line)}"
        if verdict == REC_LOST:
            return f"away_covers_{_fmt_line(line)}"
        return f"push_{_fmt_line(line)}"
    if prediction_type == "clean_sheet":
        return clean_sheet_outcome(predicted_outcome, home_score, away_score)
    if prediction_type == "win_to_nil":
        return win_to_nil_outcome(predicted_outcome, home_score, away_score)
    if prediction_type == "team_total":
        return team_total_outcome(predicted_outcome, home_score, away_score)
    if prediction_type == "odd_even":
        return odd_even_outcome(home_score, away_score)
    if prediction_type == "total_goals":
        return total_goals_outcome(model_name, home_score, away_score)
    if prediction_type == "winning_margin":
        return winning_margin_outcome(home_score, away_score)
    if prediction_type == "result_btts":
        return result_btts_outcome(home_score, away_score)
    if prediction_type == "result_over_under":
        return result_over_under_outcome(home_score, away_score)
    if prediction_type in HALFTIME_TYPES:
        return halftime_outcome(prediction_type, predicted_outcome, home_score, away_score, halftime)

    # Markets we don't grade (player_prop, lottery, ...) fall through to None.
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
    if prediction_type in HANDICAP_PREDICTION_TYPES:
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
    """'home_-0.75' / 'over_2.5' -> (side, line) or (selection, None).

    Also the space-separated 'home -1.5' / 'over 5.5' that the June-2026
    NHL generator wrote: five of those recs sat won/lost with no P&L (one
    mis-settled) because this returned no line for them."""
    sel = (selection or "").strip()
    for sep in ("_", " "):
        if sep in sel:
            side, _, raw = sel.partition(sep)
            try:
                return side.lower(), float(raw)
            except ValueError:
                pass
    return sel.lower(), None


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
