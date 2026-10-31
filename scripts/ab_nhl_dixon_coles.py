"""A/B gate for the NHL Dixon-Coles artifact (dixon_coles_nhl).

WHAT IS BEING TESTED
--------------------
The served NHL artifact (/app/models/production/dixon_coles_nhl/1.0.0/model.bin,
written 2026-06-04, sha256 ac4e7e10...) has never been gated. It knows 33 teams
(no 2025-26 data: 'Utah Mammoth' is unknown and serves attack = defense = 1.0),
was fitted by the pre-2026-09 fitter (attack AND defense normalised, a positive
level bias), carries no fitted baselines — every match sits on league_avg_goals
3.10 plus a CONFIGURED 0.25 home bump — and has rho forced to 0. It feeds 8
served markets: precompute_predictions_nhl loads it into
DixonColesPredictor(DIXON_COLES_CONFIG), takes lambdas_for_match(home, away)
with NO league, builds build_dc_matrix(h, a, rho, MAX_GOALS_DERIVE), reads
derive_markets("nhl", P) and drops its moneyline: regulation_1x2, puck_line,
over_under, total_goals, double_chance, clean_sheet, win_to_nil, correct_score.
scripts/train_hockey_dixon_coles.py now fits every row and writes to STAGING
only; this harness decides whether anything it could write may replace the
served file.

PRE-REGISTERED DESIGN (fixed 2026-10-31, before the first run of any kind)
Amended once, 2026-10-31, after the 3-fold smoke run and before the full run:
the AMENDMENT section below supersedes every passage marked [A1], [A2], [A3],
and its "RULE AS AMENDED" is what decide() implements.
  Frame    train_hockey_dixon_coles.NHL_DC_TRAINING_QUERY (finished NHL,
           preseason excluded: regular season + postseason, FINAL scores incl.
           OT/SO) joined on match_id with metadata.regulation_winner
           (REGULATION_SQL). A frame row missing from that join aborts the run;
           duplicated match_ids are dropped and counted. season = the year an
           NHL season starts (August cut-over), e.g. 2024 for 2024-25.
  Folds    monthly walk-forward from --start 2024-10-01 to the newest match
           (ft.build_folds / split_fold): every fit uses match_date < the
           fold's month start only. A month with fewer than MIN_FOLD_EVAL (50)
           eval rows is POOLED into the preceding kept fold when that fold is in
           the same season and contiguous (the playoff tails: May and June join
           April, still fitted at April 1 — strictly before every pooled row),
           otherwise skipped and listed (the empty offseason months, a thin
           September). [A3: only the folds from DECISION_START decide.]
  Arms     every fitted arm is round-tripped through save() and the serve loader
           (ht.load_serving == precompute_predictions_nhl._load_hockey_dixon_coles)
           and read through the serve path above — no league passed.
    served        the production file (or --served-artifact): the incumbent.
    pinned        walk-forward refit with the trainer's own config
                  (train_hockey_dixon_coles.nhl_config(): the pinned legacy
                  config), rho forced to 0 (trainer.force_rho_zero), on all rows
                  before the fold — "the trainer as it now runs".
    refit         walk-forward refit with DIXON_COLES_CONFIG under the name
                  dixon_coles_nhl (730-day decay, 20-match prior, fitted
                  baselines, max_goals 10), rho forced to 0.
    refit_frozen  refit's config fitted ONCE at the first kept fold's cutoff and
                  scored on every fold [A3: fitted at the first DECISION fold's
                  cutoff, scored on decision folds only]: this artifact is
                  fitted by hand and then served for months, so this staleness
                  is what would ship.
    base_global / base_season / base_season_dw
                  walk-forward constants: the empirical FINAL-scoreline
                  distribution (each side capped at MAX_GOALS_DERIVE) read through
                  derive_hockey_markets, with regulation_1x2 and double_chance
                  replaced by the regulation_winner class rates; global, per
                  season shrunk toward global with k=200 effective matches, and
                  the same with the refit arms' decay weights. Context only.
  Units    every served market, scored as precompute_predictions_nhl serves it
           and GRADED by grading_outcomes itself (actual_outcome +
           grade_prediction, model_name dixon_coles_nhl, called per row and per
           key): regulation_1x2 and double_chance on metadata.regulation_winner
           (a row without one is not gradable for them), the rest on the final
           score. A binary proposition is its own unit (binary Brier); a
           partition is one unit (Brier summed over its keys):
             over_4.5 / over_5.5 / over_6.5      over_under 'over_<line>'
             regulation_1x2                      home/draw/away (3-way)
             double_chance_1X / _12 / _X2        each coverage set
             puck_line_-1.5 / puck_line_+1.5     '-1.5_home' / '1.5_home'
             total_goals                         '0'..'7', '8+' (9-way)
             clean_sheet_home / _away            'home_yes' / 'away_yes'
             win_to_nil_home / _away             'home_yes' / 'away_yes'
             correct_score                       the arm's OWN 12 listed
                                                 scorelines + 'other' (13-way),
                                                 'other' graded by
                                                 grade_correct_score_other
                                                 [A1: now correct_score_listed,
                                                 context only; the blocking
                                                 unit is correct_score_grid]
  Primary  over_5.5 (binary Brier) — NHL's canonical total line.
  Disqualification
           served and the constants must yield finite, normalised matrices and
           valid probabilities on every row, or the run aborts. A challenger is
           DISQUALIFIED — not scored on any row, its earlier folds dropped,
           never fitted again — the first time the NHL trainer would refuse its
           fit (halftime.fit_refusal after force_rho_zero: not converged, fitted
           vs observed mean total goals outside +/-2%, or an invalid served
           matrix for a fit fixture) or it emits an invalid probability.

DECISION RULE (pre-registered 2026-10-31, before any run)
  SHIP only if refit_frozen
    (a) passes ship_strict vs served on over_5.5: ft.ship_verdict (dBrier <=
        -0.005, the per-match AND the fold-clustered z=2 95% upper bounds < 0)
        AND the t(n_folds-1) fold-clustered upper bound < 0; and
    (b) regresses on NO other unit vs served. A REGRESSION is a per-unit dBrier
        (challenger - served, pooled per-match mean) > +0.002 with a 95% upper
        bound > 0 — the fold-clustered t(n_folds-1) bound, or the per-match
        delta + 2*SE when no clustered SE exists. A unit that cannot be assessed
        (no gradable row) blocks too. [A2: replaced by the lower-bound rule.]
  refit_frozen disqualified -> DO-NOT-SHIP. pinned and refit are reported and
  never decide; verdicts vs the constants are context and never decide.
  SHIP would mean: train_hockey_dixon_coles.py moves to DIXON_COLES_CONFIG (rho
  still forced to 0), scripts/promote_dc_model.py gets an "nhl" entry, and the
  staged fit is promoted after a backup. This script does none of that.

AMENDMENT 2026-10-31, made after the 3-fold smoke run (n=665) and before the full run
  The smoke (--start 2026-03-01: folds 2026-03, 2026-04 with May and June
  pooled, 2026-10) read DO-NOT-SHIP: refit_frozen over_5.5 -0.00056 vs served,
  failing ship_strict, and two "regressions" (over_4.5 +0.0024, correct_score
  +0.012). Its review found three defects in the rule as written; each change
  below answers one of them. No threshold, margin, arm, config or market was
  chosen by how it moves a verdict: the primary unit, the -0.005 gate, the
  ship_strict bounds, the arms and the disqualification rule are unchanged.
  [A1] correct_score was scored with an IMPROPER score. Each arm was scored on
       its OWN top-12 scorelines + 'other', so arms were scored on different
       partitions, and listing less likely scorelines sends more outcomes to
       'other', which the same arm prices high. Exact expectation under true
       lambdas (3.15, 2.95) through derive_hockey_markets: the true
       distribution scores 0.7067, the same lambdas inflated 5% 0.7055 and
       inflated 10% 0.6925 — and inflated lambdas are the served file's known
       defect (smoke: P(over 5.5) 0.611 predicted vs 0.574 realised). So every
       level-correct challenger "regressed" there (+0.012 to +0.015) and rule
       (b) could block exactly the fix this gate exists to test.
       Replacement (BLOCKING): correct_score_grid — the full capped
       (MAX_GOALS_DERIVE+1)^2 = 121-cell scoreline matrix each arm serves every
       market from (rows home goals, columns away), read from that arm's
       matrix, scored as a 121-way multi-class Brier against the realised FINAL
       scoreline (each side capped at MAX_GOALS_DERIVE, as the constants
       already were; grading_outcomes grades correct_score on the final score
       too). One partition, shared by every arm. Strictly proper:
       E[score] = sum_k (q_k - p_k)^2 + const, minimised only at q = p (true
       0.97265 vs inflated 5% 0.97287 / 10% 0.97347 in the example above).
       The multi-class Brier rather than the log score of the realised cell:
       (i) every other unit and the regression rule are Brier deltas, so one
       machinery reads them all; (ii) it is bounded — the empirical constants
       hold exact zeros in never-observed cells, where a log score is -inf.
       The log score is printed beside it as context (correct_score_grid_log;
       paired only where both arms are finite on every row, never on a subset).
       The product's top-12 + 'other' score stays in the report as
       correct_score_listed, CONTEXT ONLY: it never blocks.
  [A2] The regression wording did not say what was meant. "dBrier > +0.002
       with a 95% upper bound > 0" is in effect "point estimate > +0.002" (the
       upper bound of a positive delta is almost always positive): the smoke's
       over_4.5 +0.0024 "regressed" with a t-interval reaching -0.0018. And one
       fixed margin is blind to scale across units whose Brier runs from ~0.04
       (clean_sheet) to ~0.97 (the grid). A blocking unit now REGRESSES only
       when it is SIGNIFICANTLY worse: the 95% LOWER bound of its dBrier
       (challenger - served) is > 0 — the same fold-clustered t(n_folds-1)
       machinery (delta_cluster - t * se_cluster), or the per-match delta -
       2*SE when no clustered SE exists. No margin. A unit with no gradable
       row, or with no bound at all, still blocks. With 14 blocking units this
       fails closed by design.
  [A3] The decision pooled IN-SAMPLE folds. The served file's fit ends
       2025-04-04 (its trainer took the oldest 80% of the 2026-06-04 frame: by
       today's row count 6,352 of 7,940 rows), so the 2024-10..2025-04 folds
       score served on its own training rows, and pooling every fold also let a
       fit frozen on 2024-10-01 — two years stale by the window's end — decide.
       The DECISION now reads only the kept folds whose month starts on or
       after DECISION_START = 2025-05-01: every row out of sample for served.
       (May and June 2025 hold 41 and 6 games, so they pool into the April 2025
       fold and the first decision fold is 2025-10.) refit_frozen is fitted
       ONCE at the first decision fold's cutoff, on every row before it, and
       scored on the decision folds only. The default --start stays 2024-10-01
       and every fold from it is still fitted and printed — full-window pairs,
       per-fold and per-season deltas — as CONTEXT that never decides.
       Team coverage is stratified by the frozen arm's known teams as well as
       served's, and the teams each of them does not know are listed. Watch
       the Utah naming: 2024-25 games carry 'Utah Utah Hockey Club', 2025-26
       games 'Utah Mammoth' — a distinct team to every fit — so served and a
       2025-10-01 frozen fit both serve Utah at attack = defense = 1.0 for the
       whole decision window, while the walk-forward arms learn it.
  Under the amended rule the smoke's two regressions do not count (over_4.5's
  95% lower bound is below 0; correct_score_listed no longer blocks); its
  decision stays DO-NOT-SHIP, because over_5.5 fails ship_strict. A re-run of
  the same smoke with the amended harness (2026-10-31, read-only, to prove it
  runs; it decides nothing) reproduced the 665 rows and every over_5.5 number,
  found no blocking regression for any challenger, and put correct_score_grid
  at -0.00026 (pinned) .. -0.00051 (refit) vs served, where the listed score
  had read +0.012 .. +0.015.

RULE AS AMENDED (what decide() implements)
  Decision window: the kept folds whose month starts on or after 2025-05-01.
  SHIP only if refit_frozen — one fit at the first decision fold's cutoff —
  over the decision window
    (a) passes ship_strict vs served on over_5.5 (unchanged: ft.ship_verdict
        AND the t(n_folds-1) fold-clustered upper bound < 0), and
    (b) is significantly worse than served on NO blocking unit: for each unit
        other than over_5.5 and correct_score_listed (BLOCKING_UNITS), the 95%
        lower bound of dBrier is not > 0, and every one of them is assessable.
  No decision fold, or refit_frozen disqualified -> DO-NOT-SHIP. pinned,
  refit, the constants, correct_score_listed, correct_score_grid_log and every
  full-window number are context and never decide.

CONFOUNDS, PRINTED NOT HIDDEN
  * The served artifact's training window is not recorded in it; by the row
    count above it ends 2025-04-04. The decision excludes it [A3]; the context
    folds before 2025-05 are IN-SAMPLE for served — a bias AGAINST the
    challengers there. Per-fold and per-season deltas are printed.
  * Team coverage: strata by how many of a match's teams served, and the frozen
    fit, know; the unknown teams are listed by name [A3].
  * correct_score_listed is scored on each arm's own top-12 + 'other'
    partition, because that is what is served and graded; it is improper
    across arms and is printed only as the product's score [A1].
  * correct_score_grid is graded, like the served market, on FINAL scores,
    which are never tied; every fitted arm (served included) puts its
    regulation-tie mass on the diagonal, which never wins. That costs served
    and the challengers alike — the empirical constants put nothing there,
    hence their large grid lead — so it does not tilt the decision pair.
  * Every fitted arm forces rho to 0 and fits FINAL scores, so none of them
    addresses the regulation-tie deficit (draw cells ~15% vs ~23% observed);
    the regulation units measure how much each arm moves it, nothing fixes it.

Read-only: SELECTs, plus temp files for the round-trip. Usage (api container):
    python /app/scripts/ab_nhl_dixon_coles.py --out-json /tmp/ab_nhl.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_halftime_dixon_coles as ht  # noqa: E402
import ab_soccer_dixon_coles_refit as ft  # noqa: E402
import grading_outcomes as go  # noqa: E402
import train_hockey_dixon_coles as trainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_nhl_dixon_coles")

SERVED_ARTIFACT = "/app/models/production/dixon_coles_nhl/1.0.0/model.bin"
LEGACY_SERVED_TEAMS = 33
MODEL_NAME = trainer.MODEL_NAME  # "dixon_coles_nhl" — what grading_outcomes sees as model_name
DEFAULT_START = "2024-10-01"  # the run window: every kept fold from here is fitted and printed
# [A3] Only the kept folds whose month starts on or after DECISION_START decide:
# every one of their rows is out of sample for the served file, whose fit ends
# SERVED_FIT_END. refit_frozen is fitted at the first of them.
DECISION_START = "2025-05-01"
SERVED_FIT_END = "2025-04-04"
MIN_FOLD_EVAL = ht.MIN_FOLD_EVAL  # 50
SEASON_START_MONTH = 8  # August: after every final (2021's ended 7 July), before every opener
FIT_COLS = ["home_team", "away_team", "home_score", "away_score", "match_date", "league_id"]

# precompute_predictions_nhl.HOCKEY_DERIVE_TO_PREDICTION's keys: what
# derive_markets("nhl", P) yields once its moneyline is dropped.
SERVED_MARKETS = (
    "regulation_1x2",
    "puck_line",
    "over_under",
    "total_goals",
    "double_chance",
    "clean_sheet",
    "win_to_nil",
    "correct_score",
)


@dataclass(frozen=True)
class Unit:
    """One scored proposition set of a served market. ``keys`` are the
    selections whose probabilities are scored: one key = a binary unit, several
    = a partition (exactly one key wins). The two correct_score units have no
    fixed keys: correct_score_grid is scored on the full capped scoreline grid
    every arm's matrix covers, correct_score_listed on the arm's own listed
    scorelines + 'other'."""

    name: str
    market: str  # the served market_type == predictions.prediction_type
    keys: Tuple[str, ...] = ()


PRIMARY = "over_5.5"
# [A1] The blocking correct_score unit: the whole (MAX_GOALS_DERIVE+1)^2 matrix
# each arm serves from, a 121-way Brier on one partition shared by every arm.
CORRECT_SCORE_GRID = "correct_score_grid"
# The product's top-12 + 'other' as served and graded — improper across arms
# (each arm's own partition), so CONTEXT ONLY: never a regression.
CORRECT_SCORE_LISTED = "correct_score_listed"
UNITS: Tuple[Unit, ...] = (
    Unit("over_5.5", "over_under", ("over_5.5",)),
    Unit("over_4.5", "over_under", ("over_4.5",)),
    Unit("over_6.5", "over_under", ("over_6.5",)),
    Unit("regulation_1x2", "regulation_1x2", ("home", "draw", "away")),
    Unit("double_chance_1X", "double_chance", ("1X",)),
    Unit("double_chance_12", "double_chance", ("12",)),
    Unit("double_chance_X2", "double_chance", ("X2",)),
    Unit("puck_line_-1.5", "puck_line", ("-1.5_home",)),
    Unit("puck_line_+1.5", "puck_line", ("1.5_home",)),
    Unit("total_goals", "total_goals", tuple(str(k) for k in range(8)) + ("8+",)),
    Unit("clean_sheet_home", "clean_sheet", ("home_yes",)),
    Unit("clean_sheet_away", "clean_sheet", ("away_yes",)),
    Unit("win_to_nil_home", "win_to_nil", ("home_yes",)),
    Unit("win_to_nil_away", "win_to_nil", ("away_yes",)),
    Unit(CORRECT_SCORE_GRID, "correct_score"),
    Unit(CORRECT_SCORE_LISTED, "correct_score"),
)
UNIT_NAMES = tuple(u.name for u in UNITS)
FIXED_UNITS = tuple(u for u in UNITS if u.keys)  # graded by grading_outcomes; truth independent of the arm
CONTEXT_UNITS = (CORRECT_SCORE_LISTED,)  # scored and printed, never decide
# [A2] The units a significant regression on blocks SHIP: all but the primary and the context.
BLOCKING_UNITS = tuple(u for u in UNIT_NAMES if u != PRIMARY and u not in CONTEXT_UNITS)
REGULATION_CLASSES = ("home", "draw", "away")

CHALLENGERS = ("pinned", "refit", "refit_frozen")
WALK_FORWARD_ARMS = ("pinned", "refit")
CONSTANT_ARMS = ("base_global", "base_season", "base_season_dw")
DECAYED_ARMS = ("refit", "refit_frozen")  # compared with base_season_dw; the rest with base_season
DECIDING_ARM = "refit_frozen"
CONSTANT_SHRINKAGE = 200.0  # effective matches per season — ft.fit_base_rates' k

REGULATION_SQL = """
    SELECT m.id::text AS match_id, m.metadata->>'regulation_winner' AS regulation_winner
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'nhl' AND m.status = 'finished'
"""


# ── Pure ─────────────────────────────────────────────────────────────


def season_of(ts: Any) -> int:
    """The year the NHL season containing ``ts`` starts (2024 for 2024-25)."""
    t = pd.Timestamp(ts)
    return int(t.year if t.month >= SEASON_START_MONTH else t.year - 1)


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def pool_thin_folds(
    folds: Sequence[Any], counts: Sequence[int], min_eval: int = MIN_FOLD_EVAL
) -> Tuple[List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(kept folds, pooled, skipped). A fold with < ``min_eval`` eval rows joins
    the preceding kept fold when that one is contiguous and in the same season
    — its fit cutoff (month start) is unchanged, so it stays strictly before
    every pooled row — and is skipped otherwise (an empty month included)."""
    kept: List[Any] = []
    pooled: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for fold, n in zip(folds, counts):
        if n >= min_eval:
            kept.append(fold)
            continue
        prev = kept[-1] if kept else None
        if (
            n > 0
            and prev is not None
            and prev.eval_end == fold.eval_start
            and season_of(prev.month_start) == season_of(fold.month_start)
        ):
            kept[-1] = ft.Fold(month_start=prev.month_start, eval_start=prev.eval_start, eval_end=fold.eval_end)
            pooled.append({"fold": fold.label, "n_eval": int(n), "into": prev.label})
        else:
            skipped.append({"fold": fold.label, "n_eval": int(n)})
    return kept, pooled, skipped


def _regulation_metadata(regulation_winner: Any) -> Dict[str, Any]:
    """The metadata grading_outcomes reads: only regulation_winner, absent when unknown."""
    if regulation_winner is None or (isinstance(regulation_winner, float) and math.isnan(regulation_winner)):
        return {}
    return {"regulation_winner": str(regulation_winner)}


def graded_vector(
    market: str,
    keys: Sequence[str],
    home: int,
    away: int,
    regulation_winner: Any,
    probabilities: Optional[Dict[str, float]] = None,
) -> Optional[np.ndarray]:
    """1.0 / 0.0 per key — whether grading_outcomes grades a ``market`` pick of
    that key correct on this match — or None when any key is ungradable (a
    regulation market without regulation_winner, a push). A partition (more
    than one key) must have exactly one winner, or this raises: the grading and
    the served keys would disagree on what the market is."""
    metadata = _regulation_metadata(regulation_winner)
    out = []
    for key in keys:
        actual = go.actual_outcome(
            prediction_type=market,
            model_name=MODEL_NAME,
            predicted_outcome=key,
            home_score=int(home),
            away_score=int(away),
            metadata=metadata,
        )
        verdict = go.grade_prediction(key, actual, market, probabilities)
        if verdict is None:
            return None
        out.append(1.0 if verdict else 0.0)
    y = np.array(out, dtype=float)
    if len(keys) > 1 and y.sum() != 1.0:
        raise ValueError(f"{market}: {int(y.sum())} of keys {list(keys)} graded correct on {home}-{away}")
    return y


def grid_cap() -> int:
    """The side cap of the scoreline grid every arm is read on:
    market_derivation.MAX_GOALS_DERIVE, the size precompute_predictions_nhl
    builds every served matrix at."""
    from predictors.market_derivation import MAX_GOALS_DERIVE  # type: ignore

    return int(MAX_GOALS_DERIVE)


def grid_truth(home: Sequence[int], away: Sequence[int], cap: int) -> np.ndarray:
    """(n, (cap+1)^2) one-hot of each realised FINAL scoreline — what
    grading_outcomes.correct_score_outcome grades correct_score on — each side
    capped at ``cap`` so it lands on the grid every arm's matrix covers."""
    idx = scoreline_index(home, away, cap)
    out = np.zeros((len(idx), (cap + 1) ** 2))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def fixed_truth(ev: pd.DataFrame) -> Dict[str, np.ndarray]:
    """(n, k) graded outcome per fixed unit, a NaN row where ungradable, plus
    correct_score_grid's one-hot realised cell (every row gradable)."""
    hs = ev["home_score"].to_numpy(dtype=int)
    as_ = ev["away_score"].to_numpy(dtype=int)
    rw = ev["regulation_winner"].to_numpy(dtype=object)
    out: Dict[str, np.ndarray] = {}
    for unit in FIXED_UNITS:
        arr = np.full((len(ev), len(unit.keys)), np.nan)
        for i in range(len(ev)):
            y = graded_vector(unit.market, unit.keys, hs[i], as_[i], rw[i])
            if y is not None:
                arr[i] = y
        out[unit.name] = arr
    out[CORRECT_SCORE_GRID] = grid_truth(hs, as_, grid_cap())
    return out


def regulation_label(regulation_winner: Any) -> Optional[str]:
    """home/draw/away exactly as grading_outcomes reads regulation_winner, or None."""
    return go.nhl_regulation_3way(_regulation_metadata(regulation_winner))


def readout(
    markets: Sequence[Dict[str, Dict[str, float]]], ev: pd.DataFrame, mats: Sequence[np.ndarray]
) -> Dict[str, np.ndarray]:
    """Per-unit (n, k) probabilities of each row's served markets and matrix:
    the fixed units from the market dicts; correct_score_grid = the row's whole
    scoreline matrix (rows home goals, columns away) flattened — the
    distribution every served market is derived from, on the grid all arms
    share; correct_score_listed = the served top-12 + 'other'; and
    '_cs_listed_truth', the graded outcome on the arm's own listed keys."""
    n = len(markets)
    if len(mats) != n or len(ev) != n:
        raise ValueError(f"readout needs one matrix and one eval row per market set ({n}, {len(mats)}, {len(ev)})")
    cap = grid_cap()
    out: Dict[str, np.ndarray] = {u.name: np.full((n, len(u.keys)), np.nan) for u in FIXED_UNITS}
    grid = np.full((n, (cap + 1) ** 2), np.nan)
    cs_p = np.full((n, 13), np.nan)
    cs_y = np.full((n, 13), np.nan)
    hs = ev["home_score"].to_numpy(dtype=int)
    as_ = ev["away_score"].to_numpy(dtype=int)
    rw = ev["regulation_winner"].to_numpy(dtype=object)
    for i, (m, P) in enumerate(zip(markets, mats)):
        for unit in FIXED_UNITS:
            out[unit.name][i] = [m[unit.market][k] for k in unit.keys]
        P = np.asarray(P, dtype=float)
        if P.shape != (cap + 1, cap + 1):
            raise RuntimeError(
                f"scoreline matrix {P.shape}: every arm must be read on the shared {cap + 1}x{cap + 1} grid"
            )
        grid[i] = P.ravel()
        cs = m["correct_score"]
        keys = list(cs)
        if len(keys) != 13:
            raise RuntimeError(f"correct_score served {len(keys)} keys, expected 12 scorelines + 'other'")
        cs_p[i] = [cs[k] for k in keys]
        y = graded_vector("correct_score", keys, hs[i], as_[i], rw[i], probabilities=cs)
        if y is not None:
            cs_y[i] = y
    out[CORRECT_SCORE_GRID] = grid
    out[CORRECT_SCORE_LISTED] = cs_p
    out["_cs_listed_truth"] = cs_y
    return out


def score_arm(probs: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-row Brier per unit — summed over a partition's keys (the grid's
    121 cells included), binary for one key — NaN where the row is ungradable
    for that unit."""
    full = {**truth, CORRECT_SCORE_LISTED: probs["_cs_listed_truth"]}
    return {u: np.sum((np.asarray(probs[u], dtype=float) - full[u]) ** 2, axis=1) for u in UNIT_NAMES}


def grid_log_score(probs: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-row -log of the probability the arm's matrix put on the realised
    (capped) cell — the log score on correct_score_grid's partition. +inf
    where that probability is 0 (an empirical constant's never-observed cell):
    context only, and never averaged over a subset of rows."""
    q = np.sum(np.asarray(probs[CORRECT_SCORE_GRID], dtype=float) * truth[CORRECT_SCORE_GRID], axis=1)
    with np.errstate(divide="ignore"):
        return -np.log(q)


def invalid_matrices(mats: Sequence[np.ndarray]) -> Optional[str]:
    """None when every scoreline matrix is finite, non-negative and sums to 1."""
    bad = [i for i, P in enumerate(mats) if not (np.isfinite(P).all() and (P >= 0).all() and abs(P.sum() - 1.0) < 1e-6)]
    return f"{len(bad)} of {len(mats)} scoreline matrices non-finite, negative or non-normalised" if bad else None


def invalid_probabilities(probs: Dict[str, np.ndarray]) -> Optional[str]:
    """None when every unit's every row is a probability (a partition's rows sum to 1)."""
    for u in UNITS:
        v = np.asarray(probs[u.name], dtype=float)
        bad = ~np.isfinite(v).all(axis=1) | (v < -1e-12).any(axis=1) | (v > 1 + 1e-12).any(axis=1)
        if v.shape[1] > 1:
            bad |= ~np.isclose(v.sum(axis=1), 1.0, atol=1e-6)
        if bad.any():
            return f"{int(bad.sum())} of {len(v)} {u.name} rows invalid"
    return None


def disqualification(
    arm: str, probs: Optional[Dict[str, np.ndarray]] = None, refusal: Optional[str] = None
) -> Optional[str]:
    """Why ``arm`` cannot be scored, or None. The trainer's own refusal first —
    an unconverged fit can still emit finite matrices. For served or a constant
    a reason RAISES: the experiment itself is broken."""
    reason = None
    if refusal is not None:
        reason = f"the NHL trainer would refuse this fit: {refusal}"
    elif probs is not None:
        reason = invalid_probabilities(probs)
    if reason is not None and arm not in CHALLENGERS:
        raise RuntimeError(f"{arm}: {reason}")
    return reason


def matched_constant(arm: str) -> str:
    return "base_season_dw" if arm in DECAYED_ARMS else "base_season"


def t_lower(delta: Optional[float], se: Optional[float], n_folds: Optional[int]) -> Optional[float]:
    """Fold-clustered 95% LOWER bound with a t(n_folds-1) multiplier — the
    mirror of ht.t_upper."""
    if delta is None or se is None or not n_folds or n_folds < 2:
        return None
    if not (math.isfinite(delta) and math.isfinite(se)):
        return None
    from scipy.stats import t as student_t

    return float(delta - float(student_t.ppf(0.975, n_folds - 1)) * se)


def lower_bound(p: Dict[str, Any]) -> Optional[float]:
    """The 95% lower bound the regression rule reads [A2]: the fold-clustered
    t(n_folds-1) bound, else the per-match delta - 2*SE, else None."""
    if p.get("lower_95_cluster_t") is not None:
        return float(p["lower_95_cluster_t"])
    d, se = p.get("delta"), p.get("se")
    if d is None or se is None or not math.isfinite(d) or not math.isfinite(se):
        return None
    return float(d - 2.0 * se)


def regression(p: Dict[str, Any]) -> Optional[str]:
    """Why this per-unit pair (challenger - served) is a regression under the
    amended rule [A2], or None. A regression is SIGNIFICANTLY worse: the 95%
    lower bound of dBrier above 0 — no margin. No delta at all (no gradable
    paired row), or no bound, counts too: an unassessed unit is not shown not
    to regress."""
    d = p.get("delta")
    if d is None or not math.isfinite(d):
        return "unassessable (no gradable paired rows)"
    lb = lower_bound(p)
    if lb is None:
        return f"unassessable (dBrier {d:+.5f} with no 95% bound)"
    if lb <= 0:
        return None
    return f"significantly worse: dBrier {d:+.5f}, 95% lower bound {lb:+.5f} > 0"


def blocking_regressions(per_unit: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """The regressions that block SHIP: every BLOCKING unit — never the
    primary (rule (a) judges it) nor correct_score_listed (context only)."""
    out: Dict[str, str] = {}
    for unit in BLOCKING_UNITS:
        why = regression(per_unit.get(unit, {}))
        if why is not None:
            out[unit] = why
    return out


def decide(
    verdicts_vs_served: Dict[str, Dict[str, Any]],
    regressions: Dict[str, Dict[str, str]],
    disqualified: Dict[str, Any],
    arm: str = DECIDING_ARM,
    unavailable: Optional[str] = None,
) -> Dict[str, Any]:
    """The amended pre-registered decision for ``arm`` over the decision
    window: primary ship_strict AND no significant regression on a blocking
    unit. ``unavailable`` (no decision fold) is a DO-NOT-SHIP of its own."""
    if unavailable is not None:
        return {"arm": arm, "ship": False, "primary_pass": False, "regressions": {}, "reason": unavailable}
    if arm in disqualified or arm not in verdicts_vs_served:
        why = disqualified.get(arm, {}).get("reason", "not scored")
        return {"arm": arm, "ship": False, "primary_pass": False, "regressions": {}, "reason": f"disqualified: {why}"}
    primary_pass = bool(verdicts_vs_served[arm]["ship_strict"])
    regs = dict(regressions.get(arm, {}))
    ship = primary_pass and not regs
    if ship:
        reason = f"{PRIMARY} passes ship_strict vs served and no blocking unit is significantly worse"
    elif not primary_pass:
        reason = f"{PRIMARY} fails ship_strict vs served" + (f"; {len(regs)} unit(s) also regress" if regs else "")
    else:
        reason = f"{PRIMARY} passes, but {len(regs)} unit(s) regress: {', '.join(regs)}"
    return {"arm": arm, "ship": ship, "primary_pass": primary_pass, "regressions": regs, "reason": reason}


def expected_totals(mats: Sequence[np.ndarray]) -> np.ndarray:
    """E[total goals] of each scoreline matrix (rows home goals, columns away)."""
    out = np.empty(len(mats))
    for r, P in enumerate(mats):
        i, j = np.indices(P.shape)
        out[r] = float((P * (i + j)).sum())
    return out


def scoreline_index(home: Sequence[int], away: Sequence[int], cap: int) -> np.ndarray:
    """Class index of a final scoreline in a (cap+1)^2 grid, each side capped."""
    h = np.minimum(np.asarray(home, dtype=int), cap)
    a = np.minimum(np.asarray(away, dtype=int), cap)
    return h * (cap + 1) + a


def known_count(home: Sequence[str], away: Sequence[str], teams: Any) -> np.ndarray:
    """How many of each match's two teams an artifact knows (0, 1 or 2)."""
    return np.array([int(h in teams) + int(a in teams) for h, a in zip(home, away)], dtype=float)


def unknown_teams(home: Sequence[str], away: Sequence[str], teams: Any) -> Dict[str, int]:
    """{team name: matches} for every team an artifact does not know — it
    serves them at attack = defense = 1.0 — most frequent first."""
    counts: Dict[str, int] = {}
    for name in [*home, *away]:
        if name not in teams:
            counts[str(name)] = counts.get(str(name), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# ── Model plumbing (needs ml-models on the path) ─────────────────────


def arm_configs() -> Dict[str, Any]:
    """The ModelConfig each fitted arm uses: pinned is exactly the trainer's."""
    from dataclasses import replace

    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore

    return {"pinned": trainer.nhl_config(), "refit": replace(DIXON_COLES_CONFIG, name=MODEL_NAME)}


def served_markets(P: np.ndarray) -> Dict[str, Dict[str, float]]:
    """derive_markets('nhl', P) minus its moneyline — exactly the rows
    precompute_predictions_nhl stores."""
    from predictors.market_derivation import derive_markets  # type: ignore

    m = derive_markets("nhl", P)
    m.pop("moneyline", None)
    if set(m) != set(SERVED_MARKETS):
        raise RuntimeError(f"derive_hockey_markets yields {sorted(m)}, this harness scores {sorted(SERVED_MARKETS)}")
    return m


def model_matrices(model: Any, ev: pd.DataFrame) -> List[np.ndarray]:
    """The served scoreline matrix per eval row: lambdas_for_match(home, away)
    with NO league, rho from the artifact, MAX_GOALS_DERIVE."""
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix  # type: ignore

    rho = float(getattr(model, "rho", 0.0) or 0.0)
    out = []
    for home, away in zip(ev["home_team"], ev["away_team"]):
        h_lam, a_lam = model.lambdas_for_match(home, away)
        out.append(build_dc_matrix(h_lam, a_lam, rho, max_goals=MAX_GOALS_DERIVE))
    return out


def fit_arm(fit_df: pd.DataFrame, cfg: Any, tmpdir: str, tag: str) -> Tuple[Any, Dict[str, Any], str, Optional[str]]:
    """Fit one artifact exactly as the trainer does (train, force rho to 0,
    its refusal over the fit's fixtures as served), then round-trip it through
    save() + the serve loader. Returns (served_model, train_result, path, refusal)."""
    from predictors.poisson_models import DixonColesPredictor  # type: ignore

    model = DixonColesPredictor(cfg)
    result = model.train(fit_df[FIT_COLS].copy())
    trainer.force_rho_zero(model, result)
    refusal = trainer.halftime.fit_refusal(result, model, list(trainer.served_fixtures(fit_df)))
    path = os.path.join(tmpdir, f"{tag}.bin")
    model.save(path)
    return ht.load_serving(path), result, path, refusal


def constant_markets(
    fit_df: pd.DataFrame, ev: pd.DataFrame, weights, by_season: bool
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """Walk-forward constant markets per eval row (see Arms in the docstring),
    and the constant scoreline matrix each final-score market was read from."""
    from predictors.market_derivation import MAX_GOALS_DERIVE  # type: ignore

    cap = MAX_GOALS_DERIVE
    group = (lambda df: df["season"].to_numpy(dtype=object)) if by_season else (lambda df: np.full(len(df), None))
    fit = fit_df.assign(
        _y=scoreline_index(fit_df["home_score"], fit_df["away_score"], cap),
        _g=group(fit_df),
        _r=[regulation_label(v) for v in fit_df["regulation_winner"]],
    )
    w = None if weights is None else np.asarray(weights, dtype=float)
    groups = group(ev)
    scores = ft.fit_class_base_rates(fit, "_y", (cap + 1) ** 2, league_col="_g", k=CONSTANT_SHRINKAGE, weights=w)
    P = ft.predict_class_base_rates(scores, groups)
    has_reg = fit["_r"].notna().to_numpy()
    reg = fit[has_reg].assign(_r=fit.loc[has_reg, "_r"].map(REGULATION_CLASSES.index))
    rates = ft.fit_class_base_rates(
        reg, "_r", 3, league_col="_g", k=CONSTANT_SHRINKAGE, weights=None if w is None else w[has_reg]
    )
    R = ft.predict_class_base_rates(rates, groups)
    mats = [P[i].reshape(cap + 1, cap + 1) for i in range(len(ev))]
    out = []
    for i, mat in enumerate(mats):
        m = served_markets(mat)
        h, d, a = (float(x) for x in R[i])
        m["regulation_1x2"] = {"home": h, "draw": d, "away": a}
        m["double_chance"] = {"1X": h + d, "12": h + a, "X2": d + a}
        out.append(m)
    return out, mats


def assemble_frame(frame: pd.DataFrame, regulation: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Join the trainer's frame with regulation_winner and prove the join."""
    out = frame.merge(regulation, on="match_id", how="left", validate="many_to_one", indicator=True)
    missing = out["_merge"] != "both"
    if missing.any():
        raise RuntimeError(
            f"{int(missing.sum())} NHL frame row(s) absent from REGULATION_SQL "
            f"(e.g. {out.loc[missing, 'match_id'].head(5).tolist()}) — the two queries disagree"
        )
    out = out.drop(columns="_merge")
    labels = [regulation_label(v) for v in out["regulation_winner"]]
    audit = {
        "rows": int(len(out)),
        "without_regulation_winner": int(sum(lab is None for lab in labels)),
        "regulation_labels": {str(k): int(v) for k, v in pd.Series(out["regulation_winner"]).value_counts().items()},
    }
    return out, audit


def load_frame(database_url: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from sqlalchemy import create_engine
    from utils.training_data import load_training_frame  # type: ignore

    frame = load_training_frame(database_url=database_url, query=trainer.NHL_DC_TRAINING_QUERY)
    engine = create_engine(database_url)
    try:
        regulation = pd.read_sql(REGULATION_SQL, engine)
    finally:
        engine.dispose()
    return assemble_frame(frame, regulation)


# ── Run ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    sys.path.insert(0, ht.ML_SRC)
    frame, frame_audit = load_frame(args.database_url)
    if frame_audit["without_regulation_winner"]:
        logger.warning(
            "%d NHL row(s) carry no usable regulation_winner — not gradable for regulation_1x2 / double_chance",
            frame_audit["without_regulation_winner"],
        )
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    frame = frame.sort_values("match_date", kind="stable").reset_index(drop=True)
    dups = int(frame["match_id"].duplicated().sum())
    if dups:
        # The trainer fits this same frame, so this is a finding about it too.
        logger.error("NHL frame carries %d duplicated match_id rows — dropped for scoring", dups)
        frame = frame.drop_duplicates("match_id", keep="first").reset_index(drop=True)
    frame["season"] = [season_of(t) for t in frame["match_date"]]
    end = args.end or frame["match_date"].max().strftime("%Y-%m-%d")
    raw_folds = ft.build_folds(args.start, end)
    counts = [len(ft.split_fold(frame, f)[1]) for f in raw_folds]
    folds, pooled, skipped = pool_thin_folds(raw_folds, counts)
    for p in pooled:
        logger.info("fold %s (%d eval rows) pooled into %s", p["fold"], p["n_eval"], p["into"])
    for s in skipped:
        logger.warning("fold %s skipped: %d eval rows < %d", s["fold"], s["n_eval"], MIN_FOLD_EVAL)
    if not folds:
        raise RuntimeError(f"no fold has >= {MIN_FOLD_EVAL} eval rows between {args.start} and {end}")
    # [A3] The decision window: kept folds whose month starts on or after
    # DECISION_START (a pooled fold is judged by its month start, so a fold that
    # starts before it — and its pooled tail — stays context).
    decision_from = pd.Timestamp(DECISION_START, tz="UTC")
    decision_folds = [f for f in folds if f.month_start >= decision_from]
    decision_labels = [f.label for f in decision_folds]
    unavailable = None
    if not decision_folds:
        unavailable = f"no kept fold starts on or after the decision start {DECISION_START}: nothing can decide"
        logger.error("%s — DO-NOT-SHIP", unavailable)
    logger.info(
        "frame: %d NHL matches %s..%s; %d folds (%d pooled, %d skipped); decision folds (>= %s): %s",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
        len(folds),
        len(pooled),
        len(skipped),
        DECISION_START,
        decision_labels or "none",
    )

    served_fp = ht.fingerprint(args.served_artifact)
    legacy = served_fp["teams"] == LEGACY_SERVED_TEAMS and not served_fp["has_league_baselines"]
    if args.served_artifact == SERVED_ARTIFACT and not legacy:
        raise RuntimeError(
            f"production NHL artifact is not the 2026-06-04 legacy fit ({served_fp}) — it may have been replaced; "
            "point --served-artifact at the backup to score the real incumbent"
        )
    served = ht.load_serving(args.served_artifact)
    served_teams = set(getattr(served, "team_attack", {}) or {})

    cfgs = arm_configs()
    decay = float(cfgs["refit"].hyperparameters.get("time_decay", 0.0))
    arms: List[str] = ["served", *CHALLENGERS, *CONSTANT_ARMS]
    scores: Dict[str, Dict[str, List[np.ndarray]]] = {a: {u: [] for u in UNIT_NAMES} for a in arms}
    log_scores: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    p_primary: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    p_reg: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    e_total: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    y_primary: List[np.ndarray] = []
    y_reg: List[np.ndarray] = []
    totals: List[np.ndarray] = []
    fold_of: List[np.ndarray] = []
    season_arr: List[np.ndarray] = []
    home_arr: List[np.ndarray] = []
    away_arr: List[np.ndarray] = []
    fits: List[Dict[str, Any]] = []
    disqualified: Dict[str, Dict[str, Any]] = {}
    cache: Dict[Tuple[str, pd.Timestamp], Tuple[Any, Dict[str, Any], str, Optional[str]]] = {}

    def fit(key: str, fit_df: pd.DataFrame, cutoff: pd.Timestamp, tmp: str):
        # Fits are deterministic and every fit frame is frame[match_date <
        # cutoff], so the frozen fit IS the first decision fold's refit — fitted once.
        if (key, cutoff) not in cache:
            cache[(key, cutoff)] = fit_arm(fit_df, cfgs[key], tmp, f"{key}_{cutoff:%Y%m%d}")
        return cache[(key, cutoff)]

    def record(label: str, arm: str, fit_df: pd.DataFrame, cutoff: pd.Timestamp, result, path, refusal) -> None:
        fits.append(
            {
                "fold": label,
                "arm": arm,
                "n_fit": int(len(fit_df)),
                "teams": int(result.get("team_count") or 0),
                "converged_at": result.get("iterations"),
                "leagues": int(result.get("league_count") or 0),
                "rho_fitted": result.get("rho_fitted"),
                "fitted_total": result.get("fitted_mean_total_goals"),
                "observed_total": result.get("observed_mean_total_goals"),
                "refusal": refusal,
                "newest_fit_gap_days": float((cutoff - fit_df["match_date"].max()).total_seconds() / 86400),
                "sha256": ht.fingerprint(path)["sha256"][:12],
            }
        )

    def disqualify(arm: str, label: str, reason: str) -> None:
        disqualified[arm] = {"fold": label, "reason": reason}
        logger.error("%s DISQUALIFIED at fold %s: %s", arm, label, reason)
        arms.remove(arm)

    def model_arm(model: Any, ev: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, Optional[str]]:
        mats = model_matrices(model, ev)
        bad = invalid_matrices(mats)
        if bad:
            return {}, np.array([]), bad
        return readout([served_markets(P) for P in mats], ev, mats), expected_totals(mats), None

    frozen: Any = None
    frozen_fp: Optional[Dict[str, Any]] = None
    frozen_teams: set = set()
    frozen_cutoff: Optional[pd.Timestamp] = None
    with tempfile.TemporaryDirectory() as tmp:
        if decision_folds:
            frozen_cutoff = decision_folds[0].fit_cutoff
            first_fit = frame[frame["match_date"] < frozen_cutoff]
            frozen, frozen_result, frozen_path, frozen_refusal = fit("refit", first_fit, frozen_cutoff, tmp)
            frozen_label = f"frozen@{frozen_cutoff.date()}"
            record(frozen_label, DECIDING_ARM, first_fit, frozen_cutoff, frozen_result, frozen_path, frozen_refusal)
            frozen_fp = ht.fingerprint(frozen_path)
            frozen_teams = set(getattr(frozen, "team_attack", {}) or {})
            reason = disqualification(DECIDING_ARM, refusal=frozen_refusal)
            if reason is not None:
                disqualify(DECIDING_ARM, frozen_label, reason)
        else:
            arms.remove(DECIDING_ARM)  # never fitted: there is no decision fold to fit it at
        for fold in folds:
            decides = fold.label in decision_labels
            fit_df, ev = ft.split_fold(frame, fold)
            truth = fixed_truth(ev)
            arm_probs: Dict[str, Dict[str, np.ndarray]] = {}
            arm_totals: Dict[str, np.ndarray] = {}

            probs, tot, bad = model_arm(served, ev)
            if bad:
                raise RuntimeError(f"served fold {fold.label}: {bad}")
            arm_probs["served"], arm_totals["served"] = probs, tot
            for arm in WALK_FORWARD_ARMS:
                if arm not in arms:
                    continue  # disqualified at an earlier fold: never fitted again
                model, result, path, refusal = fit(arm, fit_df, fold.fit_cutoff, tmp)
                record(fold.label, arm, fit_df, fold.fit_cutoff, result, path, refusal)
                reason = disqualification(arm, refusal=refusal)
                if reason is None:
                    probs, tot, reason = model_arm(model, ev)
                if reason is not None:
                    disqualify(arm, fold.label, reason)
                    continue
                arm_probs[arm], arm_totals[arm] = probs, tot
            if decides and DECIDING_ARM in arms:
                probs, tot, bad = model_arm(frozen, ev)
                if bad:
                    disqualify(DECIDING_ARM, fold.label, bad)
                else:
                    arm_probs[DECIDING_ARM], arm_totals[DECIDING_ARM] = probs, tot
            w = ft.decay_weights(fit_df, fold.fit_cutoff, decay)
            for arm, weights, by_season in (
                ("base_global", None, False),
                ("base_season", None, True),
                ("base_season_dw", w, True),
            ):
                markets, mats = constant_markets(fit_df, ev, weights, by_season)
                bad = invalid_matrices(mats)
                if bad:
                    raise RuntimeError(f"{arm} fold {fold.label}: {bad}")
                arm_probs[arm], arm_totals[arm] = readout(markets, ev, mats), expected_totals(mats)
            for arm in list(arm_probs):
                reason = disqualification(arm, probs=arm_probs[arm])  # raises for served / the constants
                if reason is not None:
                    disqualify(arm, fold.label, reason)
                    del arm_probs[arm]
            # Every live arm is scored on every row of the fold, except
            # refit_frozen before the decision window (it is fitted at its start).
            expected = set(arms) - (set() if decides else {DECIDING_ARM})
            if set(arm_probs) != expected:
                raise RuntimeError(f"fold {fold.label}: scored arms {sorted(arm_probs)} != expected {sorted(expected)}")
            n = len(ev)
            for arm in arms:
                if arm in arm_probs:
                    for unit, s in score_arm(arm_probs[arm], truth).items():
                        scores[arm][unit].append(s)
                    log_scores[arm].append(grid_log_score(arm_probs[arm], truth))
                    p_primary[arm].append(arm_probs[arm][PRIMARY][:, 0])
                    p_reg[arm].append(arm_probs[arm]["regulation_1x2"])
                    e_total[arm].append(arm_totals[arm])
                else:  # refit_frozen in a context fold: not fitted yet, NaN rows that no pair reads
                    for unit in UNIT_NAMES:
                        scores[arm][unit].append(np.full(n, np.nan))
                    log_scores[arm].append(np.full(n, np.nan))
                    p_primary[arm].append(np.full(n, np.nan))
                    p_reg[arm].append(np.full((n, 3), np.nan))
                    e_total[arm].append(np.full(n, np.nan))
            y_primary.append(truth[PRIMARY][:, 0])
            y_reg.append(truth["regulation_1x2"])
            totals.append((ev["home_score"] + ev["away_score"]).to_numpy(dtype=float))
            fold_of.append(np.full(n, fold.label, dtype=object))
            season_arr.append(ev["season"].to_numpy(dtype=int))
            home_arr.append(ev["home_team"].to_numpy(dtype=object))
            away_arr.append(ev["away_team"].to_numpy(dtype=object))
            logger.info(
                "fold %s%s: fit=%d eval=%d arms=%s",
                fold.label,
                " [decides]" if decides else " [context]",
                len(fit_df),
                n,
                ",".join(a for a in arms if a in arm_probs),
            )

    cat = {a: {u: np.concatenate(scores[a][u]) for u in UNIT_NAMES} for a in arms}
    cat_log = {a: np.concatenate(log_scores[a]) for a in arms}
    folds_arr = np.concatenate(fold_of)
    seasons = np.concatenate(season_arr)
    homes, aways = np.concatenate(home_arr), np.concatenate(away_arr)
    yp = np.concatenate(y_primary)
    yr = np.vstack(y_reg)
    realised_total = np.concatenate(totals)
    fold_labels = list(dict.fromkeys(folds_arr.tolist()))
    decision_mask = np.isin(folds_arr, decision_labels)
    every_row = np.ones(len(folds_arr), dtype=bool)
    challengers = [a for a in arms if a in CHALLENGERS]
    # Arms scored on every row of the full window (refit_frozen is not: context folds).
    full_window_arms = [a for a in arms if a != DECIDING_ARM]

    def pair(a: str, b: str, unit: str = PRIMARY, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        sel = every_row if mask is None else np.asarray(mask, dtype=bool)
        out = ft.paired_delta(cat[a][unit][sel], cat[b][unit][sel], folds_arr[sel])
        out["upper_95_cluster_t"] = ht.t_upper(out["delta_cluster"], out["se_cluster"], out["n_folds"])
        out["lower_95_cluster_t"] = t_lower(out["delta_cluster"], out["se_cluster"], out["n_folds"])
        return out

    def verdict(a: str, b: str, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        p = pair(a, b, mask=mask)
        v = ft.ship_verdict(
            p["delta"], p["se"], b, delta_cluster=p["delta_cluster"], se_cluster=p["se_cluster"], n_folds=p["n_folds"]
        )
        v["upper_95_cluster_t"] = p["upper_95_cluster_t"]
        v["ship_strict"] = bool(v["ship"] and p["upper_95_cluster_t"] is not None and p["upper_95_cluster_t"] < 0)
        return v

    def mean_or_none(x: np.ndarray) -> Optional[float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else None

    def brier_table(mask: np.ndarray, which: Sequence[str]) -> Dict[str, Dict[str, Optional[float]]]:
        return {a: {u: mean_or_none(cat[a][u][mask]) for u in UNIT_NAMES} for a in which}

    def calibration(a: str, mask: np.ndarray) -> Dict[str, Any]:
        pp, pr = np.concatenate(p_primary[a]), np.vstack(p_reg[a])
        rows = mask & np.isfinite(pp)
        ok = rows & np.isfinite(yr).all(axis=1) & np.isfinite(pr).all(axis=1)
        if not rows.any():
            return {}
        return {
            "n": int(rows.sum()),
            "over_5.5_pred": float(pp[rows].mean()),
            "over_5.5_real": float(np.nanmean(yp[rows])),
            "regulation_pred": (
                {c: float(pr[ok, k].mean()) for k, c in enumerate(REGULATION_CLASSES)} if ok.any() else {}
            ),
            "regulation_real": (
                {c: float(yr[ok, k].mean()) for k, c in enumerate(REGULATION_CLASSES)} if ok.any() else {}
            ),
            "expected_total": float(np.concatenate(e_total[a])[rows].mean()),
            "realised_total": float(realised_total[rows].mean()),
        }

    def log_score_block(mask: np.ndarray) -> Dict[str, Any]:
        """correct_score_grid_log (context): per arm the mean -log q(realised
        cell) where every row is finite, else None with the count of rows the
        arm priced at 0; vs served paired only when both are finite on EVERY
        row of the window — never on a subset."""
        out: Dict[str, Any] = {"arms": {}, "vs_served": {}}
        rows = {a: cat_log[a][mask & ~np.isnan(cat_log[a])] for a in arms}
        for a in arms:
            v = rows[a]
            out["arms"][a] = {
                "n": int(v.size),
                "mean": float(v.mean()) if v.size and np.isfinite(v).all() else None,
                "zero_probability_rows": int(np.isinf(v).sum()),
            }
        s = cat_log["served"][mask]
        for a in arms:
            if a == "served":
                continue
            c = cat_log[a][mask]
            scored = ~np.isnan(c)
            if not scored.any() or not (np.isfinite(c[scored]).all() and np.isfinite(s[scored]).all()):
                out["vs_served"][a] = None
                continue
            p = ft.paired_delta(c[scored], s[scored], folds_arr[mask][scored])
            p["lower_95_cluster_t"] = t_lower(p["delta_cluster"], p["se_cluster"], p["n_folds"])
            p["upper_95_cluster_t"] = ht.t_upper(p["delta_cluster"], p["se_cluster"], p["n_folds"])
            out["vs_served"][a] = p
        return out

    def strata(known: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        return {
            str(k): {
                "n": int((mask & (known == k)).sum()),
                **{
                    a: pair(a, "served", mask=mask & (known == k))
                    for a in challengers
                    if (mask & (known == k)).sum() > 1
                },
            }
            for k in (0, 1, 2)
        }

    vs_served = {a: {u: pair(a, "served", u, decision_mask) for u in UNIT_NAMES} for a in arms if a != "served"}
    regressions = {a: blocking_regressions(vs_served[a]) for a in challengers}
    verdicts_vs_served = {a: verdict(a, "served", decision_mask) for a in challengers}
    report: Dict[str, Any] = {
        "n_eval": int(len(folds_arr)),
        "decision_window": {
            "start": DECISION_START,
            "served_fit_end": SERVED_FIT_END,
            "folds": decision_labels,
            "n_eval": int(decision_mask.sum()),
            "frozen_fit_cutoff": None if frozen_cutoff is None else frozen_cutoff.isoformat(),
        },
        "n_gradable": {u: int(np.isfinite(cat["served"][u][decision_mask]).sum()) for u in UNIT_NAMES},
        "n_gradable_full_window": {u: int(np.isfinite(cat["served"][u]).sum()) for u in UNIT_NAMES},
        "folds": fold_labels,
        "pooled_folds": pooled,
        "skipped_folds": skipped,
        "disqualified": disqualified,
        "duplicated_match_ids": dups,
        "frame": frame_audit,
        "served_artifact": served_fp,
        "frozen_artifact": frozen_fp,
        "configs": {k: {"name": c.name, **c.hyperparameters} for k, c in cfgs.items()},
        "brier": brier_table(decision_mask, arms),
        "brier_full_window": brier_table(every_row, full_window_arms),
        "correct_score_grid_log": log_score_block(decision_mask),
        "calibration": {a: calibration(a, decision_mask) for a in arms},
        "calibration_full_window": {a: calibration(a, every_row) for a in full_window_arms},
        "fits": fits,
        "vs_served": vs_served,
        "vs_served_full_window": {
            a: {u: pair(a, "served", u) for u in UNIT_NAMES} for a in full_window_arms if a != "served"
        },
        "vs_matched_constant": {
            a: {"constant": matched_constant(a), **pair(a, matched_constant(a), mask=decision_mask)}
            for a in ["served"] + challengers
        },
        "head_to_head": {
            f"{a} - {b}": pair(a, b)
            for a, b in (("refit_frozen", "refit"), ("refit", "pinned"), ("pinned", "base_season"))
            if a in arms and b in arms
        },
        "per_fold_vs_served": {
            a: [
                {
                    "fold": lab,
                    "decides": lab in decision_labels,
                    "n": int((folds_arr == lab).sum()),
                    "delta": mean_or_none(cat[a][PRIMARY][folds_arr == lab] - cat["served"][PRIMARY][folds_arr == lab]),
                }
                for lab in fold_labels
            ]
            for a in arms
            if a != "served"
        },
        "by_season": {
            season_label(int(s)): {
                "n": int((seasons == s).sum()),
                **{a: pair(a, "served", mask=seasons == s) for a in challengers},
            }
            for s in sorted(set(seasons.tolist()))
            if (seasons == s).sum() > 1
        },
        "served_known_strata": strata(known_count(homes, aways, served_teams), decision_mask),
        "frozen_known_strata": strata(known_count(homes, aways, frozen_teams), decision_mask),
        "unknown_teams": {
            "served": unknown_teams(homes[decision_mask], aways[decision_mask], served_teams),
            DECIDING_ARM: (
                unknown_teams(homes[decision_mask], aways[decision_mask], frozen_teams) if frozen is not None else {}
            ),
        },
        "verdicts_vs_served": verdicts_vs_served,
        "verdicts_vs_matched_constant": {a: verdict(a, matched_constant(a), decision_mask) for a in challengers},
        "regressions": regressions,
        "decision": decide(verdicts_vs_served, regressions, disqualified, unavailable=unavailable),
    }
    return report


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def _unit_tag(unit: str) -> str:
    return " P" if unit == PRIMARY else (" C" if unit in CONTEXT_UNITS else "  ")


def _log_cell(entry: Dict[str, Any]) -> str:
    """One arm's mean grid log score, 'inf (k zero)' when it priced k realised cells at 0."""
    if entry["mean"] is not None:
        return format(entry["mean"], ".5f")
    return f"inf ({entry['zero_probability_rows']} zero)" if entry["zero_probability_rows"] else "n/a"


def format_report(r: Dict[str, Any]) -> str:
    pooled = [p["fold"] + "->" + p["into"] for p in r["pooled_folds"]] or "none"
    skipped = [s["fold"] for s in r["skipped_folds"]] or "none"
    dw = r["decision_window"]
    L = [
        f"NHL Dixon-Coles A/B — {r['n_eval']} eval matches, folds {r['folds'][0]}..{r['folds'][-1]}"
        f" ({len(r['folds'])}); pooled {pooled}; skipped {skipped}; duplicated match_ids {r['duplicated_match_ids']}",
        f"DECISION WINDOW (amended, A3): folds from {dw['start']} (served's fit ends {dw['served_fit_end']}):"
        f" {dw['folds'] or 'none'}, {dw['n_eval']} matches; refit_frozen fitted at {dw['frozen_fit_cutoff'] or 'n/a'}."
        " Everything outside it is context.",
        f"frame: {r['frame']['rows']} rows, {r['frame']['without_regulation_winner']} without regulation_winner;"
        f" gradable per unit (decision window): {r['n_gradable']}",
    ]
    for arm, d in r["disqualified"].items():
        L.append(f"DISQUALIFIED {arm} at fold {d['fold']}: {d['reason']} (cannot be served; not scored)")
    for name in ("served_artifact", "frozen_artifact"):
        a = r[name]
        if a is None:
            L.append(f"{name}: not fitted (no decision fold)")
            continue
        L.append(
            f"{name}: {a['teams']} teams, league baselines {a['has_league_baselines']}, rho {_f(a['rho'], '.4f')},"
            f" sha256 {a['sha256'][:12]}"
        )
    L += ["", "Brier by arm and unit, decision window (lower is better; C = context only, never decides):"]
    arms = list(r["brier"])
    L.append(f"  {'unit':<22}" + "".join(f"{a:>15}" for a in arms))
    for u in UNIT_NAMES:
        tag = " C" if u in CONTEXT_UNITS else "  "
        L.append(f"  {u + tag:<22}" + "".join(f"{_f(r['brier'][a][u], '.5f'):>15}" for a in arms))
    g = r["correct_score_grid_log"]
    L.append(f"  {'grid log score C':<22}" + "".join(f"{_log_cell(g['arms'][a]):>15}" for a in arms))
    L += ["", f"{PRIMARY} vs served, decision window:"]
    L += [ht._pair_line(a, d[PRIMARY]) for a, d in r["vs_served"].items()]
    L += [
        "",
        "every unit vs served, decision window (pooled dBrier [95% lower bound]; R = a regression under the"
        " amended rule: lower bound > 0 on a blocking unit; P = the primary, judged by rule (a); C = context):",
    ]
    others = list(r["vs_served"])
    L.append(f"  {'unit':<22}" + "".join(f"{a:>27}" for a in others))
    for u in UNIT_NAMES:
        cells = []
        for a in others:
            p = r["vs_served"][a][u]
            mark = " R" if u in r["regressions"].get(a, {}) else _unit_tag(u)
            lb = p.get("lower_95_cluster_t")
            cells.append(f"{_f(p.get('delta')) + ' [' + _f(lb) + ']' + mark:>27}")
        L.append(f"  {u:<22}" + "".join(cells))
    L += ["", "correct_score_grid log score vs served, decision window (context):"]
    for a, p in g["vs_served"].items():
        L.append(
            ht._pair_line(a, p)
            if p is not None
            else f"  {a:<28} n/a (a zero-probability cell; never paired on a subset)"
        )
    L += ["", f"{PRIMARY} vs the matched constants, decision window (decayed arms vs base_season_dw):"]
    L += [ht._pair_line(f"{a} vs {d['constant']}", d) for a, d in r["vs_matched_constant"].items()]
    L += [
        "",
        f"context — {PRIMARY} vs served over the FULL window (folds before {dw['start']} are in-sample for served):",
    ]
    L += [ht._pair_line(a, d[PRIMARY]) for a, d in r["vs_served_full_window"].items()]
    L += ["", "head to head (full window; refit_frozen is scored on decision folds only):"]
    L += [ht._pair_line(k, v) for k, v in r["head_to_head"].items()]
    L += ["", f"{PRIMARY} by season (context):"]
    for s, block in r["by_season"].items():
        L.append(f"  [{s}] n={block['n']}")
        L += [ht._pair_line(f"{a} vs served", v) for a, v in block.items() if isinstance(v, dict)]
    if r["per_fold_vs_served"]:
        L += ["", f"{PRIMARY} per-fold delta vs served (* = decision fold):"]
        first = next(iter(r["per_fold_vs_served"].values()))
        L.append(
            f"  {'arm':<15}" + "".join(f"{row['fold'][2:] + ('*' if row['decides'] else ' '):>9}" for row in first)
        )
        for a, rows in r["per_fold_vs_served"].items():
            L.append(f"  {a:<15}" + "".join(f"{_f(row['delta'], '+.4f'):>9}" for row in rows))
        L.append(f"  {'n':<15}" + "".join(f"{row['n']:>9}" for row in first))
    L += [
        "",
        "calibration, decision window (over 5.5 mean predicted vs realised; regulation H/D/A; E[total] vs realised):",
    ]
    for a, c in r["calibration"].items():
        if not c:
            L.append(f"  {a:<15} n/a")
            continue
        rp, rr = c["regulation_pred"], c["regulation_real"]
        L.append(
            f"  {a:<15} over5.5 {c['over_5.5_pred']:.3f}/{c['over_5.5_real']:.3f}"
            + (
                f"  reg pred {rp['home']:.3f}/{rp['draw']:.3f}/{rp['away']:.3f}"
                f" real {rr['home']:.3f}/{rr['draw']:.3f}/{rr['away']:.3f}"
                if rp
                else ""
            )
            + f"  E[total] {c['expected_total']:.3f}/{c['realised_total']:.3f}"
        )
    for title, key in (("served", "served_known_strata"), (DECIDING_ARM, "frozen_known_strata")):
        L += ["", f"{PRIMARY} vs served by teams {title} knows (decision window):"]
        for k, blk in r[key].items():
            L.append(
                f"  known={k} n={blk['n']}: "
                + ", ".join(f"{a} {_f(v.get('delta'))}" for a, v in blk.items() if isinstance(v, dict))
            )
    for arm, names in r["unknown_teams"].items():
        L.append(f"  teams {arm} does not know (matches, decision window): {names or 'none'}")
    refused: Dict[str, int] = {}
    for f in r["fits"]:
        if f.get("refusal"):
            refused[f["arm"]] = refused.get(f["arm"], 0) + 1
    L += ["", f"fits the trainer would refuse (per arm): {refused or 'none'}"]
    d = r["decision"]
    L += [
        "",
        f"DECISION (pre-registered, amended 2026-10-31: {DECIDING_ARM} must pass ship_strict vs served on {PRIMARY}"
        " over the decision window AND be significantly worse on no blocking unit):"
        f" {'SHIP' if d['ship'] else 'DO-NOT-SHIP'} — {d['reason']}",
    ]
    for u, why in d["regressions"].items():
        L.append(f"  regression {u}: {why}")
    vs = r["verdicts_vs_served"]
    for a in vs:
        v = vs[a]
        L.append(
            f"  {a:<15} {'PASS' if v['ship_strict'] else 'FAIL'}  {v['line'].split(': ', 1)[-1]}"
            f"  (t-bound {_f(v.get('upper_95_cluster_t'))})"
        )
    if not vs:
        L.append("  no challenger survived — nothing can ship")
    L += ["", "context only — never decides (vs the matched constants, decision window):"]
    for a, v in r["verdicts_vs_matched_constant"].items():
        L.append(f"  {a:<15} {'PASS' if v['ship_strict'] else 'FAIL'}  {v['line'].split(': ', 1)[-1]}")
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=DEFAULT_START, help="first eval day (monthly folds from its month)")
    p.add_argument("--end", default=None, help="last eval day (default: newest NHL match)")
    p.add_argument("--served-artifact", default=SERVED_ARTIFACT)
    p.add_argument("--out-json", default=None)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    report = run(args)
    if args.out_json:  # before printing: a formatting bug must not lose the run
        Path(args.out_json).write_text(json.dumps(ht.clean_json(report), indent=2, allow_nan=False))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
