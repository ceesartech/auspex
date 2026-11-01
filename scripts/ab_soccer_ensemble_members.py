"""A/B gate for the soccer 1x2 ensemble's FULL-TIME Dixon-Coles and Poisson
MEMBERS, which the trainer fits on its oldest-70% split and never refits.

WHAT IS BEING TESTED
--------------------
services/ml-models/src/training/train_all_models.py fits every soccer base
model on the temporal TRAIN split only (the oldest 70% of the frame); val
picks the ensemble weights and test gates the promotion. For the learned
members that is the honest design. For the two scoreline members it freezes
a team-strength table at the split and never updates it: the artifacts
served since 2026-10-26 were fitted on the oldest 119,482 of 170,689
matches, ending 2021-10-21, and know 951 teams, exactly the teams in that
slice (checked, identical sets). A team whose history starts later gets
attack = defense = 1.0: its LEAGUE baseline in the Dixon-Coles member
(per-league baselines) and the one GLOBAL baseline in the Poisson member
(POISSON_CONFIG fits none). That covers promoted sides, new leagues and every
ESPN-spelling twin of the 2026 identity split. Both members look teams up by
teams.name. Their blend weights were learned on val with these same fits:
Dixon-Coles 0.046 and Poisson 0.046 (XGBoost 0.448, NN 0.354, LightGBM
0.105), so together they make up 9.2% of every served 1x2 probability. Both
sit at the trainer's floor: EnsemblePredictor's log-loss fit put each below
min_weight 0.05, which lifted them to 0.05 before renormalising (0.05/1.0869
= 0.046, identically for both).

SAME ARTIFACT AS THE DERIVED MARKETS. scripts/precompute_predictions.py
derives the ~15 soccer markets from
ensemble.models["dixon_coles_soccer_match_result"], which is this member.
The 2026-09 fitter fix (234bbb1, scripts/ab_soccer_dixon_coles_refit.py)
was gated on WALK-FORWARD refits, but the artifact the trainer writes is
fitted on the train split like every other member, so the derivation also
serves the 2021-10 table. This harness does not re-test that fitter: it uses
DIXON_COLES_CONFIG and POISSON_CONFIG exactly as they are. What it measures
is how stale the deployed fit is. The derived over-2.5 / BTTS effect is
printed as information only, because a trainer change that refits the member
would move those markets too.

ARMS (fixed in-session on 2026-11-01 before the first full run, but committed
together with the results, so git cannot show the order; every arm came out
null and none was picked on its result). Every arm is scored on the identical
eval rows, through the serve path.
  served         the production ensemble as it is: all five on-disk
                 artifacts and the production weights.
  refit          the SAME fitters (the trainer's classes, configs and frame
                 columns) refit on every finished match before each eval
                 month. Production weights, and the three learned members
                 untouched. If the trainer refit the members on all data,
                 the weekly retrain would serve them at most a week stale, so
                 monthly is the conservative proxy.
  drop           both members removed and the remaining production weights
                 renormalised (EnsemblePredictor's own arithmetic). This arm
                 is not deployable as it stands, because the derived markets
                 read the Dixon-Coles member out of the ensemble.
  refit_pretest  (secondary) both members fitted ONCE on the incumbent's
                 train+val rows (its seen ids) and then frozen. This is the
                 literal "refit on all pre-test data" trainer change.
  NOT RUN: re-learned weights. The trainer learns its weights once, on val,
  from members fitted on train. A trainer that refits the members AFTER the
  split evaluation leaves those weights unchanged, so for refit and
  refit_pretest the faithful weights ARE the production weights.
  Re-learning them on refit members would need out-of-sample refit
  predictions across val, which is a walk-forward stacking trainer that does
  not exist. Re-learning drop's weights would need the trainer's own val
  frame (its SQL odds columns), which the serve-path frame used here does not
  reproduce. Neither is a faithful reproduction, so neither runs. The
  in-sample-optimal BOUND below covers both from above.

READOUTS
  PRIMARY    the served soccer 1x2 Brier (3-class sum), comparing refit with
             served and drop with served. Gate (§4.3): ΔBrier <= -0.005, with
             the per-match AND fold-clustered 95% upper bounds below zero
             (ab_soccer_dixon_coles_refit.ship_verdict). The t(n_folds-1)
             clustered bound must also be below zero ("strict").
  SECONDARY  each member standalone (Brier, log loss, and the ECE from
             ProbabilityCalibrator.calibration_error) for served, refit and
             refit_pretest. Also: strata by how many teams the served members
             know; the two eras (football-data rows to 2026-05, ESPN rows
             from 2026-06); fixtures touching a team the merge map
             (scripts/data/soccer_team_merge_map.csv) will fold; per-fold
             deltas; refit_pretest vs served; refit vs drop.
  BOUND      (diagnostic, not part of the verdict; the in-sample-optimal
             readout was added after review, post hoc)
             in_sample_optimal_weights: the five simplex weights refitted ON
             THE EVAL ROWS THEMSELVES to minimise the Brier, solved exactly,
             for the served, refit and refit_pretest members and for the
             learned members alone. Leaky and optimistic by construction, so
             it is an upper bound on what ANY linear re-weighting of these
             member predictions could gain: the re-learned-weights arm that is
             not run, and the trainer's own log-loss fit with its 0.05 floor,
             are both inside it. It does not bound a better member.
             members_as_market: both members replaced by the de-vigged
             closing 1x2 consensus (>= --min-books books, pre-kickoff rows
             only, prematch_odds()). It bounds a refit only under the
             ASSUMPTION that a scoreline member cannot beat that market, and
             only on the rows that carry a quote (about 42%), which are the
             rows where refit gains least; refit - served is printed on and
             off those rows. The oracle (members = the realised outcome)
             shows that without the assumption there is no hard bound. Also
             printed: the |Δp| the refit actually produced.
  INFO       derived over-2.5 and BTTS through derive_from_lambdas, with each
             arm's Dixon-Coles reconciled to that arm's 1x2 (drop uses the
             served Dixon-Coles). No O/U-2.5 market anchor is applied here.
             Production applies one wherever >= 2 fresh books quote the line,
             and that pins over-2.5 to the market there. These numbers are
             not part of the verdict.

PROTOCOL
  * Eval rows are the matches the incumbent never saw. Its
    seen_match_ids.json.gz holds its train+val ids. Every row dated after the
    newest seen row is out of sample for the learned members and for the
    ensemble weights (the stale members end even earlier, at the train
    split). Monthly folds run from the first such month
    (ab_soccer_dixon_coles_refit.build_folds / split_fold). A refit fits
    strictly on match_date < the month start.
  * Serve-path fidelity: the ensemble is reconstituted exactly as the API
    does it (prediction_service._build_ensemble_for_task). Each eval row is
    built the way precompute_predictions builds it: the latest baseline
    features_cache JSON, non-numeric values replaced by corpus medians,
    plain keys mirrored to feature__ keys, then home_team / away_team /
    league_id. Rows that lack a feature a learned member needs are counted
    and excluded, because the serve path cannot predict them either. Two
    checks fail the run loudly: the per-member blend computed here must
    equal ensemble.predict_proba on the batch, and a sample predicted one
    row at a time (as production does) must equal the batch. A NaN on either
    side fails them too. Refit members are round-tripped through save() and
    the serving loader.
  * Every arm except members_as_market must be finite on every eval row: a
    non-finite probability fails the run rather than scoring as 0.
  * Paired statistics clustered by fold (ab_soccer_dixon_coles_refit.
    paired_delta). A fold with fewer than MIN_FOLD_EVAL rows is skipped
    rather than counted as a full cluster.

CORPUS CAVEAT
  As of 2026-11-01 the soccer corpus is mid-repair. A team-identity split
  (171 ESPN-spelling twins of established teams, pending merge in
  scripts/ops/soccer_repair_runbook.sh) leaves the twins unknown even to a
  refit member, and football-data kickoffs are stored an hour late (a shift
  that is also pending). --simulate-merge-map re-runs every arm with the merge
  map applied in memory: the twins' rows take the kept team's id and name,
  and twin fixtures (same teams within 36 h, same score) collapse to one row,
  keeping the row that carries features. It approximates the repaired
  corpus for the members, whose frame holds finished, scored rows only: 2,115
  twin sides on 1,585 rows and 171 twin pairs, where the runbook's step 3
  counts (2,469 sides, 308 folds) span every match status. It does NOT
  rebuild the learned members' features (runbook step 10), and it ignores
  the one-hour kickoff shift, which moves no row across a fold boundary that
  matters to a month-level refit. Nor does it simulate step 4's row dedupe
  (repair_football_data_kickoffs folds about 98 soccer rows: dedupe 68,
  alias 4, fold 26); those rows stay in fit and eval, but the comparison is
  paired, so a duplicate counts in every arm.

RESULT (2026-11-01; read-only export of prod at 02:40Z, pre-runbook corpus;
incumbent = the 2026-10-26 artifacts): DO-NOT-SHIP, every arm.
  n = 25,729 matches (2024-04-06 .. 2026-10-31, 31 monthly folds).
  Served 1x2 Brier 0.60215.
  refit - served          -0.00032 ± 0.00007, clustered -0.00039 ± 0.00010,
                          t-bound -0.00019. Real, but about 13x short of
                          the gate.
  drop - served           +0.00017 ± 0.00011, clustered +0.00006 ± 0.00018.
  refit_pretest - served  -0.00008 ± 0.00005, clustered -0.00013 ± 0.00006.
  The bound (diagnostic; the verdict rests on the direct measurement above):
  weights refitted on the eval rows themselves, leaky and so optimistic, gain
  at most -0.00058 with the refit members (whose combined weight rises only
  to 0.161), -0.00021 with the served members and -0.00026 with
  refit_pretest, so every linear re-weighting of these members, re-learned
  weights included, falls at least 8x short of the gate. members_as_market
  moves the blend -0.00054 (clustered -0.00101 ± 0.00059), but only if no
  member can beat the market, and only on the 10,774 rows (42%) with a quote,
  which are where refit gains least: refit - served is -0.00003 there and
  -0.00053 off them (clustered -0.00067 ± 0.00021, n=14,955). The market
  alone is -0.00146 ahead of served on its rows.
  Standalone, the refit Dixon-Coles member beats the served one by 0.0145
  (0.61463 vs 0.62911) and the refit Poisson member by 0.0066, but only about a
  third of the weighted first-order gain survives the blend.
  The simulated merge (--simulate-merge-map) gives refit - served -0.00024,
  clustered -0.00032 ± 0.00009 (in-sample-optimal weights -0.00056): the
  repair moves it AWAY from the gate, because the served members already know
  the merged teams' football-data names.
  INFO, unanchored derived markets, refit - served: over-2.5 -0.0039 ± 0.0004
  (t-bound -0.0029), BTTS -0.0026 ± 0.0004.
  Fidelity checks: all 164 live predictions stored by this incumbent for
  matches 2026-10-26..31 equal the served arm to 7.5e-8, and prod's library
  versions reproduce the local member outputs to 1.2e-7 (n=400).
  Every refit of the Poisson member stops at 1000 iterations without
  converging (POISSON_CONFIG, regularization 0.001). The served artifact is
  the same.

ARM ADDED 2026-11-01 EVENING: poisson_converged (PRE-REGISTERED: this arm,
its readout and its decision rule were committed before its first run).
  The question. The served Poisson member never converges. Its train split
  has 27 connected components (one per country: football-data domestic
  leagues 2007-2021, no cups), the fitter normalises attack over all teams,
  which fixes only one of their 27 scales, and the simultaneous update flips
  each remaining scale every iteration. The artifact is iteration 1000 of an
  exact 2-cycle. Within a league its expected goals sit up to 18% off the
  converged fit on its own training rows. Normalising attack within each
  component (PoissonMatchPredictor gauge="per_component", an ablation knob)
  converges in 32 iterations to the true fixed point. It moves the member's
  1x2 by up to 0.08 on recent fixtures (0.0038 in the served blend), so it is
  a model change and must pass the gate before it ships.
  The arm. poisson_converged = the served ensemble with ONE change: the
  Poisson member refitted with POISSON_CONFIG plus gauge="per_component" on
  the served member's own train split. Every other member and every weight is
  the production one. The split is rebuilt as the trainer built it: the
  incumbent's seen ids (its train+val) sorted by match_date, the first
  int(rows x TRAIN_RATIO) rows, where rows is the incumbent's fingerprint.
  Rows that share the boundary kickoff are resolved by refitting with the
  production config and keeping the assignment that reproduces the served
  artifact to SPLIT_ATOL. If no assignment does, the run fails, so the arm
  can never be fitted on rows the served member did not see.
  Readout. Primary: the served soccer 1x2 Brier, poisson_converged - served,
  on the same eval rows as every other arm.
  Decision rule (fixed before the run):
    SHIP only if all of:
      (1) ship_strict: delta <= -0.005, the per-match and fold-clustered 95%
          upper bounds below zero, and the t(n_folds-1) clustered bound below
          zero (verdict());
      (2) no blocking unit regresses. Blocking units are the strata the report
          already prints: served_knows_both / one / neither, the football-data
          and ESPN eras, identity-split and other fixtures. A2 rule: a unit
          regresses only when its 95% lower bound is above zero, using
          delta_cluster - 1.96 x se_cluster when the unit spans at least two
          folds and delta - 1.96 x se otherwise (unit_lower_95).
    Otherwise DO-NOT-SHIP: the trainer keeps the global gauge, and the 2-cycle
    stays a documented, accepted condition (train_all_models.
    ACCEPTED_NONCONVERGENCE logs it as a WARNING and the artifact records it).
  Expectation, stated before the run: DO-NOT-SHIP. At 4.65% blend weight the
  arm moves the served 1x2 by about 0.0007 on average, and the in-sample-
  optimal bound above caps any change to these members near -0.0006.
  RESULT (2026-11-01 evening; read-only export of prod at 21:22Z, runbook not
  started; incumbent = the 2026-11-01 04:06Z artifacts, whose Poisson member
  the rebuilt split reproduced to 0.0; how many boundary assignments that
  took depends on the database's row order: 10 here, 14 on a second replica):
  DO-NOT-SHIP, because the gate is not met.
    n = 25,604 matches (2024-04-07 .. 2026-10-31, 31 monthly folds), served
    1x2 Brier 0.60173.
    poisson_converged - served  +0.0000077 ± 0.0000089, clustered
                                +0.0000105 ± 0.0000096, t-bound +0.0000301.
                                Null, and nowhere near -0.005.
    Blocking units: under the pre-registered bound, identity_split_fixtures
    (n=1,582) regresses only marginally and not robustly: its clustered delta
    +0.000074 has a z lower bound of +0.000022, but that SE comes from 4 folds
    (and is smaller than the per-match SE), and both the per-match bound
    (delta +0.000090, lower -0.0000016) and the t(3) clustered bound
    (-0.000011) include zero. The decision does not rest on it. Later arms
    should pre-register the t(n_folds-1) quantile for clustered unit bounds;
    unit_regression() now reports all three.
    The arm moves the served 1x2 by mean L1 0.0013 (max |dp| 0.0040).
    Standalone the converged member is slightly WORSE than the 2-cycle one
    (Brier 0.63380 vs 0.63329): converging fixes the fit, not the forecast.
    The other arms, re-run on this incumbent, repeat their earlier readings:
    refit -0.00032 (clustered -0.00040 ± 0.00010), drop +0.00013,
    refit_pretest -0.00009.

Read-only: SELECTs, plus temp files for the round-trip. Usage (api container):
    python /app/scripts/ab_soccer_ensemble_members.py --end 2026-10-31 \\
        --out-json /tmp/ab_members.json --out-dir /tmp/ab_members
    python /app/scripts/ab_soccer_ensemble_members.py --end 2026-10-31 --simulate-merge-map \\
        /app/scripts/data/soccer_team_merge_map.csv --out-json /tmp/ab_members_merged.json
A local replica works too: point --database-url at it and --model-path at a
copy of the production registry. On macOS, set OMP_NUM_THREADS=1, because
torch deadlocks beside the xgboost/lightgbm OpenMP runtimes otherwise.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_halftime_dixon_coles as ht  # noqa: E402
import ab_soccer_dixon_coles_refit as ft  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_ensemble_members")

PRODUCTION_MODEL_PATH = "/app/models/production"
SOCCER_TASK_KEY = "soccer:match_result"
ENSEMBLE_NAME = "ensemble_soccer_match_result"
SEEN_IDS_FILENAME = "seen_match_ids.json.gz"  # train_all_models.SEEN_IDS_FILENAME
POISSON = "poisson_soccer_match_result"
DIXON_COLES = "dixon_coles_soccer_match_result"
SCORELINE_MEMBERS = (POISSON, DIXON_COLES)
LEARNED_MEMBERS = (
    "xgboost_soccer_match_result",
    "lightgbm_soccer_match_result",
    "neural_network_soccer_match_result",
)
ALL_MEMBERS = LEARNED_MEMBERS + SCORELINE_MEMBERS

# Arms that carry their own scoreline members; "drop" carries none.
CONVERGED = "poisson_converged"
MEMBER_ARMS = ("served", "refit", "refit_pretest", CONVERGED)
ARMS = ("served", "refit", "drop", "refit_pretest", CONVERGED)
PRIMARY_CHALLENGERS = ("refit", "drop")
SECONDARY_CHALLENGERS = ("refit_pretest",)
# The pre-registered 2026-11-01-evening question (see the docstring): does a
# CONVERGED Poisson member pass the gate? Decided by convergence_decision().
CONVERGENCE_CHALLENGERS = (CONVERGED,)
BOUND_ARM = "members_as_market"
# The trainer's split (train_all_models --train-ratio default) and the knob
# the converged arm sets on POISSON_CONFIG.
TRAIN_RATIO = 0.7
CONVERGED_HYPERPARAMETERS = {"gauge": "per_component"}
# A boundary-tie assignment counts as the served member's split only if a
# production-config refit on it reproduces every team strength to this.
SPLIT_ATOL = 1e-9
MAX_TIE_ASSIGNMENTS = 500
UNIT_Z = 1.96  # the A2 blocking-unit rule's two-sided 95% bound

CLASSES = ("home", "draw", "away")  # TaskSpec labels == match_outcome 0/1/2
# The trainer's member frame: train_all_models passes the whole train split,
# and the Poisson family reads only these columns from it (league_id is the
# one LEAGUE_COLUMN_CANDIDATES key the training query selects).
MEMBER_FIT_COLUMNS = ("home_team", "away_team", "home_score", "away_score", "match_date", "league_id")
MIN_FOLD_EVAL = 50
ESPN_ERA_START = pd.Timestamp("2026-06-01", tz="UTC")
TWIN_WINDOW = pd.Timedelta(hours=36)  # merge_soccer_teams.TWIN_WINDOW
SINGLE_ROW_SAMPLE = 64
# EnsemblePredictor accumulates the blend in the first member's dtype
# (np.zeros_like of XGBoost's float32 output), and the arms here are blended
# in float64. They agree to ~1e-7, about four orders of magnitude below any
# Brier difference this harness reports.
FIDELITY_ATOL = 1e-6

# Same population and same name columns as utils/training_data.
# DEFAULT_TRAINING_QUERY (a test pins the predicates), without the odds
# subqueries and feature JSON the scoreline members never read.
MATCHES_SQL = """
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.league_id::text AS league_id,
        l.name AS league_name,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        m.home_score,
        m.away_score
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE l.sport = 'soccer'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.match_date ASC
"""

# The row precompute_predictions predicts from: the newest baseline
# features_cache row per match.
FEATURES_SQL = """
    SELECT m.id::text AS match_id, fc.features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    JOIN LATERAL (
        SELECT features
        FROM features_cache
        WHERE match_id = m.id AND feature_set = 'baseline'
        ORDER BY computed_at DESC
        LIMIT 1
    ) fc ON true
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
"""

# precompute_predictions.compute_feature_medians, in SQL: the median of every
# numeric value per key across ALL features_cache rows. percentile_cont(0.5)
# averages the two middle values exactly as statistics.median does. Only
# needed when an eval row carries a non-numeric value, so it runs lazily.
MEDIANS_SQL = """
    SELECT e.key, percentile_cont(0.5) WITHIN GROUP (ORDER BY (e.value::text)::float8) AS median
    FROM features_cache fc, jsonb_each(fc.features) e
    WHERE fc.features IS NOT NULL AND jsonb_typeof(e.value) = 'number'
    GROUP BY e.key
"""

# Closing 1x2 per (match, bookmaker, selection): the newest non-live quote,
# read through prematch_odds() at run time. prematch_odds() keeps the rows
# first seen before kickoff, or every row of a match that has none: that is
# the football-data backfill, whose closing prices were inserted in 2026, so a
# plain `timestamp <= match_date` filter would discard every historical
# quote. Its 'Average' and 'Max' rows are aggregates over books, not books,
# so they are excluded: 'Max' is the best price per outcome and is no
# consensus at all.
ODDS_1X2_SQL = """
    SELECT DISTINCT ON (o.match_id, o.bookmaker, o.selection)
        o.match_id::text      AS match_id,
        o.bookmaker           AS bookmaker,
        lower(o.selection)    AS selection,
        o.odds_decimal::float AS odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
      AND o.market_type = '1x2'
      AND o.is_live = false
      AND o.odds_decimal IS NOT NULL
      AND o.odds_decimal > 1.0
      AND o.bookmaker NOT IN ('Average', 'Max')
    ORDER BY o.match_id, o.bookmaker, o.selection, o.timestamp DESC
"""


# ── Pure ─────────────────────────────────────────────────────────────


def normalised_weights(weights: Dict[str, float], members: Sequence[str]) -> Dict[str, float]:
    """The weights EnsemblePredictor.predict_proba applies to `members`:
    each raw weight over the sum of the members' raw weights."""
    missing = [m for m in members if m not in weights]
    if missing:
        raise ValueError(f"ensemble carries no weight for {missing}")
    total = float(sum(float(weights[m]) for m in members))
    if not total > 0:
        raise ValueError(f"member weights sum to {total}")
    return {m: float(weights[m]) / total for m in members}


def blend(member_probs: Dict[str, np.ndarray], weights: Dict[str, float], members: Sequence[str]) -> np.ndarray:
    """sum_m w_m p_m / sum_m w_m over `members`, the weighted average
    EnsemblePredictor serves when every listed member predicts. Dropping a
    member is calling this without it."""
    w = normalised_weights(weights, members)
    out = np.zeros_like(np.asarray(member_probs[members[0]], dtype=float))
    for m in members:
        out = out + w[m] * np.asarray(member_probs[m], dtype=float)
    return out


def assemble_arms(
    learned: Dict[str, np.ndarray],
    scoreline: Dict[str, Dict[str, np.ndarray]],
    market: np.ndarray,
    y_idx: np.ndarray,
    weights: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Every arm's served 1x2 for one fold's eval rows, built from the
    per-member probabilities. `learned` holds the three learned members
    (shared by every arm), `scoreline[arm]` the two scoreline members of each
    MEMBER_ARMS arm, and `market` the de-vigged consensus (NaN rows where there
    is none).
      served, refit, refit_pretest  all five members at the production weights
      drop               the learned members alone, their weights renormalised
      members_as_market  both scoreline members replaced by the market (NaN
                         off the market rows)
      oracle             the served members' share replaced by the outcome"""
    missing = [a for a in MEMBER_ARMS if a not in scoreline]
    if missing:
        raise ValueError(f"no scoreline members for arm(s) {missing}")
    out = {arm: blend({**learned, **scoreline[arm]}, weights, ALL_MEMBERS) for arm in MEMBER_ARMS}
    out["drop"] = blend(learned, weights, LEARNED_MEMBERS)
    mk = np.asarray(market, dtype=float)
    out[BOUND_ARM] = blend({**learned, POISSON: mk, DIXON_COLES: mk}, weights, ALL_MEMBERS)
    w = normalised_weights(weights, ALL_MEMBERS)
    member_sum = np.zeros_like(out["served"])
    for m in SCORELINE_MEMBERS:
        member_sum = member_sum + w[m] * np.asarray(scoreline["served"][m], dtype=float)
    out["oracle"] = oracle_blend(out["served"], member_sum, y_idx, float(sum(w[m] for m in SCORELINE_MEMBERS)))
    return out


def require_finite(what: str, proba: np.ndarray) -> None:
    """Refuse loudly if any probability is NaN or infinite. A tolerance test
    alone lets NaN through: np.max of an array holding NaN is NaN, and
    `NaN > atol` is False."""
    arr = np.asarray(proba, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        rows = int(bad.reshape(len(arr), -1).any(axis=1).sum()) if arr.ndim else 1
        raise RuntimeError(f"{what}: {rows} row(s) carry non-finite probabilities")


def max_gap(what: str, a: np.ndarray, b: np.ndarray, atol: float = FIDELITY_ATOL) -> float:
    """max |a - b| between two probability arrays, refusing loudly unless both
    are finite, the same shape, and within `atol`."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise RuntimeError(f"{what}: shapes {a.shape} and {b.shape} differ")
    require_finite(what, a)
    require_finite(what, b)
    gap = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not math.isfinite(gap) or gap > atol:
        raise RuntimeError(f"{what} differ by {gap:.2e} (tolerance {atol:.0e})")
    return gap


def serve_path_fidelity(
    ensemble: Any, serve_x: pd.DataFrame, p_blend: np.ndarray, sample_size: int = SINGLE_ROW_SAMPLE
) -> Tuple[float, float, int]:
    """The two serve-path checks that fail the run loudly: the per-member
    blend this harness computes must equal ensemble.predict_proba on the
    batch, and a sample predicted one row at a time (as production does) must
    equal the batch. Returns (batch gap, single-row gap, sample size)."""
    p_ensemble = np.asarray(ensemble.predict_proba(serve_x), dtype=float)
    fid_batch = max_gap("per-member blend vs ensemble.predict_proba", p_blend, p_ensemble)
    rng = np.random.default_rng(20261101)
    sample = rng.choice(len(serve_x), size=min(sample_size, len(serve_x)), replace=False)
    single = np.vstack([np.asarray(ensemble.predict_proba(serve_x.iloc[[i]]), dtype=float) for i in sample])
    fid_single = max_gap("one-row-at-a-time predictions vs the batch", single, p_ensemble[sample])
    return fid_batch, fid_single, int(len(sample))


def check_fold_disjoint(fit_df: pd.DataFrame, ev: pd.DataFrame, fold: Any) -> None:
    """Refuse a fold whose fit reaches its own eval rows: every fit row must
    be strictly before the fold's cutoff and every eval row on or after it."""
    if not (fit_df["match_date"] < fold.fit_cutoff).all() or (ev["match_date"] < fold.fit_cutoff).any():
        raise RuntimeError(f"fold {fold.label}: fit and eval overlap")


def optimal_simplex_weights(sources: Sequence[np.ndarray], y_idx: np.ndarray) -> Tuple[np.ndarray, float]:
    """(weights, mean Brier): the convex weights (w >= 0, sum 1) that minimise
    the mean 3-class Brier of sum_k w_k P_k against y_idx, solved exactly.

    The Brier is quadratic in w: mean ||sum_k w_k P_k - Y||^2 = w'Gw - 2b'w + 1
    with G_jk = mean <P_j, P_k> and b_j = mean <P_j, Y>. Its minimum over the
    simplex lies on some support S, where it is the minimum of the same
    quadratic under sum(w_S) = 1 alone: one bordered linear solve. Every
    support is tried (31 for five sources), a solution with a negative weight
    is discarded, and the best feasible one wins, so there is no optimiser
    tolerance to argue about."""
    P = np.stack([np.asarray(p, dtype=float) for p in sources])
    k, n = P.shape[0], P.shape[1]
    require_finite("optimal_simplex_weights sources", P.reshape(k * n, -1))
    Y = np.zeros(P.shape[1:])
    Y[np.arange(n), np.asarray(y_idx, dtype=int)] = 1.0
    G = np.einsum("inc,jnc->ij", P, P) / n
    b = np.einsum("inc,nc->i", P, Y) / n
    best_w: Optional[np.ndarray] = None
    best_obj = math.inf
    for size in range(1, k + 1):
        for support in itertools.combinations(range(k), size):
            S = list(support)
            A = np.zeros((size + 1, size + 1))
            A[:size, :size] = 2.0 * G[np.ix_(S, S)]
            A[:size, size] = 1.0
            A[size, :size] = 1.0
            rhs = np.concatenate([2.0 * b[S], [1.0]])
            w_s = np.linalg.lstsq(A, rhs, rcond=None)[0][:size]
            if (w_s < -1e-12).any():
                continue
            w = np.zeros(k)
            w[S] = np.clip(w_s, 0.0, None)
            w = w / w.sum()
            obj = float(w @ G @ w - 2.0 * b @ w)
            if obj < best_obj:
                best_w, best_obj = w, obj
    if best_w is None:
        raise RuntimeError("optimal_simplex_weights found no feasible support")
    brier = float(ft.brier_multiclass(np.tensordot(best_w, P, axes=1), y_idx).mean())
    return best_w, brier


def serve_row(features: Dict[str, Any], identity: Dict[str, Any], medians: Dict[str, float]) -> Tuple[dict, int]:
    """One eval row, exactly as precompute_predictions builds it. Returns
    (row, number of values replaced by a corpus median).

    Order matters and mirrors the serve path: (1) non-numeric values are
    replaced by the corpus median for the key; (2) plain keys are mirrored
    to feature__ keys without overwriting an existing one; (3) the identity
    columns (home_team, away_team, league_id) are set LAST, so the median
    fill cannot replace them."""
    filled: Dict[str, Any] = {}
    n_filled = 0
    for k, v in (features or {}).items():
        if isinstance(v, (int, float)) and v is not None:
            filled[k] = v
        else:
            filled[k] = medians.get(k)
            n_filled += 1
    for k in list(filled.keys()):
        prefixed = k if k.startswith("feature__") else f"feature__{k}"
        if prefixed not in filled:
            filled[prefixed] = filled[k]
    for k, v in identity.items():
        filled[k] = v
    return filled, n_filled


def needs_medians(features: Dict[str, Any]) -> bool:
    return any(not (isinstance(v, (int, float)) and v is not None) for v in (features or {}).values())


def rows_missing_features(frame: pd.DataFrame, required: Sequence[str]) -> np.ndarray:
    """True for rows the batch cannot reproduce faithfully: a needed column
    absent from the row, or present but not finite. Production predicts
    each match alone. There an absent key raises KeyError in
    `X[feature_names]` and the match is skipped, and a NaN survives the
    single-row median fill (so the neural member returns NaN and drops out of
    the blend). In a batch both would be quietly median-filled from OTHER
    matches, which is not what serves, so these rows are excluded and
    counted instead."""
    missing = np.zeros(len(frame), dtype=bool)
    for col in required:
        if col not in frame.columns:
            return np.ones(len(frame), dtype=bool)
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        missing |= ~np.isfinite(vals)
    return missing


def out_of_sample_mask(frame: pd.DataFrame, seen_ids: set) -> Tuple[np.ndarray, pd.Timestamp, int]:
    """(mask, cutoff, n_late) for rows the incumbent never saw: dated
    strictly after the newest row it trained or validated on, and not in
    its seen ids. n_late counts unseen rows dated INSIDE the seen span (rows
    inserted after the incumbent's frame was read). They are excluded,
    because the learned members' weights were tuned on that period."""
    seen = frame["match_id"].isin(seen_ids).to_numpy()
    if not seen.any():
        raise RuntimeError("none of the incumbent's seen ids are in the frame — wrong artifact or wrong database")
    cutoff = pd.Timestamp(frame.loc[seen, "match_date"].max())
    after = (frame["match_date"] > cutoff).to_numpy()
    n_late = int((~seen & ~after).sum())
    return after & ~seen, cutoff, n_late


def simulate_team_merge(frame: pd.DataFrame, merge_map: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The merge map applied in memory: every new_team_id row takes the old
    team's id and name, then twin fixtures (same home and away team within
    TWIN_WINDOW, at least one row re-pointed) collapse to one row. The row
    kept is the one with features (has_features column, when present),
    then the earlier kickoff. Twins with different scores are refused, as
    merge_soccer_teams refuses the pair."""
    need = {"new_team_id", "old_team_id", "new_name", "old_name"}
    if not need.issubset(merge_map.columns):
        raise ValueError(f"merge map lacks {sorted(need - set(merge_map.columns))}")
    id_map = dict(zip(merge_map["new_team_id"].astype(str), merge_map["old_team_id"].astype(str)))
    name_map = dict(zip(merge_map["old_team_id"].astype(str), merge_map["old_name"].astype(str)))
    out = frame.copy()
    touched = np.zeros(len(out), dtype=bool)
    for side in ("home", "away"):
        ids = out[f"{side}_team_id"].astype(str)
        hit = ids.isin(id_map).to_numpy()
        touched |= hit
        new_ids = ids.map(lambda t: id_map.get(t, t))
        out[f"{side}_team_id"] = new_ids
        out.loc[hit, f"{side}_team"] = new_ids[hit].map(name_map).to_numpy()
    out["_touched"] = touched
    if "has_features" not in out.columns:
        out["has_features"] = False
    order = out.sort_values(["home_team_id", "away_team_id", "match_date"], kind="stable")
    drop_idx: List[Any] = []
    conflicts: List[Dict[str, Any]] = []
    folded = 0
    for _, grp in order.groupby(["home_team_id", "away_team_id"], sort=False):
        if len(grp) < 2:
            continue
        rows = grp.to_dict("index")
        idx = list(rows.keys())
        cluster = [idx[0]]
        for prev, cur in zip(idx, idx[1:]):
            if rows[cur]["match_date"] - rows[prev]["match_date"] <= TWIN_WINDOW:
                cluster.append(cur)
                continue
            folded += _fold_cluster(rows, cluster, drop_idx, conflicts)
            cluster = [cur]
        folded += _fold_cluster(rows, cluster, drop_idx, conflicts)
    if conflicts:
        raise RuntimeError(f"{len(conflicts)} twin fixture(s) disagree on the score: {conflicts[:5]}")
    merged = out.drop(index=drop_idx).drop(columns=["_touched"])
    stats = {
        "pairs_in_map": int(len(id_map)),
        "rows_repointed": int(touched.sum()),
        "twin_rows_folded": int(len(drop_idx)),
        "twin_clusters": int(folded),
        "rows_after": int(len(merged)),
    }
    return merged.sort_values("match_date", kind="stable").reset_index(drop=True), stats


def _fold_cluster(rows: Dict[Any, dict], cluster: List[Any], drop_idx: List[Any], conflicts: List[dict]) -> int:
    if len(cluster) < 2 or not any(rows[i]["_touched"] for i in cluster):
        return 0
    scores = {(rows[i]["home_score"], rows[i]["away_score"]) for i in cluster}
    if len(scores) > 1:
        conflicts.append({"match_ids": [rows[i]["match_id"] for i in cluster], "scores": sorted(scores)})
        return 0
    keep = sorted(cluster, key=lambda i: (not bool(rows[i]["has_features"]), rows[i]["match_date"]))[0]
    drop_idx.extend(i for i in cluster if i != keep)
    return 1


def known_teams(model: Any, home: Sequence[str], away: Sequence[str]) -> np.ndarray:
    """0, 1 or 2 per row: how many of the row's teams the member's fit holds."""
    attack = getattr(model, "team_attack", {}) or {}
    return np.array([int(h in attack) + int(a in attack) for h, a in zip(home, away)], dtype=float)


def log_loss_rows(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float)[np.arange(len(y_idx)), np.asarray(y_idx, dtype=int)]
    return -np.log(np.clip(p, 1e-12, 1.0))


def standalone_metrics(proba: np.ndarray, y_idx: np.ndarray) -> Dict[str, Any]:
    """Brier (3-class sum), log loss, ECE / MCE (the repo's own definition,
    ProbabilityCalibrator.calibration_error) and mean predicted vs realised
    class rates for one probability source."""
    from training.calibration import ProbabilityCalibrator  # type: ignore

    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y_idx, dtype=int)
    require_finite("standalone_metrics (mask the rows out first)", proba)
    if not len(y):
        return {"n": 0}
    cal = ProbabilityCalibrator().calibration_error(proba, y)
    return {
        "n": int(len(y)),
        "brier": float(ft.brier_multiclass(proba, y).mean()),
        "log_loss": float(log_loss_rows(proba, y).mean()),
        "ece": float(cal["ece"]),
        "mce": float(cal["mce"]),
        **{f"pred_{c}": float(proba[:, k].mean()) for k, c in enumerate(CLASSES)},
        **{f"real_{c}": float((y == k).mean()) for k, c in enumerate(CLASSES)},
    }


def oracle_blend(p_served: np.ndarray, member_sum: np.ndarray, y_idx: np.ndarray, weight: float) -> np.ndarray:
    """The served blend with the scoreline members' combined contribution
    (`member_sum` = sum_m w_m p_m, total weight `weight`) replaced by the
    realised outcome: the absolute ceiling on what those members could add."""
    onehot = np.zeros_like(np.asarray(p_served, dtype=float))
    onehot[np.arange(len(y_idx)), np.asarray(y_idx, dtype=int)] = 1.0
    return np.asarray(p_served, dtype=float) - np.asarray(member_sum, dtype=float) + weight * onehot


def verdict(pair: Dict[str, Any], baseline: str) -> Dict[str, Any]:
    """ft.ship_verdict on a paired delta, plus the t(n_folds-1) clustered
    bound. "ship_strict" needs both."""
    v = ft.ship_verdict(
        pair["delta"],
        pair["se"],
        baseline,
        delta_cluster=pair["delta_cluster"],
        se_cluster=pair["se_cluster"],
        n_folds=pair["n_folds"],
    )
    v["upper_95_cluster_t"] = ht.t_upper(pair["delta_cluster"], pair["se_cluster"], pair["n_folds"])
    v["ship_strict"] = bool(v["ship"] and v["upper_95_cluster_t"] is not None and v["upper_95_cluster_t"] < 0)
    return v


def _finite(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def unit_lower_95(pair: Dict[str, Any]) -> Optional[float]:
    """The A2 rule's 95% lower bound on one unit's paired delta (challenger -
    served): fold-clustered when the unit spans at least two folds, per match
    otherwise. None when neither SE exists (too few rows)."""
    n_folds = pair.get("n_folds") or 0
    if n_folds >= 2 and _finite(pair.get("delta_cluster")) and _finite(pair.get("se_cluster")):
        return float(pair["delta_cluster"]) - UNIT_Z * float(pair["se_cluster"])
    if _finite(pair.get("delta")) and _finite(pair.get("se")):
        return float(pair["delta"]) - UNIT_Z * float(pair["se"])
    return None


def unit_regression(unit: str, pair: Dict[str, Any]) -> Dict[str, Any]:
    """One unit's A2 reading: the pre-registered bound (unit_lower_95) with
    the delta that bound is built on, plus two bounds that say how much to
    trust it. A fold-clustered SE from a handful of folds is itself noisy, so
    a clustered z bound can sit above zero while the per-match bound and the
    t(n_folds-1) clustered bound both include it; `robust` is True only when
    every bound that exists is above zero. The decision uses lower_95 alone,
    as pre-registered; the other two are reported, never decided on."""
    n_folds = int(pair.get("n_folds") or 0)
    clustered = n_folds >= 2 and _finite(pair.get("delta_cluster")) and _finite(pair.get("se_cluster"))
    per_match = None
    if _finite(pair.get("delta")) and _finite(pair.get("se")):
        per_match = float(pair["delta"]) - UNIT_Z * float(pair["se"])
    cluster_t = None
    if clustered:
        from scipy.stats import t as student_t

        cluster_t = float(pair["delta_cluster"]) - float(student_t.ppf(0.975, n_folds - 1)) * float(pair["se_cluster"])
    lower = unit_lower_95(pair)
    bounds = [b for b in (lower, per_match, cluster_t) if b is not None]
    return {
        "unit": unit,
        "n": pair.get("n"),
        "n_folds": n_folds,
        "basis": f"fold-clustered over {n_folds} folds, z={UNIT_Z}" if clustered else f"per match, z={UNIT_Z}",
        "delta_used": pair.get("delta_cluster") if clustered else pair.get("delta"),
        "lower_95": lower,
        "delta_per_match": pair.get("delta"),
        "lower_95_per_match": per_match,
        "lower_95_cluster_t": cluster_t,
        "robust": bool(bounds) and all(b > 0 for b in bounds),
    }


def convergence_decision(report: Dict[str, Any], arm: str = CONVERGED) -> Dict[str, Any]:
    """The pre-registered decision for the converged-Poisson arm (docstring):
    SHIP only when verdict() is ship_strict AND no blocking unit (stratum)
    regresses, a unit regressing only when its 95% lower bound is above zero.
    Each regressing unit also says whether that reading is robust
    (unit_regression)."""
    v = report["verdicts_vs_served"][arm]
    regressed = []
    for unit, block in report["strata"].items():
        pair = block.get(arm)
        if not pair:
            continue
        lower = unit_lower_95(pair)
        if lower is not None and lower > 0:
            regressed.append(unit_regression(unit, pair))
    ship = bool(v["ship_strict"]) and not regressed
    if ship:
        line = f"SHIP: {v['line']}; no blocking unit regresses"
    else:
        reasons = []
        if not v["ship_strict"]:
            reasons.append(f"the gate is not met ({v['line']})")
        if regressed:
            reasons.append(
                "blocking unit(s) regress under the pre-registered bound: "
                + ", ".join(_unit_text(r) for r in regressed)
            )
        line = "DO-NOT-SHIP: " + "; ".join(reasons)
    return {"arm": arm, "ship": ship, "ship_strict": bool(v["ship_strict"]), "units_regressed": regressed, "line": line}


def _unit_text(r: Dict[str, Any]) -> str:
    text = f"{r['unit']} (delta {_f(r['delta_used'], '+.6f')}, lower {_f(r['lower_95'], '+.6f')}, {r['basis']}"
    if not r["robust"]:
        text += (
            f"; NOT robust: per-match lower {_f(r['lower_95_per_match'], '+.6f')}, "
            f"t({max(r['n_folds'] - 1, 0)}) clustered lower {_f(r['lower_95_cluster_t'], '+.6f')}"
        )
    return text + ")"


# ── Model plumbing (needs ml-models + api on the path) ───────────────


def _add_src_paths() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (
        "/app/services/ml-models/src",
        "/app/services/api/src",
        os.path.join(root, "services", "ml-models", "src"),
        os.path.join(root, "services", "api", "src"),
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def load_ensemble(model_path: str):
    """The soccer ensemble reconstituted from `model_path` exactly as the API
    does it. Refuses anything but the five expected members."""
    from services.prediction_service import TASKS, _build_ensemble_for_task, _klass_registry  # type: ignore

    ensemble, meta = _build_ensemble_for_task(Path(model_path), TASKS[SOCCER_TASK_KEY], _klass_registry())
    if ensemble is None:
        raise RuntimeError(f"no soccer ensemble under {model_path}/{ENSEMBLE_NAME}/")
    names = set(ensemble.models)
    if names != set(ALL_MEMBERS):
        raise RuntimeError(f"ensemble members {sorted(names)} are not the expected {sorted(ALL_MEMBERS)}")
    for name in ALL_MEMBERS:
        if not getattr(ensemble.models[name], "is_fitted", False):
            raise RuntimeError(f"member {name} is not fitted")
        if float(ensemble.weights.get(name, 0.0)) <= 0:
            raise RuntimeError(f"member {name} has no positive weight — this harness assumes all five blend")
    if ensemble.calibrator is not None:
        raise RuntimeError("the soccer ensemble carries a calibrator; this harness blends raw members")
    return ensemble, meta


def member_fingerprint(model_path: str, member: str) -> Dict[str, Any]:
    """sha256 and fit summary of the artifact the API loader resolves for
    `member` (_latest_model_bin: the newest */model.bin by mtime), so the
    provenance names the file that was actually scored."""
    from services.prediction_service import _latest_model_bin  # type: ignore

    path = _latest_model_bin(Path(model_path) / member)
    if path is None:
        raise RuntimeError(f"no {member} artifact under {model_path}/{member}/")
    raw = path.read_bytes()
    data = json.loads(raw)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "teams": len(data.get("team_attack") or {}),
        "league_baselines": len(data.get("league_baselines") or {}),
        "rho": data.get("rho"),
        "time_decay": data.get("time_decay"),
    }


def fit_member(
    name: str, fit_df: pd.DataFrame, tmpdir: str, tag: str, hyperparameters: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Fit one scoreline member with the trainer's class, config and frame
    columns, then round-trip it through save() and the serving class's
    load() so serialisation cannot flatter the refit. `hyperparameters`
    overrides config keys (the converged arm's gauge); the serving load uses
    the plain production config, as the API would."""
    from dataclasses import replace

    from predictors.model_config import DIXON_COLES_CONFIG, POISSON_CONFIG  # type: ignore
    from predictors.poisson_models import DixonColesPredictor, PoissonMatchPredictor  # type: ignore

    cls, cfg = {
        POISSON: (PoissonMatchPredictor, POISSON_CONFIG),
        DIXON_COLES: (DixonColesPredictor, DIXON_COLES_CONFIG),
    }[name]
    fit_cfg = replace(cfg, hyperparameters={**cfg.hyperparameters, **hyperparameters}) if hyperparameters else cfg
    model = cls(fit_cfg)
    result = model.train(fit_df[list(MEMBER_FIT_COLUMNS)].copy())
    path = os.path.join(tmpdir, f"{name}_{tag}.bin")
    model.save(path)
    served = cls(cfg)
    served.load(path)
    return served, result


def strength_gap(a: Any, b: Any) -> float:
    """max |difference| over every team's attack and defense; inf when the two
    fits do not hold the same teams."""
    if set(a.team_attack) != set(b.team_attack) or set(a.team_defense) != set(b.team_defense):
        return math.inf
    gaps = [abs(float(a.team_attack[t]) - float(b.team_attack[t])) for t in a.team_attack]
    gaps += [abs(float(a.team_defense[t]) - float(b.team_defense[t])) for t in a.team_defense]
    return max(gaps) if gaps else 0.0


def served_member_split(
    frame: pd.DataFrame, seen_ids: set, incumbent_rows: int, served_poisson: Any, tmpdir: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The served Poisson member's exact train split, rebuilt as the trainer
    built it, and PROVEN by refitting: the incumbent's train+val rows (its
    seen ids) in kickoff order, the first int(incumbent_rows x TRAIN_RATIO).

    The trainer's sort is not stable, so rows sharing the boundary kickoff
    may fall on either side. Every assignment of those rows is tried with the
    PRODUCTION config, and the one whose fit reproduces the served artifact to
    SPLIT_ATOL wins. No reproduction, or too many assignments to try, fails
    the run: the converged arm must be fitted on exactly what served saw."""
    seen_rows = frame[frame["match_id"].isin(seen_ids)].sort_values("match_date", kind="stable")
    seen_rows = seen_rows.reset_index(drop=True)
    n_fit = int(int(incumbent_rows) * TRAIN_RATIO)
    if not 0 < n_fit < len(seen_rows):
        raise RuntimeError(f"train split of {n_fit} rows does not fit inside {len(seen_rows)} seen rows")
    boundary = seen_rows["match_date"].iloc[n_fit - 1]
    ties = [int(i) for i in seen_rows.index[seen_rows["match_date"] == boundary]]
    k = sum(1 for i in ties if i < n_fit)
    n_assign = math.comb(len(ties), k)
    if n_assign > MAX_TIE_ASSIGNMENTS:
        raise RuntimeError(f"{n_assign} boundary assignments ({len(ties)} tied rows) exceed {MAX_TIE_ASSIGNMENTS}")
    head = seen_rows.iloc[: min(ties)]
    best = math.inf
    for tried, combo in enumerate(itertools.combinations(ties, k), start=1):
        candidate = pd.concat([head, seen_rows.loc[list(combo)]])
        refit, _ = fit_member(POISSON, candidate, tmpdir, f"split{tried}")
        gap = strength_gap(refit, served_poisson)
        best = min(best, gap)
        if gap <= SPLIT_ATOL:
            info = {
                "incumbent_rows": int(incumbent_rows),
                "seen_rows": int(len(seen_rows)),
                "n_fit": n_fit,
                "fit_newest": str(candidate["match_date"].max()),
                "boundary_tied_rows": len(ties),
                "boundary_rows_in_train": k,
                "assignments_tried": tried,
                "reproduction_max_abs": gap,
            }
            return candidate, info
    raise RuntimeError(
        f"no boundary assignment of {len(ties)} tied rows reproduces the served Poisson member "
        f"(closest max |d| {best:.3g} > {SPLIT_ATOL}) — wrong seen ids, wrong fingerprint or a changed corpus"
    )


def load_incumbent_rows(model_path: str) -> int:
    """The incumbent's frame size from its held_out_metrics.json fingerprint:
    the number the trainer's int(rows x 0.7) split was taken of."""
    meta_path = Path(model_path) / ENSEMBLE_NAME / "held_out_metrics.json"
    if not meta_path.exists():
        raise RuntimeError(f"{meta_path} is missing — cannot rebuild the served member's train split")
    rows = (json.loads(meta_path.read_text()).get("fingerprint") or {}).get("rows")
    if not rows:
        raise RuntimeError(f"{meta_path} carries no fingerprint rows")
    return int(rows)


def member_proba(model: Any, ev: pd.DataFrame) -> np.ndarray:
    """A scoreline member's 1x2 exactly as the ensemble calls it on a serve
    row (home_team, away_team, league_id)."""
    x = pd.DataFrame({col: ev[col].to_numpy() for col in ("home_team", "away_team", "league_id")})
    p = np.asarray(model.predict_proba(x), dtype=float)
    if p.shape != (len(ev), 3) or not np.isfinite(p).all():
        raise RuntimeError(f"scoreline member emitted {p.shape} / non-finite probabilities")
    return p


def derived_markets(dc: Any, ev: pd.DataFrame, target_1x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(P(over 2.5), P(btts yes)) through the serve path's derivation, with
    the scoreline reconciled to `target_1x2` (no O/U market anchor)."""
    from predictors.market_derivation import derive_from_lambdas  # type: ignore

    rho = float(getattr(dc, "rho", 0.0) or 0.0)
    over = np.empty(len(ev))
    btts = np.empty(len(ev))
    for i, (h, a, lg) in enumerate(zip(ev["home_team"], ev["away_team"], ev["league_id"])):
        h_lam, a_lam = dc.lambdas_for_match(h, a, league=lg)
        m = derive_from_lambdas(float(h_lam), float(a_lam), rho, target_1x2=tuple(float(x) for x in target_1x2[i]))
        over[i] = float(m["over_under"]["over_2.5"])
        btts[i] = float(m["btts"]["yes"])
    return over, btts


# ── Data ─────────────────────────────────────────────────────────────


def _read_sql(database_url: str, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        return pd.read_sql(sql, engine, params=params)
    finally:
        engine.dispose()


def load_frame(database_url: str) -> pd.DataFrame:
    frame = _read_sql(database_url, MATCHES_SQL)
    if frame.empty:
        raise RuntimeError("no finished soccer matches — check --database-url")
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    dups = int(frame["match_id"].duplicated().sum())
    if dups:
        raise RuntimeError(f"the match frame carries {dups} duplicated match_id rows")
    frame["outcome"] = np.where(
        frame["home_score"] > frame["away_score"], 0, np.where(frame["home_score"] == frame["away_score"], 1, 2)
    )
    return frame.sort_values("match_date", kind="stable").reset_index(drop=True)


def load_features(database_url: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, dict]:
    rows = _read_sql(database_url, FEATURES_SQL, {"start": start.to_pydatetime(), "end": end.to_pydatetime()})
    return {str(r.match_id): (r.features or {}) for r in rows.itertuples()}


def load_medians(database_url: str) -> Dict[str, float]:
    rows = _read_sql(database_url, MEDIANS_SQL)
    return {str(k): float(v) for k, v in zip(rows["key"], rows["median"])}


def load_market_1x2(
    database_url: str, start: pd.Timestamp, end: pd.Timestamp, min_books: int
) -> Dict[str, ft.MarketQuote]:
    from utils.training_data import prematch_odds  # type: ignore

    rows = _read_sql(
        database_url, prematch_odds(ODDS_1X2_SQL), {"start": start.to_pydatetime(), "end": end.to_pydatetime()}
    )
    out: Dict[str, ft.MarketQuote] = {}
    for match_id, grp in rows.groupby("match_id"):
        quotes = []
        for _book, bg in grp.groupby("bookmaker"):
            price = dict(zip(bg["selection"], bg["odds_decimal"]))
            if all(s in price for s in CLASSES):
                quotes.append([float(price[s]) for s in CLASSES])
        quote = ft.consensus_devigged(quotes, min_books=min_books)
        if quote is not None:
            out[str(match_id)] = quote
    return out


def load_seen_ids(model_path: str) -> set:
    path = Path(model_path) / ENSEMBLE_NAME / SEEN_IDS_FILENAME
    if not path.exists():
        raise RuntimeError(f"{path} is missing — without the incumbent's train+val ids there is no honest eval window")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    import modal_bundles  # the gate's own provenance check (scripts/modal_bundles.py)

    model = modal_bundles.newest_model_bin(path.parent)
    ok, reason = modal_bundles.seen_ids_provenance(doc, model.read_bytes() if model is not None else None)
    if ok is False:
        raise RuntimeError(f"{path} {reason} — these are another artifact's ids; there is no honest eval window")
    logger.info("seen ids: %s", reason)
    return set(str(i) for i in doc["match_ids"])


def served_fit_slice(frame: pd.DataFrame, model_path: str, member_teams: set) -> Dict[str, Any]:
    """Reconstruct the incumbent's train split from its recorded fingerprint
    (frame rows, newest date) and report whether its teams are exactly the
    served members' teams, i.e. whether "fitted on the oldest 70%" holds."""
    meta_path = Path(model_path) / ENSEMBLE_NAME / "held_out_metrics.json"
    if not meta_path.exists():
        return {"checked": False, "reason": f"{meta_path} missing"}
    fp = json.loads(meta_path.read_text()).get("fingerprint") or {}
    rows, date_max = fp.get("rows"), fp.get("date_max")
    if not rows or not date_max:
        return {"checked": False, "reason": "no fingerprint rows/date_max"}
    asof = frame[frame["match_date"] < pd.Timestamp(date_max, tz="UTC") + pd.Timedelta(days=1)]
    n_fit = int(int(rows) * 0.7)
    head = asof.iloc[:n_fit]
    teams = set(head["home_team"]) | set(head["away_team"])
    return {
        "checked": True,
        "incumbent_rows": int(rows),
        "rows_in_frame_asof": int(len(asof)),
        "n_fit": n_fit,
        "fit_newest": str(head["match_date"].max()),
        "teams_in_slice": len(teams),
        "member_teams": len(member_teams),
        "identical": teams == member_teams,
        "symmetric_difference": len(teams ^ member_teams),
    }


# ── Run ──────────────────────────────────────────────────────────────


@dataclass
class Collected:
    """Per-row arrays gathered fold by fold, concatenated at the end."""

    cols: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def add(self, key: str, values: Any) -> None:
        self.cols.setdefault(key, []).append(np.asarray(values))

    def stack(self) -> Dict[str, np.ndarray]:
        return {k: (np.concatenate(v) if np.asarray(v[0]).ndim == 1 else np.vstack(v)) for k, v in self.cols.items()}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _add_src_paths()
    frame = load_frame(args.database_url)
    ensemble, meta = load_ensemble(args.model_path)
    weights = {m: float(ensemble.weights[m]) for m in ALL_MEMBERS}
    w_norm = normalised_weights(weights, ALL_MEMBERS)
    member_weight = float(sum(w_norm[m] for m in SCORELINE_MEMBERS))
    served_members = {m: ensemble.models[m] for m in SCORELINE_MEMBERS}
    fingerprints = {m: member_fingerprint(args.model_path, m) for m in SCORELINE_MEMBERS}
    fit_slice = served_fit_slice(frame, args.model_path, set(served_members[DIXON_COLES].team_attack))
    seen_ids = load_seen_ids(args.model_path)

    merge_stats = None
    merge_new_ids: set = set()
    if args.simulate_merge_map:
        merge_map = pd.read_csv(args.simulate_merge_map, comment="#", dtype=str)
        merge_new_ids = set(merge_map["new_team_id"].astype(str))
    elif args.merge_map and Path(args.merge_map).exists():
        merge_new_ids = set(pd.read_csv(args.merge_map, comment="#", dtype=str)["new_team_id"].astype(str))
    frame["identity_split"] = frame["home_team_id"].isin(merge_new_ids) | frame["away_team_id"].isin(merge_new_ids)

    oos, seen_cutoff, n_late = out_of_sample_mask(frame, seen_ids)
    span_start = pd.Timestamp(frame.loc[oos, "match_date"].min())
    end_day = args.end or frame.loc[oos, "match_date"].max().strftime("%Y-%m-%d")
    start_day = args.start or span_start.strftime("%Y-%m-%d")
    folds = ft.build_folds(start_day, end_day)
    span_end = folds[-1].eval_end
    features = load_features(args.database_url, folds[0].eval_start, span_end)
    frame["has_features"] = frame["match_id"].isin(features.keys())

    if args.simulate_merge_map:
        frame, merge_stats = simulate_team_merge(frame, merge_map)
        oos = (frame["match_date"] > seen_cutoff).to_numpy() & ~frame["match_id"].isin(seen_ids).to_numpy()
        logger.info("simulated merge: %s", merge_stats)

    # Eval rows: out of sample, inside the requested span, with a features row.
    in_span = (frame["match_date"] >= folds[0].eval_start) & (frame["match_date"] < span_end)
    eval_df = frame[oos & in_span.to_numpy() & frame["has_features"].to_numpy()].copy()
    no_features = int((oos & in_span.to_numpy() & ~frame["has_features"].to_numpy()).sum())

    medians: Dict[str, float] = {}
    eval_feats = [features[mid] for mid in eval_df["match_id"]]
    if any(needs_medians(f) for f in eval_feats):
        medians = load_medians(args.database_url)
    rows = []
    n_filled = 0
    for f, (h, a, lg) in zip(eval_feats, eval_df[["home_team", "away_team", "league_id"]].itertuples(index=False)):
        row, k = serve_row(f, {"home_team": h, "away_team": a, "league_id": lg}, medians)
        rows.append(row)
        n_filled += k
    serve_x = pd.DataFrame(rows, index=eval_df.index)
    required = sorted({c for m in LEARNED_MEMBERS for c in ensemble.models[m].feature_names})
    unservable = rows_missing_features(serve_x, required)
    eval_df = eval_df[~unservable]
    serve_x = serve_x[~unservable]
    logger.info(
        "eval rows %d (excluded: %d without features, %d unservable, %d unseen inside the seen span); %d folds",
        len(eval_df),
        no_features,
        int(unservable.sum()),
        n_late,
        len(folds),
    )

    # Learned members once (shared by every arm), then the fidelity checks.
    learned = {m: np.asarray(ensemble.models[m].predict_proba(serve_x), dtype=float) for m in LEARNED_MEMBERS}
    served_scoreline = {m: member_proba(served_members[m], eval_df) for m in SCORELINE_MEMBERS}
    p_served_all = blend({**learned, **served_scoreline}, weights, ALL_MEMBERS)
    fidelity_batch, fidelity_single, n_sample = serve_path_fidelity(ensemble, serve_x, p_served_all)

    market = load_market_1x2(args.database_url, folds[0].eval_start, span_end, args.min_books)
    pos = {idx: i for i, idx in enumerate(eval_df.index)}

    c = Collected()
    fits: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        pretest_rows = frame[frame["match_id"].isin(seen_ids)]
        pretest = {}
        for m in SCORELINE_MEMBERS:
            pretest[m], res = fit_member(m, pretest_rows, tmp, "pretest")
            fits.append(_fit_record("pretest", m, pretest_rows, res))
        # The pre-registered converged arm: the served Poisson's own split,
        # proven by reproducing the artifact, refitted with the per-component
        # gauge. The Dixon-Coles member stays the served one.
        split_df, split_info = served_member_split(
            frame, seen_ids, load_incumbent_rows(args.model_path), served_members[POISSON], tmp
        )
        converged_poisson, res = fit_member(POISSON, split_df, tmp, "converged", CONVERGED_HYPERPARAMETERS)
        fits.append(_fit_record("converged", POISSON, split_df, res))
        if not res.get("converged"):
            raise RuntimeError(f"the converged arm's Poisson fit did not converge: {res.get('fit_diagnostics')}")
        converged = {POISSON: converged_poisson, DIXON_COLES: served_members[DIXON_COLES]}
        for fold in folds:
            fit_df, _ = ft.split_fold(frame, fold)
            ev = eval_df[(eval_df["match_date"] >= fold.eval_start) & (eval_df["match_date"] < fold.eval_end)]
            if args.limit:
                ev = ev.head(int(args.limit))
            if len(ev) < MIN_FOLD_EVAL:
                skipped.append({"fold": fold.label, "n_eval": int(len(ev))})
                logger.warning("fold %s skipped: %d eval rows < %d", fold.label, len(ev), MIN_FOLD_EVAL)
                continue
            check_fold_disjoint(fit_df, ev, fold)
            idx = np.array([pos[i] for i in ev.index])
            refit = {}
            for m in SCORELINE_MEMBERS:
                refit[m], res = fit_member(m, fit_df, tmp, fold.label)
                fits.append(_fit_record(fold.label, m, fit_df, res, fold))
            arm_members = {"served": served_members, "refit": refit, "refit_pretest": pretest, CONVERGED: converged}
            lp = {m: learned[m][idx] for m in LEARNED_MEMBERS}
            y = ev["outcome"].to_numpy(dtype=int)
            c.add("y", y)
            goals = ev["home_score"].to_numpy(dtype=float) + ev["away_score"].to_numpy(dtype=float)
            c.add("y_over25", (goals > 2.5).astype(float))
            c.add("y_btts", ((ev["home_score"] >= 1) & (ev["away_score"] >= 1)).to_numpy(dtype=float))
            c.add("fold", np.full(len(ev), fold.label, dtype=object))
            c.add("match_id", ev["match_id"].to_numpy(dtype=object))
            c.add("match_date", ev["match_date"].astype(str).to_numpy(dtype=object))
            c.add("league", ev["league_name"].to_numpy(dtype=object))
            c.add("home_team", ev["home_team"].to_numpy(dtype=object))
            c.add("away_team", ev["away_team"].to_numpy(dtype=object))
            c.add("identity_split", ev["identity_split"].to_numpy(dtype=float))
            c.add("espn_era", (ev["match_date"] >= ESPN_ERA_START).to_numpy(dtype=float))
            scoreline: Dict[str, Dict[str, np.ndarray]] = {}
            for arm in MEMBER_ARMS:
                scoreline[arm] = {m: member_proba(arm_members[arm][m], ev) for m in SCORELINE_MEMBERS}
                for m in SCORELINE_MEMBERS:
                    c.add(f"q::{arm}::{m}", scoreline[arm][m])
                c.add(f"known::{arm}", known_teams(arm_members[arm][DIXON_COLES], ev["home_team"], ev["away_team"]))
            mk = np.vstack([market[mid].probs if mid in market else np.full(3, np.nan) for mid in ev["match_id"]])
            c.add("market", mk)
            p_arm = assemble_arms(lp, scoreline, mk, y, weights)
            for arm, p in p_arm.items():
                c.add(f"p::{arm}", p)
            for m in LEARNED_MEMBERS:
                c.add(f"q::learned::{m}", lp[m])
            dc_for = {"served": served_members[DIXON_COLES], "refit": refit[DIXON_COLES]}
            dc_for.update({"refit_pretest": pretest[DIXON_COLES], "drop": served_members[DIXON_COLES]})
            dc_for[CONVERGED] = served_members[DIXON_COLES]
            for arm, dc in dc_for.items():
                over, btts = derived_markets(dc, ev, p_arm[arm])
                c.add(f"over25::{arm}", over)
                c.add(f"btts::{arm}", btts)
            logger.info("fold %s: fit=%d eval=%d", fold.label, len(fit_df), len(ev))

    return build_report(
        c.stack(),
        meta={
            "model_path": args.model_path,
            "ensemble_artifact": str(meta),
            "weights": weights,
            "weights_normalised": w_norm,
            "scoreline_member_weight": member_weight,
            "member_fingerprints": fingerprints,
            "served_fit_slice": fit_slice,
            "converged_arm_split": split_info,
            "seen_ids": len(seen_ids),
            "seen_cutoff": str(seen_cutoff),
            "unseen_rows_inside_seen_span": n_late,
            "eval_rows_without_features": no_features,
            "eval_rows_unservable": int(unservable.sum()),
            "median_filled_values": n_filled,
            "fidelity_batch_max_abs": fidelity_batch,
            "fidelity_single_row_max_abs": fidelity_single,
            "fidelity_single_row_sample": n_sample,
            "folds_requested": [f.label for f in folds],
            "skipped_folds": skipped,
            "fits": fits,
            "min_books": int(args.min_books),
            "simulated_merge": merge_stats,
            "corpus_rows": int(len(frame)),
        },
        out_dir=args.out_dir,
    )


def _fit_record(label: str, member: str, fit_df: pd.DataFrame, res: Dict[str, Any], fold: Any = None) -> Dict[str, Any]:
    rec = {
        "fold": label,
        "member": member,
        "n_fit": int(len(fit_df)),
        "fit_newest": str(fit_df["match_date"].max()),
        "teams": int(res.get("team_count") or 0),
        "converged_at": res.get("iterations"),
        "rho": res.get("rho"),
        "fitted_total": res.get("fitted_mean_total_goals"),
        "observed_total": res.get("observed_mean_total_goals"),
    }
    if fold is not None:
        rec["newest_fit_gap_days"] = float((fold.fit_cutoff - fit_df["match_date"].max()).total_seconds() / 86400)
    return rec


def build_report(s: Dict[str, np.ndarray], meta: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Score every arm, pair the challengers against served, and assemble the
    report. Pure given the stacked per-row arrays (so a reviewer can rebuild
    it from the per-match CSV)."""
    y = s["y"].astype(int)
    folds_arr = s["fold"]
    labels = list(dict.fromkeys(folds_arr.tolist()))
    arms = list(ARMS) + [BOUND_ARM, "oracle"]
    # Every arm but the bound must be finite on every row: a NaN scored as 0
    # would be a Brier of 1.0 that fails nothing. The bound is NaN off the
    # market rows by design, and is masked to exactly those rows.
    scored = [a for a in arms if a != BOUND_ARM]
    for a in scored:
        require_finite(f"arm {a}", s[f"p::{a}"])
    has_market = np.isfinite(s["market"]).all(axis=1)
    bound_finite = np.isfinite(s[f"p::{BOUND_ARM}"]).all(axis=1)
    if not np.array_equal(bound_finite, has_market):
        raise RuntimeError(
            f"{BOUND_ARM} is finite on {int(bound_finite.sum())} rows but the market on {int(has_market.sum())}"
        )
    briers = {a: ft.brier_multiclass(s[f"p::{a}"], y) for a in scored}
    briers[BOUND_ARM] = np.where(has_market, ft.brier_multiclass(np.nan_to_num(s[f"p::{BOUND_ARM}"]), y), np.nan)

    def pair(a: str, b: str, mask: Optional[np.ndarray] = None, score: Optional[Dict[str, np.ndarray]] = None):
        sc = score or briers
        sel = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        out = ft.paired_delta(sc[a][sel], sc[b][sel], folds_arr[sel])
        out["upper_95_cluster_t"] = ht.t_upper(out["delta_cluster"], out["se_cluster"], out["n_folds"])
        return out

    known = s["known::served"]
    espn = s["espn_era"].astype(bool)
    split = s["identity_split"].astype(bool)
    challengers = list(PRIMARY_CHALLENGERS) + list(SECONDARY_CHALLENGERS) + list(CONVERGENCE_CHALLENGERS)
    strata = {
        "served_knows_both": known == 2,
        "served_knows_one": known == 1,
        "served_knows_neither": known == 0,
        "football_data_era_to_2026_05": ~espn,
        "espn_era_2026_06_on": espn,
        "identity_split_fixtures": split,
        "not_identity_split": ~split,
    }
    report: Dict[str, Any] = {
        "meta": meta,
        "n_eval": int(len(y)),
        "folds": labels,
        "brier": {a: float(np.nanmean(briers[a])) for a in arms},
        "market_rows": int(has_market.sum()),
        "vs_served": {a: pair(a, "served") for a in challengers},
        "verdicts_vs_served": {a: verdict(pair(a, "served"), "served") for a in challengers},
        "head_to_head": {
            "refit - drop": pair("refit", "drop"),
            "refit_pretest - refit": pair("refit_pretest", "refit"),
        },
        "bound": {
            "members_as_market_vs_served": pair(BOUND_ARM, "served", mask=has_market),
            "refit_vs_served_on_market_rows": pair("refit", "served", mask=has_market),
            "refit_vs_served_off_market_rows": pair("refit", "served", mask=~has_market),
            "drop_vs_served_on_market_rows": pair("drop", "served", mask=has_market),
            "market_alone_vs_served": _market_alone(s, y, briers, folds_arr, has_market),
            "oracle_vs_served": pair("oracle", "served"),
            "refit_abs_shift_mean": float(np.abs(s["p::refit"] - s["p::served"]).sum(axis=1).mean()),
            "refit_abs_shift_p99": float(np.quantile(np.abs(s["p::refit"] - s["p::served"]).sum(axis=1), 0.99)),
            "in_sample_optimal_weights": in_sample_optimal_weights(s, y, float(briers["served"].mean())),
        },
        "strata": {
            name: {
                "n": int(mask.sum()),
                "brier_served": float(briers["served"][mask].mean()) if mask.any() else None,
                **{a: pair(a, "served", mask=mask) for a in challengers if mask.sum() > 1},
            }
            for name, mask in strata.items()
        },
        "per_fold_vs_served": {
            a: [
                {
                    "fold": lab,
                    "n": int((folds_arr == lab).sum()),
                    "delta": float(np.mean(briers[a][folds_arr == lab] - briers["served"][folds_arr == lab])),
                }
                for lab in labels
            ]
            for a in challengers
        },
        "members_standalone": {
            arm: {m: standalone_metrics(s[f"q::{arm}::{m}"], y) for m in SCORELINE_MEMBERS} for arm in MEMBER_ARMS
        },
        "members_standalone_served_knows_both": {
            arm: {m: standalone_metrics(s[f"q::{arm}::{m}"][known == 2], y[known == 2]) for m in SCORELINE_MEMBERS}
            for arm in MEMBER_ARMS
        },
        "blend_standalone": {a: standalone_metrics(s[f"p::{a}"], y) for a in ARMS},
        "learned_standalone": {m: standalone_metrics(s[f"q::learned::{m}"], y) for m in LEARNED_MEMBERS},
        "market_standalone": standalone_metrics(s["market"][has_market], y[has_market]),
        "team_coverage": {
            arm: {
                "both": float((s[f"known::{arm}"] == 2).mean()),
                "one": float((s[f"known::{arm}"] == 1).mean()),
                "neither": float((s[f"known::{arm}"] == 0).mean()),
            }
            for arm in MEMBER_ARMS
        },
    }
    report["convergence_decision"] = convergence_decision(report)
    report["bound"]["converged_abs_shift_mean"] = float(
        np.abs(s[f"p::{CONVERGED}"] - s["p::served"]).sum(axis=1).mean()
    )
    report["bound"]["converged_abs_shift_max"] = float(np.abs(s[f"p::{CONVERGED}"] - s["p::served"]).max())
    if "y_over25" in s and "y_btts" in s:
        report["derived_info"] = _derived_info(s, s["y_over25"], s["y_btts"], folds_arr)
    if out_dir:
        report["per_match_csv"] = write_per_match(s, briers, out_dir)
    return report


def in_sample_optimal_weights(s: Dict[str, np.ndarray], y: np.ndarray, brier_served: float) -> Dict[str, Any]:
    """The BOUND on any linear re-weighting: for the learned members alone
    and for each MEMBER_ARMS arm's five members, the simplex weights that
    minimise the Brier on the eval rows themselves (optimal_simplex_weights).
    Fitted on the rows it scores, so it is optimistic by construction and no
    SE is attached: it is an upper bound, not a test."""
    sets: Dict[str, List[Tuple[str, str]]] = {"learned_only": [("learned", m) for m in LEARNED_MEMBERS]}
    for arm in MEMBER_ARMS:
        sets[arm] = sets["learned_only"] + [(arm, m) for m in SCORELINE_MEMBERS]
    out: Dict[str, Any] = {"note": "weights fitted on the eval rows themselves: optimistic upper bound, not a test"}
    for name, cols in sets.items():
        w, brier = optimal_simplex_weights([s[f"q::{src}::{m}"] for src, m in cols], y)
        out[name] = {
            "weights": {m: float(wi) for (_src, m), wi in zip(cols, w)},
            "scoreline_weight": float(sum(wi for (_src, m), wi in zip(cols, w) if m in SCORELINE_MEMBERS)),
            "brier": brier,
            "delta_vs_served": brier - brier_served,
        }
    return out


def _market_alone(s, y, briers, folds_arr, has_market) -> Dict[str, Any]:
    mb = np.where(has_market, ft.brier_multiclass(np.nan_to_num(s["market"]), y), np.nan)
    out = ft.paired_delta(mb[has_market], briers["served"][has_market], folds_arr[has_market])
    out["upper_95_cluster_t"] = ht.t_upper(out["delta_cluster"], out["se_cluster"], out["n_folds"])
    return out


def _derived_info(s, y_over, y_btts, folds_arr) -> Dict[str, Any]:
    out: Dict[str, Any] = {"note": "1x2-reconciled, no O/U-2.5 market anchor; information only, not gated"}
    for market, truth in (("over25", y_over), ("btts", y_btts)):
        for a in ARMS:
            require_finite(f"derived {market} ({a})", s[f"{market}::{a}"])
        sc = {a: ft.brier_binary(s[f"{market}::{a}"], truth) for a in ARMS}
        block: Dict[str, Any] = {
            "brier": {a: float(sc[a].mean()) for a in ARMS},
            "mean_pred": {a: float(np.mean(s[f"{market}::{a}"])) for a in ARMS},
            "realised": float(np.mean(truth)),
        }
        for a in ("refit", "refit_pretest", "drop", CONVERGED):
            p = ft.paired_delta(sc[a], sc["served"], folds_arr)
            p["upper_95_cluster_t"] = ht.t_upper(p["delta_cluster"], p["se_cluster"], p["n_folds"])
            block[f"{a} - served"] = p
        out[market] = block
    return out


def write_per_match(s: Dict[str, np.ndarray], briers: Dict[str, np.ndarray], out_dir: str) -> str:
    """One row per eval match with every arm's 1x2, every member's 1x2 per
    arm, the market, team coverage and per-arm Brier, so a reviewer can
    recompute every number in the report."""
    os.makedirs(out_dir, exist_ok=True)
    cols: Dict[str, Any] = {}
    for key in ("match_id", "match_date", "fold", "league", "home_team", "away_team", "y"):
        cols[key] = s[key]
    for key in ("identity_split", "espn_era"):
        cols[key] = s[key].astype(int)
    for key, arr in s.items():
        if arr.ndim == 2:
            for k, cls in enumerate(CLASSES):
                cols[f"{key}::{cls}"] = arr[:, k]
        elif key.startswith(("known::", "over25::", "btts::", "y_")):
            cols[key] = arr
    for a, b in briers.items():
        cols[f"brier::{a}"] = b
    path = os.path.join(out_dir, "per_match.csv.gz")
    pd.DataFrame(cols).to_csv(path, index=False, compression="gzip")
    return path


# ── Formatting ───────────────────────────────────────────────────────


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def _pair_line(name: str, p: Dict[str, Any]) -> str:
    return (
        f"  {name:<34} {_f(p.get('delta'))} ±{_f(p.get('se'), '.5f')}   cluster {_f(p.get('delta_cluster'))} "
        f"±{_f(p.get('se_cluster'), '.5f')}   t-bound {_f(p.get('upper_95_cluster_t'))}   n={p.get('n')}"
    )


def format_report(r: Dict[str, Any]) -> str:
    m = r["meta"]
    L = [
        f"Soccer 1x2 ensemble scoreline members A/B: {r['n_eval']} eval matches, folds "
        f"{r['folds'][0]}..{r['folds'][-1]} ({len(r['folds'])}; skipped {m['skipped_folds'] or 'none'})",
        f"incumbent: {m['ensemble_artifact']}; seen ids {m['seen_ids']} (newest {m['seen_cutoff']}); "
        f"scoreline members carry {m['scoreline_member_weight']:.4f} of the blend",
        f"served fit slice: {m['served_fit_slice']}",
        f"excluded: {m['eval_rows_without_features']} without features, {m['eval_rows_unservable']} unservable, "
        f"{m['unseen_rows_inside_seen_span']} unseen rows inside the seen span; "
        f"median-filled values {m['median_filled_values']}",
        f"fidelity: batch blend vs ensemble.predict_proba max |d| {m['fidelity_batch_max_abs']:.1e}; "
        f"one-row-at-a-time vs batch {m['fidelity_single_row_max_abs']:.1e} (n={m['fidelity_single_row_sample']})",
    ]
    if m.get("simulated_merge"):
        L.append(f"SIMULATED MERGE (corpus caveat run): {m['simulated_merge']}")
    L += ["", "team coverage of the scoreline members (both / one / neither known):"]
    for arm, cov in r["team_coverage"].items():
        L.append(f"  {arm:<15} {cov['both']:.1%} / {cov['one']:.1%} / {cov['neither']:.1%}")
    L += ["", "served 1x2 Brier by arm (3-class sum, lower is better):"]
    for a, b in r["brier"].items():
        note = f"  (n={r['market_rows']} market rows)" if a == BOUND_ARM else ""
        L.append(f"  {a:<18} {b:.5f}{note}")
    L += ["", "PRIMARY: served 1x2 Brier vs served (paired, per match ± SE; fold-clustered ± SE; t-bound):"]
    L += [_pair_line(a, p) for a, p in r["vs_served"].items()]
    L += ["", "head to head:"]
    L += [_pair_line(k, v) for k, v in r["head_to_head"].items()]
    b = r["bound"]
    L += ["", "BOUND (diagnostic, not the verdict):"]
    opt = b["in_sample_optimal_weights"]
    L.append(f"  in-sample-optimal simplex weights ({opt['note']}):")
    for name in ("learned_only",) + MEMBER_ARMS:
        x = opt[name]
        ws = " ".join(f"{k.split('_')[0][:4]} {v:.3f}" for k, v in x["weights"].items())
        L.append(
            f"    {name:<15} Brier {x['brier']:.5f}  vs served {_f(x['delta_vs_served'])}  "
            f"scoreline weight {x['scoreline_weight']:.3f}   [{ws}]"
        )
    L.append("  members_as_market assumes no scoreline member beats the market; it covers the market rows only:")
    L.append(_pair_line("members_as_market - served", b["members_as_market_vs_served"]))
    L.append(_pair_line("refit - served (market rows)", b["refit_vs_served_on_market_rows"]))
    L.append(_pair_line("refit - served (off market rows)", b["refit_vs_served_off_market_rows"]))
    L.append(_pair_line("drop - served (market rows)", b["drop_vs_served_on_market_rows"]))
    L.append(_pair_line("market alone - served", b["market_alone_vs_served"]))
    L.append(_pair_line("oracle - served", b["oracle_vs_served"]))
    L.append(
        f"  refit moves the served 1x2 by mean L1 {b['refit_abs_shift_mean']:.4f} (p99 {b['refit_abs_shift_p99']:.4f})"
    )
    L.append(
        f"  {CONVERGED} moves the served 1x2 by mean L1 {b['converged_abs_shift_mean']:.5f} "
        f"(max |dp| {b['converged_abs_shift_max']:.5f})"
    )
    L += ["", "strata (challenger - served):"]
    for name, blk in r["strata"].items():
        L.append(f"  [{name}] n={blk['n']} served Brier {_f(blk['brier_served'], '.5f')}")
        for a in list(PRIMARY_CHALLENGERS) + list(SECONDARY_CHALLENGERS) + list(CONVERGENCE_CHALLENGERS):
            if a in blk:
                L.append(_pair_line(f"    {a}", blk[a]))
    L += ["", "per-fold delta vs served:"]
    L.append(f"  {'arm':<15}" + "".join(f"{lab[2:]:>9}" for lab in r["folds"]))
    for a, rows in r["per_fold_vs_served"].items():
        L.append(f"  {a:<15}" + "".join(f"{row['delta']:>+9.4f}" for row in rows))
    L.append(f"  {'n':<15}" + "".join(f"{row['n']:>9}" for row in next(iter(r["per_fold_vs_served"].values()))))
    L += ["", "SECONDARY: members standalone (Brier / log loss / ECE; pred vs real H/D/A):"]
    for scope, key in (
        ("all rows", "members_standalone"),
        ("served knows both", "members_standalone_served_knows_both"),
    ):
        L.append(f"  [{scope}]")
        for arm, block in r[key].items():
            for mem, x in block.items():
                if x.get("n"):
                    L.append(
                        f"    {arm:<14} {mem:<34} {x['brier']:.5f} / {x['log_loss']:.5f} / {x['ece']:.4f}   "
                        f"pred {x['pred_home']:.3f}/{x['pred_draw']:.3f}/{x['pred_away']:.3f} "
                        f"real {x['real_home']:.3f}/{x['real_draw']:.3f}/{x['real_away']:.3f}  n={x['n']}"
                    )
    L.append("  [blends, learned members, market]")
    sources = list(r["blend_standalone"].items()) + list(r["learned_standalone"].items())
    if r["market_standalone"].get("n"):
        sources.append(("market (de-vigged consensus)", r["market_standalone"]))
    for name, x in sources:
        L.append(f"    {name:<49} {x['brier']:.5f} / {x['log_loss']:.5f} / {x['ece']:.4f}  n={x['n']}")
    if "derived_info" in r:
        d = r["derived_info"]
        L += ["", f"INFO derived markets ({d['note']}):"]
        for market in ("over25", "btts"):
            blk = d[market]
            L.append(
                f"  {market}: realised {blk['realised']:.4f}; "
                + ", ".join(f"{a} {blk['brier'][a]:.5f} (pred {blk['mean_pred'][a]:.4f})" for a in ARMS)
            )
            for a in ("refit", "refit_pretest", "drop", CONVERGED):
                L.append(_pair_line(f"    {a} - served", blk[f"{a} - served"]))
    unconv: Dict[str, int] = {}
    for fr in m["fits"]:
        if fr.get("converged_at") is None:
            unconv[fr["member"]] = unconv.get(fr["member"], 0) + 1
    L += ["", f"fits that did not converge (per member, of {len(m['fits'])} fits): {unconv or 'none'}"]
    if m.get("converged_arm_split"):
        L.append(f"{CONVERGED} split (rebuilt and proven by reproducing the served member): {m['converged_arm_split']}")
    L += ["", "VERDICTS (§4.3 on the primary readout; 'strict' also needs the t(n_folds-1) clustered bound < 0):"]
    for a, v in r["verdicts_vs_served"].items():
        tag = (
            "secondary"
            if a in SECONDARY_CHALLENGERS
            else "pre-registered" if a in CONVERGENCE_CHALLENGERS else "primary"
        )
        L.append(f"  {a:<17} [{tag}] strict={'SHIP' if v['ship_strict'] else 'no'}  {v['line']}")
    cd = r.get("convergence_decision")
    if cd:
        L += ["", f"PRE-REGISTERED DECISION ({cd['arm']}: gate + A2 blocking units): {cd['line']}"]
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help="first eval day (default: the incumbent's first unseen match)")
    p.add_argument("--end", default=None, help="last eval day, inclusive (default: newest unseen match)")
    p.add_argument("--model-path", default=PRODUCTION_MODEL_PATH, help="registry holding the incumbent ensemble")
    p.add_argument(
        "--merge-map",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "soccer_team_merge_map.csv"),
        help="merge map whose new_team_id rows define the identity-split stratum (read only)",
    )
    p.add_argument(
        "--simulate-merge-map", default=None, help="apply this merge map in memory first (the corpus-caveat run)"
    )
    p.add_argument("--min-books", type=int, default=2, help="books needed for a 1x2 market consensus")
    p.add_argument("--limit", type=int, default=0, help="cap eval rows per fold (smoke run); 0 = all")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-dir", default=None, help="write per_match.csv.gz here")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        report = run(args)
    except Exception as exc:
        logger.error("ab_soccer_ensemble_members failed: %s", exc, exc_info=True)
        return 1
    if args.out_json:  # before printing: a formatting bug must not lose the run
        Path(args.out_json).write_text(json.dumps(ht.clean_json(report), indent=2, allow_nan=False))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
