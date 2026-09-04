"""A/B gate for the soccer Dixon-Coles REFIT (level bias, per-league
baselines, time decay, shrinkage prior, league fallback, max_goals).

WHAT IS BEING TESTED
--------------------
The served soccer Dixon-Coles artifact is measurably worse than a
constant base rate on over/under 2.5 even on matches where it knows
both teams. The 2026-09-04 read-only investigation traced the dominant
cause to a FITTING bug, not to team-identity coverage:

  1. LEVEL BIAS — the iterative MLE normalises BOTH attack and defense
     to mean 1 every iteration. A multiplicative Poisson has exactly ONE
     scale indeterminacy (attack -> c*attack, defense -> defense/c), so
     normalising both over-constrains it by a degree of freedom and pins
     mean(lambda) to the hard-coded baseline. Served artifact implies
     E[total] = 2*1.3615 + 0.25 = 2.9730 against 2.6814 actual on its own
     both-known population (n=46,663): +10.9%.
  2. ONE GLOBAL league_avg_goals across 41 leagues running 2.222 to 3.094
     goals/match and 0.389 to 0.591 over-2.5 rates.
  3. time_decay is serialised but NEVER APPLIED — the fit is unweighted
     over a 2007-2026 frame whose home goal advantage moved 0.405 -> 0.216
     -> 0.327.
  4. regularization (0.001) is inert against weighted counts of order
     30-500, so it neither shrinks thin-history teams nor helps the
     iteration converge.
  5. Unseen teams fall back to a GLOBAL neutral lambda instead of their
     LEAGUE's baseline.
  6. max_goals 6 truncates the DC's own 1x2 tail (derivation already
     uses MAX_GOALS_DERIVE=10).

This harness does NOT implement those fixes — it scores them. The
challenger arms load whatever
``services/ml-models/src/predictors/poisson_models.py`` currently
provides and refit it walk-forward; the served arm is the on-disk
production artifact, untouched.

WHAT ``refit`` MINUS ``served`` IS AND IS NOT. The served artifact is a
2023-vintage fit on the PRE-backfill corpus and knows a small fraction of
the teams in this eval window; where it knows neither team it emits its
unknown-team constant. Every refit arm is fitted fold-by-fold on the
current corpus. So ``refit - served`` is dominated by "retrain on a much
larger corpus" and by team coverage — both real, neither the fitting bug.
The arm that isolates the FITTING fixes is ``refit_legacy``: the same
folds, the same corpus, the old fitter (double normalisation, one global
baseline, no live decay, inert regularization, max_goals 6). State the
fitting verdict against ``refit_legacy``; the report additionally
re-reads the primary market stratified by how many teams the served arm
knows, and ``served`` is kept only as "what production emits today".

NOT A CLOSED LEVER. The ``--decay`` weight here is an exponential weight
INSIDE a 4-parameter multiplicative Poisson MLE (it reweights the
sufficient statistics of a closed-form fixed-point iteration and the rho
NLL). It is NOT the GBM recency weighting and NOT the GBM training-frame
horizon, both of which are closed in docs/SYSTEM_AUDIT_AND_ROADMAP.md
§4. Do not read a result here as re-opening either. Likewise the
per-league baseline is a fitted intercept inside the MLE, not a post-hoc
probability calibrator (calibration is also closed) — nothing in this
harness applies a transform to an emitted probability.

THE PROTOCOL (each piece is here for a reason)
----------------------------------------------
* MONTHLY WALK-FORWARD. For each eval month M the challenger is refit on
  matches with ``match_date < start(M)`` ONLY, then predicts every match
  inside M. No month is ever in its own fit. A single split would let one
  lucky season carry the verdict; monthly refit also matches how a shipped
  model would actually be retrained. Time-decay weights are aged against
  the fit frame's own newest match — soccer plays somewhere every day, so
  that is within a day or two of ``start(M)``; ``reference_date`` is
  passed to ``train`` regardless so a trainer that honours it uses the
  exact boundary.

* SERVE-PATH FIDELITY, AND ITS ONE KNOWN GAP. The number that matters is
  the probability the SERVE PATH emits, not the raw Dixon-Coles marginal.
  Production runs ``derive_from_lambdas(h_lam, a_lam, rho,
  target_1x2=<ensemble 1x2>)`` (scripts/precompute_predictions.py), which
  IPF-reconciles the scoreline matrix to the ensemble's blended 1x2 before
  reading off over/under and BTTS. Skipping that reconciliation understates
  the served model by 0.0021 and the challenger by 0.0009 — a harness
  without it measures the wrong object. The reconciliation target is
  computed once per match and held IDENTICAL across every arm, so it is a
  shared input to the paired comparison and cannot favour either side.
  THE GAP: this harness builds the ensemble's input frame from
  features_cache PLUS home_team/away_team, while production builds it from
  features_cache alone — no feature key carries a team name — so in
  production the Dixon-Coles and Poisson ensemble MEMBERS fall to their
  unknown-team constant and the served 1x2 is not the one reconciled to
  here. That is a production bug worth its own ticket (the serve path
  should pass the team columns through); it shifts absolute numbers, not
  paired deltas. (The target is optimistic in ABSOLUTE terms anyway — that
  ensemble's training frame overlaps the eval window — which is why the
  market reference, not the ensemble, is the yardstick for absolute
  quality.)

* BASELINES, ALL PRINTED, flattering or not:
    (a) served      — what production emits today. NOT the fitting-fix
        comparator; see the corpus/coverage note above.
    (b) refit_legacy — same folds, same corpus, OLD fitter. This is the
        arm the fitting-fix verdict is stated against.
    (c) base_global / base_league — walk-forward constant base rates,
        global and per-league (shrunk toward global with k effective
        matches). base_league IS THE HONEST COMPARATOR and the refit is
        EXPECTED TO FAIL against it (prototype: -0.0025 +/- 0.0006). A
        model that beats a broken model but loses to a per-league constant
        has not earned a ship.
    (d) base_global_dw / base_league_dw — the same constants given the
        SAME recency weighting the challenger gets (--base-rate-decay,
        default = --decay). Without these the comparator is denied the
        weighting the model is granted, which at the first fold cutoff is
        worth ~0.0005 of Brier — about 11% of the whole ship gate, handed
        over by a protocol choice rather than earned. Both weightings are
        scored and a verdict is printed against each.
    (e) market      — de-vigged closing prices where odds exist. A
        thin-coverage REFERENCE, not a ceiling: the report prints its
        coverage, book depth and median overround per market so the reader
        can judge it. --min-books (default 2) keeps a single book's opinion
        from being scored as "the market".

* PAIRED STATISTICS, CLUSTERED BY FOLD. Every delta is the mean of
  PER-MATCH score differences — never the difference of two
  independently-computed means. Two arms scored on the same matches are
  enormously correlated; an unpaired SE would be ~10x too wide and would
  hide both wins and losses. But the per-match SE is not the one that
  gates: the challenger is one fitted model per monthly fold, so ~775
  differences inside a fold share that fit's error. At n~10k the per-match
  SE is ~0.0006, which makes "95% upper bound below zero" automatic for
  anything already clearing -0.005 — the noise check collapses into the
  threshold. So a FOLD-CLUSTERED SE (sd of the per-fold mean differences /
  sqrt(n_folds)) is computed too, both are printed, and SHIP requires the
  CLUSTERED bound to be negative.

* BIAS IN BOTH DIRECTIONS. Realised vs mean-predicted is printed for
  every configuration and every baseline. The served model is long overs
  (0.5672 predicted vs 0.5236 realised); the prototype refit flipped to
  long unders (0.4964). A one-sided bias readout would let a sign flip
  pass as a fix.

* MARKETS. over_2.5 is PRIMARY. BTTS and the ensemble 1x2 are reported
  too. The Dixon-Coles member carries only ~4.55% of the ensemble blend,
  so the 1x2 effect is roughly -0.0016 and fails the gate on its own —
  it is reported anyway. The 1x2 readout swaps the DC member's
  contribution at its blend weight rather than re-deriving 1x2 from the
  reconciled matrix, because after IPF the derived 1x2 IS the target by
  construction.

* VERDICT. An explicit SHIP / DO-NOT-SHIP line per baseline, per §4.3:
  dBrier <= -0.005 AND clear of the noise floor — both the per-match 95%
  upper bound and the fold-clustered one must be negative. The verdict
  names the baseline, because it genuinely differs by baseline.

WHAT THIS HARNESS DOES NOT DECIDE
---------------------------------
Not ROI. ("soccer", "over_under") stays enabled=False in
scripts/rec_gating.py: every EV configuration tested loses money against
stored best-of-book prices (served -2.44% +/- 1.78%, refit -2.24% +/-
2.72%, per-league base rate -6.66% +/- 2.26% at EV>5%). A Brier win is
not an ROI win; un-gating needs a separate positive-ROI result. Nor does
it touch team identity/coverage — a separate workstream.

    # smoke (60 eval matches per month)
    docker compose exec api python /app/scripts/ab_soccer_dixon_coles_refit.py --limit 60

    # full run
    docker compose exec api python /app/scripts/ab_soccer_dixon_coles_refit.py \
        --start 2025-08-01 --end 2026-08-31 --decay 0.00095 --reg 20
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ml-models (predictors) + api (prediction_service) live outside scripts/.
# /app/... is the container layout; the repo-relative paths let the pure
# helpers be imported and unit-tested from a checkout.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_PATHS = (
    "/app/services/ml-models/src",
    "/app/services/api/src",
    os.path.join(_REPO_ROOT, "services", "ml-models", "src"),
    os.path.join(_REPO_ROOT, "services", "api", "src"),
)
for _p in _SRC_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_dixon_coles_refit")


# ── Protocol constants ───────────────────────────────────────────────

# The audit §4.3 ship gate.
SHIP_THRESHOLD = -0.005
# Historic reference noise floor quoted in §4.3 (soccer 1x2 Brier SE at
# n~3.5k). Printed for context; the verdict uses the ACTUAL paired SE
# computed on this run's per-match differences.
REFERENCE_NOISE_FLOOR_SE = 0.009

DC_MEMBER = "dixon_coles_soccer_match_result"
SOCCER_TASK_KEY = "soccer:match_result"
PRODUCTION_MODEL_PATH = "/app/models/production"

PRIMARY_LINE = 2.5
PRIMARY_MARKET = "over_2.5"

# Challenger arms. Value = hyperparameter overrides applied on top of the
# CLI --decay / --reg / --max-goals base. ``None`` means "no refit, use
# the on-disk production artifact".
CONFIG_SPECS: Dict[str, Optional[Dict[str, Any]]] = {
    # The on-disk production artifact — no refit. Its saved payload carries
    # no fitted baselines, so it keeps serving its single global level.
    "served": None,
    # Every defect fixed: attack-only scale gauge, per-league baselines,
    # recency decay, a real shrinkage prior, league fallback for unseen
    # teams, max_goals. Values come from --decay / --reg / --max-goals /
    # --league-shrinkage on top of DIXON_COLES_CONFIG.
    "refit": {},
    # THE LEGACY-FIT ARM. Everything the pre-2026-09 fitter did — the
    # over-constrained attack+defense gauge, one global baseline, no live
    # decay, the inert 0.001 regularization, max_goals 6 — refit on the SAME
    # walk-forward folds as `refit`. This, not `served`, is the arm the
    # fitting-fix verdict is stated against: `served` is a 2023-vintage
    # artifact fit on a pre-backfill frame, so `refit` minus `served` scores
    # "retrain on a much larger corpus" as much as it scores any fix.
    "refit_legacy": {
        "normalise_defense": True,
        "per_league_baselines": False,
        "time_decay": 0.0,
        "regularization": 0.001,
        "max_goals": 6,
    },
    # Ablations — which defect actually carried the result. Each turns ONE
    # fix off relative to `refit`.
    "refit_double_norm": {"normalise_defense": True},
    "refit_no_decay": {"time_decay": 0.0},
    "refit_no_reg": {"regularization": 0.0},
    "refit_global_baseline": {"per_league_baselines": False},
    "refit_max_goals_6": {"max_goals": 6},
}
# `refit_legacy` runs by default: without it there is no arm that isolates
# the fitting fixes from the corpus/coverage difference between the served
# 2023-vintage artifact and a fold-by-fold refit on the current corpus.
DEFAULT_CONFIGS = "served,refit_legacy,refit"

# Every baseline a challenger is scored against. `refit_legacy` is the
# FITTING-FIX baseline (same folds, same corpus, old fitter); `served` is kept
# only as "what production actually emits today". The *_dw baselines are the
# recency-weighted twins of the constants — see decay_weights for why scoring
# only the un-decayed ones would hand the challenger ~11% of the ship gate.
BASELINE_KEYS = (
    "served",
    "refit_legacy",
    "base_global",
    "base_global_dw",
    "base_league",
    "base_league_dw",
    "market",
)


# ── SQL ──────────────────────────────────────────────────────────────

MATCHES_SQL = """
    SELECT
        m.id::text          AS match_id,
        m.match_date,
        l.id::text          AS league_id,
        l.name              AS league,
        ht.name             AS home_team,
        at.name             AS away_team,
        m.home_score::float AS home_score,
        m.away_score::float AS away_score
    FROM matches m
    JOIN leagues l  ON l.id = m.league_id AND l.sport = 'soccer'
    JOIN teams   ht ON ht.id = m.home_team_id
    JOIN teams   at ON at.id = m.away_team_id
    WHERE m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.match_date < %(end)s
    ORDER BY m.match_date
"""

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

# Closing line per (match, bookmaker, selection): the latest non-live
# quote at or before kickoff.
ODDS_SQL = """
    SELECT DISTINCT ON (o.match_id, o.bookmaker, o.selection)
        o.match_id::text        AS match_id,
        o.bookmaker             AS bookmaker,
        lower(o.selection)      AS selection,
        o.odds_decimal::float   AS odds_decimal
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
      AND o.is_live = false
      AND o.odds_decimal IS NOT NULL
      AND o.odds_decimal > 1.0
      AND o.timestamp <= m.match_date
      AND o.market_type = %(market_type)s
      AND (%(line)s::numeric IS NULL OR o.line = %(line)s::numeric)
    ORDER BY o.match_id, o.bookmaker, o.selection, o.timestamp DESC
"""


# ── Pure: fold construction ──────────────────────────────────────────


@dataclass(frozen=True)
class Fold:
    """One eval month. ``month_start`` is BOTH the fit cutoff (fit uses
    match_date < month_start, strictly) and the time-decay reference
    point, per the protocol."""

    month_start: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp  # exclusive

    @property
    def fit_cutoff(self) -> pd.Timestamp:
        return self.month_start

    @property
    def label(self) -> str:
        return self.month_start.strftime("%Y-%m")


def _utc(ts: Any) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def month_floor(ts: Any) -> pd.Timestamp:
    t = _utc(ts)
    return pd.Timestamp(year=t.year, month=t.month, day=1, tz="UTC")


def next_month(ts: pd.Timestamp) -> pd.Timestamp:
    t = month_floor(ts)
    return pd.Timestamp(year=t.year + (t.month == 12), month=1 if t.month == 12 else t.month + 1, day=1, tz="UTC")


def build_folds(start: Any, end: Any) -> List[Fold]:
    """Monthly folds covering [start, end] INCLUSIVE of the end date.

    Each fold's eval window is clipped to the requested span, but the fit
    cutoff stays at the month boundary — which is <= eval_start, so a fold
    can never see its own eval month (or any later match) while fitting.
    """
    span_start = _utc(start)
    span_end_exclusive = _utc(end) + pd.Timedelta(days=1)
    if span_end_exclusive <= span_start:
        raise ValueError(f"--end ({end}) must be on or after --start ({start})")

    folds: List[Fold] = []
    cursor = month_floor(span_start)
    while cursor < span_end_exclusive:
        nxt = next_month(cursor)
        eval_start = max(cursor, span_start)
        eval_end = min(nxt, span_end_exclusive)
        if eval_end > eval_start:
            folds.append(Fold(month_start=cursor, eval_start=eval_start, eval_end=eval_end))
        cursor = nxt
    return folds


def split_fold(frame: pd.DataFrame, fold: Fold) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(fit, eval) for one fold. Fit is STRICTLY before the fold's month
    start; eval is the clipped month window. The two are disjoint by
    construction and the eval month is never in its own fit."""
    md = frame["match_date"]
    fit = frame[md < fold.fit_cutoff]
    ev = frame[(md >= fold.eval_start) & (md < fold.eval_end)]
    return fit, ev


# ── Pure: scoring ────────────────────────────────────────────────────


def brier_binary(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match squared error for a binary event. Returned per-match
    (not averaged) because every comparison downstream is paired."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return (p - y) ** 2


def brier_multiclass(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Per-match multi-class Brier: sum_k (p_k - onehot_k)^2."""
    proba = np.asarray(proba, dtype=float)
    y_idx = np.asarray(y_idx, dtype=int)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return np.sum((proba - onehot) ** 2, axis=1)


def paired_delta(
    challenger: Sequence[float],
    baseline: Sequence[float],
    folds: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Paired mean difference (challenger - baseline) of PER-MATCH scores,
    with BOTH a per-match SE and a fold-CLUSTERED SE.

    The per-match SE is sd(per-match difference)/sqrt(n), NOT built from
    the two arms' independent means: the arms are scored on identical
    matches and are heavily correlated, so an unpaired SE would be far too
    wide and would mask both real wins and real regressions.

    But that SE treats ~10k per-match differences as ~10k independent
    draws, and they are not. The challenger is only ONE fitted model per
    monthly fold, so every match inside a fold shares that fit's error: a
    fold whose league baselines happen to sit well moves ~775 differences
    together. Left uncorrected, the per-match SE runs ~0.0006 at this n,
    which makes the §4.3 gate's second condition (95% upper bound below
    zero) automatic for any delta that already clears -0.005 — the two
    conditions collapse into one and the noise check stops doing its job.

    So when ``folds`` is supplied (one fold label per row) the fold-level
    mean differences are also returned:
      * ``delta_cluster`` — the unweighted mean of the per-fold means.
      * ``se_cluster``    — sd(fold means)/sqrt(n_folds).
    ``ship_verdict`` requires the CLUSTERED upper bound to be negative,
    which is a bound a lucky month cannot buy.
    """
    c = np.asarray(challenger, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if c.shape != b.shape:
        raise ValueError(f"paired_delta needs aligned arrays, got {c.shape} vs {b.shape}")
    mask = np.isfinite(c) & np.isfinite(b)
    fold_labels = np.asarray(folds, dtype=object)[mask] if folds is not None else None
    c, b = c[mask], b[mask]
    n = int(c.size)
    empty = {
        "n": 0,
        "challenger_brier": None,
        "baseline_brier": None,
        "delta": None,
        "se": None,
        "t": None,
        "delta_cluster": None,
        "se_cluster": None,
        "n_folds": None,
    }
    if n == 0:
        return empty
    diff = c - b
    delta = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    t = float(delta / se) if se and math.isfinite(se) and se > 0 else float("nan")
    delta_cluster: Optional[float] = None
    se_cluster: Optional[float] = None
    n_folds: Optional[int] = None
    if fold_labels is not None and fold_labels.size:
        labels = list(dict.fromkeys(fold_labels.tolist()))
        n_folds = len(labels)
        fold_means = np.array([float(diff[fold_labels == lab].mean()) for lab in labels], dtype=float)
        delta_cluster = float(fold_means.mean())
        # <2 folds cannot estimate between-fold spread; report None rather
        # than a zero SE that would read as certainty.
        if n_folds > 1:
            se_cluster = float(fold_means.std(ddof=1) / math.sqrt(n_folds))
    return {
        "n": n,
        "challenger_brier": float(c.mean()),
        "baseline_brier": float(b.mean()),
        "delta": delta,
        "se": se,
        "t": t,
        "delta_cluster": delta_cluster,
        "se_cluster": se_cluster,
        "n_folds": n_folds,
    }


def ship_verdict(
    delta: Optional[float],
    se: Optional[float],
    baseline: str,
    threshold: float = SHIP_THRESHOLD,
    delta_cluster: Optional[float] = None,
    se_cluster: Optional[float] = None,
    n_folds: Optional[int] = None,
) -> Dict[str, Any]:
    """§4.3 gate, and it NAMES the baseline because the answer differs by
    baseline.

    SHIP requires all of:
      * delta <= threshold (-0.005),
      * the per-match paired 95% upper bound (delta + 2*se) below zero, and
      * the FOLD-CLUSTERED 95% upper bound (delta_cluster + 2*se_cluster)
        below zero, whenever a clustered SE could be computed.

    The clustered condition is the one that actually bites. At n~10k the
    per-match SE is ~0.0006, so `delta + 2*se < 0` is automatic for
    anything that already clears -0.005 and the noise check degenerates
    into a restatement of the threshold. The challenger is 13 fitted
    models, not 10,072 independent ones; sd(fold means)/sqrt(13) is the
    spread a lucky month cannot buy. A run that cannot produce a clustered
    SE (single fold) reports that in the reason rather than passing on the
    per-match bound alone.
    """
    if delta is None or se is None or not math.isfinite(delta) or not math.isfinite(se):
        return {
            "baseline": baseline,
            "ship": False,
            "reason": "insufficient data",
            "line": f"DO-NOT-SHIP (vs {baseline}): insufficient paired data",
        }
    upper = delta + 2.0 * se
    meets_threshold = delta <= threshold
    clear_of_noise = upper < 0.0
    have_cluster = (
        delta_cluster is not None
        and se_cluster is not None
        and math.isfinite(delta_cluster)
        and math.isfinite(se_cluster)
    )
    upper_cluster = (float(delta_cluster) + 2.0 * float(se_cluster)) if have_cluster else None
    clear_of_fold_noise = bool(upper_cluster is not None and upper_cluster < 0.0)
    ship = bool(meets_threshold and clear_of_noise and (clear_of_fold_noise if have_cluster else False))
    cluster_note = (
        f"clustered 95% upper bound {upper_cluster:+.5f} (n_folds={n_folds})"
        if upper_cluster is not None
        else "no fold-clustered SE (needs >= 2 folds) — the per-match bound alone is not enough to ship"
    )
    if ship:
        reason = (
            f"dBrier {delta:+.5f} <= {threshold:+.4f}, per-match 95% upper bound {upper:+.5f} < 0, "
            f"and {cluster_note} < 0"
        )
    elif not meets_threshold:
        reason = f"dBrier {delta:+.5f} does not clear the {threshold:+.4f} gate"
    elif not clear_of_noise:
        reason = f"dBrier {delta:+.5f} meets the gate but 95% upper bound {upper:+.5f} is not clear of zero"
    elif not have_cluster:
        reason = f"dBrier {delta:+.5f} clears the gate and the per-match bound, but there is " f"{cluster_note}"
    else:
        reason = (
            f"dBrier {delta:+.5f} clears the gate and the per-match bound, but the {cluster_note} "
            "is not clear of zero"
        )
    verb = "SHIP" if ship else "DO-NOT-SHIP"
    return {
        "baseline": baseline,
        "ship": ship,
        "delta": delta,
        "se": se,
        "upper_95": upper,
        "delta_cluster": delta_cluster,
        "se_cluster": se_cluster,
        "n_folds": n_folds,
        "upper_95_cluster": upper_cluster,
        "reason": reason,
        "line": f"{verb} (vs {baseline}): {reason}",
    }


# ── Pure: walk-forward base-rate baselines ───────────────────────────


@dataclass
class BaseRates:
    """Walk-forward base rates. Built ONLY from the fold's fit frame —
    never from the eval month — so the "constant" comparator sees exactly
    the information the model it judges sees, INCLUDING the recency
    weighting (see ``decay_weights``)."""

    global_rate: float
    by_league: Dict[str, float] = field(default_factory=dict)
    k: float = 200.0
    n_fit: int = 0
    decay: float = 0.0
    effective_n: float = 0.0

    def predict(self, leagues: Sequence[Any]) -> np.ndarray:
        return np.array([self.by_league.get(lg, self.global_rate) for lg in leagues], dtype=float)


def decay_weights(fit_df: pd.DataFrame, reference: Any, decay: float, date_col: str = "match_date") -> np.ndarray:
    """w = exp(-decay * age_days) aged against ``reference``, matching what
    PoissonMatchPredictor._time_weights does inside the MLE.

    This exists so the CONSTANT comparators can be given the same weighting
    the challenger gets. Denying it to them is not neutrality: at the first
    fold cutoff the unweighted over-2.5 rate and the 730-day-half-life
    weighted rate differ by ~0.011 of probability, worth ~0.0005 of Brier
    against a realised rate near 0.524 — about 11% of the whole -0.005 ship
    gate, handed to the challenger by a protocol choice rather than earned.
    """
    n = len(fit_df)
    if decay <= 0 or n == 0 or date_col not in fit_df.columns:
        return np.ones(n, dtype=float)
    dates = pd.to_datetime(fit_df[date_col], errors="coerce", utc=True)
    if bool(dates.isna().all()):
        return np.ones(n, dtype=float)
    ref = _utc(reference) if reference is not None else dates.max()
    age_days = (ref - dates).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    if np.isnan(age_days).any():
        # An undated row is treated as the OLDEST, never the newest.
        age_days = np.where(np.isnan(age_days), float(np.nanmax(age_days)), age_days)
    return np.exp(-float(decay) * np.clip(age_days, 0.0, None))


def _weights_for(fit_df: pd.DataFrame, weights: Optional[Sequence[float]]) -> np.ndarray:
    if weights is None:
        return np.ones(len(fit_df), dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(fit_df),):
        raise ValueError(f"weights must align with fit_df, got {w.shape} vs {len(fit_df)}")
    return np.clip(w, 0.0, None)


def fit_base_rates(
    fit_df: pd.DataFrame,
    outcome_col: str,
    league_col: str = "league",
    k: float = 200.0,
    weights: Optional[Sequence[float]] = None,
    decay: float = 0.0,
) -> BaseRates:
    """Per-league rate shrunk toward the global rate with ``k`` effective
    matches: (sum_w*y + k*global) / (sum_w + k).

    k~200 keeps Copa Libertadores (n=16) and Chile (n=47) — and any
    brand-new league — from inheriting a noisy or empty estimate, while
    letting Eredivisie (n=5,741) sit essentially at its own rate.

    ``weights`` (normally ``decay_weights(...)``) makes the comparator
    recency-matched to the challenger. Passing None gives the un-decayed
    constant; the harness scores BOTH and prints them side by side, so the
    verdict is never stated only against the handicapped one.
    """
    if len(fit_df) == 0:
        return BaseRates(global_rate=0.5, by_league={}, k=k, n_fit=0, decay=decay, effective_n=0.0)
    w = _weights_for(fit_df, weights)
    y = fit_df[outcome_col].astype(float).to_numpy()
    total_w = float(w.sum())
    if total_w <= 0:
        raise ValueError("base-rate weights summed to zero")
    global_rate = float((w * y).sum() / total_w)
    by_league: Dict[str, float] = {}
    if league_col in fit_df.columns:
        grouped = pd.DataFrame({"league": fit_df[league_col].to_numpy(), "w": w, "wy": w * y}).dropna(subset=["league"])
        for league, sub in grouped.groupby("league", sort=False):
            n_w = float(sub["w"].sum())
            by_league[league] = float((float(sub["wy"].sum()) + k * global_rate) / (n_w + k))
    return BaseRates(
        global_rate=global_rate,
        by_league=by_league,
        k=k,
        n_fit=int(len(fit_df)),
        decay=float(decay),
        effective_n=total_w,
    )


def fit_class_base_rates(
    fit_df: pd.DataFrame,
    outcome_col: str,
    n_classes: int,
    league_col: str = "league",
    k: float = 200.0,
    weights: Optional[Sequence[float]] = None,
    decay: float = 0.0,
) -> Dict[str, Any]:
    """Multi-class version for the 1x2 readout — same shrinkage and the
    same optional recency weighting, applied per class. Returns
    {'global': (n_classes,), 'by_league': {...}}."""
    if len(fit_df) == 0:
        uniform = np.full(n_classes, 1.0 / n_classes)
        return {"global": uniform, "by_league": {}, "k": k, "n_fit": 0, "decay": decay, "effective_n": 0.0}
    w = _weights_for(fit_df, weights)
    y = fit_df[outcome_col].astype(int).to_numpy()
    counts = np.bincount(y, weights=w, minlength=n_classes).astype(float)
    total_w = float(counts.sum())
    if total_w <= 0:
        raise ValueError("base-rate weights summed to zero")
    global_rate = counts / total_w
    by_league: Dict[str, np.ndarray] = {}
    if league_col in fit_df.columns:
        leagues = fit_df[league_col].to_numpy()
        for league in pd.unique(pd.Series(leagues).dropna()):
            sel = leagues == league
            c = np.bincount(y[sel], weights=w[sel], minlength=n_classes).astype(float)
            by_league[league] = (c + k * global_rate) / (c.sum() + k)
    return {
        "global": global_rate,
        "by_league": by_league,
        "k": k,
        "n_fit": int(len(fit_df)),
        "decay": float(decay),
        "effective_n": total_w,
    }


def predict_class_base_rates(rates: Dict[str, Any], leagues: Sequence[Any]) -> np.ndarray:
    glob = np.asarray(rates["global"], dtype=float)
    by_league = rates["by_league"]
    return np.vstack([np.asarray(by_league.get(lg, glob), dtype=float) for lg in leagues])


# ── Pure: market de-vig ──────────────────────────────────────────────


def devig(odds: Sequence[float]) -> Optional[np.ndarray]:
    """Proportional (multiplicative) de-vig of a complete quote set."""
    arr = np.asarray(list(odds), dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 1.0):
        return None
    raw = 1.0 / arr
    total = float(raw.sum())
    if total <= 0:
        return None
    return raw / total


@dataclass(frozen=True)
class MarketQuote:
    """One match's de-vigged consensus, WITH the depth it was built from.

    ``n_books`` and ``book_sum_median`` (the median raw implied-probability
    sum, i.e. 1 + overround) are carried rather than discarded because they
    are what tells a reader whether this "ceiling" is one. A proportional
    de-vig of a ~6% overround off a handful of books on a small,
    non-random subset of matches is a reference, not a ceiling.
    """

    probs: np.ndarray
    n_books: int
    book_sum_median: float


def consensus_devigged(quotes: Sequence[Sequence[float]], min_books: int = 1) -> Optional[MarketQuote]:
    """Median across books of each book's own de-vigged probability
    vector. De-vigging per book BEFORE averaging keeps a wide-margin book
    from dragging the consensus; the median then blunts stale quotes.
    Renormalised because a per-element median need not sum to 1.

    ``min_books`` drops matches too thin to be a consensus at all — a
    single book is one book's opinion plus its own margin, and scoring it
    as "the market" flatters or damns the model by that book's vig.
    """
    raw_sums: List[float] = []
    vecs: List[np.ndarray] = []
    for quote in quotes:
        vec = devig(quote)
        if vec is None:
            continue
        vecs.append(vec)
        raw_sums.append(float(np.sum(1.0 / np.asarray(list(quote), dtype=float))))
    if len(vecs) < max(1, int(min_books)):
        return None
    med = np.median(np.vstack(vecs), axis=0)
    total = float(med.sum())
    if total <= 0:
        return None
    return MarketQuote(probs=med / total, n_books=len(vecs), book_sum_median=float(np.median(raw_sums)))


# ── Pure: ensemble blend swap ────────────────────────────────────────


def blend_swap_1x2(
    p_ensemble: np.ndarray, p_member_old: np.ndarray, p_member_new: np.ndarray, weight: float
) -> np.ndarray:
    """Replace one weighted-average member's contribution in a blended
    probability vector.

    The ensemble emits sum_i w_i p_i / sum_i w_i, so swapping member m is
    exactly ``p + w_eff * (p_new - p_old)`` with w_eff the member's
    normalised weight. Used instead of re-deriving 1x2 from the
    reconciled matrix, because after IPF the derived 1x2 IS the target by
    construction and would show no effect at all.
    """
    p = np.asarray(p_ensemble, dtype=float) + float(weight) * (
        np.asarray(p_member_new, dtype=float) - np.asarray(p_member_old, dtype=float)
    )
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


# ── Impure: DB loading ───────────────────────────────────────────────


def _connect(database_url: str):
    import psycopg2

    return psycopg2.connect(database_url)


def load_matches(database_url: str, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    """Every finished soccer match before ``end_exclusive`` — the fit
    corpus AND the eval population come out of this one frame so the two
    can never disagree about a match's date, league or score."""
    with _connect(database_url) as conn:
        frame = pd.read_sql(MATCHES_SQL, conn, params={"end": end_exclusive.to_pydatetime()})
    if frame.empty:
        raise RuntimeError("No finished soccer matches returned — check --database-url and the corpus.")
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    frame["league"] = frame["league"].fillna("__unknown__")
    # poisson_models resolves a league key from the first of
    # ("league_id", "league", "league_name") that carries real values, and
    # the serve path (scripts/precompute_predictions.py) passes league_id.
    # Carrying all three keeps the harness on the identifier production
    # actually uses while leaving readable names for the report.
    frame["league_id"] = frame["league_id"].fillna(frame["league"])
    frame["league_name"] = frame["league"]
    frame["total_goals"] = frame["home_score"] + frame["away_score"]
    frame["over_2_5"] = (frame["total_goals"] > 2.5).astype(int)
    frame["btts"] = ((frame["home_score"] >= 1) & (frame["away_score"] >= 1)).astype(int)
    # 1x2 index order matches TaskSpec labels ["home", "draw", "away"].
    frame["outcome_1x2"] = np.where(
        frame["home_score"] > frame["away_score"], 0, np.where(frame["home_score"] == frame["away_score"], 1, 2)
    )
    return frame.sort_values("match_date").reset_index(drop=True)


def load_features(database_url: str, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> Dict[str, dict]:
    with _connect(database_url) as conn:
        rows = pd.read_sql(
            FEATURES_SQL, conn, params={"start": start.to_pydatetime(), "end": end_exclusive.to_pydatetime()}
        )
    return {r["match_id"]: (r["features"] or {}) for _, r in rows.iterrows()}


def load_market_consensus(
    database_url: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    market_type: str,
    selections: Sequence[str],
    line: Optional[float],
    min_books: int = 1,
) -> Dict[str, MarketQuote]:
    """match_id -> de-vigged consensus (probability vector ordered to match
    ``selections``, plus book depth and median overround). Books that don't
    quote the complete set are dropped (a partial quote can't be de-vigged),
    and matches with fewer than ``min_books`` usable books are dropped
    entirely rather than scored as a one-book "consensus"."""
    with _connect(database_url) as conn:
        rows = pd.read_sql(
            ODDS_SQL,
            conn,
            params={
                "start": start.to_pydatetime(),
                "end": end_exclusive.to_pydatetime(),
                "market_type": market_type,
                "line": line,
            },
        )
    out: Dict[str, MarketQuote] = {}
    if rows.empty:
        return out
    wanted = list(selections)
    for match_id, grp in rows.groupby("match_id"):
        quotes: List[List[float]] = []
        for _book, bg in grp.groupby("bookmaker"):
            price = dict(zip(bg["selection"], bg["odds_decimal"]))
            if all(s in price for s in wanted):
                quotes.append([float(price[s]) for s in wanted])
        quote = consensus_devigged(quotes, min_books=min_books)
        if quote is not None:
            out[str(match_id)] = quote
    return out


def market_depth_summary(quotes: Dict[str, MarketQuote], eval_ids: Sequence[str]) -> Dict[str, Any]:
    """Coverage + depth + overround for one market's consensus, so the
    report can say how thin the "ceiling" actually is instead of printing a
    Brier next to it and letting the reader assume depth.

    Restricted to ``eval_ids`` — the matches actually scored — because a
    quote for a match that never entered the eval set is not coverage.
    """
    empty = {
        "coverage": None,
        "n_matches": 0,
        "books_median": None,
        "books_min": None,
        "books_max": None,
        "overround_median": None,
    }
    ids = [str(i) for i in eval_ids]
    if not ids:
        return empty
    hits = [quotes[i] for i in ids if i in quotes]
    books = np.array([q.n_books for q in hits], dtype=float)
    sums = np.array([q.book_sum_median for q in hits], dtype=float)
    return {
        "coverage": float(len(hits) / len(ids)),
        "n_matches": int(len(hits)),
        "books_median": (float(np.median(books)) if books.size else None),
        "books_min": (int(books.min()) if books.size else None),
        "books_max": (int(books.max()) if books.size else None),
        "overround_median": (float(np.median(sums) - 1.0) if sums.size else None),
    }


# ── Impure: model plumbing ───────────────────────────────────────────


def load_production_ensemble():
    """The live soccer ensemble, reconstituted from the production
    registry exactly as the API does. Gives us both the served
    Dixon-Coles member and the 1x2 reconciliation target in one load."""
    from pathlib import Path

    from services.prediction_service import TASKS, _build_ensemble_for_task, _klass_registry  # type: ignore

    task = TASKS[SOCCER_TASK_KEY]
    ensemble, meta = _build_ensemble_for_task(Path(PRODUCTION_MODEL_PATH), task, _klass_registry())
    if ensemble is None:
        raise RuntimeError(
            f"No soccer ensemble under {PRODUCTION_MODEL_PATH}/{task.ensemble_name}/. "
            "This harness must run where the production artifacts are mounted."
        )
    return ensemble, task, meta


def member_weight(ensemble, name: str = DC_MEMBER) -> float:
    """The member's NORMALISED blend weight (ensemble.predict_proba
    divides by the surviving weight total, so raw weights need not sum
    to 1)."""
    weights = getattr(ensemble, "weights", {}) or {}
    total = float(sum(float(w) for w in weights.values())) or 1.0
    return float(weights.get(name, 0.0)) / total


def _supports_league(fn) -> bool:
    try:
        return "league" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C funcs
        return False


def fit_dixon_coles(fit_df: pd.DataFrame, overrides: Dict[str, Any], reference_date: pd.Timestamp):
    """Refit a DixonColesPredictor on the fold's fit frame.

    The predictor module is imported here, not at module import time, so
    the pure pieces stay unit-testable without ml-models on the path.
    The concurrent poisson_models change owns the fitting semantics —
    this harness only supplies hyperparameters, the frame and the decay
    reference point.
    """
    from dataclasses import replace as dc_replace

    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore
    from predictors.poisson_models import DixonColesPredictor  # type: ignore

    hyper = dict(DIXON_COLES_CONFIG.hyperparameters)
    hyper.update(overrides)
    cfg = dc_replace(DIXON_COLES_CONFIG, hyperparameters=hyper)
    model = DixonColesPredictor(cfg)
    # The predictor ages its decay weights against the fit frame's own max
    # match_date. Soccer plays somewhere every day, so that max sits within
    # a day or two of the fold's month start; reference_date is passed
    # through anyway so a trainer that honours it uses the exact boundary.
    wanted = (
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "match_date",
        "league_id",
        "league",
        "league_name",
    )
    cols = [c for c in wanted if c in fit_df.columns]
    model.train(fit_df[cols].copy(), reference_date=reference_date)
    return model


def arm_probabilities(
    model,
    ev: pd.DataFrame,
    targets_1x2: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Serve-path probabilities for one arm on one eval fold.

    Mirrors scripts/precompute_predictions.py: lambdas from the model,
    then ``derive_from_lambdas(..., target_1x2=<ensemble 1x2>)`` so the
    scoreline matrix is IPF-reconciled before over/under and BTTS are
    read off. Also returns the model's OWN (unreconciled) 1x2 as the
    ensemble member would emit it, for the blend-swap readout.

    ``known_teams`` (0, 1 or 2 per match) is returned because it is the
    single biggest confound in this experiment. An arm that does not know
    either team emits its baseline constant for that match, so an arm
    fitted on a small old corpus and an arm refitted on the current one are
    not being compared on the same object at all on those rows — the
    difference there is corpus and team coverage, not the fitting fixes.
    ``run`` reports per-arm coverage and states the fitting verdict on the
    both-known stratum.
    """
    from predictors.market_derivation import MAX_GOALS_DERIVE, derive_from_lambdas  # type: ignore

    rho = float(getattr(model, "rho", 0.0) or 0.0)
    use_league = _supports_league(model.lambdas_for_match)

    p_over = np.full(len(ev), np.nan)
    p_btts = np.full(len(ev), np.nan)
    home = ev["home_team"].to_numpy()
    away = ev["away_team"].to_numpy()
    # league_id is the key the production serve path passes;
    # poisson_models resolves it ahead of the name columns.
    leagues = ev["league_id"].to_numpy()

    for i in range(len(ev)):
        if use_league:
            h_lam, a_lam = model.lambdas_for_match(home[i], away[i], league=leagues[i])
        else:
            h_lam, a_lam = model.lambdas_for_match(home[i], away[i])
        markets = derive_from_lambdas(
            float(h_lam),
            float(a_lam),
            rho,
            target_1x2=tuple(float(x) for x in targets_1x2[i]),
            max_goals=MAX_GOALS_DERIVE,
        )
        p_over[i] = markets["over_under"][f"over_{PRIMARY_LINE}"]
        p_btts[i] = markets["btts"]["yes"]

    member_x = pd.DataFrame({"home_team": home, "away_team": away, "league_id": leagues})
    member_1x2 = np.asarray(model.predict_proba(member_x), dtype=float)
    attack = getattr(model, "team_attack", {}) or {}
    known_teams = np.array(
        [int(home[i] in attack) + int(away[i] in attack) for i in range(len(ev))],
        dtype=float,
    )
    return {"over": p_over, "btts": p_btts, "member_1x2": member_1x2, "known_teams": known_teams}


def ensemble_targets_1x2(ensemble, ev: pd.DataFrame, features: Dict[str, dict], batch: int = 1000) -> np.ndarray:
    """The ensemble's blended (home, draw, away) for each eval match —
    the IPF target the serve path reconciles to.

    Identical across every arm by construction, so it is a shared input
    to the paired comparison and cannot tilt the result. Matches without
    a features_cache row are dropped upstream (see ``run``); we fail
    loudly here rather than silently substituting a prior.
    """
    rows: List[dict] = []
    for match_id, home, away in zip(ev["match_id"], ev["home_team"], ev["away_team"]):
        feats = dict(features.get(str(match_id)) or {})
        payload: Dict[str, Any] = {}
        for k, v in feats.items():
            payload[k] = v
            prefixed = k if k.startswith("feature__") else f"feature__{k}"
            payload.setdefault(prefixed, v)
        # NOTE: production does NOT do this. scripts/precompute_predictions.py
        # builds its ensemble frame from the features_cache JSONB plus
        # feature__ mirrors only, and no feature key carries a team name — so
        # there the Dixon-Coles and Poisson members serve their unknown-team
        # constant inside the blend. Injecting the names here gives a
        # team-aware 1x2 target, which is a better IPF target but NOT the
        # served one. It is identical across arms, so paired deltas are
        # unaffected; the absolute numbers are not production's.
        payload["home_team"] = home
        payload["away_team"] = away
        rows.append(payload)

    out = np.empty((len(rows), 3))
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        proba = np.asarray(ensemble.predict_proba(pd.DataFrame(chunk)), dtype=float)
        if proba.shape[1] != 3:
            raise RuntimeError(f"Soccer ensemble emitted {proba.shape[1]} classes, expected 3")
        out[start : start + len(chunk)] = proba
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Ensemble emitted non-finite 1x2 probabilities — refusing to score against a broken target")
    return out


# ── Reporting ────────────────────────────────────────────────────────


def summarise(scores: np.ndarray, probs: np.ndarray, realised: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(scores) & np.isfinite(probs)
    if not mask.any():
        return {"n": 0, "brier": None, "mean_predicted": None, "realised": None, "bias": None}
    p = probs[mask]
    y = realised[mask].astype(float)
    return {
        "n": int(mask.sum()),
        "brier": float(scores[mask].mean()),
        "mean_predicted": float(p.mean()),
        "realised": float(y.mean()),
        "bias": float(p.mean() - y.mean()),
    }


def _fmt(v: Optional[float], width: int = 9, places: int = 5) -> str:
    return " " * width if v is None else f"{v:>{width}.{places}f}"


def format_market_section(
    name: str,
    primary: bool,
    arms: Dict[str, Dict[str, Any]],
    deltas: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    lines = [""]
    lines.append(f"--- MARKET: {name}{'  (PRIMARY)' if primary else ''} ---")
    lines.append(f"{'config':<24}{'n':>7}{'brier':>10}{'mean_pred':>11}{'realised':>10}{'bias':>10}")
    for key, s in arms.items():
        lines.append(
            f"{key:<24}{s['n']:>7}{_fmt(s['brier'], 10)}{_fmt(s['mean_predicted'], 11, 4)}"
            f"{_fmt(s['realised'], 10, 4)}{_fmt(s['bias'], 10, 4)}"
        )
    for challenger, per_baseline in deltas.items():
        lines.append("")
        lines.append(f"paired dBrier: '{challenger}' minus baseline (per-match differences)")
        lines.append(f"{'baseline':<24}{'n':>7}{'dBrier':>11}{'SE':>10}{'SE_fold':>10}{'t':>8}  verdict")
        for baseline, d in per_baseline.items():
            verdict = d.get("verdict", {})
            lines.append(
                f"{baseline:<24}{d['n']:>7}{_fmt(d['delta'], 11)}{_fmt(d['se'], 10)}"
                f"{_fmt(d.get('se_cluster'), 10)}{_fmt(d['t'], 8, 2)}  "
                f"{'SHIP' if verdict.get('ship') else 'DO-NOT-SHIP'}"
            )
    return lines


STRATA = (
    ("both_known", 2.0),
    ("one_known", 1.0),
    ("neither_known", 0.0),
)


def stratified_deltas(
    scores: Dict[str, np.ndarray],
    configs: Sequence[str],
    known_teams: np.ndarray,
    fold_labels: np.ndarray,
) -> Tuple[Dict[str, Any], List[str]]:
    """Re-read the primary market split by how many teams the SERVED arm
    knows, because that is the confound the headline table cannot separate.

    On rows where an arm knows neither team it emits a constant, so a
    served-vs-refit delta there is a statement about corpus size and team
    coverage — a real and important effect, but a DIFFERENT one from the
    fitting fixes this harness is named after. The both-known stratum is the
    only one on which the two arms are predicting the same kind of object,
    and it is where the fitting-fix verdict belongs.
    """
    block: Dict[str, Any] = {}
    lines: List[str] = ["", "--- PRIMARY market by SERVED-arm team coverage ---"]
    lines.append("(strata defined by the served artifact, the arm whose coverage is in question)")
    for label, value in STRATA:
        mask = known_teams == value
        n = int(mask.sum())
        stratum: Dict[str, Any] = {"n": n, "arms": {}, "deltas": {}}
        block[label] = stratum
        lines.append("")
        lines.append(f"[{label}] n={n}")
        if n == 0:
            lines.append("  (no rows)")
            continue
        for key, per_match in scores.items():
            vals = per_match[mask]
            finite = np.isfinite(vals)
            stratum["arms"][key] = float(vals[finite].mean()) if finite.any() else None
        lines.append(f"  {'arm':<24}{'brier':>10}")
        for key, brier in stratum["arms"].items():
            lines.append(f"  {key:<24}{_fmt(brier, 10)}")
        for challenger in configs:
            if challenger == "served" or challenger not in scores:
                continue
            per_baseline: Dict[str, Dict[str, Any]] = {}
            for baseline in BASELINE_KEYS:
                if baseline not in scores or baseline == challenger:
                    continue
                d = paired_delta(scores[challenger][mask], scores[baseline][mask], folds=fold_labels[mask])
                d["verdict"] = ship_verdict(
                    d["delta"],
                    d["se"],
                    baseline,
                    delta_cluster=d["delta_cluster"],
                    se_cluster=d["se_cluster"],
                    n_folds=d["n_folds"],
                )
                per_baseline[baseline] = d
                if label == "both_known":
                    lines.append(f"  [{label}] {challenger}: {d['verdict']['line']}")
            stratum["deltas"][challenger] = per_baseline
    return block, lines


# ── Orchestration ────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIG_SPECS]
    if unknown:
        raise ValueError(f"Unknown --configs entries {unknown}; known: {sorted(CONFIG_SPECS)}")
    # 'served' is the reference arm for the blend-swap 1x2 readout and
    # baseline (a), so it always runs and always runs FIRST.
    configs = ["served"] + [c for c in configs if c != "served"]

    span_start = _utc(args.start)
    span_end_exclusive = _utc(args.end) + pd.Timedelta(days=1)
    folds = build_folds(args.start, args.end)

    frame = load_matches(args.database_url, span_end_exclusive)
    features = load_features(args.database_url, span_start, span_end_exclusive)
    min_books = int(args.min_books)
    market_over = load_market_consensus(
        args.database_url, span_start, span_end_exclusive, "over_under", ("over", "under"), PRIMARY_LINE, min_books
    )
    market_btts = load_market_consensus(
        args.database_url, span_start, span_end_exclusive, "btts", ("yes", "no"), None, min_books
    )
    market_1x2 = load_market_consensus(
        args.database_url, span_start, span_end_exclusive, "1x2", ("home", "draw", "away"), None, min_books
    )

    ensemble, _task, meta = load_production_ensemble()
    dc_weight = member_weight(ensemble)
    served_dc = ensemble.models.get(DC_MEMBER)
    if served_dc is None or not getattr(served_dc, "is_fitted", False):
        raise RuntimeError(f"Production ensemble has no fitted '{DC_MEMBER}' member — nothing to compare against")

    logger.info(
        "corpus=%d matches  folds=%d  eval span %s..%s  dc_blend_weight=%.4f",
        len(frame),
        len(folds),
        span_start.date(),
        (span_end_exclusive - pd.Timedelta(days=1)).date(),
        dc_weight,
    )

    # The comparator's recency weighting. Defaults to the challenger's own
    # --decay so the constants are held to the SAME weighting, not merely the
    # same rows; --base-rate-decay 0 restores the un-decayed protocol.
    base_rate_decay = float(args.decay if args.base_rate_decay is None else args.base_rate_decay)

    base_overrides = {
        "time_decay": float(args.decay),
        "regularization": float(args.reg),
        "max_goals": int(args.max_goals),
        "per_league_baselines": True,
        "league_shrinkage": float(args.league_shrinkage),
    }

    collected: Dict[str, List[np.ndarray]] = {}
    base_rate_effective_n: Dict[str, List[float]] = {}
    # Match ids in scoring order, so market coverage is measured against the
    # rows actually scored rather than against everything the odds query
    # returned.
    eval_ids: List[np.ndarray] = []

    def _collect(key: str, values: np.ndarray) -> None:
        collected.setdefault(key, []).append(np.asarray(values, dtype=float))

    for fold_index, fold in enumerate(folds):
        fit_df, ev = split_fold(frame, fold)
        ev = ev[ev["match_id"].astype(str).isin(features.keys())]
        if args.limit:
            ev = ev.head(int(args.limit))
        if ev.empty:
            logger.warning("fold %s: no eval matches with features — skipped", fold.label)
            continue
        if len(fit_df) < 500:
            logger.warning("fold %s: only %d fit rows — skipped", fold.label, len(fit_df))
            continue

        logger.info("fold %s: fit=%d eval=%d", fold.label, len(fit_df), len(ev))
        targets = ensemble_targets_1x2(ensemble, ev, features)

        y_over = ev["over_2_5"].to_numpy(dtype=float)
        y_btts = ev["btts"].to_numpy(dtype=float)
        y_1x2 = ev["outcome_1x2"].to_numpy(dtype=int)
        leagues = ev["league"].to_numpy()
        ids = ev["match_id"].astype(str).to_numpy()
        eval_ids.append(ids)

        _collect("y_over", y_over)
        _collect("y_btts", y_btts)
        _collect("y_1x2", y_1x2)
        # Fold label per row: the clustered SE needs to know which single
        # fitted model produced each of these ~775 correlated differences.
        _collect("fold", np.full(len(ev), fold_index, dtype=float))

        # (b) + (c): walk-forward constant baselines, built from THIS fold's
        # fit frame only, in BOTH weightings — un-decayed, and recency-matched
        # to the challenger's own decay. Printing only the un-decayed pair
        # would hand the challenger a comparator handicapped by a protocol
        # choice rather than by its own quality.
        w_fit = decay_weights(fit_df, fold.fit_cutoff, base_rate_decay)
        for suffix, weights, dec in (("", None, 0.0), ("_dw", w_fit, base_rate_decay)):
            r_over = fit_base_rates(fit_df, "over_2_5", k=args.shrink_k, weights=weights, decay=dec)
            r_btts = fit_base_rates(fit_df, "btts", k=args.shrink_k, weights=weights, decay=dec)
            r_1x2 = fit_class_base_rates(fit_df, "outcome_1x2", 3, k=args.shrink_k, weights=weights, decay=dec)
            _collect(f"over::base_global{suffix}", np.full(len(ev), r_over.global_rate))
            _collect(f"over::base_league{suffix}", r_over.predict(leagues))
            _collect(f"btts::base_global{suffix}", np.full(len(ev), r_btts.global_rate))
            _collect(f"btts::base_league{suffix}", r_btts.predict(leagues))
            _collect(f"1x2::base_global{suffix}", np.tile(np.asarray(r_1x2["global"], dtype=float), (len(ev), 1)))
            _collect(f"1x2::base_league{suffix}", predict_class_base_rates(r_1x2, leagues))
            base_rate_effective_n.setdefault(suffix or "flat", []).append(float(r_over.effective_n))

        # (d): de-vigged closing market where it exists (and where enough
        # books quoted it — see --min-books).
        _collect("over::market", np.array([market_over[i].probs[0] if i in market_over else np.nan for i in ids]))
        _collect("btts::market", np.array([market_btts[i].probs[0] if i in market_btts else np.nan for i in ids]))
        _collect(
            "1x2::market",
            np.vstack([market_1x2[i].probs if i in market_1x2 else np.full(3, np.nan) for i in ids]),
        )

        served_member_1x2: Optional[np.ndarray] = None
        for cfg_name in configs:
            spec = CONFIG_SPECS[cfg_name]
            if spec is None:
                model = served_dc
            else:
                overrides = dict(base_overrides)
                overrides.update(spec)
                model = fit_dixon_coles(fit_df, overrides, fold.fit_cutoff)
            out = arm_probabilities(model, ev, targets)
            if cfg_name == "served":
                served_member_1x2 = out["member_1x2"]
            _collect(f"over::{cfg_name}", out["over"])
            _collect(f"btts::{cfg_name}", out["btts"])
            _collect(f"known::{cfg_name}", out["known_teams"])
            assert served_member_1x2 is not None  # 'served' is always first
            _collect(
                f"1x2::{cfg_name}",
                blend_swap_1x2(targets, served_member_1x2, out["member_1x2"], dc_weight),
            )

    if "y_over" not in collected:
        raise RuntimeError("No fold produced eval rows — check --start/--end and features_cache coverage.")

    stacked = {k: (np.concatenate(v) if v[0].ndim == 1 else np.vstack(v)) for k, v in collected.items()}
    y_over = stacked["y_over"]
    y_btts = stacked["y_btts"]
    y_1x2 = stacked["y_1x2"].astype(int)
    # One label per eval row saying which monthly fit produced it — the
    # clustering unit for the SE that actually gates the verdict.
    fold_labels = stacked["fold"]
    scored_ids = np.concatenate(eval_ids) if eval_ids else np.array([], dtype=object)

    results: Dict[str, Any] = {
        "meta": {
            "start": str(span_start.date()),
            "end": str((span_end_exclusive - pd.Timedelta(days=1)).date()),
            "folds": [f.label for f in folds],
            "configs": configs,
            "decay_per_day": float(args.decay),
            "decay_half_life_days": (round(math.log(2) / float(args.decay), 1) if float(args.decay) > 0 else None),
            "regularization_effective_matches": float(args.reg),
            "max_goals": int(args.max_goals),
            "league_shrinkage": float(args.league_shrinkage),
            "base_rate_shrink_k": float(args.shrink_k),
            "base_rate_decay_per_day": base_rate_decay,
            "base_rate_effective_n": {
                key: (round(float(np.mean(vals)), 1) if vals else None) for key, vals in base_rate_effective_n.items()
            },
            "dc_blend_weight": dc_weight,
            "production_artifact": str(meta) if meta is not None else None,
            "ship_threshold": SHIP_THRESHOLD,
            "reference_noise_floor_se": REFERENCE_NOISE_FLOOR_SE,
            "n_eval": int(len(y_over)),
            # Team coverage per arm. `served` is expected to be far thinner
            # than any refit arm; that gap, not the fitting fixes, is what
            # `refit` minus `served` mostly measures.
            "arm_team_coverage": {
                cfg: {
                    "both_known": float((stacked[f"known::{cfg}"] == 2).mean()),
                    "one_known": float((stacked[f"known::{cfg}"] == 1).mean()),
                    "neither_known": float((stacked[f"known::{cfg}"] == 0).mean()),
                }
                for cfg in configs
                if f"known::{cfg}" in stacked
            },
            "min_books": min_books,
            "market_depth": {
                "over_2.5": market_depth_summary(market_over, scored_ids),
                "btts_yes": market_depth_summary(market_btts, scored_ids),
                "ensemble_1x2": market_depth_summary(market_1x2, scored_ids),
            },
            "market_coverage_over_2_5": float(np.isfinite(stacked["over::market"]).mean()),
        },
        "markets": {},
    }

    lines: List[str] = []
    lines.append("=== ab_soccer_dixon_coles_refit ===")
    lines.append(
        f"eval {results['meta']['start']}..{results['meta']['end']}  folds={len(folds)}  "
        f"n={results['meta']['n_eval']}  dc_blend_weight={dc_weight:.4f}"
    )
    lines.append("reconciliation target: production ensemble 1x2 (identical across arms; IPF via derive_from_lambdas)")
    lines.append(
        "  NOTE: that target is computed with home_team/away_team present. Production builds its "
        "ensemble frame from features_cache alone (no team columns), so the SERVED 1x2 differs. "
        "Shared across arms, so paired deltas are unaffected; absolute numbers are not the served ones."
    )
    lines.append(
        f"decay={args.decay}/day (half-life {results['meta']['decay_half_life_days']}d)  "
        f"reg={args.reg} eff. matches  league_shrinkage={args.league_shrinkage}  "
        f"base_rate_k={args.shrink_k}  base_rate_decay={base_rate_decay}/day  max_goals={args.max_goals}"
    )
    lines.append("")
    lines.append("team coverage per arm (both / one / neither team known to that arm):")
    for cfg, cov in results["meta"]["arm_team_coverage"].items():
        lines.append(
            f"  {cfg:<22}{cov['both_known']:>7.1%} both{cov['one_known']:>8.1%} one{cov['neither_known']:>8.1%} neither"
        )
    lines.append(
        "  An arm emits its baseline CONSTANT where it knows neither team, so 'refit' minus 'served' "
        "on the full population scores corpus + coverage as much as it scores any fitting fix. The "
        "fitting-fix verdict is the one against 'refit_legacy' (same folds, same corpus, old fitter) "
        "and the both-known stratum below."
    )
    lines.append("")
    lines.append(f"market reference (NOT a ceiling anyone can lean on) — min_books={min_books}:")
    for label, depth in results["meta"]["market_depth"].items():
        cov = depth["coverage"]
        overround = depth["overround_median"]
        lines.append(
            f"  {label:<14}coverage {('n/a' if cov is None else format(cov, '.1%')):>6} "
            f"(n={depth['n_matches']})  books median={depth['books_median']} "
            f"min={depth['books_min']} max={depth['books_max']}  "
            f"overround median={('n/a' if overround is None else format(overround, '.2%'))}"
        )

    market_defs = [
        (PRIMARY_MARKET, "over", y_over, True),
        ("btts_yes", "btts", y_btts, False),
        ("ensemble_1x2", "1x2", y_1x2, False),
    ]
    for market_name, prefix, truth, is_primary in market_defs:
        arms: Dict[str, Dict[str, Any]] = {}
        raw: Dict[str, np.ndarray] = {}
        scores: Dict[str, np.ndarray] = {}
        baseline_only = [k for k in BASELINE_KEYS if k not in configs]
        for key in list(configs) + baseline_only:
            probs = stacked.get(f"{prefix}::{key}")
            if probs is None:
                continue
            if prefix == "1x2":
                finite = np.all(np.isfinite(probs), axis=1)
                per_match = np.where(finite, brier_multiclass(np.where(finite[:, None], probs, 0.0), truth), np.nan)
                summary_probs = probs[:, 0]
                summary_truth = (truth == 0).astype(float)
            else:
                per_match = brier_binary(probs, truth)
                summary_probs = probs
                summary_truth = truth
            raw[key] = summary_probs
            scores[key] = per_match
            arms[key] = summarise(per_match, summary_probs, summary_truth)
            if prefix == "1x2":
                arms[key]["note"] = "mean_predicted/realised shown for the HOME class"

        deltas: Dict[str, Dict[str, Dict[str, Any]]] = {}
        verdict_lines: List[str] = []
        for challenger in configs:
            if challenger == "served" or challenger not in scores:
                continue
            per_baseline: Dict[str, Dict[str, Any]] = {}
            for baseline in BASELINE_KEYS:
                if baseline not in scores or baseline == challenger:
                    continue
                d = paired_delta(scores[challenger], scores[baseline], folds=fold_labels)
                d["verdict"] = ship_verdict(
                    d["delta"],
                    d["se"],
                    baseline,
                    delta_cluster=d["delta_cluster"],
                    se_cluster=d["se_cluster"],
                    n_folds=d["n_folds"],
                )
                per_baseline[baseline] = d
                if is_primary:
                    verdict_lines.append(f"[{market_name}] {challenger}: {d['verdict']['line']}")
            deltas[challenger] = per_baseline

        results["markets"][market_name] = {"arms": arms, "deltas": deltas}
        lines.extend(format_market_section(market_name, is_primary, arms, deltas))
        if verdict_lines:
            lines.append("")
            lines.extend(verdict_lines)

        # The primary market gets a coverage-stratified re-read. `served` is
        # the arm whose coverage is in question, so its known_teams is what
        # defines the strata; the fitting-fix verdict is the both-known one.
        if is_primary and "known::served" in stacked:
            strata_block, strata_lines = stratified_deltas(
                scores,
                configs,
                stacked["known::served"],
                fold_labels,
            )
            results["markets"][market_name]["strata"] = strata_block
            lines.extend(strata_lines)

    lines.append("")
    lines.append(
        "REMINDER: a Brier win is not an ROI win. ('soccer','over_under') stays gated off in "
        "scripts/rec_gating.py until a separate positive-ROI result exists."
    )
    report = "\n".join(lines)
    print(report)
    print(json.dumps(results, indent=2, default=float))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        logger.info("Wrote %s", args.json_out)
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--start", default="2025-08-01", help="First eval day (inclusive).")
    p.add_argument("--end", default="2026-08-31", help="Last eval day (inclusive).")
    p.add_argument(
        "--decay",
        type=float,
        default=0.00095,
        help=(
            "Exponential decay per day inside the Dixon-Coles MLE (0.00095 = 730-day half-life). "
            "Sweep of over-2.5 Brier by half-life: 180d 0.2492, 385d 0.2468, 730d 0.2460, "
            "1095d 0.2459, 1825d 0.2461, none 0.2473."
        ),
    )
    p.add_argument(
        "--reg",
        type=float,
        default=20.0,
        help="Shrinkage prior in EFFECTIVE MATCHES applied as (num + reg)/(den + reg).",
    )
    p.add_argument("--max-goals", type=int, default=10, help="Goals grid for the DC's own 1x2 (derivation uses 10).")
    p.add_argument(
        "--league-shrinkage",
        type=float,
        default=200.0,
        help="Effective matches of shrinkage on the MODEL's fitted per-league scoring baselines.",
    )
    p.add_argument(
        "--shrink-k",
        type=float,
        default=200.0,
        help="Effective matches of shrinkage on the per-league BASE-RATE comparator (baseline c).",
    )
    p.add_argument(
        "--base-rate-decay",
        type=float,
        default=None,
        help=(
            "Per-day recency decay for the CONSTANT comparators. Defaults to --decay so they are "
            "held to the same weighting as the challenger; the un-decayed constants are scored and "
            "printed alongside either way. 0 = un-decayed only."
        ),
    )
    p.add_argument(
        "--min-books",
        type=int,
        default=2,
        help=(
            "Minimum books quoting a complete set before a match counts toward the market reference. "
            "1 would score one book's opinion plus its own margin as 'the market'."
        ),
    )
    p.add_argument(
        "--configs",
        default=DEFAULT_CONFIGS,
        help=f"Comma-separated arms to run. Known: {','.join(sorted(CONFIG_SPECS))}. 'served' is always included.",
    )
    p.add_argument("--limit", type=int, default=0, help="Cap eval matches PER MONTH (fast smoke run). 0 = no cap.")
    p.add_argument("--json-out", default="", help="Optional path to also write the results JSON.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not given — refusing to run.")
        return 2
    try:
        run(args)
    except Exception as e:
        logger.error("ab_soccer_dixon_coles_refit failed: %s", e, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
