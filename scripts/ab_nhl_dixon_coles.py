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
           September).
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
                  scored on every fold: this artifact is fitted by hand and then
                  served for months, so this staleness is what would ship.
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
        (no gradable row) blocks too.
  refit_frozen disqualified -> DO-NOT-SHIP. pinned and refit are reported and
  never decide; verdicts vs the constants are context and never decide.
  SHIP would mean: train_hockey_dixon_coles.py moves to DIXON_COLES_CONFIG (rho
  still forced to 0), scripts/promote_dc_model.py gets an "nhl" entry, and the
  staged fit is promoted after a backup. This script does none of that.

CONFOUNDS, PRINTED NOT HIDDEN
  * The served artifact's training window is not recorded in it; its trainer
    fitted the oldest 80% of the 2026-06-04 frame, which by today's row count
    (6,352 of 7,940 rows before 2026-06-05) ends 2025-04-04. The 2024-10..2025-03
    folds are then IN-SAMPLE for served — a bias AGAINST the challengers.
    Per-fold and per-season deltas are printed.
  * Team coverage: strata by how many of a match's teams the served file knows.
  * correct_score is scored on each arm's own top-12 + 'other' partition,
    because that is what is served and graded; two arms' partitions differ, so
    it is a product score, not a like-for-like distribution score.
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
DEFAULT_START = "2024-10-01"
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
    = a partition (exactly one key wins). correct_score has no fixed keys — each
    row is scored on the arm's own listed scorelines + 'other'."""

    name: str
    market: str  # the served market_type == predictions.prediction_type
    keys: Tuple[str, ...] = ()


PRIMARY = "over_5.5"
CORRECT_SCORE = "correct_score"
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
    Unit(CORRECT_SCORE, "correct_score"),
)
UNIT_NAMES = tuple(u.name for u in UNITS)
FIXED_UNITS = tuple(u for u in UNITS if u.keys)  # truth independent of the arm
REGULATION_CLASSES = ("home", "draw", "away")

CHALLENGERS = ("pinned", "refit", "refit_frozen")
WALK_FORWARD_ARMS = ("pinned", "refit")
CONSTANT_ARMS = ("base_global", "base_season", "base_season_dw")
DECAYED_ARMS = ("refit", "refit_frozen")  # compared with base_season_dw; the rest with base_season
DECIDING_ARM = "refit_frozen"
REGRESSION_DELTA = 0.002
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


def fixed_truth(ev: pd.DataFrame) -> Dict[str, np.ndarray]:
    """(n, k) graded outcome per fixed unit, a NaN row where ungradable."""
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
    return out


def regulation_label(regulation_winner: Any) -> Optional[str]:
    """home/draw/away exactly as grading_outcomes reads regulation_winner, or None."""
    return go.nhl_regulation_3way(_regulation_metadata(regulation_winner))


def readout(markets: Sequence[Dict[str, Dict[str, float]]], ev: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Per-unit (n, k) probabilities from each row's served markets, plus
    '_cs_truth': the graded correct_score outcome on the arm's own keys."""
    n = len(markets)
    out: Dict[str, np.ndarray] = {u.name: np.full((n, len(u.keys)), np.nan) for u in FIXED_UNITS}
    cs_p: List[np.ndarray] = []
    cs_y = np.full((n, 13), np.nan)
    hs = ev["home_score"].to_numpy(dtype=int)
    as_ = ev["away_score"].to_numpy(dtype=int)
    rw = ev["regulation_winner"].to_numpy(dtype=object)
    for i, m in enumerate(markets):
        for unit in FIXED_UNITS:
            out[unit.name][i] = [m[unit.market][k] for k in unit.keys]
        cs = m[CORRECT_SCORE]
        keys = list(cs)
        if len(keys) != 13:
            raise RuntimeError(f"correct_score served {len(keys)} keys, expected 12 scorelines + 'other'")
        cs_p.append(np.array([cs[k] for k in keys], dtype=float))
        y = graded_vector("correct_score", keys, hs[i], as_[i], rw[i], probabilities=cs)
        if y is not None:
            cs_y[i] = y
    out[CORRECT_SCORE] = np.vstack(cs_p) if cs_p else np.zeros((0, 13))
    out["_cs_truth"] = cs_y
    return out


def score_arm(probs: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-row Brier per unit — summed over a partition's keys, binary for one
    key — NaN where the row is ungradable for that unit."""
    full = {**truth, CORRECT_SCORE: probs["_cs_truth"]}
    return {u: np.sum((np.asarray(probs[u], dtype=float) - full[u]) ** 2, axis=1) for u in UNIT_NAMES}


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


def upper_bound(p: Dict[str, Any]) -> Optional[float]:
    """The 95% upper bound the regression rule reads: the fold-clustered
    t(n_folds-1) bound, else the per-match delta + 2*SE."""
    if p.get("upper_95_cluster_t") is not None:
        return float(p["upper_95_cluster_t"])
    d, se = p.get("delta"), p.get("se")
    if d is None or se is None or not math.isfinite(d) or not math.isfinite(se):
        return None
    return float(d + 2.0 * se)


def regression(p: Dict[str, Any], threshold: float = REGRESSION_DELTA) -> Optional[str]:
    """Why this per-unit pair (challenger - served) is a regression, or None.
    No delta at all (no gradable row) counts: an unassessed unit is not shown
    not to regress."""
    d = p.get("delta")
    if d is None or not math.isfinite(d):
        return "unassessable (no gradable paired rows)"
    if d <= threshold:
        return None
    ub = upper_bound(p)
    if ub is not None and ub <= 0:
        return None
    return f"dBrier {d:+.5f} > +{threshold:.3f} with 95% upper bound {'n/a' if ub is None else f'{ub:+.5f}'} > 0"


def decide(
    verdicts_vs_served: Dict[str, Dict[str, Any]],
    regressions: Dict[str, Dict[str, str]],
    disqualified: Dict[str, Any],
    arm: str = DECIDING_ARM,
) -> Dict[str, Any]:
    """The pre-registered decision for ``arm``: primary ship_strict AND no regression."""
    if arm in disqualified or arm not in verdicts_vs_served:
        why = disqualified.get(arm, {}).get("reason", "not scored")
        return {"arm": arm, "ship": False, "primary_pass": False, "regressions": {}, "reason": f"disqualified: {why}"}
    primary_pass = bool(verdicts_vs_served[arm]["ship_strict"])
    regs = dict(regressions.get(arm, {}))
    ship = primary_pass and not regs
    if ship:
        reason = f"{PRIMARY} passes ship_strict vs served and no other unit regresses"
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
    logger.info(
        "frame: %d NHL matches %s..%s; %d folds (%d pooled, %d skipped)",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
        len(folds),
        len(pooled),
        len(skipped),
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
    p_primary: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    p_reg: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    e_total: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    y_primary: List[np.ndarray] = []
    y_reg: List[np.ndarray] = []
    totals: List[np.ndarray] = []
    fold_of: List[np.ndarray] = []
    season_arr: List[np.ndarray] = []
    served_known: List[np.ndarray] = []
    fits: List[Dict[str, Any]] = []
    disqualified: Dict[str, Dict[str, Any]] = {}
    cache: Dict[Tuple[str, pd.Timestamp], Tuple[Any, Dict[str, Any], str, Optional[str]]] = {}

    def fit(key: str, fit_df: pd.DataFrame, cutoff: pd.Timestamp, tmp: str):
        # Fits are deterministic and every fit frame is frame[match_date <
        # cutoff], so the frozen fit IS the first fold's refit — fitted once.
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
        return readout([served_markets(P) for P in mats], ev), expected_totals(mats), None

    with tempfile.TemporaryDirectory() as tmp:
        first_cut = folds[0].fit_cutoff
        first_fit = frame[frame["match_date"] < first_cut]
        frozen, frozen_result, frozen_path, frozen_refusal = fit("refit", first_fit, first_cut, tmp)
        frozen_label = f"frozen@{first_cut.date()}"
        record(frozen_label, "refit_frozen", first_fit, first_cut, frozen_result, frozen_path, frozen_refusal)
        frozen_fp = ht.fingerprint(frozen_path)
        reason = disqualification("refit_frozen", refusal=frozen_refusal)
        if reason is not None:
            disqualify("refit_frozen", frozen_label, reason)
        for fold in folds:
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
            if "refit_frozen" in arms:
                probs, tot, bad = model_arm(frozen, ev)
                if bad:
                    disqualify("refit_frozen", fold.label, bad)
                else:
                    arm_probs["refit_frozen"], arm_totals["refit_frozen"] = probs, tot
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
                arm_probs[arm], arm_totals[arm] = readout(markets, ev), expected_totals(mats)
            for arm in list(arm_probs):
                reason = disqualification(arm, probs=arm_probs[arm])  # raises for served / the constants
                if reason is not None:
                    disqualify(arm, fold.label, reason)
                    del arm_probs[arm]
            for arm in arms:
                for unit, s in score_arm(arm_probs[arm], truth).items():
                    scores[arm][unit].append(s)
                p_primary[arm].append(arm_probs[arm][PRIMARY][:, 0])
                p_reg[arm].append(arm_probs[arm]["regulation_1x2"])
                e_total[arm].append(arm_totals[arm])
            y_primary.append(truth[PRIMARY][:, 0])
            y_reg.append(truth["regulation_1x2"])
            totals.append((ev["home_score"] + ev["away_score"]).to_numpy(dtype=float))
            fold_of.append(np.full(len(ev), fold.label, dtype=object))
            season_arr.append(ev["season"].to_numpy(dtype=int))
            served_known.append(
                np.array(
                    [int(h in served_teams) + int(a in served_teams) for h, a in zip(ev["home_team"], ev["away_team"])],
                    dtype=float,
                )
            )
            logger.info("fold %s: fit=%d eval=%d arms=%s", fold.label, len(fit_df), len(ev), ",".join(arms))

    cat = {a: {u: np.concatenate(scores[a][u]) for u in UNIT_NAMES} for a in arms}
    folds_arr = np.concatenate(fold_of)
    seasons = np.concatenate(season_arr)
    known = np.concatenate(served_known)
    yp = np.concatenate(y_primary)
    yr = np.vstack(y_reg)
    realised_total = np.concatenate(totals)
    fold_labels = list(dict.fromkeys(folds_arr.tolist()))

    def pair(a: str, b: str, unit: str = PRIMARY, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        sel = np.ones(len(folds_arr), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        out = ft.paired_delta(cat[a][unit][sel], cat[b][unit][sel], folds_arr[sel])
        out["upper_95_cluster_t"] = ht.t_upper(out["delta_cluster"], out["se_cluster"], out["n_folds"])
        return out

    def verdict(a: str, b: str) -> Dict[str, Any]:
        p = pair(a, b)
        v = ft.ship_verdict(
            p["delta"], p["se"], b, delta_cluster=p["delta_cluster"], se_cluster=p["se_cluster"], n_folds=p["n_folds"]
        )
        v["upper_95_cluster_t"] = p["upper_95_cluster_t"]
        v["ship_strict"] = bool(v["ship"] and p["upper_95_cluster_t"] is not None and p["upper_95_cluster_t"] < 0)
        return v

    def calibration(a: str) -> Dict[str, Any]:
        pp, pr = np.concatenate(p_primary[a]), np.vstack(p_reg[a])
        ok = np.isfinite(yr).all(axis=1)
        return {
            "over_5.5_pred": float(pp.mean()),
            "over_5.5_real": float(np.nanmean(yp)),
            "regulation_pred": (
                {c: float(pr[ok, k].mean()) for k, c in enumerate(REGULATION_CLASSES)} if ok.any() else {}
            ),
            "regulation_real": (
                {c: float(yr[ok, k].mean()) for k, c in enumerate(REGULATION_CLASSES)} if ok.any() else {}
            ),
            "expected_total": float(np.concatenate(e_total[a]).mean()),
            "realised_total": float(realised_total.mean()),
        }

    challengers = [a for a in arms if a in CHALLENGERS]
    vs_served = {a: {u: pair(a, "served", u) for u in UNIT_NAMES} for a in arms if a != "served"}
    regressions = {
        a: {u: r for u in UNIT_NAMES if u != PRIMARY and (r := regression(vs_served[a][u])) is not None}
        for a in challengers
    }
    verdicts_vs_served = {a: verdict(a, "served") for a in challengers}
    report: Dict[str, Any] = {
        "n_eval": int(len(folds_arr)),
        "n_gradable": {u: int(np.isfinite(cat["served"][u]).sum()) for u in UNIT_NAMES},
        "folds": fold_labels,
        "pooled_folds": pooled,
        "skipped_folds": skipped,
        "disqualified": disqualified,
        "duplicated_match_ids": dups,
        "frame": frame_audit,
        "served_artifact": served_fp,
        "frozen_artifact": frozen_fp,
        "configs": {k: {"name": c.name, **c.hyperparameters} for k, c in cfgs.items()},
        "brier": {a: {u: float(np.nanmean(cat[a][u])) for u in UNIT_NAMES} for a in arms},
        "calibration": {a: calibration(a) for a in arms},
        "fits": fits,
        "vs_served": vs_served,
        "vs_matched_constant": {
            a: {"constant": matched_constant(a), **pair(a, matched_constant(a))} for a in ["served"] + challengers
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
                    "n": int((folds_arr == lab).sum()),
                    "delta": float(
                        np.mean(cat[a][PRIMARY][folds_arr == lab] - cat["served"][PRIMARY][folds_arr == lab])
                    ),
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
        "served_known_strata": {
            str(k): {
                "n": int((known == k).sum()),
                **{a: pair(a, "served", mask=known == k) for a in challengers if (known == k).sum() > 1},
            }
            for k in (0, 1, 2)
        },
        "verdicts_vs_served": verdicts_vs_served,
        "verdicts_vs_matched_constant": {a: verdict(a, matched_constant(a)) for a in challengers},
        "regressions": regressions,
        "decision": decide(verdicts_vs_served, regressions, disqualified),
    }
    return report


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def format_report(r: Dict[str, Any]) -> str:
    pooled = [p["fold"] + "->" + p["into"] for p in r["pooled_folds"]] or "none"
    skipped = [s["fold"] for s in r["skipped_folds"]] or "none"
    L = [
        f"NHL Dixon-Coles A/B — {r['n_eval']} eval matches, folds {r['folds'][0]}..{r['folds'][-1]}"
        f" ({len(r['folds'])}); pooled {pooled}; skipped {skipped}; duplicated match_ids {r['duplicated_match_ids']}",
        f"frame: {r['frame']['rows']} rows, {r['frame']['without_regulation_winner']} without regulation_winner;"
        f" gradable per unit: {r['n_gradable']}",
    ]
    for arm, d in r["disqualified"].items():
        L.append(f"DISQUALIFIED {arm} at fold {d['fold']}: {d['reason']} (cannot be served; not scored)")
    for name in ("served_artifact", "frozen_artifact"):
        a = r[name]
        L.append(
            f"{name}: {a['teams']} teams, league baselines {a['has_league_baselines']}, rho {_f(a['rho'], '.4f')},"
            f" sha256 {a['sha256'][:12]}"
        )
    L += ["", "Brier by arm and unit (lower is better):"]
    arms = list(r["brier"])
    L.append(f"  {'unit':<18}" + "".join(f"{a:>15}" for a in arms))
    for u in UNIT_NAMES:
        L.append(f"  {u:<18}" + "".join(f"{r['brier'][a][u]:>15.5f}" for a in arms))
    L += ["", f"{PRIMARY} vs served:"]
    L += [ht._pair_line(a, d[PRIMARY]) for a, d in r["vs_served"].items()]
    L += ["", "every unit vs served (pooled dBrier; R = a regression under the pre-registered rule):"]
    others = list(r["vs_served"])
    L.append(f"  {'unit':<18}" + "".join(f"{a:>17}" for a in others))
    for u in UNIT_NAMES:
        cells = []
        for a in others:
            mark = " R" if u in r["regressions"].get(a, {}) else "  "
            cells.append(f"{_f(r['vs_served'][a][u].get('delta')) + mark:>17}")
        L.append(f"  {u:<18}" + "".join(cells))
    L += ["", f"{PRIMARY} vs the matched constants (decayed arms vs base_season_dw):"]
    L += [ht._pair_line(f"{a} vs {d['constant']}", d) for a, d in r["vs_matched_constant"].items()]
    L += ["", "head to head:"]
    L += [ht._pair_line(k, v) for k, v in r["head_to_head"].items()]
    L += ["", f"{PRIMARY} by season:"]
    for s, block in r["by_season"].items():
        L.append(f"  [{s}] n={block['n']}")
        L += [ht._pair_line(f"{a} vs served", v) for a, v in block.items() if isinstance(v, dict)]
    if r["per_fold_vs_served"]:
        L += ["", f"{PRIMARY} per-fold delta vs served:"]
        L.append(f"  {'arm':<15}" + "".join(f"{lab[2:]:>9}" for lab in r["folds"]))
        for a, rows in r["per_fold_vs_served"].items():
            L.append(f"  {a:<15}" + "".join(f"{row['delta']:>+9.4f}" for row in rows))
        first = next(iter(r["per_fold_vs_served"].values()))
        L.append(f"  {'n':<15}" + "".join(f"{row['n']:>9}" for row in first))
    L += ["", "calibration (over 5.5 mean predicted vs realised; regulation H/D/A; E[total] vs realised):"]
    for a, c in r["calibration"].items():
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
    L += ["", f"{PRIMARY} vs served by teams the served artifact knows:"]
    for k, blk in r["served_known_strata"].items():
        L.append(
            f"  known={k} n={blk['n']}: "
            + ", ".join(f"{a} {_f(v.get('delta'))}" for a, v in blk.items() if isinstance(v, dict))
        )
    refused: Dict[str, int] = {}
    for f in r["fits"]:
        if f.get("refusal"):
            refused[f["arm"]] = refused.get(f["arm"], 0) + 1
    L += ["", f"fits the trainer would refuse (per arm): {refused or 'none'}"]
    d = r["decision"]
    L += [
        "",
        f"DECISION (pre-registered: {DECIDING_ARM} must pass ship_strict vs served on {PRIMARY} AND regress on no"
        f" other unit): {'SHIP' if d['ship'] else 'DO-NOT-SHIP'} — {d['reason']}",
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
    L += ["", "context only — never decides (vs the matched constants):"]
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
