"""A/B gate for the soccer SECOND-HALF Dixon-Coles artifact (dixon_coles_h2_soccer).

WHAT IS BEING TESTED
--------------------
The served second-half artifact was written 2026-06-04 by
scripts/train_second_half_dixon_coles.py and has never been gated: 175 teams,
the pre-2026-09 fitter, every unknown team on one global baseline (2H home
0.766 + a CONFIGURED 0.25 home bump, away 0.766: a 1.33 home/away lambda
ratio), sha256 d72a221e.... It feeds
exactly one served market, ht_ft_double_result: precompute_predictions builds
the HT scoreline matrix, builds this artifact's 2H matrix (loaded through
_load_dixon_coles_artifact into DixonColesPredictor(DIXON_COLES_CONFIG),
build_dc_matrix(..., MAX_GOALS_DERIVE)) and convolves the two
(derive_soccer_htft_markets). The trainer as written fits the OLDEST 80% of
the frame with the pinned legacy config — which never converges on the 2H
frame (15/15 walk-forward cutoffs) yet still emits finite matrices, so the
invalid-probability rule that caught the HT pinned config cannot catch it
here: non-convergence is a disqualification of its own.

PRE-REGISTERED DESIGN (fixed before the first run)
  Frame    SECOND_HALF_TRAINING_QUERY rows (2H goals = FT - HT, FT >= HT)
           joined on match_id with their HT and FT scores (the HT_TRAINING_QUERY
           population with the 2H query's FT filters), so every row carries HT
           goals, 2H goals and the FT outcome. The query's 2H targets are
           re-derived from the scores and must agree (asserted). HT rows the
           FT filters drop are counted and logged as data defects.
  Folds    monthly walk-forward 2025-08..newest (ft.build_folds / split_fold);
           every fit uses match_date < month start only. A fold with fewer
           than MIN_FOLD_EVAL eval rows is skipped and listed.
  HT side  ht_ft needs an HT matrix. For EVERY arm it is a per-fold walk-forward
           refit of the GATED HT config (replace(DIXON_COLES_CONFIG,
           name="dixon_coles_ht_soccer"), what train_halftime_dixon_coles.py
           fits) on the HT goals of the rows before the fold — NOT the
           production HT file, which was fitted on every row including the
           eval window. One HT matrix per eval row, shared by every arm, so the
           paired deltas isolate the second half. An unconverged HT-side fit or
           an invalid HT matrix aborts the run: the HT trainer would refuse it.
  Arms     (second half; every model arm is round-tripped through save() and
           the serve loader, ht.load_serving == _load_dixon_coles_artifact)
    served        the production H2 file (or --served-artifact): the incumbent.
    pinned        walk-forward refit with the trainer's pinned legacy config
                  (dixon_coles_config_pinned_to_legacy_fit) — "the trainer,
                  fitted on all data before the fold".
    refit         walk-forward refit with DIXON_COLES_CONFIG (per-league
                  baselines, 20-match prior, 730-day decay, max_goals 10).
    refit_frozen  refit's config fitted ONCE at the first fold cutoff and scored
                  on every fold: the H2 artifact is fitted by hand, not by the
                  weekly DAG, so this staleness is what would ship.
    base_league / base_league_dw
                  per-league 9-class ht_ft base rates (k=200 shrinkage toward
                  the global rate), undecayed / with the refit arms' recency
                  weighting. Also the 2H 1x2 and 2H O/U constants for the
                  diagnostics.
    base_cond / base_cond_dw
                  HT-state-conditional constant: ht_ft[h, f] = P_HT(h) x
                  P(FT outcome f | HT outcome h, league), shrunk with k=200
                  effective matches per (league, HT outcome) toward P(f | h)
                  over all leagues; P_HT(h) is the shared HT matrix's 1x2. The
                  convolution assumes the halves are independent given the
                  teams; this arm keeps the same HT marginal and replaces the
                  second-half model with the empirical HT -> FT transition, so
                  it tests that assumption.
  Primary  ht_ft 9-way Brier (sum over classes). Gate (§4.3): challenger vs
           served — ft.ship_verdict (dBrier <= -0.005, per-match and
           fold-clustered z=2 bounds < 0) AND the stricter t(n_folds-1)
           clustered bound (ship_strict). Reported beside it, never used to
           rescue a miss: vs the matched constant and vs the matched
           conditional constant (decayed arms vs the *_dw twins).
  Diagnostics (from the 2H matrix alone — no 2H market is served, none gates)
           2H 1x2 3-way Brier, 2H over 0.5 / 1.5 Brier, 2H 1x2 calibration;
           ht_ft calibration (mean predicted vs realised per cell); the shared
           HT side's own HT 1x2 Brier (a sanity anchor against
           ab_halftime_dixon_coles); per-fold deltas; era split at 2026-06-01;
           strata by how many of a match's teams the served artifact knows;
           artifact fingerprints.
  Disqualification
           served, the constants and the HT side must emit finite, normalised
           probabilities on every row — otherwise the run aborts (a broken
           incumbent or comparator is a broken experiment). A challenger
           (pinned, refit, refit_frozen) is DISQUALIFIED — not scored on any
           row, its earlier folds dropped, never fitted again — the first time
           a fit does not converge (train result 'iterations' is None) or it
           emits an invalid probability: the trainer must refuse that fit, so
           it could not be served. Scoring a subset of rows would run the
           paired comparisons on different row sets.

READING THE RESULT
  A challenger ships only on ship_strict vs served. If the conditional constant
  beats every model arm (served included), the lever is the independence
  assumption in the convolution, not this artifact's fit — a finding to act on
  separately, not a reason to ship any arm here.

CONFOUNDS, PRINTED NOT HIDDEN
  * The served artifact's training window is not recorded in it (its trainer
    fitted the oldest 80% of the 2026-06-04 frame). If that overlaps the early
    folds the served arm is in-sample there — a bias AGAINST the challengers.
    Per-fold deltas are printed so an early-fold effect is visible.
  * Team coverage: the served arm knows 175 teams; strata by how many of a
    match's teams it knows are printed.
  * Source: football-data rows until ~2026-05, ESPN results-path rows from
    2026-06 (what production now serves); the two eras are reported apart.
  * The HT side is the gated config refit per fold — fresher than the frozen
    production HT file. It is shared by every arm, so it favours none; but the
    ht_ft levels here are not production's, and the paired deltas depend on it
    too (a 2H matrix is scored through its convolution with the HT one).

Read-only: SELECTs, plus temp files for the round-trip. Usage (api container):
    python /app/scripts/ab_second_half_dixon_coles.py --out-json /tmp/ab_h2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_halftime_dixon_coles as ht  # noqa: E402
import ab_soccer_dixon_coles_refit as ft  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_second_half_dixon_coles")

SERVED_ARTIFACT = ht.H2_ARTIFACT  # /app/models/production/dixon_coles_h2_soccer/1.0.0/model.bin
LEGACY_SERVED_TEAMS = 175

CLASSES = ht.CLASSES  # home/draw/away == match_outcome 0/1/2
HTFT_KEYS = ht.HTFT_KEYS  # index = ht_outcome * 3 + ft_outcome
PRIMARY = "ht_ft"
H2_1X2 = "h2_1x2"
H2_OU_LINES = (0.5, 1.5)
MARKETS = (PRIMARY, H2_1X2) + tuple(f"h2_over_{line}" for line in H2_OU_LINES)
MULTICLASS = {PRIMARY: 9, H2_1X2: 3}
HT_1X2 = "ht_1x2"  # the shared HT side's own market — a sanity anchor, not an arm's

CHALLENGERS = ("pinned", "refit", "refit_frozen")
WALK_FORWARD_ARMS = ("pinned", "refit")
BASE_ARMS = ("base_league", "base_league_dw")
COND_ARMS = ("base_cond", "base_cond_dw")
CONSTANT_ARMS = BASE_ARMS + COND_ARMS
DECAYED_ARMS = ("refit", "refit_frozen")  # compared with the *_dw constants; the rest with the undecayed ones
COND_SHRINKAGE = 200.0  # effective matches per (league, HT outcome) — ft.fit_base_rates' k
MIN_FOLD_EVAL = ht.MIN_FOLD_EVAL
ESPN_ERA_START = ht.ESPN_ERA_START

SCORES_SQL = """
    SELECT m.id::text AS match_id,
           m.half_time_home AS ht_home, m.half_time_away AS ht_away,
           m.home_score AS ft_home, m.away_score AS ft_away
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'soccer' AND m.status = 'finished'
      AND m.half_time_home IS NOT NULL AND m.half_time_away IS NOT NULL
"""


# ── Pure ─────────────────────────────────────────────────────────────


def second_half_goals(
    ht_home: Sequence[int], ht_away: Sequence[int], ft_home: Sequence[int], ft_away: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """2H goals per side = FT - HT, what SECOND_HALF_TRAINING_QUERY trains on.
    The query drops FT < HT rows, so one here is a defect: raise, never clip."""
    hth, hta = np.asarray(ht_home, dtype=int), np.asarray(ht_away, dtype=int)
    h2h = np.asarray(ft_home, dtype=int) - hth
    h2a = np.asarray(ft_away, dtype=int) - hta
    neg = (h2h < 0) | (h2a < 0)
    if neg.any():
        raise ValueError(f"{int(neg.sum())} row(s) have a full-time score below the half-time score")
    return h2h, h2a


def assemble_frame(h2_frame: pd.DataFrame, scores: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Join the trainer's 2H frame with the HT/FT scores and prove the join.

    Returns the frame with ht_home/ht_away, ft_home/ft_away and h2_home/h2_away
    columns, and an audit of the HT rows the 2H query's FT filters dropped."""
    frame = h2_frame.merge(scores, on="match_id", how="left", validate="many_to_one")
    missing = frame[["ht_home", "ht_away", "ft_home", "ft_away"]].isna().any(axis=1)
    if missing.any():
        raise RuntimeError(
            f"{int(missing.sum())} second-half row(s) have no HT/FT score in SCORES_SQL — the two queries disagree"
        )
    h2h, h2a = second_half_goals(frame["ht_home"], frame["ht_away"], frame["ft_home"], frame["ft_away"])
    off = (h2h != frame["home_score"].to_numpy(dtype=int)) | (h2a != frame["away_score"].to_numpy(dtype=int))
    if off.any():
        raise RuntimeError(
            f"{int(off.sum())} row(s): SECOND_HALF_TRAINING_QUERY's 2H goals != FT - HT "
            f"(e.g. {frame.loc[off, 'match_id'].head(5).tolist()}) — the frame is not what this harness assumes"
        )
    frame = frame.assign(h2_home=h2h, h2_away=h2a)
    dropped = scores[~scores["match_id"].isin(set(frame["match_id"]))]
    no_ft = dropped[["ft_home", "ft_away"]].isna().any(axis=1)
    below = ~no_ft & ((dropped["ft_home"] < dropped["ht_home"]) | (dropped["ft_away"] < dropped["ht_away"]))
    audit = {
        "ht_rows": int(len(scores)),
        "second_half_rows": int(len(frame)),
        "dropped_no_full_time": dropped.loc[no_ft, "match_id"].tolist(),
        "dropped_ft_below_ht": dropped.loc[below, "match_id"].tolist(),
        # e.g. a team row missing from the trainer query's team joins
        "dropped_other": dropped.loc[~no_ft & ~below, "match_id"].tolist(),
    }
    return frame, audit


def goals_frame(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    """A fit frame with ``home_col``/``away_col`` routed into home_score/
    away_score — the trainers' column-rename pattern."""
    out = df[["home_team", "away_team", "match_date", "league_id"]].copy()
    out["home_score"] = df[home_col].to_numpy(dtype=int)
    out["away_score"] = df[away_col].to_numpy(dtype=int)
    return out


def realised(ev: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Outcomes per market: ht_ft from the HT and FT scores, the 2H markets from
    the 2H goals, and the HT 1x2 for the shared HT side's sanity anchor."""
    hth, hta = ev["ht_home"].to_numpy(dtype=int), ev["ht_away"].to_numpy(dtype=int)
    h2h, h2a = ev["h2_home"].to_numpy(dtype=int), ev["h2_away"].to_numpy(dtype=int)
    ht_out = ht._outcome(hth, hta)
    ft_out = ht._outcome(ev["ft_home"].to_numpy(dtype=int), ev["ft_away"].to_numpy(dtype=int))
    out: Dict[str, np.ndarray] = {PRIMARY: ht_out * 3 + ft_out, H2_1X2: ht._outcome(h2h, h2a), HT_1X2: ht_out}
    for line in H2_OU_LINES:
        out[f"h2_over_{line}"] = ((h2h + h2a) > line).astype(float)
    return out


def one_x_two(P: np.ndarray) -> np.ndarray:
    """(home, draw, away) mass of a scoreline matrix whose rows are home goals."""
    P = np.asarray(P, dtype=float)
    return np.array([np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()], dtype=float)


def h2_markets(P_2h: np.ndarray) -> Dict[str, Any]:
    """The 2H diagnostics from the 2H matrix alone (never served)."""
    P = np.asarray(P_2h, dtype=float)
    i, j = np.indices(P.shape)
    out: Dict[str, Any] = {H2_1X2: one_x_two(P)}
    for line in H2_OU_LINES:
        out[f"h2_over_{line}"] = float(P[(i + j) > line].sum())
    return out


def htft_vector(P_ht: np.ndarray, P_2h: np.ndarray) -> np.ndarray:
    """The served ht_ft_double_result, in HTFT_KEYS order."""
    from predictors.market_derivation import derive_soccer_htft_markets  # type: ignore

    m = derive_soccer_htft_markets(P_ht, P_2h)["ht_ft_double_result"]
    return np.array([m[k] for k in HTFT_KEYS], dtype=float)


def fit_conditional_htft(
    ht_idx: Sequence[int],
    ft_idx: Sequence[int],
    leagues: Sequence[Any],
    weights: Optional[Sequence[float]] = None,
    k: float = COND_SHRINKAGE,
) -> Dict[str, Any]:
    """P(FT outcome | HT outcome, league) as 3x3 row-stochastic matrices.

    Each league's row for HT outcome h is (c[h, :] + k * global[h, :]) /
    (sum_f c[h, f] + k), where c are the (weighted) transition counts and
    global[h, :] = P(f | h) over every league — so a thin league, or an HT
    state a league has barely seen, inherits the global transition."""
    hi, fi = np.asarray(ht_idx, dtype=int), np.asarray(ft_idx, dtype=int)
    n = len(hi)
    w = np.ones(n, dtype=float) if weights is None else np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if w.shape != (n,):
        raise ValueError(f"weights must align with the outcomes, got {w.shape} vs {n}")
    counts = np.zeros((3, 3))
    np.add.at(counts, (hi, fi), w)
    rows = counts.sum(axis=1, keepdims=True)
    glob = np.where(rows > 0, counts / np.where(rows > 0, rows, 1.0), 1.0 / 3.0)
    lg = np.asarray(leagues, dtype=object)
    by_league: Dict[Any, np.ndarray] = {}
    for league in pd.unique(pd.Series(lg).dropna()):
        sel = lg == league
        c = np.zeros((3, 3))
        np.add.at(c, (hi[sel], fi[sel]), w[sel])
        by_league[league] = (c + k * glob) / (c.sum(axis=1, keepdims=True) + k)
    return {"global": glob, "by_league": by_league, "k": k, "n_fit": n, "effective_n": float(w.sum())}


def predict_conditional_htft(cond: Dict[str, Any], p_ht: np.ndarray, leagues: Sequence[Any]) -> np.ndarray:
    """ht_ft[h * 3 + f] = P_HT(h) * P(f | h, league); an unseen league uses the global transition."""
    p_ht = np.asarray(p_ht, dtype=float)
    out = np.empty((len(p_ht), 9), dtype=float)
    for r, league in enumerate(leagues):
        T = cond["by_league"].get(league, cond["global"])
        out[r] = (p_ht[r][:, None] * T).ravel()
    return out


def score_arm(probs: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-match Brier for every market the arm emits (multi-class sum for ht_ft and 2H 1x2)."""
    return {
        mk: (ft.brier_multiclass if mk in MULTICLASS else ft.brier_binary)(probs[mk], truth[mk])
        for mk in MARKETS
        if mk in probs
    }


def htft_calibration(proba: np.ndarray, y_idx: np.ndarray) -> Dict[str, Any]:
    """Mean predicted vs realised rate per ht_ft cell, the worst cell, and argmax hit."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y_idx, dtype=int)
    if not len(y):
        return {"n": 0}
    pred = proba.mean(axis=0)
    real = np.bincount(y, minlength=9)[:9] / len(y)
    gap = np.abs(pred - real)
    return {
        "n": int(len(y)),
        "pred": {key: float(pred[i]) for i, key in enumerate(HTFT_KEYS)},
        "real": {key: float(real[i]) for i, key in enumerate(HTFT_KEYS)},
        "max_abs_gap": float(gap.max()),
        "worst_cell": HTFT_KEYS[int(gap.argmax())],
        "argmax_hit": float((proba.argmax(axis=1) == y).mean()),
    }


def invalid_matrices(mats: Sequence[np.ndarray]) -> Optional[str]:
    """None when every scoreline matrix is finite, non-negative and sums to 1."""
    bad = [i for i, P in enumerate(mats) if not (np.isfinite(P).all() and (P >= 0).all() and abs(P.sum() - 1.0) < 1e-6)]
    return f"{len(bad)} of {len(mats)} scoreline matrices non-finite or non-normalised" if bad else None


def invalid_probabilities(probs: Dict[str, np.ndarray]) -> Optional[str]:
    """None when every row of every market the arm emits is a valid probability, else a description."""
    for mk in MARKETS:
        if mk not in probs:
            continue
        v = np.asarray(probs[mk], dtype=float)
        if mk in MULTICLASS:
            bad = ~np.isfinite(v).all(axis=1) | (v < 0).any(axis=1) | ~np.isclose(v.sum(axis=1), 1.0, atol=1e-6)
            if bad.any():
                return f"{int(bad.sum())} of {len(v)} {mk} rows non-finite, negative or non-normalised"
        else:
            bad = ~np.isfinite(v) | (v < 0) | (v > 1)
            if bad.any():
                return f"{int(bad.sum())} of {len(v)} {mk} rows invalid"
    return None


def disqualification(
    arm: str, probs: Optional[Dict[str, np.ndarray]] = None, train_result: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Why ``arm`` cannot be scored, or None.

    Non-convergence first — checked on the train result before any
    probability is computed, because the pinned legacy config emits FINITE
    matrices from a fit that never converged, which the probability check
    alone would wave through. A missing 'iterations' key counts as not
    converged. Only a challenger can be disqualified: for served, a constant
    or the shared HT side a reason RAISES, because the experiment itself is
    broken."""
    reason = None
    if train_result is not None and train_result.get("iterations") is None:
        reason = "fit did not converge (train result 'iterations' is None) — the trainer must refuse it"
    elif probs is not None:
        reason = invalid_probabilities(probs)
    if reason is not None and arm not in CHALLENGERS:
        raise RuntimeError(f"{arm}: {reason}")
    return reason


def matched_constant(arm: str) -> str:
    return "base_league_dw" if arm in DECAYED_ARMS else "base_league"


def matched_conditional(arm: str) -> str:
    return "base_cond_dw" if arm in DECAYED_ARMS else "base_cond"


def arm_configs() -> Dict[str, Any]:
    """The ModelConfig each fitted arm uses — exactly the trainers' own."""
    from dataclasses import replace

    from predictors.model_config import DIXON_COLES_CONFIG, dixon_coles_config_pinned_to_legacy_fit  # type: ignore

    return {
        "ht_side": replace(DIXON_COLES_CONFIG, name="dixon_coles_ht_soccer"),  # train_halftime_dixon_coles.py
        "pinned": dixon_coles_config_pinned_to_legacy_fit("dixon_coles_h2_soccer"),  # train_second_half_dixon_coles
        "refit": replace(DIXON_COLES_CONFIG, name="dixon_coles_h2_soccer"),
    }


# ── Model plumbing (needs ml-models on the path) ─────────────────────


def fit_config(fit_df: pd.DataFrame, home_col: str, away_col: str, cfg: Any, tmpdir: str, tag: str):
    """Fit one artifact on the fold's fit frame, then round-trip it through
    save() + the serve loader. Returns (served_model, train_result, path).

    No reference date is passed: train() ages decay weights from the fit
    frame's newest match, which is what a production fit does too."""
    from predictors.poisson_models import DixonColesPredictor  # type: ignore

    model = DixonColesPredictor(cfg)
    result = model.train(goals_frame(fit_df, home_col, away_col))
    path = os.path.join(tmpdir, f"{tag}.bin")
    model.save(path)
    return ht.load_serving(path), result, path


def model_arm_probabilities(ht_mats: Sequence[np.ndarray], h2_mats: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """ht_ft through the served convolution + the 2H diagnostics, per eval row."""
    n = len(ht_mats)
    out: Dict[str, np.ndarray] = {mk: np.full((n, k), np.nan) for mk, k in MULTICLASS.items()}
    for line in H2_OU_LINES:
        out[f"h2_over_{line}"] = np.full(n, np.nan)
    for i in range(n):
        out[PRIMARY][i] = htft_vector(ht_mats[i], h2_mats[i])
        for mk, v in h2_markets(h2_mats[i]).items():
            out[mk][i] = v
    return out


def base_probabilities(fit_df: pd.DataFrame, ev: pd.DataFrame, weights) -> Dict[str, np.ndarray]:
    """Per-league shrunk constants for every market (weights None = undecayed)."""
    fit_truth = realised(fit_df)
    frame = fit_df.assign(**{f"_y_{mk}": fit_truth[mk] for mk in MARKETS})
    leagues = ev["league_id"].to_numpy()
    out: Dict[str, np.ndarray] = {}
    for mk in MARKETS:
        if mk in MULTICLASS:
            rates = ft.fit_class_base_rates(frame, f"_y_{mk}", MULTICLASS[mk], league_col="league_id", weights=weights)
            out[mk] = ft.predict_class_base_rates(rates, leagues)
        else:
            out[mk] = ft.fit_base_rates(frame, f"_y_{mk}", league_col="league_id", weights=weights).predict(leagues)
    return out


def conditional_probabilities(fit_df: pd.DataFrame, ev: pd.DataFrame, weights, p_ht: np.ndarray) -> Dict[str, Any]:
    """ht_ft only: the shared HT 1x2 times the walk-forward HT -> FT transition."""
    y = realised(fit_df)[PRIMARY]
    cond = fit_conditional_htft(y // 3, y % 3, fit_df["league_id"].to_numpy(), weights=weights)
    return {PRIMARY: predict_conditional_htft(cond, p_ht, ev["league_id"].to_numpy())}


def load_frame(database_url: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from sqlalchemy import create_engine
    from train_second_half_dixon_coles import SECOND_HALF_TRAINING_QUERY  # type: ignore
    from utils.training_data import load_training_frame  # type: ignore

    h2_frame = load_training_frame(database_url=database_url, query=SECOND_HALF_TRAINING_QUERY)
    engine = create_engine(database_url)
    try:
        scores = pd.read_sql(SCORES_SQL, engine)
    finally:
        engine.dispose()
    return assemble_frame(h2_frame, scores)


# ── Run ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    sys.path.insert(0, ht.ML_SRC)
    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore

    frame, frame_audit = load_frame(args.database_url)
    for key in ("dropped_no_full_time", "dropped_ft_below_ht"):
        if frame_audit[key]:
            # Finished matches whose scores cannot make a second half: data
            # defects, dropped for every arm (and by the trainer), counted.
            logger.error("%d HT row(s) %s — dropped: %s", len(frame_audit[key]), key, frame_audit[key][:20])
    if frame_audit["dropped_other"]:
        logger.warning(
            "%d HT row(s) absent from the 2H frame for another reason: %s",
            len(frame_audit["dropped_other"]),
            frame_audit["dropped_other"][:20],
        )
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    frame = frame.sort_values("match_date", kind="stable").reset_index(drop=True)
    dups = int(frame["match_id"].duplicated().sum())
    if dups:
        # The trainer fits this same frame, so this is a finding about it too.
        logger.error("2H frame carries %d duplicated match_id rows — dropped for scoring", dups)
        frame = frame.drop_duplicates("match_id", keep="first").reset_index(drop=True)
    if not frame["match_outcome"].isin([0, 1, 2]).all():
        raise RuntimeError("match_outcome outside {0,1,2} — the 2H frame is not what this harness assumes")
    end = args.end or frame["match_date"].max().strftime("%Y-%m-%d")
    folds = ft.build_folds(args.start, end)
    logger.info(
        "frame: %d second-half matches %s..%s; %d folds",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
        len(folds),
    )

    served_fp = ht.fingerprint(args.served_artifact)
    legacy = served_fp["teams"] == LEGACY_SERVED_TEAMS and not served_fp["has_league_baselines"]
    if args.served_artifact == SERVED_ARTIFACT and not legacy:
        raise RuntimeError(
            f"production H2 artifact is not the 2026-06-04 legacy fit ({served_fp}) — it may have been replaced; "
            "point --served-artifact at the backup to score the real incumbent"
        )
    served = ht.load_serving(args.served_artifact)
    served_teams = set(getattr(served, "team_attack", {}) or {})

    cfgs = arm_configs()
    decay = float(DIXON_COLES_CONFIG.hyperparameters.get("time_decay", 0.0))
    if float(cfgs["refit"].hyperparameters.get("time_decay", 0.0)) != decay:
        raise RuntimeError("the *_dw constants must carry the refit arm's decay")
    arms: List[str] = ["served", *CHALLENGERS, *CONSTANT_ARMS]
    scores: Dict[str, Dict[str, List[np.ndarray]]] = {a: {mk: [] for mk in MARKETS} for a in arms}
    probs_htft: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    probs_h2: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    truth_htft: List[np.ndarray] = []
    truth_h2: List[np.ndarray] = []
    truth_ht: List[np.ndarray] = []
    ht_side_probs: List[np.ndarray] = []
    fold_of: List[np.ndarray] = []
    dates: List[np.ndarray] = []
    served_known: List[np.ndarray] = []
    fits: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    disqualified: Dict[str, Dict[str, Any]] = {}
    cache: Dict[Tuple[str, pd.Timestamp], Tuple[Any, Dict[str, Any], str]] = {}

    def fit(key: str, fit_df: pd.DataFrame, cutoff: pd.Timestamp, tmp: str) -> Tuple[Any, Dict[str, Any], str]:
        # Fits are deterministic and every fit frame is frame[match_date <
        # cutoff], so the frozen fit IS the first fold's refit — fitted once.
        if (key, cutoff) not in cache:
            cols = ("ht_home", "ht_away") if key == "ht_side" else ("h2_home", "h2_away")
            cache[(key, cutoff)] = fit_config(fit_df, cols[0], cols[1], cfgs[key], tmp, f"{key}_{cutoff:%Y%m%d}")
        return cache[(key, cutoff)]

    def record(label: str, arm: str, fit_df: pd.DataFrame, cutoff: pd.Timestamp, result: Dict[str, Any], path: str):
        fits.append(
            {
                "fold": label,
                "arm": arm,
                "n_fit": int(len(fit_df)),
                "teams": int(result.get("team_count") or 0),
                "converged_at": result.get("iterations"),
                "leagues": int(result.get("league_count") or 0),
                "rho": result.get("rho"),
                "fitted_total": result.get("fitted_mean_total_goals"),
                "observed_total": result.get("observed_mean_total_goals"),
                "newest_fit_gap_days": float((cutoff - fit_df["match_date"].max()).total_seconds() / 86400),
                "sha256": ht.fingerprint(path)["sha256"][:12],
            }
        )

    def disqualify(arm: str, label: str, reason: str) -> None:
        disqualified[arm] = {"fold": label, "reason": reason}
        logger.error("%s DISQUALIFIED at fold %s: %s", arm, label, reason)
        arms.remove(arm)

    with tempfile.TemporaryDirectory() as tmp:
        first_cut = folds[0].fit_cutoff
        first_fit = frame[frame["match_date"] < first_cut]
        frozen, frozen_result, frozen_path = fit("refit", first_fit, first_cut, tmp)
        record(f"frozen@{first_cut.date()}", "refit_frozen", first_fit, first_cut, frozen_result, frozen_path)
        frozen_fp = ht.fingerprint(frozen_path)
        reason = disqualification("refit_frozen", train_result=frozen_result)
        if reason is not None:
            disqualify("refit_frozen", f"frozen@{first_cut.date()}", reason)
        for fold in folds:
            fit_df, ev = ft.split_fold(frame, fold)
            if len(ev) < MIN_FOLD_EVAL:
                skipped.append({"fold": fold.label, "n_eval": int(len(ev))})
                logger.warning("fold %s skipped: %d eval rows < %d", fold.label, len(ev), MIN_FOLD_EVAL)
                continue
            truth = realised(ev)

            # The shared HT side: abort on a fit the HT trainer would refuse.
            ht_model, ht_result, ht_path = fit("ht_side", fit_df, fold.fit_cutoff, tmp)
            record(fold.label, "ht_side", fit_df, fold.fit_cutoff, ht_result, ht_path)
            disqualification("ht_side", train_result=ht_result)
            ht_mats = ht.second_half_matrices(ht_model, ev)  # model-agnostic serve-path matrix builder
            bad = invalid_matrices(ht_mats)
            if bad:
                raise RuntimeError(f"ht_side fold {fold.label}: {bad}")
            p_ht = np.vstack([one_x_two(P) for P in ht_mats])

            arm_probs: Dict[str, Dict[str, np.ndarray]] = {
                "served": model_arm_probabilities(ht_mats, ht.second_half_matrices(served, ev))
            }
            for arm in WALK_FORWARD_ARMS:
                if arm not in arms:
                    continue  # disqualified at an earlier fold: never fitted again
                model, result, path = fit(arm, fit_df, fold.fit_cutoff, tmp)
                record(fold.label, arm, fit_df, fold.fit_cutoff, result, path)
                reason = disqualification(arm, train_result=result)
                if reason is not None:
                    disqualify(arm, fold.label, reason)
                    continue
                arm_probs[arm] = model_arm_probabilities(ht_mats, ht.second_half_matrices(model, ev))
            if "refit_frozen" in arms:
                arm_probs["refit_frozen"] = model_arm_probabilities(ht_mats, ht.second_half_matrices(frozen, ev))
            w = ft.decay_weights(fit_df, fold.fit_cutoff, decay)
            arm_probs["base_league"] = base_probabilities(fit_df, ev, None)
            arm_probs["base_league_dw"] = base_probabilities(fit_df, ev, w)
            arm_probs["base_cond"] = conditional_probabilities(fit_df, ev, None, p_ht)
            arm_probs["base_cond_dw"] = conditional_probabilities(fit_df, ev, w, p_ht)
            for arm in list(arm_probs):
                reason = disqualification(arm, probs=arm_probs[arm])  # raises for served / the constants
                if reason is not None:
                    disqualify(arm, fold.label, reason)
                    del arm_probs[arm]
            for arm in arms:
                for mk, s in score_arm(arm_probs[arm], truth).items():
                    scores[arm][mk].append(s)
                probs_htft[arm].append(arm_probs[arm][PRIMARY])
                if H2_1X2 in arm_probs[arm]:
                    probs_h2[arm].append(arm_probs[arm][H2_1X2])
            truth_htft.append(truth[PRIMARY])
            truth_h2.append(truth[H2_1X2])
            truth_ht.append(truth[HT_1X2])
            ht_side_probs.append(p_ht)
            fold_of.append(np.full(len(ev), fold.label, dtype=object))
            dates.append(ev["match_date"].to_numpy())
            served_known.append(
                np.array(
                    [int(h in served_teams) + int(a in served_teams) for h, a in zip(ev["home_team"], ev["away_team"])],
                    dtype=float,
                )
            )
            logger.info("fold %s: fit=%d eval=%d arms=%s", fold.label, len(fit_df), len(ev), ",".join(arms))

    if not fold_of:
        raise RuntimeError(f"no fold had >= {MIN_FOLD_EVAL} eval rows — nothing was scored")
    cat = {a: {mk: np.concatenate(v) for mk, v in scores[a].items() if v} for a in arms}
    folds_arr = np.concatenate(fold_of)
    y = np.concatenate(truth_htft)
    y_h2 = np.concatenate(truth_h2)
    known = np.concatenate(served_known)
    espn_era = pd.to_datetime(np.concatenate(dates), utc=True) >= ESPN_ERA_START
    fold_labels = list(dict.fromkeys(folds_arr.tolist()))
    p_ht_all = np.vstack(ht_side_probs)
    y_ht = np.concatenate(truth_ht)

    def pair(a: str, b: str, mk: str = PRIMARY, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        sel = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        out = ft.paired_delta(cat[a][mk][sel], cat[b][mk][sel], folds_arr[sel])
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

    challengers = [a for a in arms if a in CHALLENGERS]
    report: Dict[str, Any] = {
        "n_eval": int(len(y)),
        "folds": fold_labels,
        "skipped_folds": skipped,
        "disqualified": disqualified,
        "duplicated_match_ids": dups,
        "frame": {**frame_audit, "rows_scored_from": int(len(frame))},
        "served_artifact": served_fp,
        "frozen_artifact": frozen_fp,
        "ht_side": {
            "config": cfgs["ht_side"].name,
            "brier_ht_1x2": float(np.mean(ft.brier_multiclass(p_ht_all, y_ht))),
            "calibration_ht_1x2": ht.draw_calibration(p_ht_all, y_ht),
        },
        "brier": {a: {mk: (float(np.mean(cat[a][mk])) if mk in cat[a] else None) for mk in MARKETS} for a in arms},
        "calibration_ht_ft": {a: htft_calibration(np.vstack(probs_htft[a]), y) for a in arms},
        "calibration_h2_1x2": {a: ht.draw_calibration(np.vstack(probs_h2[a]), y_h2) for a in arms if probs_h2[a]},
        "fits": fits,
        "vs_served": {a: {mk: pair(a, "served", mk) for mk in MARKETS if mk in cat[a]} for a in arms if a != "served"},
        "vs_matched_constant": {
            a: {"constant": matched_constant(a), **pair(a, matched_constant(a))} for a in ["served"] + challengers
        },
        "vs_conditional_constant": {
            a: {"constant": matched_conditional(a), **pair(a, matched_conditional(a))} for a in ["served"] + challengers
        },
        "head_to_head": {
            f"{a} - {b}": pair(a, b)
            for a, b in (
                ("refit_frozen", "refit"),
                ("refit", "pinned"),
                ("refit", "base_cond_dw"),
                ("served", "base_cond"),
                ("base_cond", "base_league"),
                ("base_cond_dw", "base_league_dw"),
            )
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
        "era": {
            era: {
                "n": int(mask.sum()),
                **{
                    a: {
                        "vs_served": pair(a, "served", mask=mask),
                        "vs_matched_constant": pair(a, matched_constant(a), mask=mask),
                        "vs_conditional_constant": pair(a, matched_conditional(a), mask=mask),
                    }
                    for a in challengers
                },
            }
            for era, mask in (("football_data_to_2026_05", ~espn_era), ("espn_2026_06_on", espn_era))
            if mask.sum() > 1
        },
        "served_known_strata": {
            str(k): {
                "n": int((known == k).sum()),
                **{a: pair(a, "served", mask=known == k) for a in challengers if (known == k).sum() > 1},
            }
            for k in (0, 1, 2)
        },
        "verdicts_vs_served": {a: verdict(a, "served") for a in challengers},
        "verdicts_vs_matched_constant": {a: verdict(a, matched_constant(a)) for a in challengers},
        "verdicts_vs_conditional_constant": {a: verdict(a, matched_conditional(a)) for a in challengers},
    }
    return report


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def format_report(r: Dict[str, Any]) -> str:
    fr = r["frame"]
    L = [
        f"2H Dixon-Coles A/B — {r['n_eval']} eval matches, folds {r['folds'][0]}..{r['folds'][-1]}"
        f" ({len(r['folds'])}; skipped {r['skipped_folds'] or 'none'}); duplicated match_ids: "
        f"{r['duplicated_match_ids']}",
        f"frame: {fr['second_half_rows']} 2H rows of {fr['ht_rows']} HT rows; dropped: no FT "
        f"{len(fr['dropped_no_full_time'])}, FT<HT {len(fr['dropped_ft_below_ht'])}, other {len(fr['dropped_other'])}",
    ]
    for arm, d in r["disqualified"].items():
        L.append(f"DISQUALIFIED {arm} at fold {d['fold']}: {d['reason']} (cannot be served; not scored)")
    for name in ("served_artifact", "frozen_artifact"):
        a = r[name]
        L.append(
            f"{name}: {a['teams']} teams, league baselines {a['has_league_baselines']}, rho {_f(a['rho'], '.4f')},"
            f" sha256 {a['sha256'][:12]}"
        )
    hs = r["ht_side"]
    c = hs["calibration_ht_1x2"]
    L.append(
        f"shared HT side ({hs['config']}, walk-forward): HT 1x2 Brier {hs['brier_ht_1x2']:.5f}"
        + (
            f"  pred H/D/A {c['pred_home']:.3f}/{c['pred_draw']:.3f}/{c['pred_away']:.3f}"
            f"  real {c['real_home']:.3f}/{c['real_draw']:.3f}/{c['real_away']:.3f}"
            if c.get("n")
            else ""
        )
    )
    L += ["", "Brier by arm (lower is better):", f"  {'arm':<15}" + "".join(f"{mk:>14}" for mk in MARKETS)]
    for a, b in r["brier"].items():
        L.append(f"  {a:<15}" + "".join(f"{_f(b[mk], '.5f'):>14}" for mk in MARKETS))
    L += ["", f"{PRIMARY} vs served:"]
    L += [ht._pair_line(a, d[PRIMARY]) for a, d in r["vs_served"].items()]
    L += ["", "2H diagnostics vs served (delta; from the 2H matrix alone, never a gate):"]
    for a, d in r["vs_served"].items():
        if len(d) > 1:
            L.append(f"  {a:<15} " + "  ".join(f"{mk} {_f(d[mk].get('delta'))}" for mk in MARKETS[1:] if mk in d))
    L += ["", f"{PRIMARY} vs the matched constants (decayed arms vs the *_dw twins):"]
    L += [ht._pair_line(f"{a} vs {d['constant']}", d) for a, d in r["vs_matched_constant"].items()]
    L += [ht._pair_line(f"{a} vs {d['constant']}", d) for a, d in r["vs_conditional_constant"].items()]
    L += ["", "head to head:"]
    L += [ht._pair_line(k, v) for k, v in r["head_to_head"].items()]
    L += ["", f"{PRIMARY} by era:"]
    for era, block in r["era"].items():
        L.append(f"  [{era}] n={block['n']}")
        for a, v in block.items():
            if isinstance(v, dict):
                L.append(ht._pair_line(f"{a} vs served", v["vs_served"]))
                L.append(ht._pair_line(f"{a} vs constant", v["vs_matched_constant"]))
                L.append(ht._pair_line(f"{a} vs conditional", v["vs_conditional_constant"]))
    if r["per_fold_vs_served"]:
        L += ["", f"{PRIMARY} per-fold delta vs served:"]
        L.append(f"  {'arm':<15}" + "".join(f"{lab[2:]:>9}" for lab in r["folds"]))
        for a, rows in r["per_fold_vs_served"].items():
            L.append(f"  {a:<15}" + "".join(f"{row['delta']:>+9.4f}" for row in rows))
        first = next(iter(r["per_fold_vs_served"].values()))
        L.append(f"  {'n':<15}" + "".join(f"{row['n']:>9}" for row in first))
    L += ["", f"{PRIMARY} calibration (max |mean predicted - realised| over the 9 cells; draw_draw; argmax hit):"]
    for a, cal in r["calibration_ht_ft"].items():
        if cal.get("n"):
            L.append(
                f"  {a:<15} max gap {cal['max_abs_gap']:.4f} ({cal['worst_cell']})  draw_draw pred "
                f"{cal['pred']['draw_draw']:.3f} real {cal['real']['draw_draw']:.3f}  hit {cal['argmax_hit']:.3f}"
            )
    L += ["", "2H 1x2 calibration (mean predicted vs realised; argmax share; argmax hit):"]
    for a, cal in r["calibration_h2_1x2"].items():
        if cal.get("n"):
            L.append(
                f"  {a:<15} pred H/D/A {cal['pred_home']:.3f}/{cal['pred_draw']:.3f}/{cal['pred_away']:.3f}  "
                f"real {cal['real_home']:.3f}/{cal['real_draw']:.3f}/{cal['real_away']:.3f}  "
                f"argmax {cal['argmax_home']:.2f}/{cal['argmax_draw']:.2f}/{cal['argmax_away']:.2f}"
                f"  hit {cal['argmax_hit']:.3f}"
            )
    L += ["", f"{PRIMARY} vs served by teams the served artifact knows:"]
    for k, blk in r["served_known_strata"].items():
        L.append(
            f"  known={k} n={blk['n']}: "
            + ", ".join(f"{a} {_f(v.get('delta'))}" for a, v in blk.items() if isinstance(v, dict))
        )
    unconv: Dict[str, int] = {}
    for f in r["fits"]:
        if f.get("converged_at") is None:
            unconv[f["arm"]] = unconv.get(f["arm"], 0) + 1
    L += ["", f"fits that did not converge (per arm): {unconv or 'none'}"]
    L += ["", "Verdicts (§4.3 gate on ht_ft vs served; 'strict' also needs the t(n_folds-1) clustered bound < 0):"]
    for a in r["verdicts_vs_served"]:
        for key in ("verdicts_vs_served", "verdicts_vs_matched_constant", "verdicts_vs_conditional_constant"):
            v = r[key][a]
            L.append(f"  {a:<15} strict={'SHIP' if v['ship_strict'] else 'no'}  {v['line']}")
    if not r["verdicts_vs_served"]:
        L.append("  no challenger survived — nothing can ship")
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2025-08-01", help="first eval day (monthly folds from its month)")
    p.add_argument("--end", default=None, help="last eval day (default: newest second-half match)")
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
