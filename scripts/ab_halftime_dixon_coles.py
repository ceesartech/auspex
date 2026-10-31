"""A/B gate for the soccer HALF-TIME Dixon-Coles artifact (dixon_coles_ht_soccer).

WHAT IS BEING TESTED
--------------------
The served half-time artifact was written 2026-06-04 by
scripts/train_halftime_dixon_coles.py and has never been gated (the
trainer writes straight into production; model_config.py: "whatever they
fit is what serves"). It knows 175 teams, was fitted by the pre-2026-09
fitter (attack AND defense normalised every iteration), and serves every
unknown team from one global baseline: home 0.60 + a CONFIGURED 0.25 home
bump, away 0.60 — a 1.42 home/away lambda ratio. Live it picks home on
93% of matches at an average 39.5% and hits 35.0%, while the realised HT
draw rate is 39.7% (docs/SYSTEM_AUDIT_AND_ROADMAP.md, 2026-10-30 night).

The 2026-10-31 half-time backfill moved ~2.7k ESPN matches into
matches.half_time_home/away, so a refit now sees 108k matches. Re-running
the trainer as written is NOT that refit: it fits the OLDEST 80% by date
and only scores the newest 20%, so none of the new rows enter the fit.

ARMS (all scored on identical matches, through the serve path)
  served         the on-disk production artifact (or --served-artifact).
  trainer        the artifact the trainer itself wrote (--trainer-artifact),
                 i.e. the literal "re-run the trainer" option. Scored only
                 on rows after its own fit cutoff (asserted).
  pinned         walk-forward refit, the trainer's pinned legacy config
                 (max_goals 6, no decay, inert regularization, one global
                 baseline + the configured 0.25 home bump) on ALL data
                 before each eval month — "the trainer, fitted on all data".
  pinned_league  pinned, but home/away baselines fitted from the data
                 (per_league_baselines) — isolates the home-bump question.
  refit          walk-forward refit with DIXON_COLES_CONFIG as measured for
                 the FULL-TIME artifact (decay, shrinkage prior 20, per-league
                 baselines, max_goals 10). Not measured on HT goals before.
  refit_frozen   refit's config fitted ONCE at the first fold's cutoff and
                 scored on every fold: production serves a frozen artifact
                 (the HT trainer is run by hand, not by the weekly DAG), so
                 this is the arm whose staleness matches what would ship.
  base_global / base_league / base_league_dw
                 walk-forward constant HT outcome rates (global; per-league
                 shrunk with k=200 effective matches; the same with the
                 refit arm's recency weighting). base_league is the honest
                 comparator: a model that beats the served artifact but loses
                 to a per-league constant has not earned much.

PROTOCOL
  * Monthly walk-forward (ab_soccer_dixon_coles_refit.build_folds /
    split_fold): each refit arm is fitted on match_date < month start only.
  * Serve-path fidelity: precompute_predictions loads the artifact into
    DixonColesPredictor(DIXON_COLES_CONFIG), takes lambdas_for_match(home,
    away, league=league_id), builds build_dc_matrix(..., MAX_GOALS_DERIVE)
    and reads derive_soccer_halftime_markets — no reconciliation step. Every
    arm here goes through exactly that, and refit arms are round-tripped
    through save() + that loader so serialisation cannot flatter them.
  * Paired statistics clustered by fold (ab_soccer_dixon_coles_refit.
    paired_delta / ship_verdict): per-match differences, fold-clustered SE.
  * Gate (§4.3): primary market match_result_ht (3-way Brier, sum over
    classes), challenger vs served: dBrier <= -0.005 and both the per-match
    and the fold-clustered 95% upper bounds below zero. Secondary markets
    (over_under_ht 0.5/1.5/2.5, btts_ht) and draw calibration are reported,
    never used to rescue a primary-market miss. ht_ft (the 9-way HT/FT double
    result) is scored too: production builds it by convolving the HT matrix
    with the SECOND-HALF artifact's (derive_soccer_htft_markets), so every arm
    here is paired with the same served second-half artifact and a HT swap's
    effect on that market is measured, not assumed.

CONFOUNDS, PRINTED NOT HIDDEN
  * The served artifact's own training window is not recorded in it. If it
    overlaps the early eval folds the served arm is in-sample there — a bias
    AGAINST the challengers. Per-fold deltas are printed so an early-fold
    effect is visible.
  * Team coverage: the served arm knows 175 teams; strata by how many of a
    match's teams the served arm knows are printed.
  * Source: until 2026-05 the eval rows are football-data (16 leagues); from
    2026-06 they are ESPN results-path rows (up to 26 leagues), which is what
    production now serves. Both eras are reported separately.
  * The clustered 95% bound in ship_verdict uses z=2 on ~15 folds; a
    t(n_folds-1) bound is reported next to it and the stricter one decides.

Read-only: SELECTs, plus temp files for the round-trip. Usage (api container):
    python /app/scripts/ab_halftime_dixon_coles.py --out-json /tmp/ab_ht.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_soccer_dixon_coles_refit as ft  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_halftime_dixon_coles")

ML_SRC = "/app/services/ml-models/src"
SERVED_ARTIFACT = "/app/models/production/dixon_coles_ht_soccer/1.0.0/model.bin"
H2_ARTIFACT = "/app/models/production/dixon_coles_h2_soccer/1.0.0/model.bin"
TRAINER_ARTIFACT = "/app/models/staging-ht/dixon_coles_ht_soccer/1.0.0/model.bin"
TRAINER_TRAIN_FRACTION = 0.8  # train_halftime_dixon_coles.py: cutoff = int(n * 0.8)

CLASSES = ("home", "draw", "away")  # == match_outcome 0/1/2 in HT_TRAINING_QUERY
OU_LINES = (0.5, 1.5, 2.5)
MARKETS = ("match_result_ht",) + tuple(f"over_{line}" for line in OU_LINES) + ("btts_ht", "ht_ft")
PRIMARY = "match_result_ht"
HTFT = "ht_ft"
MULTICLASS = {PRIMARY: 3, HTFT: 9}  # ht_ft index = ht_outcome * 3 + ft_outcome (home/draw/away)
HTFT_KEYS = tuple(f"{a}_{b}" for a in CLASSES for b in CLASSES)
REFIT_ARMS = ("pinned", "pinned_league", "refit")
BASE_ARMS = ("base_global", "base_league", "base_league_dw")
DECAYED_ARMS = ("refit", "refit_frozen")  # compared with base_league_dw; the rest with base_league
MIN_FOLD_EVAL = 50  # a near-empty month would count as a full fold in the clustered bound
ESPN_ERA_START = pd.Timestamp("2026-06-01", tz="UTC")  # football-data HT columns end 2026-05/08
LEGACY_SERVED_TEAMS = 175


# ── Pure ─────────────────────────────────────────────────────────────


def arm_overrides() -> Dict[str, Dict[str, Any]]:
    """Hyperparameter overrides on DIXON_COLES_CONFIG per refit arm."""
    from predictors.model_config import LEGACY_DIXON_COLES_HYPERPARAMETERS  # type: ignore

    legacy = dict(LEGACY_DIXON_COLES_HYPERPARAMETERS)
    return {
        "pinned": legacy,
        "pinned_league": {**legacy, "per_league_baselines": True},
        "refit": {},
    }


def _outcome(h: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.where(h > a, 0, np.where(h == a, 1, 2))


def realised(ev: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Outcomes per market from the HT scores (home_score/away_score in the
    HT frame) and, for ht_ft, the full-time scores (ft_home/ft_away)."""
    hs = ev["home_score"].to_numpy(dtype=int)
    as_ = ev["away_score"].to_numpy(dtype=int)
    out: Dict[str, np.ndarray] = {PRIMARY: _outcome(hs, as_)}
    for line in OU_LINES:
        out[f"over_{line}"] = ((hs + as_) > line).astype(float)
    out["btts_ht"] = ((hs >= 1) & (as_ >= 1)).astype(float)
    ft_out = _outcome(ev["ft_home"].to_numpy(dtype=int), ev["ft_away"].to_numpy(dtype=int))
    out[HTFT] = out[PRIMARY] * 3 + ft_out
    return out


def markets_from_matrix(P: np.ndarray, P_2h: np.ndarray) -> Dict[str, Any]:
    """The served HT probabilities for one match, from its HT scoreline
    matrix, plus the HT/FT double result from HT (x) second-half."""
    from predictors.market_derivation import (  # type: ignore
        _fmt_line,
        derive_soccer_halftime_markets,
        derive_soccer_htft_markets,
    )

    m = derive_soccer_halftime_markets(P)
    out: Dict[str, Any] = {PRIMARY: np.array([m[PRIMARY][c] for c in CLASSES], dtype=float)}
    for line in OU_LINES:
        out[f"over_{line}"] = float(m["over_under_ht"][f"over_{_fmt_line(line)}"])
    out["btts_ht"] = float(m["btts_ht"]["yes"])
    htft = derive_soccer_htft_markets(P, P_2h)["ht_ft_double_result"]
    out[HTFT] = np.array([htft[k] for k in HTFT_KEYS], dtype=float)
    return out


def score_arm(probs: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-match Brier per market: multi-class sum for 1x2 and HT/FT, binary otherwise."""
    return {mk: (ft.brier_multiclass if mk in MULTICLASS else ft.brier_binary)(probs[mk], truth[mk]) for mk in MARKETS}


def draw_calibration(proba: np.ndarray, y_idx: np.ndarray) -> Dict[str, Any]:
    """Mean predicted vs realised class rates, and how often each class is the argmax."""
    proba = np.asarray(proba, dtype=float)
    ok = np.isfinite(proba).all(axis=1)
    proba, y = proba[ok], np.asarray(y_idx)[ok]
    if not len(y):
        return {"n": 0}
    pick = np.argmax(proba, axis=1)
    return {
        "n": int(len(y)),
        **{f"pred_{c}": float(proba[:, k].mean()) for k, c in enumerate(CLASSES)},
        **{f"real_{c}": float((y == k).mean()) for k, c in enumerate(CLASSES)},
        **{f"argmax_{c}": float((pick == k).mean()) for k, c in enumerate(CLASSES)},
        "argmax_hit": float((pick == y).mean()),
    }


def trainer_fit_cutoff(frame: pd.DataFrame, n_fit: Optional[int] = None) -> pd.Timestamp:
    """Newest match in the trainer's fit: it sorts by date and fits rows
    [0, n_fit). n_fit defaults to int(n * 0.8) of TODAY's frame, which is
    only right if no older-dated row was added or deleted since the trainer
    ran — so pass the trainer's logged Train= count, and check_trainer_slice
    confirms the slice against the artifact before anything is scored."""
    n = int(n_fit) if n_fit is not None else int(len(frame) * TRAINER_TRAIN_FRACTION)
    return pd.Timestamp(frame["match_date"].iloc[max(n - 1, 0)])


def check_trainer_slice(frame: pd.DataFrame, n_fit: Optional[int], team_attack: Dict[str, Any]) -> None:
    """Raise unless the teams in the reconstructed fit slice are exactly the
    teams the trainer artifact holds — the evidence the cutoff is right."""
    n = int(n_fit) if n_fit is not None else int(len(frame) * TRAINER_TRAIN_FRACTION)
    head = frame.iloc[:n]
    teams = set(head["home_team"]) | set(head["away_team"])
    if teams != set(team_attack):
        raise RuntimeError(
            f"trainer artifact holds {len(team_attack)} teams but rows [0, {n}) of today's frame hold {len(teams)} "
            f"(symmetric difference {len(teams ^ set(team_attack))}) — the fit slice cannot be reconstructed; "
            "pass --trainer-n-fit"
        )


def fingerprint(path: str) -> Dict[str, Any]:
    """sha256, team count and baseline keys of an artifact file."""
    import hashlib

    raw = Path(path).read_bytes()
    data = json.loads(raw)
    return {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "teams": len(data.get("team_attack") or {}),
        "has_league_baselines": bool(data.get("league_baselines")),
        "has_global_baselines": data.get("global_home_baseline") is not None,
        "max_goals": data.get("max_goals"),
        "rho": data.get("rho"),
    }


def invalid_probabilities(probs: Dict[str, np.ndarray]) -> Optional[str]:
    """None when every row of every market is a valid probability, else a
    description. An arm that emits invalid probabilities is DISQUALIFIED
    (it could not be served), never scored on a subset of rows — that would
    run the paired comparisons on different row sets."""
    for mk in MULTICLASS:
        pm = np.asarray(probs[mk], dtype=float)
        bad = ~np.isfinite(pm).all(axis=1) | ~np.isclose(pm.sum(axis=1), 1.0, atol=1e-6)
        if bad.any():
            return f"{int(bad.sum())} of {len(pm)} {mk} rows non-finite or non-normalised"
    for mk in (m for m in MARKETS if m not in MULTICLASS):
        v = np.asarray(probs[mk], dtype=float)
        bad = ~np.isfinite(v) | (v < 0) | (v > 1)
        if bad.any():
            return f"{int(bad.sum())} of {len(v)} {mk} rows invalid"
    return None


def t_upper(delta: Optional[float], se: Optional[float], n_folds: Optional[int]) -> Optional[float]:
    """Clustered 95% upper bound with a t(n_folds-1) multiplier."""
    if delta is None or se is None or not n_folds or n_folds < 2:
        return None
    from scipy.stats import t as student_t

    return float(delta + float(student_t.ppf(0.975, n_folds - 1)) * se)


def clean_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so the dump is valid JSON."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ── Model plumbing (needs ml-models on the path) ─────────────────────


def load_serving(path: str):
    """Exactly precompute_predictions._load_dixon_coles_artifact."""
    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore
    from predictors.poisson_models import DixonColesPredictor  # type: ignore

    model = DixonColesPredictor(DIXON_COLES_CONFIG)
    model.load(path)
    return model


def fit_arm(fit_df: pd.DataFrame, overrides: Dict[str, Any], tmpdir: str, tag: str):
    """Fit one refit arm on the fold's fit frame, then round-trip it through
    save() + the serve loader. Returns (served_model, train_result).

    No reference date is passed: train() ignores one and ages decay weights
    from the fit frame's newest match (PoissonMatchPredictor._time_weights),
    which is what a production fit does too."""
    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore
    from predictors.poisson_models import DixonColesPredictor  # type: ignore

    cfg = dc_replace(DIXON_COLES_CONFIG, hyperparameters={**DIXON_COLES_CONFIG.hyperparameters, **overrides})
    model = DixonColesPredictor(cfg)
    cols = [c for c in ("home_team", "away_team", "home_score", "away_score", "match_date", "league_id")]
    result = model.train(fit_df[cols].copy())
    path = os.path.join(tmpdir, f"{tag}.bin")
    model.save(path)
    return load_serving(path), result


def second_half_matrices(h2, ev: pd.DataFrame) -> List[np.ndarray]:
    """The served second-half matrix per eval row — shared by every HT arm."""
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix  # type: ignore

    rho = float(getattr(h2, "rho", 0.0) or 0.0)
    out = []
    for home, away, league in zip(ev["home_team"], ev["away_team"], ev["league_id"]):
        h_lam, a_lam = h2.lambdas_for_match(home, away, league=league)
        out.append(build_dc_matrix(float(h_lam), float(a_lam), rho, max_goals=MAX_GOALS_DERIVE))
    return out


def arm_probabilities(model, ev: pd.DataFrame, h2_mats: List[np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Serve-path HT probabilities for every eval row + teams known (0/1/2)."""
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix  # type: ignore

    rho = float(getattr(model, "rho", 0.0) or 0.0)
    n = len(ev)
    out: Dict[str, np.ndarray] = {mk: np.full((n, k), np.nan) for mk, k in MULTICLASS.items()}
    for mk in MARKETS:
        if mk not in MULTICLASS:
            out[mk] = np.full(n, np.nan)
    home, away, league = ev["home_team"].to_numpy(), ev["away_team"].to_numpy(), ev["league_id"].to_numpy()
    for i in range(n):
        h_lam, a_lam = model.lambdas_for_match(home[i], away[i], league=league[i])
        P = build_dc_matrix(float(h_lam), float(a_lam), rho, max_goals=MAX_GOALS_DERIVE)
        m = markets_from_matrix(P, h2_mats[i])
        for mk in MARKETS:
            out[mk][i] = m[mk]
    attack = getattr(model, "team_attack", {}) or {}
    known = np.array([int(home[i] in attack) + int(away[i] in attack) for i in range(n)], dtype=float)
    return out, known


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


FT_SCORES_SQL = """
    SELECT m.id::text AS match_id, m.home_score AS ft_home, m.away_score AS ft_away
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'soccer' AND m.status = 'finished'
      AND m.half_time_home IS NOT NULL AND m.half_time_away IS NOT NULL
"""


def load_full_time_scores(database_url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        return pd.read_sql(FT_SCORES_SQL, engine)
    finally:
        engine.dispose()


# ── Run ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    sys.path.insert(0, ML_SRC)
    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore
    from train_halftime_dixon_coles import HT_TRAINING_QUERY  # type: ignore
    from utils.training_data import load_training_frame  # type: ignore

    frame = load_training_frame(database_url=args.database_url, query=HT_TRAINING_QUERY)
    frame = frame.merge(load_full_time_scores(args.database_url), on="match_id", how="left", validate="many_to_one")
    no_ft = frame[["ft_home", "ft_away"]].isna().any(axis=1)
    if no_ft.any():
        # Finished matches with a half-time but no full-time score are a data
        # defect (2 of 108,228 on 2026-10-31); dropped for every arm, counted.
        logger.error(
            "%d HT row(s) have no full-time score — dropped: %s",
            int(no_ft.sum()),
            frame.loc[no_ft, "match_id"].tolist(),
        )
    frame = frame[~no_ft].copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    frame = frame.sort_values("match_date", kind="stable").reset_index(drop=True)
    dups = int(frame["match_id"].duplicated().sum())
    if dups:
        # The trainer fits this same frame, so this is a finding about it too.
        logger.error("HT frame carries %d duplicated match_id rows — dropped for scoring", dups)
        frame = frame.drop_duplicates("match_id", keep="first").reset_index(drop=True)
    if not frame["match_outcome"].isin([0, 1, 2]).all():
        raise RuntimeError("match_outcome outside {0,1,2} — the HT frame is not what this harness assumes")
    end = args.end or frame["match_date"].max().strftime("%Y-%m-%d")
    folds = ft.build_folds(args.start, end)
    logger.info(
        "frame: %d HT matches %s..%s; %d folds",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
        len(folds),
    )

    served_fp = fingerprint(args.served_artifact)
    legacy = served_fp["teams"] == LEGACY_SERVED_TEAMS and not served_fp["has_league_baselines"]
    if args.served_artifact == SERVED_ARTIFACT and not legacy:
        raise RuntimeError(
            f"production HT artifact is not the 2026-06-04 legacy fit ({served_fp}) — it may have been overwritten "
            "by the trainer; point --served-artifact at the backup to score the real incumbent"
        )
    served = load_serving(args.served_artifact)
    h2 = load_serving(args.h2_artifact)
    h2_fp = fingerprint(args.h2_artifact)
    trainer = None
    trainer_fp = None
    t_cutoff = None
    if args.trainer_artifact and Path(args.trainer_artifact).exists():
        trainer_fp = fingerprint(args.trainer_artifact)
        trainer = load_serving(args.trainer_artifact)
        check_trainer_slice(frame, args.trainer_n_fit, trainer.team_attack)
        t_cutoff = trainer_fit_cutoff(frame, args.trainer_n_fit)
    else:
        logger.warning("no trainer artifact at %s — the trainer arm is skipped", args.trainer_artifact)

    overrides = arm_overrides()
    decay = float(DIXON_COLES_CONFIG.hyperparameters.get("time_decay", 0.0))
    arms = (
        ["served"]
        + (["trainer"] if trainer is not None else [])
        + list(REFIT_ARMS)
        + ["refit_frozen"]
        + list(BASE_ARMS)
    )
    scores: Dict[str, Dict[str, List[np.ndarray]]] = {a: {mk: [] for mk in MARKETS} for a in arms}
    probs_1x2: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    truth_1x2: List[np.ndarray] = []
    fold_of: List[np.ndarray] = []
    dates: List[np.ndarray] = []
    served_known: List[np.ndarray] = []
    fits: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    disqualified: Dict[str, Dict[str, Any]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        first_cut = folds[0].fit_cutoff
        frozen, frozen_result = fit_arm(frame[frame["match_date"] < first_cut], overrides["refit"], tmp, "frozen")
        fits.append(
            {
                "fold": f"frozen@{first_cut.date()}",
                "arm": "refit_frozen",
                "n_fit": int((frame["match_date"] < first_cut).sum()),
                "teams": int(frozen_result.get("team_count") or 0),
                "converged_at": frozen_result.get("iterations"),
                "leagues": int(frozen_result.get("league_count") or 0),
            }
        )
        for fold in folds:
            fit_df, ev = ft.split_fold(frame, fold)
            if len(ev) < MIN_FOLD_EVAL:
                skipped.append({"fold": fold.label, "n_eval": int(len(ev))})
                logger.warning("fold %s skipped: %d eval rows < %d", fold.label, len(ev), MIN_FOLD_EVAL)
                continue
            truth = realised(ev)
            h2_mats = second_half_matrices(h2, ev)
            arm_probs: Dict[str, Dict[str, np.ndarray]] = {}
            arm_probs["served"], known = arm_probabilities(served, ev, h2_mats)
            if trainer is not None:
                if not (ev["match_date"] > t_cutoff).all():
                    raise RuntimeError(f"fold {fold.label} overlaps the trainer's fit (cutoff {t_cutoff})")
                arm_probs["trainer"], _ = arm_probabilities(trainer, ev, h2_mats)
            for arm in REFIT_ARMS:
                if arm in disqualified:
                    continue
                model, result = fit_arm(fit_df, overrides[arm], tmp, f"{arm}_{fold.label}")
                arm_probs[arm], arm_known = arm_probabilities(model, ev, h2_mats)
                fits.append(
                    {
                        "fold": fold.label,
                        "arm": arm,
                        "n_fit": int(len(fit_df)),
                        "teams": int(result.get("team_count") or 0),
                        "converged_at": result.get("iterations"),
                        "leagues": int(result.get("league_count") or 0),
                        "rho": result.get("rho"),
                        "fitted_total": result.get("fitted_mean_total_goals"),
                        "observed_total": result.get("observed_mean_total_goals"),
                        "both_known_share": float((arm_known == 2).mean()),
                        "newest_fit_gap_days": float(
                            (fold.fit_cutoff - fit_df["match_date"].max()).total_seconds() / 86400
                        ),
                    }
                )
            arm_probs["refit_frozen"], _ = arm_probabilities(frozen, ev, h2_mats)
            arm_probs["base_global"] = base_probabilities(
                fit_df.assign(league_id=None), ev.assign(league_id=None), None
            )
            arm_probs["base_league"] = base_probabilities(fit_df, ev, None)
            arm_probs["base_league_dw"] = base_probabilities(
                fit_df, ev, ft.decay_weights(fit_df, fold.fit_cutoff, decay)
            )
            for arm in list(arm_probs):
                reason = invalid_probabilities(arm_probs[arm])
                if reason is None:
                    continue
                if arm == "served" or arm in BASE_ARMS:
                    raise RuntimeError(f"{arm} fold {fold.label}: {reason}")
                disqualified[arm] = {"fold": fold.label, "reason": reason}
                logger.error("%s DISQUALIFIED at fold %s: %s", arm, fold.label, reason)
                arms.remove(arm)
                del arm_probs[arm]
            for arm in arms:
                s = score_arm(arm_probs[arm], truth)
                for mk in MARKETS:
                    scores[arm][mk].append(s[mk])
                probs_1x2[arm].append(arm_probs[arm][PRIMARY])
            truth_1x2.append(truth[PRIMARY])
            fold_of.append(np.full(len(ev), fold.label, dtype=object))
            dates.append(ev["match_date"].to_numpy())
            served_known.append(known)
            logger.info("fold %s: fit=%d eval=%d", fold.label, len(fit_df), len(ev))

    cat = {a: {mk: np.concatenate(scores[a][mk]) for mk in MARKETS} for a in arms}
    probs_1x2 = {a: probs_1x2[a] for a in arms}
    folds_arr = np.concatenate(fold_of)
    y = np.concatenate(truth_1x2)
    known = np.concatenate(served_known)
    espn_era = pd.to_datetime(np.concatenate(dates), utc=True) >= ESPN_ERA_START
    fold_labels = list(dict.fromkeys(folds_arr.tolist()))

    def pair(a: str, b: str, mk: str = PRIMARY, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        sel = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        out = ft.paired_delta(cat[a][mk][sel], cat[b][mk][sel], folds_arr[sel])
        out["upper_95_cluster_t"] = t_upper(out["delta_cluster"], out["se_cluster"], out["n_folds"])
        return out

    def verdict(a: str, b: str) -> Dict[str, Any]:
        p = pair(a, b)
        v = ft.ship_verdict(
            p["delta"], p["se"], b, delta_cluster=p["delta_cluster"], se_cluster=p["se_cluster"], n_folds=p["n_folds"]
        )
        v["upper_95_cluster_t"] = p["upper_95_cluster_t"]
        v["ship_strict"] = bool(v["ship"] and p["upper_95_cluster_t"] is not None and p["upper_95_cluster_t"] < 0)
        return v

    def matched_constant(a: str) -> str:
        return "base_league_dw" if a in DECAYED_ARMS else "base_league"

    challengers = [a for a in arms if a != "served" and a not in BASE_ARMS]
    report: Dict[str, Any] = {
        "n_eval": int(len(y)),
        "folds": fold_labels,
        "skipped_folds": skipped,
        "disqualified": disqualified,
        "duplicated_match_ids": dups,
        "rows_without_full_time_dropped": int(no_ft.sum()),
        "served_artifact": served_fp,
        "h2_artifact": h2_fp,
        "trainer_artifact": trainer_fp,
        "trainer_fit_cutoff": str(t_cutoff) if t_cutoff is not None else None,
        "brier": {a: {mk: float(np.mean(cat[a][mk])) for mk in MARKETS} for a in arms},
        "calibration_1x2": {a: draw_calibration(np.vstack(probs_1x2[a]), y) for a in arms},
        "fits": fits,
        "vs_served": {a: {mk: pair(a, "served", mk) for mk in MARKETS} for a in arms if a != "served"},
        "vs_matched_constant": {
            a: {"constant": matched_constant(a), **pair(a, matched_constant(a))} for a in ["served"] + challengers
        },
        "head_to_head": {
            f"{a} - {b}": pair(a, b)
            for a, b in (
                ("refit", "trainer"),
                ("refit", "pinned_league"),
                ("refit_frozen", "refit"),
                ("refit", "base_league_dw"),
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
            for a in challengers
        },
        "era": {
            era: {
                "n": int(mask.sum()),
                **{
                    a: {
                        "vs_served": pair(a, "served", mask=mask),
                        "vs_matched_constant": pair(a, matched_constant(a), mask=mask),
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
    }
    return report


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def _pair_line(name: str, p: Dict[str, Any]) -> str:
    return (
        f"  {name:<28} {_f(p.get('delta'))} ±{_f(p.get('se'), '.5f')}   cluster {_f(p.get('delta_cluster'))} "
        f"±{_f(p.get('se_cluster'), '.5f')}   t-bound {_f(p.get('upper_95_cluster_t'))}   n={p.get('n')}"
    )


def format_report(r: Dict[str, Any]) -> str:
    L = [
        f"HT Dixon-Coles A/B — {r['n_eval']} eval matches, folds {r['folds'][0]}..{r['folds'][-1]}"
        f" ({len(r['folds'])}; skipped {r['skipped_folds'] or 'none'});"
        f" duplicated match_ids: {r['duplicated_match_ids']}"
    ]
    for arm, d in r["disqualified"].items():
        L.append(f"DISQUALIFIED {arm} at fold {d['fold']}: {d['reason']} (cannot be served; not scored)")
    sf = r["served_artifact"]
    L.append(
        f"served artifact: {sf['teams']} teams, league baselines {sf['has_league_baselines']},"
        f" sha256 {sf['sha256'][:12]}"
    )
    if r["trainer_artifact"]:
        tf = r["trainer_artifact"]
        L.append(
            f"trainer artifact: {tf['teams']} teams, sha256 {tf['sha256'][:12]}, fit cutoff {r['trainer_fit_cutoff']}"
        )
    L += ["", "Brier by arm (lower is better):", f"  {'arm':<15}" + "".join(f"{mk:>17}" for mk in MARKETS)]
    for a, b in r["brier"].items():
        L.append(f"  {a:<15}" + "".join(f"{b[mk]:>17.5f}" for mk in MARKETS))
    L += ["", f"{PRIMARY} vs served:"]
    L += [_pair_line(a, d[PRIMARY]) for a, d in r["vs_served"].items()]
    L += ["", "secondary markets vs served (delta):"]
    for a, d in r["vs_served"].items():
        L.append(f"  {a:<15} " + "  ".join(f"{mk} {_f(d[mk].get('delta'))}" for mk in MARKETS[1:]))
    L += ["", f"{PRIMARY} vs the recency-matched constant (refit arms vs base_league_dw, others vs base_league):"]
    L += [_pair_line(f"{a} vs {d['constant']}", d) for a, d in r["vs_matched_constant"].items()]
    L += ["", "head to head:"]
    L += [_pair_line(k, v) for k, v in r["head_to_head"].items()]
    L += ["", f"{PRIMARY} by era:"]
    for era, block in r["era"].items():
        L.append(f"  [{era}] n={block['n']}")
        for a, v in block.items():
            if isinstance(v, dict):
                L.append(_pair_line(f"{a} vs served", v["vs_served"]))
                L.append(_pair_line(f"{a} vs constant", v["vs_matched_constant"]))
    L += ["", f"{PRIMARY} per-fold delta vs served:"]
    labels = r["folds"]
    L.append(f"  {'arm':<15}" + "".join(f"{lab[2:]:>9}" for lab in labels))
    for a, rows in r["per_fold_vs_served"].items():
        L.append(f"  {a:<15}" + "".join(f"{row['delta']:>+9.4f}" for row in rows))
    L.append(f"  {'n':<15}" + "".join(f"{row['n']:>9}" for row in next(iter(r["per_fold_vs_served"].values()))))
    L += ["", "1x2 calibration (mean predicted vs realised; argmax share; argmax hit):"]
    for a, c in r["calibration_1x2"].items():
        if c.get("n"):
            L.append(
                f"  {a:<15} pred H/D/A {c['pred_home']:.3f}/{c['pred_draw']:.3f}/{c['pred_away']:.3f}  "
                f"real {c['real_home']:.3f}/{c['real_draw']:.3f}/{c['real_away']:.3f}  "
                f"argmax {c['argmax_home']:.2f}/{c['argmax_draw']:.2f}/{c['argmax_away']:.2f}"
                f"  hit {c['argmax_hit']:.3f}"
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
    L += ["", "Verdicts (§4.3 gate on the primary market; 'strict' also needs the t(n_folds-1) clustered bound < 0):"]
    for a in r["verdicts_vs_served"]:
        for key in ("verdicts_vs_served", "verdicts_vs_matched_constant"):
            v = r[key][a]
            L.append(f"  {a:<15} strict={'SHIP' if v['ship_strict'] else 'no'}  {v['line']}")
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2025-08-01", help="first eval day (monthly folds from its month)")
    p.add_argument("--end", default=None, help="last eval day (default: newest HT match)")
    p.add_argument("--served-artifact", default=SERVED_ARTIFACT)
    p.add_argument("--trainer-artifact", default=TRAINER_ARTIFACT)
    p.add_argument("--h2-artifact", default=H2_ARTIFACT, help="second-half artifact every arm's ht_ft is built with")
    p.add_argument("--trainer-n-fit", type=int, default=None, help="the trainer's logged Train= row count")
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
        Path(args.out_json).write_text(json.dumps(clean_json(report), indent=2, allow_nan=False))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
