"""NHL regulation-scoreline structure for the Dixon-Coles artifact (dixon_coles_nhl).

WHY. The served NHL artifact is a Poisson fit (rho forced to 0) on FINAL
scores, and derive_hockey_markets reads its diagonal as "tied after
regulation". Both halves of that are wrong:

  * a final NHL score is never tied. An overtime or shootout winner is
    credited one goal, so a game tied k-k after 60 minutes ends (k+1)-k or
    k-(k+1). A fit on final scores spreads that mass over the diagonal, where
    nothing is ever realised, and over every margin;
  * even on REGULATION scores an independent Poisson has the wrong shape.
    Over the 6,551 games of 2021-01..2025-09, ties ran 1.35x what it
    predicts, one-goal margins 0.60x and three-goal margins 1.5x: the late
    goalie pull — a side trailing by one either equalises or concedes into
    the empty net, a side trailing by two concedes a third.

So the served regulation-tie mass sat at ~15% against ~22-25% realised, and
the final-score markets carried tie cells and the wrong total parity.

THE MODEL. The regulation scoreline is

    P_R(i, j)  ∝  Pois(i; λ_home) · Pois(j; λ_away) · exp(θ_b(|i - j|))

on the truncated (max_goals+1)^2 grid build_dc_matrix serves, with one
log-factor θ_b per margin bucket b of ``margin_buckets`` (an absolute margin
past the last bucket has θ = 0; all θ = 0 is the plain Poisson). λ is
attack · defense · baseline exactly as PoissonMatchPredictor. EVERY
parameter — the team strengths and θ — is fitted jointly by weighted maximum
likelihood on the regulation scorelines (fit_structure): θ is a parameter of
the likelihood, not a transform of emitted probabilities. (Probability
calibration is a closed lever — docs/SYSTEM_AUDIT_AND_ROADMAP.md §4 — and
nothing here recalibrates an output.)

The final scoreline P_F moves each regulation tie (k, k) to (k+1, k) with
probability ``ot_home_share`` and to (k, k+1) otherwise — the NHL's scoring
rule, not a fit. ``ot_home_share`` is a weighted maximum-likelihood estimate:
the home share of the fit frame's overtime / shootout games.

Regulation markets (regulation_1x2, double_chance) read P_R; every
final-score market (totals, puck line, correct score, clean sheet, win to
nil) reads P_F — see market_derivation.derive_hockey_markets(P, final=...).

An artifact whose payload has no ``regulation_structure`` serves exactly as
before (one final-score matrix for every market). The structure is gated by
scripts/ab_nhl_regulation_ties.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .poisson_models import _league_values, poisson_pmf_grid

logger = logging.getLogger(__name__)

PAYLOAD_KEY = "regulation_structure"
STRUCTURE_VERSION = 1
# A margin log-factor past +/-3 (a 20x tilt of a margin's mass) is not a
# hockey number; a payload carrying one is refused.
MAX_ABS_LOG_FACTOR = 3.0
# Newton steps on theta are capped so one step cannot leave the region the
# likelihood was evaluated in.
_MAX_THETA_STEP = 0.5


# ── Pure ─────────────────────────────────────────────────────────────


def regulation_scores(
    home_score: Sequence[int], away_score: Sequence[int], went_past_regulation: Sequence[bool]
) -> Tuple[np.ndarray, np.ndarray]:
    """(regulation home goals, regulation away goals) per game.

    A game decided in regulation ended on its regulation score. A game that
    went past regulation was tied k-k after 60 minutes and its FINAL score is
    (k+1)-k or k-(k+1) — the overtime / shootout winner is credited one goal —
    so its regulation score is (k, k), k = the loser's final goals. Raises on
    a game that went past regulation with a final margin other than one goal,
    or one decided in regulation with a tied final: either is a data error,
    and guessing its regulation score would fit on it silently.
    """
    h = np.asarray(home_score, dtype=int)
    a = np.asarray(away_score, dtype=int)
    past = np.asarray(went_past_regulation, dtype=bool)
    if not (h.shape == a.shape == past.shape):
        raise ValueError(f"regulation_scores needs aligned arrays, got {h.shape}, {a.shape}, {past.shape}")
    bad_past = past & (np.abs(h - a) != 1)
    bad_reg = ~past & (h == a)
    if bad_past.any() or bad_reg.any():
        raise ValueError(
            f"{int(bad_past.sum())} game(s) past regulation without a one-goal final margin and "
            f"{int(bad_reg.sum())} decided in regulation with a tied final — not a regulation score"
        )
    k = np.minimum(h, a)
    return np.where(past, k, h), np.where(past, k, a)


def margin_basis(margin_buckets: Sequence[int], max_goals: int) -> np.ndarray:
    """(len(buckets), N, N) indicator of each margin bucket on the grid:
    basis[b, i, j] = 1 when |i - j| == margin_buckets[b]."""
    n = max_goals + 1
    i, j = np.indices((n, n))
    m = np.abs(i - j)
    return (
        np.stack([(m == int(b)).astype(float) for b in margin_buckets]) if len(margin_buckets) else np.zeros((0, n, n))
    )


def margin_factor_grid(margin_buckets: Sequence[int], log_factors: Sequence[float], max_goals: int) -> np.ndarray:
    """(N, N) multiplicative factor exp(θ_b(|i-j|)) — 1 outside every bucket."""
    basis = margin_basis(margin_buckets, max_goals)
    theta = np.asarray(log_factors, dtype=float)
    if theta.shape != (basis.shape[0],):
        raise ValueError(f"{len(theta)} log-factor(s) for {basis.shape[0]} margin bucket(s)")
    return np.exp(np.tensordot(theta, basis, axes=1)) if basis.shape[0] else np.ones(basis.shape[1:])


def tilt(P: np.ndarray, factor_grid: np.ndarray) -> np.ndarray:
    """P (one (N, N) matrix or a stack (..., N, N)) times the factor grid,
    renormalised to sum to 1 per matrix."""
    Q = np.asarray(P, dtype=float) * factor_grid
    return Q / Q.sum(axis=(-2, -1), keepdims=True)


def resolve_overtime(R: np.ndarray, ot_home_share: float) -> np.ndarray:
    """Regulation scoreline -> final scoreline (one matrix or a stack).

    Each tie (k, k) moves to (k+1, k) with probability ``ot_home_share`` and
    to (k, k+1) otherwise. The grid's corner tie (cap, cap) has no cell one
    goal up, so its mass (~1e-10 for NHL lambdas) moves to (cap, cap-1) /
    (cap-1, cap) instead: the final matrix stays on the served grid and
    carries no tie.
    """
    R = np.asarray(R, dtype=float)
    q = float(ot_home_share)
    F = R.copy()
    n = R.shape[-1]
    for k in range(n):
        d = R[..., k, k].copy()
        F[..., k, k] = 0.0
        if k + 1 < n:
            F[..., k + 1, k] += q * d
            F[..., k, k + 1] += (1.0 - q) * d
        else:
            F[..., k, k - 1] += q * d
            F[..., k - 1, k] += (1.0 - q) * d
    return F


@dataclass(frozen=True)
class RegulationStructure:
    """The fitted regulation-scoreline structure an NHL artifact carries
    under PAYLOAD_KEY. Built by fit_structure, read back by from_payload,
    which refuses anything malformed."""

    margin_buckets: Tuple[int, ...]
    margin_log_factors: Tuple[float, ...]
    ot_home_share: float

    def __post_init__(self) -> None:
        reason = structure_refusal(self.margin_buckets, self.margin_log_factors, self.ot_home_share)
        if reason:
            raise ValueError(f"invalid regulation structure: {reason}")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "version": STRUCTURE_VERSION,
            "target": "regulation scoreline; final = regulation + overtime/shootout goal",
            "margin_buckets": [int(b) for b in self.margin_buckets],
            "margin_log_factors": [float(t) for t in self.margin_log_factors],
            "ot_home_share": float(self.ot_home_share),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "RegulationStructure":
        """Parse an artifact's PAYLOAD_KEY value. Raises ValueError on any
        malformed payload (wrong version, mismatched lengths, a non-finite
        or out-of-range number) — a serve path must refuse it, never fall
        back to the legacy matrices silently."""
        if not isinstance(payload, dict):
            raise ValueError(f"{PAYLOAD_KEY} is {type(payload).__name__}, not an object")
        if payload.get("version") != STRUCTURE_VERSION:
            raise ValueError(f"{PAYLOAD_KEY} version {payload.get('version')!r}, expected {STRUCTURE_VERSION}")
        try:
            buckets = tuple(int(b) for b in payload["margin_buckets"])
            factors = tuple(float(t) for t in payload["margin_log_factors"])
            share = float(payload["ot_home_share"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{PAYLOAD_KEY} is malformed: {e}") from e
        return cls(margin_buckets=buckets, margin_log_factors=factors, ot_home_share=share)

    def factor_grid(self, max_goals: int) -> np.ndarray:
        return margin_factor_grid(self.margin_buckets, self.margin_log_factors, max_goals)

    def matrices(self, P_poisson: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(regulation matrix, final matrix) from the untilted Poisson grid
        build_dc_matrix(λ_home, λ_away, 0.0, max_goals) of one fixture."""
        P = np.asarray(P_poisson, dtype=float)
        R = tilt(P, self.factor_grid(P.shape[-1] - 1))
        return R, resolve_overtime(R, self.ot_home_share)


def structure_refusal(buckets: Sequence[Any], factors: Sequence[Any], ot_home_share: Any) -> Optional[str]:
    """Why these numbers are not a usable structure, or None."""
    try:
        b = [int(x) for x in buckets]
    except (TypeError, ValueError):
        return f"margin buckets {buckets!r} are not integers"
    if any(x < 0 for x in b) or any(y <= x for x, y in zip(b, b[1:])):
        return f"margin buckets {b} must be non-negative and strictly increasing"
    if len(factors) != len(b):
        return f"{len(factors)} log-factor(s) for {len(b)} margin bucket(s)"
    for t in factors:
        if not (isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)):
            return f"log-factor {t!r} is not finite"
        if abs(t) > MAX_ABS_LOG_FACTOR:
            return f"|log-factor| {abs(t):.3f} exceeds {MAX_ABS_LOG_FACTOR}"
    if not (isinstance(ot_home_share, (int, float)) and not isinstance(ot_home_share, bool)):
        return f"ot_home_share {ot_home_share!r} is not a number"
    if not (0.0 < float(ot_home_share) < 1.0):
        return f"ot_home_share {ot_home_share!r} is not in (0, 1)"
    return None


def structure_of(model: Any) -> Optional[RegulationStructure]:
    """The model's RegulationStructure, None when it carries none (a legacy
    artifact). Raises ValueError when it carries a malformed one."""
    payload = getattr(model, PAYLOAD_KEY, None)
    return None if payload is None else RegulationStructure.from_payload(payload)


# ── The joint maximum-likelihood fit ─────────────────────────────────


def _row_baselines(model: Any, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(home_baseline, away_baseline) per row, resolved as the serve path does."""
    leagues = _league_values(df)
    keys = leagues if leagues is not None else [None] * len(df)
    pairs = [model._baselines_for_league(k) for k in keys]
    return np.array([p[0] for p in pairs], dtype=float), np.array([p[1] for p in pairs], dtype=float)


def _poisson_grid(lh: np.ndarray, la: np.ndarray, max_goals: int) -> np.ndarray:
    """(n, N, N) untilted, truncated, renormalised Poisson grids — what
    build_dc_matrix(λ_home, λ_away, 0.0, max_goals) returns per row."""
    P = poisson_pmf_grid(lh, max_goals)[:, :, None] * poisson_pmf_grid(la, max_goals)[:, None, :]
    return P / P.sum(axis=(1, 2), keepdims=True)


def fit_structure(
    model: Any,
    reg_df: pd.DataFrame,
    overtime_home_win: Sequence[float],
    margin_buckets: Sequence[int],
    max_goals: int,
) -> Tuple[RegulationStructure, Dict[str, Any]]:
    """Fit θ jointly with the team strengths by weighted maximum likelihood.

    ``model`` is a PoissonMatchPredictor (DixonColesPredictor) already trained
    on ``reg_df`` — REGULATION scores in home_score / away_score — with rho 0;
    its strengths are the starting point and are REPLACED by the joint fit
    (baselines, recency weights and the strength prior are the model's own).
    ``overtime_home_win`` is 1.0 / 0.0 for each game that went past
    regulation (did the home side win it) and NaN for the rest.

    Block coordinate ascent on the weighted log-likelihood
    sum_r w_r log P_R(h_r, a_r):
      * strengths: the Poisson update with the tilted model's expectations —
        at the fixed point each team's weighted goals for / against equal
        their expectation under P_R (with the model's strength prior,
        exactly as PoissonMatchPredictor.train), attack normalised to mean 1;
      * θ: one Newton step (the log-likelihood is concave in θ: its Hessian
        is minus the weighted covariance of the bucket indicators).
    Observed scores are capped at max_goals, as the served grid is.

    Returns the structure and a result dict in the trainer's vocabulary
    (iterations — None when not converged — and the fitted vs observed mean
    REGULATION total goals under the served P_R, for fit_refusal's level
    check) plus the log-likelihood gain over the Poisson start.
    """
    buckets = tuple(int(b) for b in margin_buckets)
    hs = np.minimum(reg_df["home_score"].to_numpy(dtype=int), max_goals)
    as_ = np.minimum(reg_df["away_score"].to_numpy(dtype=int), max_goals)
    w = model._time_weights(reg_df)
    hb, ab = _row_baselines(model, reg_df)
    teams = sorted(model.team_attack)
    t_idx = {t: i for i, t in enumerate(teams)}
    hi = reg_df["home_team"].map(t_idx).to_numpy()
    ai = reg_df["away_team"].map(t_idx).to_numpy()
    if pd.isna(hi).any() or pd.isna(ai).any():
        raise ValueError("fit_structure: the frame holds a team the model was not trained on")
    hi, ai = hi.astype(np.int64), ai.astype(np.int64)
    n_teams = len(teams)
    att = np.array([model.team_attack[t] for t in teams], dtype=float)
    dfn = np.array([model.team_defense[t] for t in teams], dtype=float)
    reg = float(model.config.hyperparameters.get("regularization", 0.001))
    max_iter = int(model.config.hyperparameters.get("max_iterations", 1000))
    tol = float(model.config.hyperparameters.get("convergence_threshold", 1e-6))

    basis = margin_basis(buckets, max_goals)  # (B, N, N)
    rows = np.arange(len(hs))
    obs_b = basis[:, hs, as_].T  # (n, B) observed bucket indicators
    goals = np.arange(max_goals + 1, dtype=float)
    num_att = np.bincount(hi, weights=w * hs, minlength=n_teams) + np.bincount(ai, weights=w * as_, minlength=n_teams)
    num_def = np.bincount(hi, weights=w * as_, minlength=n_teams) + np.bincount(ai, weights=w * hs, minlength=n_teams)

    def grids(att_: np.ndarray, dfn_: np.ndarray, theta_: np.ndarray) -> np.ndarray:
        G = np.exp(np.tensordot(theta_, basis, axes=1)) if len(buckets) else 1.0
        return tilt(_poisson_grid(att_[hi] * dfn_[ai] * hb, att_[ai] * dfn_[hi] * ab, max_goals), G)

    def loglik(Q: np.ndarray) -> float:
        return float(np.sum(w * np.log(Q[rows, hs, as_])))

    theta = np.zeros(len(buckets))
    Q = grids(att, dfn, theta)
    ll_start = loglik(Q)
    converged_at: Optional[int] = None
    change = float("inf")
    for iteration in range(max_iter):
        # Strengths, from the current tilted expectations.
        e_home = (Q.sum(axis=2) * goals).sum(axis=1)
        e_away = (Q.sum(axis=1) * goals).sum(axis=1)
        e_att = np.bincount(hi, weights=w * e_home, minlength=n_teams) + np.bincount(
            ai, weights=w * e_away, minlength=n_teams
        )
        e_def = np.bincount(hi, weights=w * e_away, minlength=n_teams) + np.bincount(
            ai, weights=w * e_home, minlength=n_teams
        )
        new_att = (num_att + reg) / (e_att / att + reg)
        new_dfn = (num_def + reg) / (e_def / dfn + reg)
        new_att /= new_att.mean()
        Q = grids(new_att, new_dfn, theta)
        # theta: one Newton step at the new strengths.
        step = np.zeros(len(buckets))
        if len(buckets):
            e_b = np.einsum("rij,bij->rb", Q, basis)  # (n, B) expected indicators
            grad = (w[:, None] * (obs_b - e_b)).sum(axis=0)
            e_bb = np.einsum("rij,bij,cij->rbc", Q, basis, basis)
            cov = e_bb - e_b[:, :, None] * e_b[:, None, :]
            hess = (w[:, None, None] * cov).sum(axis=0)
            step = np.clip(np.linalg.solve(hess, grad), -_MAX_THETA_STEP, _MAX_THETA_STEP)
            theta = theta + step
            Q = grids(new_att, new_dfn, theta)
        change = float(
            max(
                np.max(np.abs(new_att - att)),
                np.max(np.abs(new_dfn - dfn)),
                float(np.max(np.abs(step))) if len(buckets) else 0.0,
            )
        )
        att, dfn = new_att, new_dfn
        if change < tol:
            converged_at = iteration + 1
            break
    if converged_at is None:
        logger.warning(
            "Regulation-structure fit did not converge in %d iterations (max change %.2e > tol %.2e)",
            max_iter,
            change,
            tol,
        )

    past = ~np.isnan(np.asarray(overtime_home_win, dtype=float))
    ot = np.asarray(overtime_home_win, dtype=float)
    if not past.any():
        raise ValueError("fit_structure: no game in the frame went past regulation — no overtime share to fit")
    ot_share = float(np.sum(w[past] * ot[past]) / np.sum(w[past]))

    model.team_attack = {t: float(att[i]) for t, i in t_idx.items()}
    model.team_defense = {t: float(dfn[i]) for t, i in t_idx.items()}
    structure = RegulationStructure(
        margin_buckets=buckets, margin_log_factors=tuple(float(t) for t in theta), ot_home_share=ot_share
    )
    i_g, j_g = np.indices(Q.shape[1:])
    total_w = float(w.sum())
    fitted_total = float(np.sum(w * (Q * (i_g + j_g)).sum(axis=(1, 2))) / total_w)
    observed_total = float(np.sum(w * (hs + as_)) / total_w)
    result = {
        "iterations": converged_at,
        "fitted_mean_total_goals": fitted_total,
        "observed_mean_total_goals": observed_total,
        "fitted_tie_rate": float(np.sum(w * Q[:, i_g == j_g].sum(axis=1)) / total_w),
        "observed_tie_rate": float(np.sum(w * (hs == as_)) / total_w),
        "loglik_gain": loglik(Q) - ll_start,
        "margin_buckets": list(buckets),
        "margin_log_factors": [float(t) for t in theta],
        "ot_home_share": ot_share,
        "games_past_regulation": int(past.sum()),
    }
    logger.info(
        "Regulation structure: converged after %s iterations; theta %s on margins %s; overtime home share %.4f; "
        "fitted/observed regulation tie rate %.4f/%.4f, mean total %.4f/%.4f; log-likelihood gain %.1f",
        converged_at,
        [round(t, 4) for t in theta],
        list(buckets),
        ot_share,
        result["fitted_tie_rate"],
        result["observed_tie_rate"],
        fitted_total,
        observed_total,
        result["loglik_gain"],
    )
    return structure, result
