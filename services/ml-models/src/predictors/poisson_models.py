"""Poisson and Dixon-Coles models for score-based prediction."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln  # log(k!) = gammaln(k + 1)

from .base_model import BaseModel
from .model_config import ModelConfig, PredictionTask

logger = logging.getLogger(__name__)


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function (scalar — used by callers that
    take one match at a time)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam**k * math.exp(-lam)) / math.factorial(k)


def poisson_pmf_grid(lambdas: np.ndarray, max_k: int) -> np.ndarray:
    """Vectorised Poisson PMF over a grid of lambdas × k values.

    Returns shape (len(lambdas), max_k+1). lambdas <= 0 are treated as
    point mass at k=0 to match poisson_pmf().
    """
    ks = np.arange(max_k + 1)
    lam = np.clip(lambdas, 1e-12, None)[:, None]  # (N, 1)
    log_pmf = ks * np.log(lam) - lam - gammaln(ks + 1)
    out = np.exp(log_pmf)
    # When lambda was <=0, force k=0 → 1.0, rest → 0.0
    invalid = lambdas <= 0
    if invalid.any():
        out[invalid] = 0.0
        out[invalid, 0] = 1.0
    return out


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring matches (scalar)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_tau_vec(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Vectorised Dixon-Coles correction over arrays of (x, y, lam, mu)."""
    tau = np.ones_like(lam, dtype=np.float64)
    mask_00 = (x == 0) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_10 = (x == 1) & (y == 0)
    mask_11 = (x == 1) & (y == 1)
    tau[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho
    tau[mask_01] = 1.0 + lam[mask_01] * rho
    tau[mask_10] = 1.0 + mu[mask_10] * rho
    tau[mask_11] = 1.0 - rho
    return tau


# Column names accepted as "which league is this match in", in priority
# order. The soccer training frame carries league_id (see
# utils/training_data.DEFAULT_TRAINING_QUERY) and so do the halftime /
# second-half / NHL Dixon-Coles queries in scripts/. The serve path
# (scripts/precompute_predictions.py) passes that same league_id into
# lambdas_for_match(). `league` / `league_name` are accepted as well so a
# frame built from a fixtures-style query (which selects l.name AS league)
# resolves too — train() additionally aliases the fitted baselines under
# every other league column it can see, so the two identifiers are
# interchangeable at serve time.
LEAGUE_COLUMN_CANDIDATES = ("league_id", "league", "league_name")

# Sentinel strings that a stringified NULL can produce.
_NULL_LEAGUE_KEYS = frozenset({"", "nan", "none", "null", "<na>", "nat"})

# Default recency decay for the Dixon-Coles family, per day. 0.00095/day is
# a 730-day half-life (ln 2 / 730), the winner of the over-2.5 Brier sweep
# by half-life: 180d 0.2492 | 385d 0.2468 | 730d 0.2460 | 1095d 0.2459 |
# 1825d 0.2461 | none 0.2473. Kept here (not only in the config) so an
# artifact loaded through a config that predates the key still reports the
# decay it would be refitted with.
DEFAULT_DIXON_COLES_TIME_DECAY = 0.00095


def _league_key(value: Any) -> Optional[str]:
    """Normalise one league cell to a dict key, or None when unusable."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    key = str(value).strip()
    if key.lower() in _NULL_LEAGUE_KEYS:
        return None
    return key


def _league_key_columns(df: pd.DataFrame) -> List[str]:
    """League-identifying columns present on `df`, in priority order."""
    return [column for column in LEAGUE_COLUMN_CANDIDATES if column in df.columns]


def _league_column_and_keys(df: pd.DataFrame) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """(column, per-row keys) from the first league column with real values.

    Returns (None, None) when the frame carries no usable league column —
    callers fall back to a single global baseline.
    """
    for column in _league_key_columns(df):
        keys = np.array([_league_key(value) for value in df[column].to_numpy()], dtype=object)
        if any(key is not None for key in keys):
            return column, keys
    return None, None


def _league_values(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Per-row league keys, or None when the frame carries no league column."""
    return _league_column_and_keys(df)[1]


class PoissonMatchPredictor(BaseModel):
    """Independent Poisson model for match score prediction.

    Fitting notes (the three things that are easy to get wrong here):

    * SCALE GAUGE. The model is lambda_home = attack[h] * defense[a] *
      home_baseline, so it has exactly ONE scale indeterminacy
      (attack -> c*attack, defense -> defense/c). Normalising attack alone
      fixes it. Normalising BOTH over-constrains the fit by a degree of
      freedom: it pins mean(lambda) to whatever the hard-coded baseline
      says instead of letting the MLE match observed goals, which is how
      the served soccer artifact ended up implying 2.97 goals/match
      against 2.68 actual (+10.9%). Only `attack` is normalised below;
      `defense` absorbs the level, and at the fixed point the MLE's own
      stationarity conditions make total predicted goals equal total
      observed goals.
    * BASELINES. With `per_league_baselines` on, the home/away baselines
      are fitted from the data per league (shrunk toward the global
      weighted means), instead of one global (league_avg_goals +/-
      home_advantage) constant across every competition.
    * RECENCY. With `time_decay` > 0 every match contributes
      w = exp(-time_decay * age_days) to the MLE.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.team_attack: Dict[str, float] = {}
        self.team_defense: Dict[str, float] = {}
        self.home_advantage: float = config.hyperparameters.get("home_advantage", 0.25)
        self.league_avg_goals: float = 1.35
        self.max_goals: int = config.hyperparameters.get("max_goals", 6)
        # Exponential recency weight applied inside the MLE:
        # w = exp(-time_decay * age_days). 0.0 = unweighted, which is the
        # historical behaviour and stays the default for the plain Poisson
        # and hockey-Poisson configs.
        self.time_decay: float = float(config.hyperparameters.get("time_decay", 0.0))
        # Fit home/away baselines from the data, per league, shrunk toward
        # the global weighted means. Off by default so every model that
        # doesn't opt in keeps the single (league_avg_goals +/-
        # home_advantage) baseline it was trained with.
        self.per_league_baselines: bool = bool(config.hyperparameters.get("per_league_baselines", False))
        # Shrinkage strength for those per-league baselines, in effective
        # matches: a league with `league_shrinkage` weighted matches sits
        # halfway between its own mean and the global mean.
        self.league_shrinkage: float = float(config.hyperparameters.get("league_shrinkage", 200.0))
        # Fitted baselines. None / empty means "legacy artifact, or the
        # flag was off" and the (league_avg_goals +/- home_advantage)
        # fallback applies — this is what keeps pre-change artifacts
        # loading and serving exactly as before.
        self.global_home_baseline: Optional[float] = None
        self.global_away_baseline: Optional[float] = None
        self.league_baselines: Dict[str, Tuple[float, float]] = {}
        # Distinct leagues that got their own fitted baselines — smaller
        # than len(league_baselines), which also holds name aliases. Set by
        # train(); it is fit-time reporting only and not serialised.
        self.fitted_league_count: int = 0
        # ABLATION ONLY. True restores the over-constrained pre-2026-09 gauge
        # (normalise attack AND defense to mean 1), which pins mean(lambda) to
        # the hard-coded baseline. It exists so scripts/ab_soccer_dixon_coles_refit.py
        # can score a genuine LEGACY-fit arm on the same folds and attribute the
        # result to the gauge fix rather than to "refit on a bigger corpus".
        # Nothing in production sets it; leave it False.
        self.normalise_defense: bool = bool(config.hyperparameters.get("normalise_defense", False))

    def prepare_data(self, df: pd.DataFrame, target: str, features: Optional[List[str]] = None) -> tuple:
        required_cols = ["home_team", "away_team", "home_score", "away_score"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df[required_cols].copy(), df[target].values if target in df.columns else None

    def _time_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Exponential recency weights w = exp(-time_decay * age_days),
        aged against the training frame's own max match_date.

        NOT the closed GBM lever. This is a weight inside a 4-parameter
        multiplicative Poisson MLE — it changes how much a 2009 match
        counts toward a team's attack/defense strength. It is NOT a GBM
        training-frame horizon and NOT GBM recency weighting, both of
        which were tested and closed (docs/SYSTEM_AUDIT_AND_ROADMAP.md
        section 4). The justification is that the era is not stationary:
        the home goal advantage runs 0.405 (2007) -> 0.216 (2020) ->
        0.327 (2026 ytd).

        Returns all-ones when decay is off or no usable date column
        exists — and says so loudly in the latter case, because a
        silently unweighted fit is exactly the class of failure that
        produced the constant-prior incident.
        """
        n = len(df)
        if self.time_decay <= 0:
            return np.ones(n, dtype=np.float64)
        if "match_date" not in df.columns:
            logger.error(
                "time_decay=%.5f requested but the training frame has no match_date column; fitting UNWEIGHTED",
                self.time_decay,
            )
            return np.ones(n, dtype=np.float64)
        dates = pd.to_datetime(df["match_date"], errors="coerce")
        if bool(dates.isna().all()):
            logger.error(
                "time_decay=%.5f requested but every match_date is unparseable; fitting UNWEIGHTED",
                self.time_decay,
            )
            return np.ones(n, dtype=np.float64)
        age_days = (dates.max() - dates).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        if np.isnan(age_days).any():
            # An undated row is treated as the OLDEST in the frame, never
            # the newest — an unknown date must not win the fit.
            age_days = np.where(np.isnan(age_days), float(np.nanmax(age_days)), age_days)
        return np.exp(-self.time_decay * np.clip(age_days, 0.0, None))

    def _fit_baselines(
        self,
        df: pd.DataFrame,
        weights: np.ndarray,
        home_score: np.ndarray,
        away_score: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Set league_avg_goals plus (when enabled) the global and
        per-league baselines, and return the per-match (home_baseline,
        away_baseline) arrays the MLE multiplies through.

        Per-league baselines are the league's own weighted home/away means
        shrunk toward the global weighted means with `league_shrinkage`
        effective matches of prior, so a 16-match competition stays at the
        global level while a 6,000-match one gets its own.
        """
        total_w = float(weights.sum())
        if total_w <= 0:
            raise ValueError("Time-decay weights summed to zero; cannot fit scoring baselines")
        global_home = float((weights * home_score).sum() / total_w)
        global_away = float((weights * away_score).sum() / total_w)
        self.league_avg_goals = (global_home + global_away) / 2.0

        n = len(df)
        if not self.per_league_baselines:
            # Unchanged legacy behaviour: one global constant baseline
            # derived from league_avg_goals and the configured home bump.
            self.global_home_baseline = None
            self.global_away_baseline = None
            self.league_baselines = {}
            self.fitted_league_count = 0
            return (
                np.full(n, self.league_avg_goals + self.home_advantage, dtype=np.float64),
                np.full(n, self.league_avg_goals, dtype=np.float64),
            )

        self.global_home_baseline = global_home
        self.global_away_baseline = global_away
        self.league_baselines = {}
        self.fitted_league_count = 0

        column, keys = _league_column_and_keys(df)
        if keys is None:
            logger.warning(
                "per_league_baselines is on but the frame carries none of %s; using the global "
                "weighted baselines (home=%.3f away=%.3f) for every match",
                LEAGUE_COLUMN_CANDIDATES,
                global_home,
                global_away,
            )
            return (
                np.full(n, global_home, dtype=np.float64),
                np.full(n, global_away, dtype=np.float64),
            )

        shrink = max(self.league_shrinkage, 0.0)
        grouped = (
            pd.DataFrame(
                {
                    "league": keys,
                    "w": weights,
                    "home_goals": weights * home_score,
                    "away_goals": weights * away_score,
                }
            )
            .dropna(subset=["league"])
            .groupby("league", sort=False)[["w", "home_goals", "away_goals"]]
            .sum()
        )
        for league, row in grouped.iterrows():
            denom = float(row["w"]) + shrink
            if denom <= 0:
                continue
            self.league_baselines[str(league)] = (
                (float(row["home_goals"]) + shrink * global_home) / denom,
                (float(row["away_goals"]) + shrink * global_away) / denom,
            )
        self.fitted_league_count = len(self.league_baselines)
        self._alias_league_baselines(df, column, keys)

        pairs = [self._baselines_for_league(key) for key in keys]
        home_baseline = np.array([pair[0] for pair in pairs], dtype=np.float64)
        away_baseline = np.array([pair[1] for pair in pairs], dtype=np.float64)
        logger.info(
            "Fitted per-league baselines for %d leagues (global home=%.3f away=%.3f, shrinkage=%.0f)",
            len(grouped),
            global_home,
            global_away,
            shrink,
        )
        return home_baseline, away_baseline

    def _alias_league_baselines(
        self,
        df: pd.DataFrame,
        primary_column: Optional[str],
        primary_keys: np.ndarray,
    ) -> None:
        """Also key the fitted baselines by every OTHER league column on
        the frame, so a caller that passes a league NAME resolves against
        a fit that grouped on league_id (and vice versa). Ambiguous
        aliases — one value spanning two primary keys — are skipped.
        """
        for column in _league_key_columns(df):
            if column == primary_column:
                continue
            alt = np.array([_league_key(value) for value in df[column].to_numpy()], dtype=object)
            pairs = pd.DataFrame({"alt": alt, "primary": primary_keys}).dropna()
            for alt_key, group in pairs.groupby("alt", sort=False)["primary"]:
                distinct = set(group)
                if len(distinct) != 1:
                    continue
                primary = str(distinct.pop())
                alias = str(alt_key)
                if alias in self.league_baselines or primary not in self.league_baselines:
                    continue
                self.league_baselines[alias] = self.league_baselines[primary]

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Vectorised iterative MLE.

        Equivalent to the original per-team loop with df.iterrows() but
        runs O(matches) numpy ops per iteration instead of O(teams ×
        matches) Python row accesses. Uses Jacobi updates (all teams
        from start-of-iteration values) rather than Gauss-Seidel —
        same fixed point, possibly a few more iterations to converge,
        but each iteration is ~100-1000× faster.

        BLAST RADIUS — read this before assuming a soccer change stayed in
        soccer. Two different kinds of change live in this method:

        * THREE OPT-IN KNOBS, off by default, so a model whose config does
          not set them fits exactly as before: `time_decay` (recency
          weights), `per_league_baselines` (+ `league_shrinkage`), and
          `regularization` read as a shrinkage prior in effective matches.
          Only DIXON_COLES_CONFIG (the soccer full-time Dixon-Coles) opts
          in, and it is the only model the 2026-09 A/B measured.
        * ONE UNCONDITIONAL CORRECTION: the scale gauge (normalise attack
          only; see the class docstring). It is a correctness fix to the
          shared fitting routine, not a knob, so it re-fits EVERY
          Poisson-family model on its next retrain —
          `poisson_soccer_match_result` (POISSON_CONFIG), the four
          `hockey_poisson_nhl_{ml,reg,pl,tot}` ensemble members, and the
          hockey / halftime / second-half Dixon-Coles artifacts. The
          direction is the same everywhere (it removes a positive level
          bias, so mean lambda drops to match observed goals), but the
          magnitude is model-specific and NO non-soccer number was
          measured. The level canary below prints fitted vs observed
          E[total goals] on every fit, which is how a non-soccer retrain
          can be checked without re-deriving the maths.
        """
        logger.info("Training Poisson model...")
        df = train_df.copy()

        home_score = df["home_score"].to_numpy(dtype=np.float64)
        away_score = df["away_score"].to_numpy(dtype=np.float64)

        weights = self._time_weights(df)
        home_baseline, away_baseline = self._fit_baselines(df, weights, home_score, away_score)

        # Stable integer indexing for teams; we convert back to a
        # name-keyed dict at the end so callers see the same API.
        teams_list = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        team_to_idx = {t: i for i, t in enumerate(teams_list)}
        n_teams = len(teams_list)

        home_idx = df["home_team"].map(team_to_idx).to_numpy(dtype=np.int64)
        away_idx = df["away_team"].map(team_to_idx).to_numpy(dtype=np.int64)

        # Numerators are constant across iterations — weighted sum of
        # goals scored / conceded by each team, regardless of side.
        attack_num = np.bincount(home_idx, weights=weights * home_score, minlength=n_teams) + np.bincount(
            away_idx, weights=weights * away_score, minlength=n_teams
        )
        defense_num = np.bincount(home_idx, weights=weights * away_score, minlength=n_teams) + np.bincount(
            away_idx, weights=weights * home_score, minlength=n_teams
        )

        attack = np.ones(n_teams, dtype=np.float64)
        defense = np.ones(n_teams, dtype=np.float64)

        # `regularization` is a shrinkage prior toward strength 1.0,
        # denominated in effective matches: (num + reg) / (den + reg) is
        # the posterior mean after `reg` matches' worth of league-average
        # pseudo-observations. Both num and den run to order
        # (weighted matches × goals), so a value of ~20 is a real prior
        # over a thin-history team and the old 0.001 was inert.
        reg = float(self.config.hyperparameters.get("regularization", 0.001))
        max_iter = self.config.hyperparameters.get("max_iterations", 1000)
        tol = self.config.hyperparameters.get("convergence_threshold", 1e-6)

        converged_at: Optional[int] = None
        max_change = float("inf")
        for iteration in range(max_iter):
            old_attack = attack
            old_defense = defense

            # attack_den[T] = Σ matches T plays: w × opponent_defense × side_baseline
            #   When T is home, opponent is away_idx; side_baseline = home_baseline.
            #   When T is away, opponent is home_idx; side_baseline = away_baseline.
            attack_den = np.bincount(
                home_idx, weights=weights * old_defense[away_idx] * home_baseline, minlength=n_teams
            ) + np.bincount(away_idx, weights=weights * old_defense[home_idx] * away_baseline, minlength=n_teams)
            # defense_den[T] = Σ matches T plays: w × opponent_attack × opp_side_baseline
            #   When T is home, opp is away and the opp's lambda uses away_baseline.
            #   When T is away, opp is home and the opp's lambda uses home_baseline.
            defense_den = np.bincount(
                home_idx, weights=weights * old_attack[away_idx] * away_baseline, minlength=n_teams
            ) + np.bincount(away_idx, weights=weights * old_attack[home_idx] * home_baseline, minlength=n_teams)

            attack = np.where(attack_den > 0, (attack_num + reg) / (attack_den + reg), old_attack)
            defense = np.where(defense_den > 0, (defense_num + reg) / (defense_den + reg), old_defense)

            # Fix the ONE scale indeterminacy (attack -> c*attack,
            # defense -> defense/c) by normalising attack only. Normalising
            # defense as well removes a second degree of freedom the model
            # does not have and pins mean(lambda) to the baseline instead
            # of to the data — see the class docstring. `normalise_defense`
            # restores that bug and exists purely so the A/B harness can
            # measure a legacy arm; it is False everywhere in production.
            attack /= attack.mean()
            if self.normalise_defense:
                defense /= defense.mean()

            max_change = float(max(np.max(np.abs(attack - old_attack)), np.max(np.abs(defense - old_defense))))
            if max_change < tol:
                converged_at = iteration + 1
                logger.info("Converged after %d iterations", converged_at)
                break

        if converged_at is None:
            logger.warning(
                "Poisson MLE did not converge in %d iterations (max parameter change %.2e > tol %.2e)",
                max_iter,
                max_change,
                tol,
            )

        self.team_attack = {t: float(attack[i]) for t, i in team_to_idx.items()}
        self.team_defense = {t: float(defense[i]) for t, i in team_to_idx.items()}
        self.is_fitted = True
        self.feature_names = ["home_team", "away_team"]

        # Level canary. At the MLE fixed point the stationarity conditions
        # force fitted total goals == observed total goals; a gap means the
        # fit is over-constrained (the double-normalisation bug) or did not
        # converge. Loud, because this failure is invisible downstream —
        # it just biases every derived totals market.
        total_w = float(weights.sum())
        fitted_total = float(
            np.sum(
                weights
                * (
                    attack[home_idx] * defense[away_idx] * home_baseline
                    + attack[away_idx] * defense[home_idx] * away_baseline
                )
            )
            / total_w
        )
        observed_total = float(np.sum(weights * (home_score + away_score)) / total_w)
        level_bias = fitted_total / observed_total - 1.0 if observed_total > 0 else 0.0
        if abs(level_bias) > 0.02:
            logger.warning(
                "Poisson level check: fitted E[total goals]=%.4f vs observed %.4f (%+.1f%%)",
                fitted_total,
                observed_total,
                100.0 * level_bias,
            )
        else:
            logger.info(
                "Poisson level check: fitted E[total goals]=%.4f vs observed %.4f (%+.1f%%)",
                fitted_total,
                observed_total,
                100.0 * level_bias,
            )

        validation_metrics = {}
        if val_df is not None:
            y_val = val_df["match_outcome"].values if "match_outcome" in val_df.columns else None
            if y_val is not None:
                validation_metrics = self.evaluate(val_df, y_val)

        logger.info("Poisson model trained: %d teams, league avg goals=%.2f", n_teams, self.league_avg_goals)
        return {
            "team_count": n_teams,
            "league_avg_goals": self.league_avg_goals,
            "league_count": self.fitted_league_count,
            "time_decay": self.time_decay,
            "iterations": converged_at,
            "fitted_mean_total_goals": fitted_total,
            "observed_mean_total_goals": observed_total,
            "validation_metrics": validation_metrics,
        }

    def _baselines_for_league(self, league: Any = None) -> Tuple[float, float]:
        """(home_baseline, away_baseline) for a league.

        Resolution order: the league's own fitted baselines → the global
        fitted baselines → (league_avg_goals + home_advantage,
        league_avg_goals). The last branch is what a pre-change artifact
        always takes, which is why old artifacts keep serving identically.
        """
        key = _league_key(league)
        if key is not None:
            pair = self.league_baselines.get(key)
            if pair is not None:
                return float(pair[0]), float(pair[1])
        if self.global_home_baseline is not None and self.global_away_baseline is not None:
            return float(self.global_home_baseline), float(self.global_away_baseline)
        return self.league_avg_goals + self.home_advantage, self.league_avg_goals

    def _expected_goals(self, home_team: str, away_team: str, league: Any = None) -> Tuple[float, float]:
        home_baseline, away_baseline = self._baselines_for_league(league)

        home_attack = self.team_attack.get(home_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)

        home_lambda = home_attack * away_defense * home_baseline
        away_lambda = away_attack * home_defense * away_baseline

        return home_lambda, away_lambda

    def lambdas_for_match(self, home_team: str, away_team: str, league: Any = None) -> Tuple[float, float]:
        """Public accessor for the (home_lambda, away_lambda) expected-goals
        pair. Used by the market-derivation pipeline to build a scoreline
        matrix without reaching into the private _expected_goals().

        Unseen teams fall back to attack=defense=1.0 — which, with `league`
        supplied and a per-league fit loaded, makes the prediction that
        LEAGUE's baseline rather than a single global constant. Pass the
        same identifier the model was trained on (league_id for every query
        in scripts/ and utils/training_data.py); an unknown or omitted
        league falls back to the global baseline.
        """
        return self._expected_goals(home_team, away_team, league=league)

    def _score_matrix(self, home_lambda: float, away_lambda: float) -> np.ndarray:
        mg = self.max_goals + 1
        matrix = np.zeros((mg, mg))
        for i in range(mg):
            for j in range(mg):
                matrix[i, j] = poisson_pmf(i, home_lambda) * poisson_pmf(j, away_lambda)
        return matrix

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _lambdas_for(
        self,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        leagues: Optional[np.ndarray] = None,
    ) -> tuple:
        """Vectorised expected-goals for arrays of (home_team, away_team).

        Missing teams default to attack=defense=1.0 (same as the scalar
        _expected_goals fallback). `leagues` — one league key per row, or
        None for "use the global baseline everywhere".
        """
        ha = np.array([self.team_attack.get(t, 1.0) for t in home_teams], dtype=np.float64)
        hd = np.array([self.team_defense.get(t, 1.0) for t in home_teams], dtype=np.float64)
        aa = np.array([self.team_attack.get(t, 1.0) for t in away_teams], dtype=np.float64)
        ad = np.array([self.team_defense.get(t, 1.0) for t in away_teams], dtype=np.float64)
        if leagues is None:
            home_base, away_base = self._baselines_for_league(None)
            home_baseline = np.full(len(ha), home_base, dtype=np.float64)
            away_baseline = np.full(len(ha), away_base, dtype=np.float64)
        else:
            pairs = [self._baselines_for_league(league) for league in leagues]
            home_baseline = np.array([pair[0] for pair in pairs], dtype=np.float64)
            away_baseline = np.array([pair[1] for pair in pairs], dtype=np.float64)
        home_lambda = ha * ad * home_baseline
        away_lambda = aa * hd * away_baseline
        return home_lambda, away_lambda

    def _score_tensor(self, home_lambdas: np.ndarray, away_lambdas: np.ndarray) -> np.ndarray:
        """Batched score-matrix builder. Returns shape (N, max_goals+1, max_goals+1)."""
        home_pmf = poisson_pmf_grid(home_lambdas, self.max_goals)  # (N, mg)
        away_pmf = poisson_pmf_grid(away_lambdas, self.max_goals)  # (N, mg)
        return home_pmf[:, :, None] * away_pmf[:, None, :]  # outer product per match

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        home = X["home_team"].to_numpy() if "home_team" in X.columns else np.array([""] * len(X))
        away = X["away_team"].to_numpy() if "away_team" in X.columns else np.array([""] * len(X))

        h_lam, a_lam = self._lambdas_for(home, away, _league_values(X))
        matrices = self._score_tensor(h_lam, a_lam)  # (N, mg, mg)

        # P(home win) = sum of cells where i > j (home goals > away goals).
        # P(draw) = diagonal. P(away win) = i < j.
        mg = self.max_goals + 1
        i_idx, j_idx = np.indices((mg, mg))
        home_mask = i_idx > j_idx
        draw_mask = i_idx == j_idx
        away_mask = i_idx < j_idx

        p_home = matrices[:, home_mask].sum(axis=1)
        p_draw = matrices[:, draw_mask].sum(axis=1)
        p_away = matrices[:, away_mask].sum(axis=1)

        total = p_home + p_draw + p_away
        # Avoid divide-by-zero — fall back to a neutral prior when lambdas
        # were degenerate (e.g. unseen teams with zero historical data).
        out = np.empty((len(X), 3))
        nonzero = total > 0
        out[nonzero, 0] = p_home[nonzero] / total[nonzero]
        out[nonzero, 1] = p_draw[nonzero] / total[nonzero]
        out[nonzero, 2] = p_away[nonzero] / total[nonzero]
        out[~nonzero] = [0.4, 0.3, 0.3]
        return out

    def predict_score(self, home_team: str, away_team: str, league: Any = None) -> Dict[str, Any]:
        """Predict most likely scores and match probabilities."""
        h_lam, a_lam = self._expected_goals(home_team, away_team, league=league)
        matrix = self._score_matrix(h_lam, a_lam)

        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.sum(np.diag(matrix)))
        p_away = float(np.sum(np.triu(matrix, 1)))

        # Most likely scores
        flat_idx = np.argsort(matrix.ravel())[::-1][:5]
        top_scores = []
        mg = self.max_goals + 1
        for idx in flat_idx:
            h, a = divmod(idx, mg)
            top_scores.append(
                {
                    "home_goals": int(h),
                    "away_goals": int(a),
                    "probability": float(matrix[h, a]),
                }
            )

        # Over/Under 2.5
        p_over_25 = float(sum(matrix[i, j] for i in range(mg) for j in range(mg) if i + j > 2))

        # BTTS
        p_btts = float(sum(matrix[i, j] for i in range(1, mg) for j in range(1, mg)))

        return {
            "home_lambda": h_lam,
            "away_lambda": a_lam,
            "probabilities": {
                "home_win": p_home,
                "draw": p_draw,
                "away_win": p_away,
            },
            "top_scores": top_scores,
            "over_2_5": p_over_25,
            "btts": p_btts,
        }

    def _artifact_payload(self) -> Dict[str, Any]:
        """The serialised model state shared by Poisson and Dixon-Coles.

        `league_avg_goals` and `home_advantage` are kept even when
        per-league baselines are fitted: they are the fallback a reader
        uses when a league is unknown, and keeping them means an artifact
        written here still loads in an older checkout.
        """
        return {
            "team_attack": self.team_attack,
            "team_defense": self.team_defense,
            "home_advantage": self.home_advantage,
            "league_avg_goals": self.league_avg_goals,
            "max_goals": self.max_goals,
            "time_decay": self.time_decay,
            "per_league_baselines": self.per_league_baselines,
            "league_shrinkage": self.league_shrinkage,
            "global_home_baseline": self.global_home_baseline,
            "global_away_baseline": self.global_away_baseline,
            "league_baselines": {k: [v[0], v[1]] for k, v in self.league_baselines.items()},
            "config": self.config.to_dict(),
        }

    def _load_artifact_payload(self, data: Dict[str, Any]) -> None:
        """Restore shared state. Every key added by the level/baseline fix
        is read with a default, so an artifact saved BEFORE that change
        loads and serves exactly as it did (empty league_baselines and
        None global baselines route _baselines_for_league() to the legacy
        league_avg_goals +/- home_advantage branch)."""
        self.team_attack = data["team_attack"]
        self.team_defense = data["team_defense"]
        self.home_advantage = data["home_advantage"]
        self.league_avg_goals = data["league_avg_goals"]
        self.max_goals = data["max_goals"]
        self.time_decay = float(data.get("time_decay", self.time_decay))
        self.per_league_baselines = bool(data.get("per_league_baselines", self.per_league_baselines))
        self.league_shrinkage = float(data.get("league_shrinkage", self.league_shrinkage))
        global_home = data.get("global_home_baseline")
        global_away = data.get("global_away_baseline")
        self.global_home_baseline = float(global_home) if global_home is not None else None
        self.global_away_baseline = float(global_away) if global_away is not None else None
        self.league_baselines = {
            str(key): (float(pair[0]), float(pair[1])) for key, pair in (data.get("league_baselines") or {}).items()
        }
        self.is_fitted = True

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self._artifact_payload(), f, indent=2)
        logger.info(f"Poisson model saved to {path}")

    def load(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        self._load_artifact_payload(data)
        logger.info(f"Poisson model loaded from {path}")


class HockeyPoissonPredictor(PoissonMatchPredictor):
    """NHL-tuned Poisson model that emits the joint goal distribution
    once and derives task-specific probabilities (moneyline, regulation
    3-way, puck-line, total) from it.

    The classification models in Phase 3a-3c learned moneyline well but
    plateaued for puck-line and total — those tasks ask about MARGIN
    and TOTAL GOALS, which are naturally framed as functions of the
    joint goal distribution rather than direct classifications. By
    fitting one Poisson and integrating it differently per task, we get
    a more principled signal for the harder markets and a useful 4th
    base model for the moneyline ensemble.

    Inherits the vectorized MLE and team-attack/defense state from
    PoissonMatchPredictor. The only overrides are:
      * Hockey-tuned defaults (lower home_advantage, higher max_goals).
      * predict_proba branches on config.prediction_task and derives
        the appropriate (N, num_classes) array from the joint matrices.
      * train() reads the target from config.target_column so NHL
        target columns (nhl_moneyline, nhl_regulation, ...) work in
        place of soccer's match_outcome.

    Output shape per task:
      * NHL_MONEYLINE   (N, 2)  [P(home win incl OT/SO), P(away win)]
      * NHL_REGULATION  (N, 3)  [P(home reg), P(reg tie), P(away reg)]
      * NHL_PUCK_LINE   (N, 2)  [P(home covers -1.5), P(does not)]
      * NHL_TOTAL       (N, 2)  [P(over 5.5), P(under 5.5)]

    Approximation: OT/SO goals aren't separately modeled. For moneyline
    we split the regulation-tie mass 50/50 between home and away (OT/SO
    is approximately a coin flip at the league level). For totals,
    we predict regulation totals; the ~22% of games that go to OT/SO
    add +1 to the eventual total in NHL but we don't apply a correction
    here — the calibration step downstream absorbs the small bias.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # NHL home advantage is smaller than soccer's. League home win
        # rate is ~55% — translating that to a log-Poisson home bump:
        # roughly 0.10-0.15 added to home_lambda. Soccer's 0.25 is too
        # aggressive for hockey.
        self.home_advantage = config.hyperparameters.get("home_advantage", 0.15)
        # NHL scores cluster 0-6 per team; max_goals=10 covers the long
        # tail (10+ goal games are <0.1%). Soccer's 6 would underweight
        # blowouts.
        self.max_goals = config.hyperparameters.get("max_goals", 10)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Call parent training but skip its match_outcome-hardcoded
        # validation block (val_df=None), then run our own validation
        # against the task-specific target column.
        result = super().train(train_df, val_df=None, **kwargs)

        if val_df is not None:
            target = self.config.target_column
            if target in val_df.columns:
                try:
                    y_val = val_df[target].values
                    result["validation_metrics"] = self.evaluate(val_df, y_val)
                except Exception as e:
                    logger.warning("HockeyPoisson evaluate(%s) failed: %s", target, e)

        return result

    def _joint_matrices(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the (N, mg, mg) joint goal distribution for the
        rows in X. Shared by every task-specific predict_proba branch."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        home = X["home_team"].to_numpy() if "home_team" in X.columns else np.array([""] * len(X))
        away = X["away_team"].to_numpy() if "away_team" in X.columns else np.array([""] * len(X))
        h_lam, a_lam = self._lambdas_for(home, away, _league_values(X))
        return self._score_tensor(h_lam, a_lam)  # (N, mg, mg)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        task = self.config.prediction_task
        matrices = self._joint_matrices(X)
        mg = self.max_goals + 1
        i_idx, j_idx = np.indices((mg, mg))

        if task == PredictionTask.NHL_REGULATION:
            # 3-class: [home reg win, regulation tie, away reg win]
            home_mask = i_idx > j_idx
            tie_mask = i_idx == j_idx
            away_mask = i_idx < j_idx
            p_home = matrices[:, home_mask].sum(axis=1)
            p_tie = matrices[:, tie_mask].sum(axis=1)
            p_away = matrices[:, away_mask].sum(axis=1)
            total = p_home + p_tie + p_away
            out = np.empty((len(X), 3))
            nonzero = total > 0
            out[nonzero, 0] = p_home[nonzero] / total[nonzero]
            out[nonzero, 1] = p_tie[nonzero] / total[nonzero]
            out[nonzero, 2] = p_away[nonzero] / total[nonzero]
            out[~nonzero] = [0.42, 0.22, 0.36]  # league marginal prior
            return out

        if task == PredictionTask.NHL_MONEYLINE:
            # 2-class: [home win incl OT/SO, away win incl OT/SO].
            # Regulation ties split ~50/50 in OT/SO at the league level.
            home_mask = i_idx > j_idx
            tie_mask = i_idx == j_idx
            away_mask = i_idx < j_idx
            p_home_reg = matrices[:, home_mask].sum(axis=1)
            p_tie = matrices[:, tie_mask].sum(axis=1)
            p_away_reg = matrices[:, away_mask].sum(axis=1)
            p_home_ml = p_home_reg + 0.5 * p_tie
            p_away_ml = p_away_reg + 0.5 * p_tie
            total = p_home_ml + p_away_ml
            out = np.empty((len(X), 2))
            nonzero = total > 0
            out[nonzero, 0] = p_home_ml[nonzero] / total[nonzero]
            out[nonzero, 1] = p_away_ml[nonzero] / total[nonzero]
            out[~nonzero] = [0.55, 0.45]
            return out

        if task == PredictionTask.NHL_PUCK_LINE:
            # 2-class: [home covers -1.5 (margin >= 2), does not].
            # Note: OT/SO games end at margin=1, which doesn't cover —
            # so using regulation-goal masses is approximately right
            # (some OT games would have ended margin>=2 in regulation
            # if not for the goalie pull etc., but the effect is small).
            cover_mask = (i_idx - j_idx) >= 2
            p_cover = matrices[:, cover_mask].sum(axis=1)
            grid_total = matrices.sum(axis=(1, 2))
            out = np.empty((len(X), 2))
            nonzero = grid_total > 0
            out[nonzero, 0] = p_cover[nonzero] / grid_total[nonzero]
            out[nonzero, 1] = 1.0 - out[nonzero, 0]
            out[~nonzero] = [0.28, 0.72]
            return out

        if task == PredictionTask.NHL_TOTAL:
            # 2-class: [over 5.5 (total >= 6), under 5.5]. Per NHL
            # convention the SO winner gets +1 toward the total, but
            # we predict regulation totals here and let downstream
            # calibration absorb the small bias.
            over_mask = (i_idx + j_idx) >= 6
            p_over = matrices[:, over_mask].sum(axis=1)
            grid_total = matrices.sum(axis=(1, 2))
            out = np.empty((len(X), 2))
            nonzero = grid_total > 0
            out[nonzero, 0] = p_over[nonzero] / grid_total[nonzero]
            out[nonzero, 1] = 1.0 - out[nonzero, 0]
            out[~nonzero] = [0.55, 0.45]
            return out

        raise ValueError(f"HockeyPoissonPredictor doesn't support prediction_task={task}")


class DixonColesPredictor(PoissonMatchPredictor):
    """Dixon-Coles model — extends Poisson with low-score correction."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.rho: float = config.hyperparameters.get("rho_init", -0.13)
        # time_decay itself lives on PoissonMatchPredictor (it is applied
        # inside the shared MLE); the Dixon-Coles default is the 730-day
        # half-life that won the half-life sweep. See DIXON_COLES_CONFIG.
        self.time_decay = float(config.hyperparameters.get("time_decay", DEFAULT_DIXON_COLES_TIME_DECAY))

    def _score_matrix(self, home_lambda: float, away_lambda: float) -> np.ndarray:
        mg = self.max_goals + 1
        matrix = np.zeros((mg, mg))
        for i in range(mg):
            for j in range(mg):
                base = poisson_pmf(i, home_lambda) * poisson_pmf(j, away_lambda)
                tau = dixon_coles_tau(i, j, home_lambda, away_lambda, self.rho)
                matrix[i, j] = base * tau
        # Renormalize
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Train base Poisson parameters first.
        result = super().train(train_df, val_df=None, **kwargs)

        # Optimise rho using MLE on the full training set. Vectorised:
        # we precompute lambdas + score arrays + the per-(score) base
        # Poisson PMFs once, then each call to neg_log_likelihood is
        # just a vectorised tau computation + sum of logs.
        logger.info("Optimizing Dixon-Coles rho parameter...")

        df = train_df.copy()
        home_teams = df["home_team"].to_numpy()
        away_teams = df["away_team"].to_numpy()
        hs = df["home_score"].astype(int).to_numpy()
        as_ = df["away_score"].astype(int).to_numpy()

        # Same recency weights and same per-league baselines the
        # attack/defense MLE just used — rho is estimated on the same
        # effective sample, not on an unweighted 2007-2026 frame.
        weights = self._time_weights(df)
        h_lam, a_lam = self._lambdas_for(home_teams, away_teams, _league_values(df))

        # Base Poisson PMF per match — independent of rho, so compute once.
        log_h = hs * np.log(np.clip(h_lam, 1e-12, None)) - h_lam - gammaln(hs + 1)
        log_a = as_ * np.log(np.clip(a_lam, 1e-12, None)) - a_lam - gammaln(as_ + 1)
        log_base = log_h + log_a  # log( P_h(hs) * P_a(as) )

        def neg_log_likelihood(rho_arr: np.ndarray) -> float:
            rho = float(rho_arr[0])
            tau = dixon_coles_tau_vec(hs, as_, h_lam, a_lam, rho)
            # tau can go negative for invalid rho regions — penalise.
            valid = tau > 0
            nll = -float(np.sum(weights[valid] * (log_base[valid] + np.log(tau[valid]))))
            # Add a stiff penalty for any non-positive tau so the
            # optimiser steers back into the valid region. 100 per
            # match matches the original scalar implementation; it is
            # weighted like the likelihood so an old invalid match does
            # not outvote a recent valid one.
            nll += 100.0 * float(np.sum(weights[~valid]))
            return nll

        res = minimize(
            neg_log_likelihood,
            x0=[self.rho],
            method="Nelder-Mead",
            options={"maxiter": 100},
        )
        self.rho = float(res.x[0])
        logger.info("Optimized rho: %.4f", self.rho)

        if val_df is not None and "match_outcome" in val_df.columns:
            result["validation_metrics"] = self.evaluate(val_df, val_df["match_outcome"].values)

        result["rho"] = self.rho
        return result

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._artifact_payload()
        data["rho"] = self.rho
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Dixon-Coles model saved to {path}")

    def load(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        self._load_artifact_payload(data)
        self.rho = data["rho"]
        logger.info(f"Dixon-Coles model loaded from {path}")
