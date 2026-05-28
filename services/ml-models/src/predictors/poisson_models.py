"""Poisson and Dixon-Coles models for score-based prediction."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln  # log(k!) = gammaln(k + 1)

from .base_model import BaseModel
from .model_config import ModelConfig

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


class PoissonMatchPredictor(BaseModel):
    """Independent Poisson model for match score prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.team_attack: Dict[str, float] = {}
        self.team_defense: Dict[str, float] = {}
        self.home_advantage: float = config.hyperparameters.get("home_advantage", 0.25)
        self.league_avg_goals: float = 1.35
        self.max_goals: int = config.hyperparameters.get("max_goals", 6)

    def prepare_data(self, df: pd.DataFrame, target: str, features: Optional[List[str]] = None) -> tuple:
        required_cols = ["home_team", "away_team", "home_score", "away_score"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df[required_cols].copy(), df[target].values if target in df.columns else None

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
        """
        logger.info("Training Poisson model...")
        df = train_df.copy()

        self.league_avg_goals = (df["home_score"].mean() + df["away_score"].mean()) / 2.0
        home_baseline = self.league_avg_goals + self.home_advantage
        away_baseline = self.league_avg_goals

        # Stable integer indexing for teams; we convert back to a
        # name-keyed dict at the end so callers see the same API.
        teams_list = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        team_to_idx = {t: i for i, t in enumerate(teams_list)}
        n_teams = len(teams_list)

        home_idx = df["home_team"].map(team_to_idx).to_numpy(dtype=np.int64)
        away_idx = df["away_team"].map(team_to_idx).to_numpy(dtype=np.int64)
        home_score = df["home_score"].to_numpy(dtype=np.float64)
        away_score = df["away_score"].to_numpy(dtype=np.float64)

        # Numerators are constant across iterations — sum of goals
        # scored / conceded by each team, regardless of side.
        attack_num = np.bincount(home_idx, weights=home_score, minlength=n_teams) + np.bincount(
            away_idx, weights=away_score, minlength=n_teams
        )
        defense_num = np.bincount(home_idx, weights=away_score, minlength=n_teams) + np.bincount(
            away_idx, weights=home_score, minlength=n_teams
        )

        attack = np.ones(n_teams, dtype=np.float64)
        defense = np.ones(n_teams, dtype=np.float64)

        reg = self.config.hyperparameters.get("regularization", 0.001)
        max_iter = self.config.hyperparameters.get("max_iterations", 1000)
        tol = self.config.hyperparameters.get("convergence_threshold", 1e-6)

        for iteration in range(max_iter):
            old_attack = attack
            old_defense = defense

            # attack_den[T] = Σ matches T plays: opponent_defense × side_baseline
            #   When T is home, opponent is away_idx; side_baseline = home_baseline.
            #   When T is away, opponent is home_idx; side_baseline = away_baseline.
            attack_den = np.bincount(
                home_idx, weights=old_defense[away_idx] * home_baseline, minlength=n_teams
            ) + np.bincount(away_idx, weights=old_defense[home_idx] * away_baseline, minlength=n_teams)
            # defense_den[T] = Σ matches T plays: opponent_attack × opp_side_baseline
            #   When T is home, opp is away and the opp's lambda uses away_baseline.
            #   When T is away, opp is home and the opp's lambda uses home_baseline.
            defense_den = np.bincount(
                home_idx, weights=old_attack[away_idx] * away_baseline, minlength=n_teams
            ) + np.bincount(away_idx, weights=old_attack[home_idx] * home_baseline, minlength=n_teams)

            attack = np.where(attack_den > 0, (attack_num + reg) / (attack_den + reg), old_attack)
            defense = np.where(defense_den > 0, (defense_num + reg) / (defense_den + reg), old_defense)

            # Normalise to mean = 1 to remove the scaling indeterminacy.
            attack /= attack.mean()
            defense /= defense.mean()

            max_change = float(max(np.max(np.abs(attack - old_attack)), np.max(np.abs(defense - old_defense))))
            if max_change < tol:
                logger.info("Converged after %d iterations", iteration + 1)
                break

        self.team_attack = {t: float(attack[i]) for t, i in team_to_idx.items()}
        self.team_defense = {t: float(defense[i]) for t, i in team_to_idx.items()}
        self.is_fitted = True
        self.feature_names = ["home_team", "away_team"]

        validation_metrics = {}
        if val_df is not None:
            y_val = val_df["match_outcome"].values if "match_outcome" in val_df.columns else None
            if y_val is not None:
                validation_metrics = self.evaluate(val_df, y_val)

        logger.info("Poisson model trained: %d teams, league avg goals=%.2f", n_teams, self.league_avg_goals)
        return {
            "team_count": n_teams,
            "league_avg_goals": self.league_avg_goals,
            "validation_metrics": validation_metrics,
        }

    def _expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        home_attack = self.team_attack.get(home_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)

        home_lambda = home_attack * away_defense * (self.league_avg_goals + self.home_advantage)
        away_lambda = away_attack * home_defense * self.league_avg_goals

        return home_lambda, away_lambda

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

    def _lambdas_for(self, home_teams: np.ndarray, away_teams: np.ndarray) -> tuple:
        """Vectorised expected-goals for arrays of (home_team, away_team).

        Missing teams default to attack=defense=1.0 (same as the scalar
        _expected_goals fallback).
        """
        ha = np.array([self.team_attack.get(t, 1.0) for t in home_teams])
        hd = np.array([self.team_defense.get(t, 1.0) for t in home_teams])
        aa = np.array([self.team_attack.get(t, 1.0) for t in away_teams])
        ad = np.array([self.team_defense.get(t, 1.0) for t in away_teams])
        home_lambda = ha * ad * (self.league_avg_goals + self.home_advantage)
        away_lambda = aa * hd * self.league_avg_goals
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

        h_lam, a_lam = self._lambdas_for(home, away)
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

    def predict_score(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Predict most likely scores and match probabilities."""
        h_lam, a_lam = self._expected_goals(home_team, away_team)
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

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "team_attack": self.team_attack,
            "team_defense": self.team_defense,
            "home_advantage": self.home_advantage,
            "league_avg_goals": self.league_avg_goals,
            "max_goals": self.max_goals,
            "config": self.config.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Poisson model saved to {path}")

    def load(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        self.team_attack = data["team_attack"]
        self.team_defense = data["team_defense"]
        self.home_advantage = data["home_advantage"]
        self.league_avg_goals = data["league_avg_goals"]
        self.max_goals = data["max_goals"]
        self.is_fitted = True
        logger.info(f"Poisson model loaded from {path}")


class DixonColesPredictor(PoissonMatchPredictor):
    """Dixon-Coles model — extends Poisson with low-score correction."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.rho: float = config.hyperparameters.get("rho_init", -0.13)
        self.time_decay: float = config.hyperparameters.get("time_decay", 0.0018)

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

        h_lam, a_lam = self._lambdas_for(home_teams, away_teams)

        # Base Poisson PMF per match — independent of rho, so compute once.
        log_h = hs * np.log(np.clip(h_lam, 1e-12, None)) - h_lam - gammaln(hs + 1)
        log_a = as_ * np.log(np.clip(a_lam, 1e-12, None)) - a_lam - gammaln(as_ + 1)
        log_base = log_h + log_a  # log( P_h(hs) * P_a(as) )

        def neg_log_likelihood(rho_arr: np.ndarray) -> float:
            rho = float(rho_arr[0])
            tau = dixon_coles_tau_vec(hs, as_, h_lam, a_lam, rho)
            # tau can go negative for invalid rho regions — penalise.
            valid = tau > 0
            nll = -np.sum(log_base[valid] + np.log(tau[valid]))
            # Add a stiff penalty for any non-positive tau so the
            # optimiser steers back into the valid region. 100 per
            # match matches the original scalar implementation.
            nll += 100.0 * int(np.count_nonzero(~valid))
            return float(nll)

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

        data = {
            "team_attack": self.team_attack,
            "team_defense": self.team_defense,
            "home_advantage": self.home_advantage,
            "league_avg_goals": self.league_avg_goals,
            "max_goals": self.max_goals,
            "rho": self.rho,
            "time_decay": self.time_decay,
            "config": self.config.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Dixon-Coles model saved to {path}")

    def load(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        self.team_attack = data["team_attack"]
        self.team_defense = data["team_defense"]
        self.home_advantage = data["home_advantage"]
        self.league_avg_goals = data["league_avg_goals"]
        self.max_goals = data["max_goals"]
        self.rho = data["rho"]
        self.time_decay = data.get("time_decay", 0.0018)
        self.is_fitted = True
        logger.info(f"Dixon-Coles model loaded from {path}")
