"""Poisson and Dixon-Coles models for score-based prediction."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base_model import BaseModel
from .model_config import ModelConfig

logger = logging.getLogger(__name__)


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam**k * math.exp(-lam)) / math.factorial(k)


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring matches."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


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
        logger.info("Training Poisson model...")

        df = train_df.copy()
        self.league_avg_goals = (df["home_score"].mean() + df["away_score"].mean()) / 2.0

        teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        # Initialize attack/defense strengths
        for team in teams:
            self.team_attack[team] = 1.0
            self.team_defense[team] = 1.0

        # Iterative estimation
        reg = self.config.hyperparameters.get("regularization", 0.001)
        max_iter = self.config.hyperparameters.get("max_iterations", 1000)
        tol = self.config.hyperparameters.get("convergence_threshold", 1e-6)

        for iteration in range(max_iter):
            old_attack = dict(self.team_attack)
            old_defense = dict(self.team_defense)

            for team in teams:
                # Attack: avg goals scored / opponent defense
                home_matches = df[df["home_team"] == team]
                away_matches = df[df["away_team"] == team]

                attack_num = 0.0
                attack_den = 0.0
                defense_num = 0.0
                defense_den = 0.0

                for _, row in home_matches.iterrows():
                    opp = row["away_team"]
                    attack_num += row["home_score"]
                    attack_den += self.team_defense[opp] * (self.league_avg_goals + self.home_advantage)
                    defense_num += row["away_score"]
                    defense_den += self.team_attack[opp] * self.league_avg_goals

                for _, row in away_matches.iterrows():
                    opp = row["home_team"]
                    attack_num += row["away_score"]
                    attack_den += self.team_defense[opp] * self.league_avg_goals
                    defense_num += row["home_score"]
                    defense_den += self.team_attack[opp] * (self.league_avg_goals + self.home_advantage)

                if attack_den > 0:
                    self.team_attack[team] = (attack_num + reg) / (attack_den + reg)
                if defense_den > 0:
                    self.team_defense[team] = (defense_num + reg) / (defense_den + reg)

            # Normalize
            avg_attack = np.mean(list(self.team_attack.values()))
            avg_defense = np.mean(list(self.team_defense.values()))
            for team in teams:
                self.team_attack[team] /= avg_attack
                self.team_defense[team] /= avg_defense

            # Check convergence
            max_change = max(
                max(abs(self.team_attack[t] - old_attack[t]) for t in teams),
                max(abs(self.team_defense[t] - old_defense[t]) for t in teams),
            )
            if max_change < tol:
                logger.info(f"Converged after {iteration + 1} iterations")
                break

        self.is_fitted = True
        self.feature_names = ["home_team", "away_team"]

        validation_metrics = {}
        if val_df is not None:
            y_val = val_df["match_outcome"].values if "match_outcome" in val_df.columns else None
            if y_val is not None:
                validation_metrics = self.evaluate(val_df, y_val)

        logger.info(f"Poisson model trained: {len(teams)} teams, " f"league avg goals={self.league_avg_goals:.2f}")
        return {
            "team_count": len(teams),
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

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        results = []
        for _, row in X.iterrows():
            home_team = row.get("home_team", "")
            away_team = row.get("away_team", "")
            h_lam, a_lam = self._expected_goals(home_team, away_team)
            matrix = self._score_matrix(h_lam, a_lam)

            p_home = np.sum(np.tril(matrix, -1))
            p_draw = np.sum(np.diag(matrix))
            p_away = np.sum(np.triu(matrix, 1))

            total = p_home + p_draw + p_away
            if total > 0:
                results.append([p_home / total, p_draw / total, p_away / total])
            else:
                results.append([0.4, 0.3, 0.3])

        return np.array(results)

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
        # Train base Poisson parameters first
        result = super().train(train_df, val_df=None, **kwargs)

        # Optimize rho using MLE
        logger.info("Optimizing Dixon-Coles rho parameter...")

        df = train_df.copy()

        def neg_log_likelihood(rho_arr):
            rho = rho_arr[0]
            nll = 0.0
            for _, row in df.iterrows():
                h_lam, a_lam = self._expected_goals(row["home_team"], row["away_team"])
                hs = int(row["home_score"])
                as_ = int(row["away_score"])

                p_base = poisson_pmf(hs, h_lam) * poisson_pmf(as_, a_lam)
                tau = dixon_coles_tau(hs, as_, h_lam, a_lam, rho)
                p = p_base * tau

                if p > 0:
                    nll -= math.log(p)
                else:
                    nll += 100  # Penalty for zero probability

            return nll

        res = minimize(
            neg_log_likelihood,
            x0=[self.rho],
            method="Nelder-Mead",
            options={"maxiter": 100},
        )
        self.rho = float(res.x[0])
        logger.info(f"Optimized rho: {self.rho:.4f}")

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
