"""Hyperparameter optimization with Optuna."""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from predictors.model_config import ModelType
from training.cross_validation import TimeSeriesCV

logger = logging.getLogger(__name__)

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_xgboost_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
        "random_state": 42,
    }


def get_lightgbm_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }


def get_nn_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_layers = []
    for i in range(n_layers):
        dim = trial.suggest_int(f"hidden_dim_{i}", 32, 512, log=True)
        hidden_layers.append(dim)

    return {
        "hidden_layers": hidden_layers,
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "epochs": 50,
        "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
    }


SEARCH_SPACES = {
    ModelType.XGBOOST: get_xgboost_search_space,
    ModelType.LIGHTGBM: get_lightgbm_search_space,
    ModelType.NEURAL_NETWORK: get_nn_search_space,
}


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        model_type: ModelType,
        n_trials: int = 100,
        metric: str = "log_loss",
        direction: str = "minimize",
        timeout: Optional[int] = None,
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.metric = metric
        self.direction = direction
        self.timeout = timeout
        self.study: Optional[optuna.Study] = None

    def optimize(
        self,
        objective_fn: Callable[[optuna.Trial], float],
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run optimization with a custom objective function."""
        self.study = optuna.create_study(
            study_name=study_name or f"{self.model_type.value}_optimization",
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        self.study.optimize(
            objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        logger.info(
            f"Optimization complete: best trial #{self.study.best_trial.number}, "
            f"best {self.metric}={self.study.best_trial.value:.4f}"
        )

        return {
            "best_params": self.study.best_trial.params,
            "best_score": self.study.best_trial.value,
            "n_trials_completed": len(self.study.trials),
            "study": self.study,
        }

    def get_search_space_fn(self) -> Optional[Callable]:
        """Get the search space function for this model type."""
        return SEARCH_SPACES.get(self.model_type)

    def get_param_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance from completed study."""
        if self.study is None:
            return {}
        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception:
            return {}


def make_time_series_cv_objective(
    *,
    model_class: type,
    train_df: pd.DataFrame,
    features: List[str],
    target: str,
    n_splits: int = 5,
    date_column: str = "match_date",
    metric: str = "log_loss",
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective that scores trials by time-series CV.

    Sports data is temporal: random k-fold CV leaks future information into
    the training fold (a model can learn from games that hadn't happened yet
    at training time). TimeSeriesCV enforces train_idx < val_idx by date so
    each fold's validation set is strictly in the future of its training set.

    Returns an objective function that, given a trial's suggested params,
    fits `model_class(config_from_trial)` on each CV fold and averages the
    metric. Pass the returned callable to `HyperparameterOptimizer.optimize`.
    """
    search_space = SEARCH_SPACES.get(getattr(model_class, "MODEL_TYPE", None))
    if search_space is None:
        raise ValueError(
            f"No registered search space for {model_class.__name__}. "
            "Add it to SEARCH_SPACES or write a custom objective."
        )

    cv = TimeSeriesCV(n_splits=n_splits, date_column=date_column)

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        fold_scores: List[float] = []
        for fold in cv.split(train_df):
            train_fold = fold["train"]
            val_fold = fold["val"]
            try:
                model = model_class(params)
                model.train(train_fold, val_df=val_fold, features=features, target=target)
                metrics = getattr(model, "validation_metrics", {}) or {}
                score = metrics.get(metric)
                if score is None:
                    # Pruning: if the model doesn't report the requested metric,
                    # skip this trial rather than guess.
                    raise optuna.TrialPruned()
                fold_scores.append(float(score))
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning("CV fold failed for trial #%s: %s", trial.number, exc)
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    return objective
