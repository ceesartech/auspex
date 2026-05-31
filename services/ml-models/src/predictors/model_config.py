"""Model configuration system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ModelType(Enum):
    """Model types."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    POISSON = "poisson"
    DIXON_COLES = "dixon_coles"
    ENSEMBLE = "ensemble"


class PredictionTask(Enum):
    """Prediction tasks."""

    MATCH_OUTCOME = "match_outcome"  # 3-class: home/draw/away (soccer)
    OVER_UNDER = "over_under"  # Binary: over/under 2.5 (soccer)
    BTTS = "btts"  # Binary: both teams score (soccer)
    CORRECT_SCORE = "correct_score"  # Multi-class (soccer)
    ASIAN_HANDICAP = "asian_handicap"  # Multi-class (soccer)
    # NHL tasks. All modeled as 2-class softmax (num_class=2, NOT binary
    # objective) so they pass 2D probability arrays through the ensemble
    # — the ensemble's optimizer indexes proba[:, y] which requires 2D.
    NHL_MONEYLINE = "nhl_moneyline"  # 2-class: home/away game winner incl. OT/SO
    NHL_REGULATION = "nhl_regulation"  # 3-class: home reg / tie / away reg
    NHL_PUCK_LINE = "nhl_puck_line"  # 2-class: home covers -1.5 / not
    NHL_TOTAL = "nhl_total"  # 2-class: over 5.5 / under 5.5


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    model_type: ModelType
    prediction_task: PredictionTask
    version: str
    hyperparameters: Dict[str, Any]
    features: List[str]
    target_column: str
    loss_function: str
    metrics: List[str]
    training_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "prediction_task": self.prediction_task.value,
            "version": self.version,
            "hyperparameters": self.hyperparameters,
            "features": self.features,
            "target_column": self.target_column,
            "loss_function": self.loss_function,
            "metrics": self.metrics,
            "training_config": self.training_config,
        }


# ============= PREDEFINED MODEL CONFIGURATIONS =============

XGBOOST_MATCH_OUTCOME = ModelConfig(
    name="xgboost_match_outcome",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="match_outcome",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "verbose": 100,
    },
)

LIGHTGBM_OVER_UNDER = ModelConfig(
    name="lightgbm_over_under",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.OVER_UNDER,
    version="1.0.0",
    hyperparameters={
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="over_under_2_5",
    loss_function="binary",
    metrics=["accuracy", "log_loss", "roc_auc"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

LIGHTGBM_MATCH_OUTCOME = ModelConfig(
    name="lightgbm_match_outcome",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="match_outcome",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_CONFIG = ModelConfig(
    name="neural_network_match_outcome",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [256, 128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="match_outcome",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={
        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,
        "verbose": 1,
    },
)

POISSON_CONFIG = ModelConfig(
    name="poisson_goals",
    model_type=ModelType.POISSON,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "max_goals": 6,
        "home_advantage": 0.25,
        "regularization": 0.001,
        "max_iterations": 1000,
        "convergence_threshold": 1e-6,
    },
    features=[],
    target_column="match_outcome",
    loss_function="poisson_nll",
    metrics=["accuracy", "log_loss", "ranked_probability_score"],
    training_config={},
)

DIXON_COLES_CONFIG = ModelConfig(
    name="dixon_coles",
    model_type=ModelType.DIXON_COLES,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "max_goals": 6,
        "rho_init": -0.13,
        "time_decay": 0.0018,
        "max_iterations": 500,
        "convergence_threshold": 1e-5,
    },
    features=[],
    target_column="match_outcome",
    loss_function="dixon_coles_nll",
    metrics=["accuracy", "log_loss", "ranked_probability_score"],
    training_config={},
)

ENSEMBLE_CONFIG = ModelConfig(
    name="ensemble_match_outcome",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="match_outcome",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

MODEL_CONFIGS = {
    "match_outcome_xgboost": XGBOOST_MATCH_OUTCOME,
    "match_outcome_lightgbm": LIGHTGBM_MATCH_OUTCOME,
    "over_under_lightgbm": LIGHTGBM_OVER_UNDER,
    "match_outcome_nn": NEURAL_NETWORK_CONFIG,
    "poisson_goals": POISSON_CONFIG,
    "dixon_coles": DIXON_COLES_CONFIG,
    "ensemble_match_outcome": ENSEMBLE_CONFIG,
}


# ============= NHL MODEL CONFIGURATIONS =============
#
# Hyperparameters intentionally close to the soccer baselines so the
# first NHL training run inherits proven defaults; tune per-task once
# we have holdout numbers. Two structural differences from soccer:
#   * num_class=2 (not 3) for moneyline / puck-line / totals.
#   * objective="multi:softprob" / "multiclass" (NOT "binary") so the
#     model emits 2D (N, 2) probability arrays — the EnsemblePredictor
#     optimizer indexes proba[:, y], which requires 2D input.

XGBOOST_NHL_MONEYLINE = ModelConfig(
    name="xgboost_nhl_moneyline",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 2,
        "max_depth": 6,  # shallower than soccer — less feature interaction
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "verbose": 100,
    },
)

LIGHTGBM_NHL_MONEYLINE = ModelConfig(
    name="lightgbm_nhl_moneyline",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 2,
        "boosting_type": "gbdt",
        "num_leaves": 48,  # smaller than soccer's 64
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_MONEYLINE = ModelConfig(
    name="neural_network_nhl_moneyline",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [128, 64, 32],  # smaller than soccer — fewer features
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={
        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,
        "verbose": 1,
    },
)

ENSEMBLE_NHL_MONEYLINE = ModelConfig(
    name="ensemble_nhl_moneyline",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_MONEYLINE_CONFIGS = {
    "xgboost_nhl_ml": XGBOOST_NHL_MONEYLINE,
    "lightgbm_nhl_ml": LIGHTGBM_NHL_MONEYLINE,
    "neural_network_nhl_ml": NEURAL_NETWORK_NHL_MONEYLINE,
    "ensemble_nhl_ml": ENSEMBLE_NHL_MONEYLINE,
}


# ── NHL REGULATION 3-WAY (60-minute outcome) ─────────────────────────
#
# Three classes: 0=home reg win, 1=regulation tie (game went to OT/SO),
# 2=away reg win. Target column derived from matches.metadata->>'regulation_winner'
# (string 'home' | 'tie' | 'away'), mapped through a CASE in the query so the
# frame already has the integer class label. NHL regulation distribution from
# recent seasons:
#   * home reg win  ~42%
#   * tie (OT/SO)   ~22%
#   * away reg win  ~36%
# Class imbalance is mild — softmax handles it without explicit reweighting.
#
# Hyperparameters mirror the moneyline configs except num_class=3 and (for NN)
# the slightly wider hidden layers — 3-class softmax can usually exploit a
# bit more capacity than 2-class.

XGBOOST_NHL_REGULATION = ModelConfig(
    name="xgboost_nhl_regulation",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "verbose": 100,
    },
)

LIGHTGBM_NHL_REGULATION = ModelConfig(
    name="lightgbm_nhl_regulation",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "num_leaves": 48,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_REGULATION = ModelConfig(
    name="neural_network_nhl_regulation",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [192, 96, 48],  # slightly wider than moneyline — 3-class room
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={
        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,
        "verbose": 1,
    },
)

ENSEMBLE_NHL_REGULATION = ModelConfig(
    name="ensemble_nhl_regulation",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_REGULATION_CONFIGS = {
    "xgboost_nhl_reg": XGBOOST_NHL_REGULATION,
    "lightgbm_nhl_reg": LIGHTGBM_NHL_REGULATION,
    "neural_network_nhl_reg": NEURAL_NETWORK_NHL_REGULATION,
    "ensemble_nhl_reg": ENSEMBLE_NHL_REGULATION,
}


# ── NHL PUCK LINE (home covers -1.5) ─────────────────────────────────
#
# 2-class binary classifier framed as 2-class softmax (same reasoning
# as NHL_MONEYLINE — ensemble needs 2D proba). Target derived from
# final score margin INCLUDING OT/SO goals (puck-line bets settle on
# the full game's final score, not regulation alone):
#   * 0 = home covers -1.5 (home_score - away_score >= 2)
#   * 1 = home does NOT cover (home wins by 1 OR any away win)
# Class distribution is roughly 50/50 because NHL margins cluster
# tightly around 1-2 goals. Hyperparameters cloned from moneyline.

XGBOOST_NHL_PUCK_LINE = ModelConfig(
    name="xgboost_nhl_puck_line",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 2,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "eval_metric": "mlogloss", "verbose": 100},
)

LIGHTGBM_NHL_PUCK_LINE = ModelConfig(
    name="lightgbm_nhl_puck_line",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 2,
        "boosting_type": "gbdt",
        "num_leaves": 48,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_PUCK_LINE = ModelConfig(
    name="neural_network_nhl_puck_line",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={"early_stopping_patience": 15, "reduce_lr_patience": 5, "verbose": 1},
)

ENSEMBLE_NHL_PUCK_LINE = ModelConfig(
    name="ensemble_nhl_puck_line",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_PUCK_LINE_CONFIGS = {
    "xgboost_nhl_pl": XGBOOST_NHL_PUCK_LINE,
    "lightgbm_nhl_pl": LIGHTGBM_NHL_PUCK_LINE,
    "neural_network_nhl_pl": NEURAL_NETWORK_NHL_PUCK_LINE,
    "ensemble_nhl_pl": ENSEMBLE_NHL_PUCK_LINE,
}


# ── NHL TOTAL (over 5.5) ─────────────────────────────────────────────
#
# 2-class binary classifier for the canonical 5.5-goal NHL total. As
# with puck line, the bet settles on the full game total INCLUDING
# OT/SO goals (NHL convention: shootout winner is credited with +1
# goal toward the total). Target:
#   * 0 = over  5.5  (home_score + away_score >= 6)
#   * 1 = under 5.5  (home_score + away_score <= 5)
# League average NHL total is ~6.0, so the line is moderately tilted
# toward over but the class split is close enough to balanced that we
# don't need explicit class weighting.

XGBOOST_NHL_TOTAL = ModelConfig(
    name="xgboost_nhl_total",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 2,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_total",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "eval_metric": "mlogloss", "verbose": 100},
)

LIGHTGBM_NHL_TOTAL = ModelConfig(
    name="lightgbm_nhl_total",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 2,
        "boosting_type": "gbdt",
        "num_leaves": 48,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_total",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_TOTAL = ModelConfig(
    name="neural_network_nhl_total",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_total",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={"early_stopping_patience": 15, "reduce_lr_patience": 5, "verbose": 1},
)

ENSEMBLE_NHL_TOTAL = ModelConfig(
    name="ensemble_nhl_total",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_total",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_TOTAL_CONFIGS = {
    "xgboost_nhl_tot": XGBOOST_NHL_TOTAL,
    "lightgbm_nhl_tot": LIGHTGBM_NHL_TOTAL,
    "neural_network_nhl_tot": NEURAL_NETWORK_NHL_TOTAL,
    "ensemble_nhl_tot": ENSEMBLE_NHL_TOTAL,
}


# ── Hockey-Poisson configs (Phase 3d) ────────────────────────────────
#
# One HockeyPoissonPredictor fits per task; each emits a task-specific
# probability array derived from the same joint goal distribution.
# Hyperparameters are shared across all four task variants because the
# underlying MLE is the same — only the predict_proba derivation differs.
#
# Hockey home_advantage 0.15 (NHL home win rate ~55%) vs soccer's 0.25
# (soccer home win rate ~45-50%). max_goals=10 covers NHL's long tail.

_HOCKEY_POISSON_HYPERPARAMS = {
    "max_goals": 10,
    "home_advantage": 0.15,
    "regularization": 0.001,
    "max_iterations": 1000,
    "convergence_threshold": 1e-6,
}


def _hockey_poisson_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_type=ModelType.POISSON,
        prediction_task=task,
        version="1.0.0",
        hyperparameters=dict(_HOCKEY_POISSON_HYPERPARAMS),
        features=[],
        target_column=target,
        loss_function="poisson_nll",
        metrics=["accuracy", "log_loss", "brier_score"],
        training_config={},
    )


HOCKEY_POISSON_NHL_MONEYLINE = _hockey_poisson_config(
    "hockey_poisson_nhl_ml", PredictionTask.NHL_MONEYLINE, "nhl_moneyline"
)
HOCKEY_POISSON_NHL_REGULATION = _hockey_poisson_config(
    "hockey_poisson_nhl_reg", PredictionTask.NHL_REGULATION, "nhl_regulation"
)
HOCKEY_POISSON_NHL_PUCK_LINE = _hockey_poisson_config(
    "hockey_poisson_nhl_pl", PredictionTask.NHL_PUCK_LINE, "nhl_puck_line"
)
HOCKEY_POISSON_NHL_TOTAL = _hockey_poisson_config("hockey_poisson_nhl_tot", PredictionTask.NHL_TOTAL, "nhl_total")
