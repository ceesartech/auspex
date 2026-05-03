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

    MATCH_OUTCOME = "match_outcome"  # 3-class: home/draw/away
    OVER_UNDER = "over_under"  # Binary: over/under 2.5
    BTTS = "btts"  # Binary: both teams score
    CORRECT_SCORE = "correct_score"  # Multi-class
    ASIAN_HANDICAP = "asian_handicap"  # Multi-class


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
